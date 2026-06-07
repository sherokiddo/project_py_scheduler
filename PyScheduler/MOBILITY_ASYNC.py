"""
#------------------------------------------------------------------------------
# Модуль: MOBILITY_ASYNC - Асинхронный расчет моделей мобильности UE
#------------------------------------------------------------------------------
# Назначение:
#   Выносит вычислительное ядро MOBILITY_MODEL.py в отдельные worker-процессы.
#   Worker-процессы заранее считают mobility-step'ы и публикуют immutable snapshots.
#   Основной процесс симуляции остается единственным владельцем живых UE-объектов
#   и только применяет готовые snapshots перед расчетом канала.
#
# Зачем так:
#   - нет гонок данных: worker не трогает UserEquipment из основной симуляции;
#   - можно считать мобильность вперед всей симуляции;
#   - при промахе кэша нет скрытого расхождения траекторий: strict mode падает,
#     non-strict mode требует shutdown/restart provider'а из live UE state;
#   - изменения в проекте локальны: UE_MODULE.py + SIMULATION_MANAGER.py.
#
# Версия: 1.0.0
# Дата последнего изменения: 2026-06-07
# Автор: Македон Никита
# Версия Python Kernel: 3.12.9
#
# v1.0.0 - 2026-06-07:
# Автор изменений: Македон Никита
# - Новый модуль асинхронного расчёта моделей мобильности в отдельном worker-процессе.
# - Добавлен ahead-cache immutable MobilitySnapshot с привязкой к step_idx.
# - Реализованы strict/non-strict режимы; non-strict fallback выполняется через
#  shutdown/restart provider из live UE state.
# - Живые UserEquipment не передаются в worker, что исключает гонки данных по состоянию UE.
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class MobilitySnapshot:
    """
    Immutable-снимок результата мобильности для одного UE на одном mobility-step.

    step_idx считается не по каждому TTI, а по mobility_update_interval:
        step_idx = current_time // mobility_update_interval
    """

    ue_id: int
    step_idx: int
    position: Tuple[float, float]
    velocity: float
    direction: float


@dataclass(frozen=True, slots=True)
class _WorkerUEState:
    """Минимальное состояние UE, нужное моделям мобильности в worker-процессе."""

    ue_id: int
    position: Tuple[float, float]
    velocity: float
    direction: float
    velocity_min: float
    velocity_max: float


@dataclass(frozen=True, slots=True)
class _WorkerModelSpec:
    """
    Сериализуемое описание модели мобильности.

    Важно: сюда не попадает живой UserEquipment из основной симуляции.
    Передается только состояние самой модели и легковесное состояние UE.
    """

    ue_state: _WorkerUEState
    model_class_name: str
    model_state: Dict[str, Any]


def _export_model_state(model: Any) -> Dict[str, Any]:
    """
    Получить worker-state модели через явный контракт MobilityInterface.

    Старый _safe_copy_model_state был спрятан в MOBILITY_ASYNC и знал слишком
    много о внутренностях моделей. Теперь предпочтительный контракт находится
    рядом с моделями: model.export_worker_state(). Fallback оставлен для тестовых
    или внешних моделей, которые пока не наследуются от MobilityInterface.
    """
    if hasattr(model, "export_worker_state"):
        return model.export_worker_state()

    state: Dict[str, Any] = {}
    for key, value in getattr(model, "__dict__", {}).items():
        if key == "ue":
            continue
        try:
            state[key] = copy.deepcopy(value)
        except Exception:
            continue
    return state


def build_worker_specs(users: Iterable[Any]) -> List[_WorkerModelSpec]:
    """
    Собирает сериализуемые specs по текущим UE.

    Вызывать из main process после настройки UE и их mobility_model. Функция
    нужна также для restart/resync provider'а после non-strict fallback.
    """
    specs: List[_WorkerModelSpec] = []

    for ue in users:
        mobility_model = getattr(ue, "mobility_model", None)
        if mobility_model is None:
            continue

        ue_state = _WorkerUEState(
            ue_id=int(ue.UE_ID),
            position=tuple(ue.position),
            velocity=float(getattr(ue, "velocity", 0.0)),
            direction=float(getattr(ue, "direction", 0.0)),
            velocity_min=float(getattr(ue, "velocity_min", 0.0)),
            velocity_max=float(getattr(ue, "velocity_max", 0.0)),
        )

        specs.append(
            _WorkerModelSpec(
                ue_state=ue_state,
                model_class_name=mobility_model.__class__.__name__,
                model_state=_export_model_state(mobility_model),
            )
        )

    return specs


def _rebuild_model_from_spec(spec: _WorkerModelSpec) -> Tuple[int, Any, Any]:
    """
    Восстанавливает модель мобильности в worker-процессе.

    Создаем отдельный ue_proxy. Модель будет читать/обновлять только его.
    """
    import MOBILITY_MODEL

    cls = getattr(MOBILITY_MODEL, spec.model_class_name)

    ue_proxy = SimpleNamespace(
        UE_ID=spec.ue_state.ue_id,
        position=tuple(spec.ue_state.position),
        velocity=spec.ue_state.velocity,
        direction=spec.ue_state.direction,
        velocity_min=spec.ue_state.velocity_min,
        velocity_max=spec.ue_state.velocity_max,
    )

    if hasattr(cls, "import_worker_state"):
        model = cls.import_worker_state(ue_proxy, spec.model_state)
    else:
        # Fallback только для внешних/тестовых моделей. Для штатных моделей
        # используется явный контракт MobilityInterface.import_worker_state().
        model = cls.__new__(cls)
        model.__dict__.update(copy.deepcopy(spec.model_state))
        model.ue = ue_proxy

    return spec.ue_state.ue_id, ue_proxy, model


def _partition_specs(specs: List[_WorkerModelSpec], workers: int) -> List[List[_WorkerModelSpec]]:
    """Стабильно закрепить UE за mobility-worker по ue_id % workers."""
    worker_count = max(1, min(int(workers), len(specs)))
    buckets: List[List[_WorkerModelSpec]] = [[] for _ in range(worker_count)]
    for spec in specs:
        buckets[spec.ue_state.ue_id % worker_count].append(spec)
    return buckets


def _mobility_worker_loop(
    worker_id: int,
    specs: List[_WorkerModelSpec],
    mobility_interval_ms: int,
    start_step: int,
    max_step: int,
    out_queue: mp.Queue,
    stop_event: mp.Event,
    seed: Optional[int] = None,
) -> None:
    """
    Worker-процесс: считает mobility snapshots вперед и публикует батчами.

    Публикация идет батчем на один step:
        (worker_id, step_idx, [MobilitySnapshot, ...])
    """
    try:
        if seed is not None:
            import numpy as np
            np.random.seed(seed + int(worker_id))

        models: Dict[int, Tuple[Any, Any]] = {}
        for spec in specs:
            ue_id, ue_proxy, model = _rebuild_model_from_spec(spec)
            models[ue_id] = (ue_proxy, model)

        for step_idx in range(int(start_step), int(max_step) + 1):
            if stop_event.is_set():
                break

            batch: List[MobilitySnapshot] = []

            for ue_id, (ue_proxy, model) in models.items():
                new_pos, new_vel, new_dir = model.update(time_ms=mobility_interval_ms)

                # Важно: обновляем proxy, иначе следующий шаг модели будет считать
                # от старой позиции/скорости/направления.
                ue_proxy.position = tuple(new_pos)
                ue_proxy.velocity = float(new_vel)
                ue_proxy.direction = float(new_dir)

                batch.append(
                    MobilitySnapshot(
                        ue_id=ue_id,
                        step_idx=step_idx,
                        position=tuple(new_pos),
                        velocity=float(new_vel),
                        direction=float(new_dir),
                    )
                )

            # Если main process не успевает читать, worker слегка притормаживает.
            # Это защищает память от бесконечного роста очереди.
            while not stop_event.is_set():
                try:
                    out_queue.put((int(worker_id), int(step_idx), batch), timeout=0.1)
                    break
                except queue.Full:
                    continue

    except BaseException as exc:
        out_queue.put(
            (
                int(worker_id),
                "__error__",
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )


class MobilityAsyncProvider:
    """
    Управляет worker-процессами для расчета мобильности и кэшем snapshots.

    Типичный жизненный цикл:
        provider = MobilityAsyncProvider(...)
        provider.start(ue_collection.GET_ALL_USERS())
        ...
        batch = provider.get_batch(step_idx)
        ...
        provider.shutdown()

    Важно:
        Provider не изменяет UE напрямую. Он только отдает MobilitySnapshot.

    Race-safety policy:
        * strict_snapshots=True: если batch для step_idx не готов, get_batch()
          возвращает None, а UECollection обязан поднять RuntimeError.
        * strict_snapshots=False: UECollection может сделать sync fallback, но
          сначала должен остановить provider, а затем перезапустить его из live
          UE state через restart_from_users(..., start_step=step_idx+1).
    """

    def __init__(
        self,
        mobility_interval_ms: int,
        sim_duration_ms: int,
        *,
        workers: int = 1,
        prefetch_steps: int = 16,
        cache_steps: int = 64,
        snapshot_timeout_ms: int = 2,
        strict_snapshots: bool = True,
        restart_on_fallback: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        if mobility_interval_ms <= 0:
            raise ValueError("mobility_interval_ms должен быть > 0")
        if sim_duration_ms <= 0:
            raise ValueError("sim_duration_ms должен быть > 0")
        if workers <= 0:
            raise ValueError("workers должен быть > 0")
        if workers != 1:
            raise ValueError(
                "async mobility workers > 1 пока зарезервирован. "
                "Используйте workers=1: один producer-процесс уже считает mobility "
                "вперед симуляции без гонок данных. Масштабирование mobility на "
                "несколько producer'ов лучше добавлять отдельным этапом после "
                "стабилизации сериализации/benchmark'ов."
            )

        self.mobility_interval_ms = int(mobility_interval_ms)
        self.sim_duration_ms = int(sim_duration_ms)
        self.workers = int(workers)
        self.prefetch_steps = int(max(1, prefetch_steps))
        self.cache_steps = int(max(2, cache_steps))
        self.snapshot_timeout_ms = int(max(0, snapshot_timeout_ms))
        self.strict_snapshots = bool(strict_snapshots)
        self.restart_on_fallback = bool(restart_on_fallback)
        self.seed = seed
        self.verbose = verbose

        self._queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._processes: List[mp.Process] = []

        self._cache: Dict[int, Dict[int, MobilitySnapshot]] = {}
        self._ready_steps: Set[int] = set()
        self._partial_cache: Dict[int, Dict[int, MobilitySnapshot]] = {}
        self._partial_workers: Dict[int, Set[int]] = {}
        self._active_worker_ids: Set[int] = set()
        self._last_error: Optional[Dict[str, str]] = None
        self._last_requested_step = -1
        self._start_step = 0
        self._max_step = max(0, self.sim_duration_ms // self.mobility_interval_ms)

        self.cache_hits = 0
        self.cache_misses = 0
        self.steps_received = 0
        self.late_steps = 0
        self.fallback_steps = 0
        self.provider_restarts = 0
        self.worker_errors = 0
        self.missing_steps: List[int] = []

        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def last_error(self) -> Optional[Dict[str, str]]:
        return self._last_error

    def start(self, users: Iterable[Any], *, start_step: int = 0) -> None:
        if self._started:
            return

        specs = build_worker_specs(users)
        if not specs:
            if self.verbose:
                print("[MOBILITY_ASYNC] No UE with mobility_model found; async provider disabled.")
            return

        self._start_step = int(max(0, start_step))
        self._max_step = max(self._start_step, self.sim_duration_ms // self.mobility_interval_ms)

        self._cache.clear()
        self._ready_steps.clear()
        self._partial_cache.clear()
        self._partial_workers.clear()
        self._active_worker_ids.clear()
        self._last_error = None

        buckets = _partition_specs(specs, self.workers)
        self._queue = mp.Queue(maxsize=max(self.prefetch_steps, len(buckets)))
        self._stop_event = mp.Event()

        for worker_id, bucket in enumerate(buckets):
            if not bucket:
                continue
            proc = mp.Process(
                target=_mobility_worker_loop,
                args=(
                    worker_id,
                    bucket,
                    self.mobility_interval_ms,
                    self._start_step,
                    self._max_step,
                    self._queue,
                    self._stop_event,
                    self.seed,
                ),
                daemon=True,
                name=f"MobilityAsyncWorker-{worker_id}",
            )
            proc.start()
            self._processes.append(proc)
            self._active_worker_ids.add(worker_id)

        self._started = True

        if self.verbose:
            print(
                f"[MOBILITY_ASYNC] Started {len(self._processes)} worker(s) for {len(specs)} UE, "
                f"mobility_interval={self.mobility_interval_ms} ms, "
                f"steps={self._start_step}..{self._max_step}, strict={self.strict_snapshots}"
            )

    def restart_from_users(self, users: Iterable[Any], *, start_step: int) -> None:
        """
        Полный resync provider'а из live UE state.

        Использовать после non-strict sync fallback: сначала main process применяет
        UPD_POSITION() к живым UE, потом provider стартует с step_idx+1. Так worker
        больше не продолжает считать траекторию из старой параллельной реальности.
        """
        self.shutdown()
        self.provider_restarts += 1
        self.start(users, start_step=int(start_step))

    def _store_worker_batch(self, worker_id: int, step_idx: int, payload: List[MobilitySnapshot]) -> None:
        if step_idx < self._last_requested_step:
            self.late_steps += 1

        step_cache = self._partial_cache.setdefault(step_idx, {})
        for snap in payload:
            step_cache[int(snap.ue_id)] = snap

        workers = self._partial_workers.setdefault(step_idx, set())
        workers.add(int(worker_id))

        if self._active_worker_ids and workers >= self._active_worker_ids:
            self._cache[step_idx] = dict(step_cache)
            self._ready_steps.add(step_idx)
            self.steps_received += 1
            self._partial_cache.pop(step_idx, None)
            self._partial_workers.pop(step_idx, None)
            self._trim_cache(current_step=step_idx)

    def _drain_ready(self, *, block: bool = False, timeout_s: float = 0.0) -> None:
        if not self._queue:
            return

        first = True
        while True:
            try:
                if block and first:
                    item = self._queue.get(timeout=timeout_s)
                else:
                    item = self._queue.get_nowait()
            except queue.Empty:
                return

            first = False

            worker_id, step_idx, payload = item
            if step_idx == "__error__":
                self._last_error = payload
                self.worker_errors += 1
                if self.verbose:
                    print("[MOBILITY_ASYNC] Worker error:", payload.get("error"))
                    print(payload.get("traceback"))
                return

            self._store_worker_batch(int(worker_id), int(step_idx), payload)

    def _trim_cache(self, current_step: int) -> None:
        min_keep = current_step - self.cache_steps
        for old_step in list(self._cache.keys()):
            if old_step < min_keep:
                self._cache.pop(old_step, None)
                self._ready_steps.discard(old_step)
        for old_step in list(self._partial_cache.keys()):
            if old_step < min_keep:
                self._partial_cache.pop(old_step, None)
                self._partial_workers.pop(old_step, None)

    def _record_miss(self, step_idx: int) -> None:
        self.cache_misses += 1
        self.missing_steps.append(int(step_idx))

    def get_snapshot(self, ue_id: int, step_idx: int) -> Optional[MobilitySnapshot]:
        """
        Получить snapshot одного UE. Для производительности в UECollection лучше
        использовать get_batch(step_idx), чтобы не дергать Queue на каждый UE.
        """
        batch = self.get_batch(step_idx)
        if batch is None:
            return None
        snap = batch.get(int(ue_id))
        if snap is None:
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        return snap

    def get_batch(self, step_idx: int) -> Optional[Dict[int, MobilitySnapshot]]:
        """
        Получить все snapshots для одного mobility-step.

        Возвращает None только когда batch не готов или worker упал. Дальнейшая
        политика задается UECollection: strict -> RuntimeError, non-strict ->
        shutdown + sync fallback + restart_from_users(step_idx + 1).
        """
        if not self._started:
            return None

        step_idx = int(step_idx)
        self._last_requested_step = max(self._last_requested_step, step_idx)

        self._drain_ready(block=False)

        if step_idx not in self._ready_steps and self.snapshot_timeout_ms > 0:
            self._drain_ready(
                block=True,
                timeout_s=self.snapshot_timeout_ms / 1000.0,
            )

        if self._last_error is not None:
            self._record_miss(step_idx)
            return None

        batch = self._cache.get(step_idx)
        if batch is None:
            self._record_miss(step_idx)
        else:
            self.cache_hits += len(batch)
        return batch

    def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        for proc in self._processes:
            if proc.is_alive():
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)

        if self._queue is not None:
            try:
                self._queue.close()
                self._queue.join_thread()
            except Exception:
                pass

        self._started = False
        self._processes.clear()
        self._active_worker_ids.clear()
        self._queue = None
        self._stop_event = None

        if self.verbose:
            print(
                f"[MOBILITY_ASYNC] Worker(s) stopped. "
                f"steps_received={self.steps_received}, "
                f"cache_hits={self.cache_hits}, cache_misses={self.cache_misses}, "
                f"late_steps={self.late_steps}, fallback_steps={self.fallback_steps}, "
                f"provider_restarts={self.provider_restarts}"
            )

    def __enter__(self) -> "MobilityAsyncProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
