"""
#------------------------------------------------------------------------------
# Модуль: MOBILITY_ASYNC - Асинхронный расчет моделей мобильности UE
#------------------------------------------------------------------------------
# Назначение:
#   Выносит вычислительное ядро MOBILITY_MODEL.py в отдельный процесс.
#   Worker-процесс заранее считает mobility-step'ы и публикует immutable snapshots.
#   Основной процесс симуляции остается единственным владельцем живых UE-объектов
#   и только применяет готовые snapshots перед расчетом канала.
#
# Зачем так:
#   - нет гонок данных: worker не трогает UserEquipment из основной симуляции;
#   - можно считать мобильность вперед всей симуляции;
#   - старый синхронный режим остается fallback'ом;
#   - изменения в проекте локальны: UE_MODULE.py + SIMULATION_MANAGER.py.
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _safe_copy_model_state(model: Any) -> Dict[str, Any]:
    """
    Копирует внутреннее состояние модели без ссылки на живой UE.

    Модели мобильности хранят состояние вроде:
        pause_time, pause_timer, destination, is_first_move,
        mean_velocity, mean_direction, current_velocity, current_direction,
        bs_x/bs_y для DiagonalWalk и т.п.

    Поле 'ue' намеренно удаляем: в worker будет отдельный легковесный ue_proxy.
    """
    state = {}
    for key, value in getattr(model, "__dict__", {}).items():
        if key == "ue":
            continue

        # Не тащим тяжелые/живые объекты. Для мобильности нужны скаляры, tuple, bool.
        try:
            state[key] = copy.deepcopy(value)
        except Exception:
            # Если вдруг встретится несериализуемый объект, оставляем как есть только
            # при успешной pickle-сериализации; иначе пропускаем.
            try:
                import pickle
                pickle.dumps(value)
                state[key] = value
            except Exception:
                continue
    return state


def build_worker_specs(users: Iterable[Any]) -> List[_WorkerModelSpec]:
    """
    Собирает сериализуемые specs по текущим UE.

    Вызывать из main process после настройки UE и их mobility_model.
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
                model_state=_safe_copy_model_state(mobility_model),
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

    model = cls.__new__(cls)
    model.__dict__.update(copy.deepcopy(spec.model_state))
    model.ue = ue_proxy

    return spec.ue_state.ue_id, ue_proxy, model


def _mobility_worker_loop(
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
        (step_idx, [MobilitySnapshot, ...])
    """
    try:
        if seed is not None:
            import numpy as np
            np.random.seed(seed)

        models: Dict[int, Tuple[Any, Any]] = {}
        for spec in specs:
            ue_id, ue_proxy, model = _rebuild_model_from_spec(spec)
            models[ue_id] = (ue_proxy, model)

        for step_idx in range(start_step, max_step + 1):
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
                    out_queue.put((step_idx, batch), timeout=0.1)
                    break
                except queue.Full:
                    continue

    except BaseException as exc:
        out_queue.put(
            (
                "__error__",
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )


class MobilityAsyncProvider:
    """
    Управляет отдельным процессом для расчета мобильности и кэшем snapshots.

    Типичный жизненный цикл:
        provider = MobilityAsyncProvider(...)
        provider.start(ue_collection.GET_ALL_USERS())
        ...
        snap = provider.get_snapshot(ue_id, step_idx)
        ...
        provider.shutdown()

    Важно:
        Provider не изменяет UE напрямую. Он только отдает MobilitySnapshot.
    """

    def __init__(
        self,
        mobility_interval_ms: int,
        sim_duration_ms: int,
        *,
        prefetch_steps: int = 16,
        cache_steps: int = 64,
        snapshot_timeout_ms: int = 2,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        if mobility_interval_ms <= 0:
            raise ValueError("mobility_interval_ms должен быть > 0")
        if sim_duration_ms <= 0:
            raise ValueError("sim_duration_ms должен быть > 0")

        self.mobility_interval_ms = int(mobility_interval_ms)
        self.sim_duration_ms = int(sim_duration_ms)
        self.prefetch_steps = int(max(1, prefetch_steps))
        self.cache_steps = int(max(2, cache_steps))
        self.snapshot_timeout_ms = int(max(0, snapshot_timeout_ms))
        self.seed = seed
        self.verbose = verbose

        self._queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._process: Optional[mp.Process] = None

        self._cache: Dict[int, Dict[int, MobilitySnapshot]] = {}
        self._ready_steps = set()
        self._last_error: Optional[Dict[str, str]] = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.steps_received = 0

        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self, users: Iterable[Any]) -> None:
        if self._started:
            return

        specs = build_worker_specs(users)
        if not specs:
            if self.verbose:
                print("[MOBILITY_ASYNC] No UE with mobility_model found; async provider disabled.")
            return

        max_step = max(0, self.sim_duration_ms // self.mobility_interval_ms)

        # Очередь ограничиваем по prefetch, чтобы worker не заливал память бесконечно.
        self._queue = mp.Queue(maxsize=self.prefetch_steps)
        self._stop_event = mp.Event()

        self._process = mp.Process(
            target=_mobility_worker_loop,
            args=(
                specs,
                self.mobility_interval_ms,
                0,
                max_step,
                self._queue,
                self._stop_event,
                self.seed,
            ),
            daemon=True,
            name="MobilityAsyncWorker",
        )
        self._process.start()
        self._started = True

        if self.verbose:
            print(
                f"[MOBILITY_ASYNC] Started worker for {len(specs)} UE, "
                f"mobility_interval={self.mobility_interval_ms} ms, max_step={max_step}"
            )

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

            step_idx, payload = item
            if step_idx == "__error__":
                self._last_error = payload
                if self.verbose:
                    print("[MOBILITY_ASYNC] Worker error:", payload.get("error"))
                return

            self._cache[int(step_idx)] = {snap.ue_id: snap for snap in payload}
            self._ready_steps.add(int(step_idx))
            self.steps_received += 1
            self._trim_cache(current_step=int(step_idx))

    def _trim_cache(self, current_step: int) -> None:
        min_keep = current_step - self.cache_steps
        for old_step in list(self._cache.keys()):
            if old_step < min_keep:
                self._cache.pop(old_step, None)
                self._ready_steps.discard(old_step)

    def get_snapshot(self, ue_id: int, step_idx: int) -> Optional[MobilitySnapshot]:
        """
        Получить snapshot. Возвращает None, если snapshot не успел подготовиться.

        Основная симуляция может в этом случае:
            - либо кратко подождать,
            - либо сделать fallback на синхронный UPD_POSITION().
        """
        if not self._started:
            return None

        self._drain_ready(block=False)

        if step_idx not in self._ready_steps and self.snapshot_timeout_ms > 0:
            self._drain_ready(
                block=True,
                timeout_s=self.snapshot_timeout_ms / 1000.0,
            )

        step_cache = self._cache.get(step_idx)
        if step_cache is None:
            self.cache_misses += 1
            return None

        snap = step_cache.get(int(ue_id))
        if snap is None:
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        return snap

    def get_batch(self, step_idx: int) -> Optional[Dict[int, MobilitySnapshot]]:
        """
        Получить все snapshots для одного mobility-step.
        """
        if not self._started:
            return None

        self._drain_ready(block=False)

        if step_idx not in self._ready_steps and self.snapshot_timeout_ms > 0:
            self._drain_ready(
                block=True,
                timeout_s=self.snapshot_timeout_ms / 1000.0,
            )

        batch = self._cache.get(step_idx)
        if batch is None:
            self.cache_misses += 1
        else:
            self.cache_hits += len(batch)
        return batch

    def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        self._started = False

        if self.verbose:
            print(
                f"[MOBILITY_ASYNC] Worker stopped. "
                f"steps_received={self.steps_received}, "
                f"cache_hits={self.cache_hits}, cache_misses={self.cache_misses}"
            )

    def __enter__(self) -> "MobilityAsyncProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
