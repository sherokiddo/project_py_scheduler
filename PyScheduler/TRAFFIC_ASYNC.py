"""
#------------------------------------------------------------------------------
# Модуль: TRAFFIC_ASYNC - Асинхронная генерация legacy traffic snapshots
#------------------------------------------------------------------------------
# Назначение:
#   Выносит наиболее тяжелую фазу SimpleGenerator.generate_packets() в отдельные
#   worker-процессы. Main process остается единственным владельцем живых буферов
#   BaseStation и только применяет готовые immutable TrafficSnapshot.
#
# Race-safety:
#   - worker не трогает BaseStation/UE/buffers;
#   - каждый worker владеет своей копией SimpleGenerator для закрепленных UE;
#   - UE закрепляются за worker по ue_id % workers, поэтому stateful-модели
#     OnOff/MMPP не прыгают между процессами;
#   - sync fallback намеренно не используется: иначе main и worker разойдутся
#     по внутреннему состоянию traffic model. В non-strict режиме provider
#     просто ждет готовый batch.
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class TrafficSnapshot:
    """Immutable-снимок сгенерированных пакетов для одного UE на одном TTI."""

    ue_id: int
    step_idx: int
    packets: Tuple[Any, ...]


def _build_worker_generator(generator: Any, ue_ids: Iterable[int]) -> Any:
    """
    Собрать копию SimpleGenerator только для UE конкретного worker'а.

    В worker не передается live-ссылка на self.traffic_gen из main process.
    Каждая модель копируется отдельно, чтобы stateful-модели жили только внутри
    своего worker'а.
    """
    gen = generator.__class__()
    selected = set(int(ue_id) for ue_id in ue_ids)

    if hasattr(generator, "models") and hasattr(gen, "models"):
        for ue_id in selected:
            if ue_id in generator.models:
                gen.models[int(ue_id)] = copy.deepcopy(generator.models[ue_id])

    # Статистика worker'а не является source of truth для main process, но пусть
    # остается валидной внутри generator.generate_packets().
    if hasattr(gen, "_total_packets_generated"):
        gen._total_packets_generated = 0
    if hasattr(gen, "_packets_per_ue"):
        gen._packets_per_ue = {}

    return gen


def _traffic_worker_loop(
    worker_id: int,
    generator: Any,
    ue_ids: List[int],
    update_interval_ms: int,
    start_step: int,
    max_step: int,
    out_queue: mp.Queue,
    stop_event: mp.Event,
    seed: Optional[int] = None,
) -> None:
    """
    Worker-процесс: генерирует traffic snapshots вперед по TTI.

    Публикация идет батчем на один step:
        (worker_id, step_idx, [TrafficSnapshot, ...])
    Пустые UE не отправляются отдельными snapshots, чтобы уменьшить IPC.
    """
    try:
        if seed is not None:
            import numpy as np

            np.random.seed(int(seed) + int(worker_id))

        ue_ids = [int(ue_id) for ue_id in ue_ids]
        update_interval_ms = int(update_interval_ms)

        for step_idx in range(int(start_step), int(max_step) + 1):
            if stop_event.is_set():
                break

            batch: List[TrafficSnapshot] = []
            current_time = int(step_idx)

            for ue_id in ue_ids:
                packets = generator.generate_packets(
                    ue_id=ue_id,
                    current_time=current_time,
                    update_interval=update_interval_ms,
                )
                if packets:
                    batch.append(
                        TrafficSnapshot(
                            ue_id=ue_id,
                            step_idx=current_time,
                            packets=tuple(packets),
                        )
                    )

            while not stop_event.is_set():
                try:
                    out_queue.put((int(worker_id), int(step_idx), batch), timeout=0.1)
                    break
                except queue.Full:
                    continue

    except BaseException as exc:
        try:
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
        except Exception:
            pass


class TrafficAsyncProvider:
    """
    Управляет worker-процессами для генерации legacy-трафика.

    В отличие от mobility, у traffic нет безопасного sync fallback без resync
    состояния моделей: stateful-модели живут внутри worker'ов. Поэтому:
      * strict_snapshots=True  -> get_batch() возвращает None при timeout;
      * strict_snapshots=False -> get_batch() блокирующе ждет нужный step.
    """

    def __init__(
        self,
        update_interval_ms: int,
        sim_duration_ms: int,
        *,
        workers: int = 2,
        prefetch_steps: int = 16,
        cache_steps: int = 64,
        snapshot_timeout_ms: int = 2,
        strict_snapshots: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        if update_interval_ms <= 0:
            raise ValueError("update_interval_ms должен быть > 0")
        if sim_duration_ms <= 0:
            raise ValueError("sim_duration_ms должен быть > 0")
        if workers <= 0:
            raise ValueError("workers должен быть > 0")

        self.update_interval_ms = int(update_interval_ms)
        self.sim_duration_ms = int(sim_duration_ms)
        self.workers = int(workers)
        self.prefetch_steps = int(max(1, prefetch_steps))
        self.cache_steps = int(max(2, cache_steps))
        self.snapshot_timeout_ms = int(max(0, snapshot_timeout_ms))
        self.strict_snapshots = bool(strict_snapshots)
        self.seed = seed
        self.verbose = verbose

        self._ctx = mp.get_context()
        # Отдельная очередь на worker: иначе один быстрый worker может
        # заспамить общую очередь будущими steps, а main будет распаковывать
        # лишние batch'и, пока ждет step от другого worker.
        self._queues: List[mp.Queue] = []
        self._stop_event: Optional[mp.Event] = None
        self._processes: List[mp.Process] = []
        self._started = False

        self._cache: Dict[int, Dict[int, TrafficSnapshot]] = {}
        self._ready_steps: Set[int] = set()
        self._partial_cache: Dict[int, Dict[int, TrafficSnapshot]] = {}
        self._partial_workers: Dict[int, Set[int]] = {}
        self._active_worker_ids: Set[int] = set()
        self._last_error: Optional[Dict[str, str]] = None
        self._last_requested_step = -1
        self._start_step = 0
        self._max_step = max(0, self.sim_duration_ms - 1)

        self.steps_received = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.blocking_waits = 0
        self.blocking_wait_s = 0.0
        self.late_steps = 0
        self.worker_errors = 0
        self.snapshots_processed = 0
        self.packets_generated = 0
        self.missing_steps: List[int] = []
        # Диагностика bounded-drain: сколько queue-сообщений забрано из worker'ов.
        self.queue_messages_drained = 0
        self.queue_drain_calls = 0
        self.max_ready_step_seen = -1

    @property
    def started(self) -> bool:
        return self._started

    @property
    def last_error(self) -> Optional[Dict[str, str]]:
        return self._last_error

    @property
    def num_workers(self) -> int:
        return len(self._processes)

    def start(self, generator: Any, ue_ids: Iterable[int], *, start_step: int = 0) -> None:
        if self._started:
            return

        if not hasattr(generator, "models"):
            raise TypeError("TrafficAsyncProvider пока поддерживает только SimpleGenerator-like models")

        model_ue_ids = [int(ue_id) for ue_id in ue_ids if int(ue_id) in generator.models]
        if not model_ue_ids:
            if self.verbose:
                print("[TRAFFIC_ASYNC] No UE with traffic models found; async provider disabled.")
            return

        worker_count = max(1, min(self.workers, len(model_ue_ids)))
        buckets: List[List[int]] = [[] for _ in range(worker_count)]
        for ue_id in model_ue_ids:
            buckets[int(ue_id) % worker_count].append(int(ue_id))

        self._start_step = int(max(0, start_step))
        self._max_step = max(self._start_step, self.sim_duration_ms - 1)

        self._cache.clear()
        self._ready_steps.clear()
        self._partial_cache.clear()
        self._partial_workers.clear()
        self._active_worker_ids.clear()
        self._last_error = None

        self._queues = [
            self._ctx.Queue(maxsize=max(self.prefetch_steps, 1))
            for _ in range(worker_count)
        ]
        self._stop_event = self._ctx.Event()

        for worker_id, bucket in enumerate(buckets):
            if not bucket:
                continue

            worker_generator = _build_worker_generator(generator, bucket)
            proc = self._ctx.Process(
                target=_traffic_worker_loop,
                args=(
                    worker_id,
                    worker_generator,
                    bucket,
                    self.update_interval_ms,
                    self._start_step,
                    self._max_step,
                    self._queues[worker_id],
                    self._stop_event,
                    self.seed,
                ),
                daemon=True,
                name=f"TrafficAsyncWorker-{worker_id}",
            )
            proc.start()
            self._processes.append(proc)
            self._active_worker_ids.add(worker_id)

        self._started = bool(self._processes)

        if self.verbose:
            print(
                f"[TRAFFIC_ASYNC] Started {len(self._processes)} worker(s) for "
                f"{len(model_ue_ids)} UE, steps={self._start_step}..{self._max_step}, "
                f"strict={self.strict_snapshots}"
            )

    def _store_worker_batch(self, worker_id: int, step_idx: int, payload: List[TrafficSnapshot]) -> None:
        if step_idx < self._last_requested_step:
            self.late_steps += 1

        step_cache = self._partial_cache.setdefault(int(step_idx), {})
        for snap in payload:
            step_cache[int(snap.ue_id)] = snap

        workers = self._partial_workers.setdefault(int(step_idx), set())
        workers.add(int(worker_id))

        if self._active_worker_ids and workers >= self._active_worker_ids:
            self._cache[int(step_idx)] = dict(step_cache)
            self._ready_steps.add(int(step_idx))
            self.steps_received += 1
            self.snapshots_processed += len(step_cache)
            self.packets_generated += sum(len(snap.packets) for snap in step_cache.values())
            self._partial_cache.pop(int(step_idx), None)
            self._partial_workers.pop(int(step_idx), None)
            self._trim_cache(current_step=int(step_idx))

    def _drain_ready(
        self,
        *,
        block: bool = False,
        timeout_s: float = 0.0,
        max_items_per_worker: int = 1,
        until_step: Optional[int] = None,
    ) -> int:
        """
        Забрать сообщения из worker queues с ограничением.

        Важно: используется отдельная очередь на каждого worker'а. Это убирает
        проблему v6/v7-single-queue, где быстрый worker мог заполнить общую
        очередь будущими steps, а main process распаковывал лишние Packet-batches,
        пока ждал текущий step от другого worker.

        При последовательном запросе TTI обычно достаточно считать 1 сообщение
        из каждой очереди: worker'ы публикуют steps строго по порядку.
        """
        if not self._queues:
            return 0

        drained = 0
        self.queue_drain_calls += 1
        per_worker_limit = max(1, int(max_items_per_worker))

        for q in list(self._queues):
            if until_step is not None and int(until_step) in self._ready_steps:
                break

            for i in range(per_worker_limit):
                if until_step is not None and int(until_step) in self._ready_steps:
                    break
                try:
                    # block=True применяется только к первой попытке чтения из
                    # очереди конкретного worker'а. Это дает короткое ожидание
                    # нужного step, но не вычитывает будущие steps без меры.
                    if block and i == 0:
                        item = q.get(timeout=max(0.0, timeout_s))
                    else:
                        item = q.get_nowait()
                except queue.Empty:
                    break

                drained += 1
                self.queue_messages_drained += 1

                worker_id, step_idx, payload = item
                if step_idx == "__error__":
                    self._last_error = payload
                    self.worker_errors += 1
                    if self.verbose:
                        print("[TRAFFIC_ASYNC] Worker error:", payload.get("error"))
                        print(payload.get("traceback"))
                    break

                step_i = int(step_idx)
                if step_i > self.max_ready_step_seen:
                    self.max_ready_step_seen = step_i
                self._store_worker_batch(int(worker_id), step_i, payload)

        return drained

    def _trim_cache(self, current_step: int) -> None:
        # Важно: worker может посчитать сильно вперед, но main process еще не
        # запросил эти steps. Поэтому нельзя чистить cache относительно
        # produced current_step — так можно удалить будущий batch до применения.
        # Чистим только прошлое относительно последнего запрошенного main step.
        anchor_step = self._last_requested_step if self._last_requested_step >= 0 else int(current_step)
        min_keep = int(anchor_step) - self.cache_steps
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

    def _workers_alive(self) -> bool:
        return any(proc.is_alive() for proc in self._processes)

    def get_batch(self, step_idx: int) -> Optional[Dict[int, TrafficSnapshot]]:
        """Получить traffic snapshots для TTI step_idx."""
        if not self._started:
            return None

        step_idx = int(step_idx)
        self._last_requested_step = max(self._last_requested_step, step_idx)

        # Не вычитываем весь queue. Для текущего TTI достаточно забрать
        # сообщения текущего step от всех worker'ов плюс небольшой lookahead.
        # Это сохраняет prefetch, но не переносит всю стоимость IPC на main loop.
        small_budget = 1
        self._drain_ready(block=False, max_items_per_worker=small_budget, until_step=step_idx)

        if step_idx not in self._ready_steps and self.snapshot_timeout_ms > 0:
            self._drain_ready(
                block=True,
                timeout_s=self.snapshot_timeout_ms / 1000.0,
                max_items_per_worker=small_budget,
                until_step=step_idx,
            )

        if self._last_error is not None:
            self._record_miss(step_idx)
            return None

        batch = self._cache.get(step_idx)
        if batch is not None:
            self.cache_hits += 1
            return batch

        self._record_miss(step_idx)

        if self.strict_snapshots:
            return None

        # Нестрогий режим: не делаем sync fallback, чтобы не рассинхронизировать
        # stateful traffic models. Но ждем только нужный step, а не опустошаем
        # всю очередь с будущими batch'ами.
        wait_start = time.perf_counter()
        self.blocking_waits += 1
        wait_budget = 1
        while step_idx not in self._ready_steps:
            if self._last_error is not None:
                raise RuntimeError(
                    "Traffic async worker failed: "
                    f"{self._last_error.get('error')}\n{self._last_error.get('traceback')}"
                )
            if not self._workers_alive() and self._queues:
                # Даем шанс забрать последние сообщения из очереди, но bounded.
                self._drain_ready(block=False, max_items_per_worker=wait_budget, until_step=step_idx)
                if step_idx in self._ready_steps:
                    break
                raise RuntimeError(
                    f"Traffic async workers stopped before step {step_idx} became ready"
                )
            self._drain_ready(
                block=True,
                timeout_s=0.05,
                max_items_per_worker=wait_budget,
                until_step=step_idx,
            )

        self.blocking_wait_s += time.perf_counter() - wait_start
        batch = self._cache.get(step_idx, {})
        self.cache_hits += 1
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

        for q in self._queues:
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        self._started = False
        self._processes.clear()
        self._active_worker_ids.clear()
        self._queues = []
        self._stop_event = None
        self._cache.clear()
        self._ready_steps.clear()
        self._partial_cache.clear()
        self._partial_workers.clear()

        if self.verbose:
            print(
                f"[TRAFFIC_ASYNC] Worker(s) stopped. steps_received={self.steps_received}, "
                f"cache_hits={self.cache_hits}, cache_misses={self.cache_misses}, "
                f"blocking_waits={self.blocking_waits}, blocking_wait_s={self.blocking_wait_s:.3f}, "
                f"packets_generated={self.packets_generated}, queue_messages_drained={self.queue_messages_drained}"
            )

    def __enter__(self) -> "TrafficAsyncProvider":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
