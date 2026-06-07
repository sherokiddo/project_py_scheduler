"""
#------------------------------------------------------------------------------
# Модуль: CHANNEL_PARALLEL - Параллельный batch-расчёт качества канала UE
#------------------------------------------------------------------------------
# Назначение:
#   Выносит наиболее частую фазу UPD_CH_QUALITY() в отдельные worker-процессы.
#   Основной процесс остаётся единственным владельцем живых UserEquipment и только
#   применяет готовые immutable ChannelSnapshot.
#
# Важные ограничения:
#   - worker не трогает реальные UE-объекты;
#   - worker получает только сериализуемый ChannelInput;
#   - состояние channel_model хранится внутри worker-процесса;
#   - UE закрепляются за worker по ue_id % workers, чтобы состояние канала
#     конкретного UE не прыгало между процессами;
#   - late/stale результаты не применяются к новому step_idx.
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import importlib
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class ChannelInput:
    """Минимальный снимок состояния UE, нужный для расчёта канала."""

    ue_id: int
    step_idx: int
    position: Tuple[float, float]
    velocity: float
    ue_class: str
    is_indoor: bool
    indoor_boundaries: Tuple[float, float, float, float]
    ue_height: float


@dataclass(frozen=True, slots=True)
class ChannelSnapshot:
    """Immutable-результат расчёта канала для одного UE."""

    ue_id: int
    step_idx: int
    sinr: float
    cqi: int
    cqi_subband: Tuple[int, ...]
    ue_height: float
    dist_to_bs_2d: float
    dist_to_bs_2d_in: float
    dist_to_bs_2d_out: float
    dist_to_bs_3d: float


@dataclass(frozen=True, slots=True)
class ChannelModelSpec:
    """
    Сериализуемое описание channel_model для worker-процесса.

    Для штатных моделей используется явный контракт ChannelModel.export_worker_state()
    / import_worker_state(). Для тестовых или внешних моделей оставлен fallback:
    deepcopy-template, но он изолирован внутри spec и не смешан с live BS/UE state.
    """

    module_name: str
    class_name: str
    model_state: Optional[Dict[str, Any]] = None
    fallback_template: Optional[Any] = None


def build_channel_model_spec(channel_model: Any) -> ChannelModelSpec:
    """Собрать ChannelModelSpec без передачи live-ссылки на модель в worker."""
    cls = channel_model.__class__
    if hasattr(channel_model, "export_worker_state") and hasattr(cls, "import_worker_state"):
        return ChannelModelSpec(
            module_name=cls.__module__,
            class_name=cls.__name__,
            model_state=channel_model.export_worker_state(),
            fallback_template=None,
        )

    # Fallback для тестовых моделей, не наследующихся от ChannelModel.
    return ChannelModelSpec(
        module_name=cls.__module__,
        class_name=cls.__name__,
        model_state=None,
        fallback_template=copy.deepcopy(channel_model),
    )


def _rebuild_channel_model_from_spec(spec: ChannelModelSpec) -> Any:
    """Восстановить channel_model внутри worker-процесса."""
    if spec.model_state is not None:
        module = importlib.import_module(spec.module_name)
        cls = getattr(module, spec.class_name)
        if hasattr(cls, "import_worker_state"):
            return cls.import_worker_state(spec.model_state)

    return copy.deepcopy(spec.fallback_template)


def _sinr_to_cqi(sinr: float) -> int:
    """Та же логика, что UserEquipment.SINR_TO_CQI(), но без UE-объекта."""
    if sinr <= -6.934:
        return 1
    if sinr >= 22.976:
        return 15
    step = (22.976 + 6.934) / 14
    return int(1 + (sinr + 6.934) / step)


def _calculate_distances_for_input(
    item: ChannelInput,
    bs_position: Tuple[float, float],
    bs_height: float,
) -> Tuple[float, float, float, float]:
    """Повторяет логику расчёта дистанций из UE_MODULE без мутации UE."""
    ue_x, ue_y = item.position
    bs_x, bs_y = bs_position

    if not item.is_indoor:
        d_2d = float(np.hypot(ue_x - bs_x, ue_y - bs_y))
        d_2d_in = 0.0
        d_2d_out = d_2d
        d_3d = float(np.hypot(d_2d, bs_height - item.ue_height))
        return d_2d, d_2d_in, d_2d_out, d_3d

    x_min, x_max, y_min, y_max = item.indoor_boundaries

    if (x_min <= bs_x <= x_max) and (y_min <= bs_y <= y_max):
        d_2d = float(np.hypot(bs_x - ue_x, bs_y - ue_y))
        d_2d_in = d_2d
        d_2d_out = 0.0
        d_3d = float(np.hypot(d_2d, bs_height - item.ue_height))
        return d_2d, d_2d_in, d_2d_out, d_3d

    dx = bs_x - ue_x
    dy = bs_y - ue_y

    t_values = []
    if dx != 0:
        t_values.extend([(x_min - ue_x) / dx, (x_max - ue_x) / dx])
    if dy != 0:
        t_values.extend([(y_min - ue_y) / dy, (y_max - ue_y) / dy])

    t_valid = [t for t in t_values if t > 0]
    if not t_valid:
        d_2d = float(np.hypot(dx, dy))
        d_2d_in = d_2d
        d_2d_out = 0.0
        d_3d = float(np.hypot(d_2d, bs_height - item.ue_height))
        return d_2d, d_2d_in, d_2d_out, d_3d

    t_exit = min(t_valid)
    exit_x = ue_x + dx * t_exit
    exit_y = ue_y + dy * t_exit

    d_2d_in = float(np.hypot(exit_x - ue_x, exit_y - ue_y))
    d_2d = float(np.hypot(dx, dy))
    d_2d_out = d_2d - d_2d_in
    d_3d = float(np.hypot(d_2d, bs_height - item.ue_height))
    return d_2d, d_2d_in, d_2d_out, d_3d


def _calculate_one_channel_snapshot(
    channel_model: Any,
    item: ChannelInput,
    channel_update_interval: int,
) -> ChannelSnapshot:
    """Расчёт одного UE внутри worker-процесса."""
    import GLOBALS
    from CHANNEL_MODEL import RMaModel, UMaModel, UMiModel

    bs = channel_model.bs
    ue_height = float(item.ue_height)

    # Сохраняем старую семантику UPD_CH_QUALITY(): высота UE инициализируется лениво.
    if isinstance(channel_model, RMaModel):
        if ue_height == 0.0:
            ue_height = float(np.random.uniform(1, 10)) if item.is_indoor else 1.0

    if isinstance(channel_model, (UMaModel, UMiModel)):
        if ue_height == 0.0:
            if item.is_indoor:
                n_fl_total = float(np.random.uniform(4, 8))
                n_fl = float(np.random.uniform(1, n_fl_total))
                ue_height = 3 * (n_fl - 1) + 1.5
            else:
                ue_height = 1.5

    # Дистанции считаем после ленивой инициализации ue_height.
    item_for_distance = ChannelInput(
        ue_id=item.ue_id,
        step_idx=item.step_idx,
        position=item.position,
        velocity=item.velocity,
        ue_class=item.ue_class,
        is_indoor=item.is_indoor,
        indoor_boundaries=item.indoor_boundaries,
        ue_height=ue_height,
    )
    d_2d, d_2d_in, d_2d_out, d_3d = _calculate_distances_for_input(
        item_for_distance,
        tuple(bs.position),
        float(bs.height),
    )

    static_ue_velocity = 0.2
    effective_velocity = max(float(item.velocity), static_ue_velocity)
    displacement = effective_velocity * (channel_update_interval / 1000.0)

    sinr_on_rb = channel_model.calculate_SINR(
        int(item.ue_id),
        displacement,
        d_2d,
        d_2d_in,
        d_3d,
        ue_height,
        item.ue_class,
    )

    sinr = float(np.mean(sinr_on_rb))
    cqi = _sinr_to_cqi(sinr)

    cqi_subband: Tuple[int, ...] = ()
    if getattr(bs, "enable_tdl", False):
        sinr_array = np.atleast_1d(sinr_on_rb)
        subband_size = GLOBALS.SUBBAND_SIZE[bs.bandwidth]
        cqi_subband = tuple(
            _sinr_to_cqi(float(np.mean(sinr_array[i : i + subband_size])))
            for i in range(0, len(sinr_array), subband_size)
        )

    return ChannelSnapshot(
        ue_id=int(item.ue_id),
        step_idx=int(item.step_idx),
        sinr=sinr,
        cqi=int(cqi),
        cqi_subband=cqi_subband,
        ue_height=float(ue_height),
        dist_to_bs_2d=float(d_2d),
        dist_to_bs_2d_in=float(d_2d_in),
        dist_to_bs_2d_out=float(d_2d_out),
        dist_to_bs_3d=float(d_3d),
    )


def _channel_worker_loop(
    worker_id: int,
    channel_model_spec: ChannelModelSpec,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    stop_event: mp.Event,
    seed: Optional[int] = None,
) -> None:
    """Long-lived worker: держит свою копию channel_model и считает батчи."""
    try:
        if seed is not None:
            np.random.seed(seed + worker_id)

        channel_model = _rebuild_channel_model_from_spec(channel_model_spec)

        while not stop_event.is_set():
            try:
                msg = in_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if msg is None or msg == "__stop__":
                break

            step_idx, channel_update_interval, inputs = msg
            snapshots: List[ChannelSnapshot] = []
            for item in inputs:
                snapshots.append(
                    _calculate_one_channel_snapshot(
                        channel_model=channel_model,
                        item=item,
                        channel_update_interval=int(channel_update_interval),
                    )
                )

            out_queue.put((worker_id, int(step_idx), snapshots))

    except BaseException as exc:
        out_queue.put(
            (
                worker_id,
                "__error__",
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )


class ChannelParallelProvider:
    """
    Параллельный provider для channel phase.

    Это не background precompute, а безопасный synchronous batch parallelism:
        main process собирает ChannelInput по UE;
        workers считают SINR/CQI параллельно;
        main process применяет ChannelSnapshot к UE.

    Такой вариант подходит для канала лучше, чем очередь по одному UE, потому что
    channel_update вызывается часто, и нам нужно уменьшать overhead обмена.
    """

    def __init__(
        self,
        channel_model: Any,
        *,
        workers: int = 2,
        timeout_s: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        if workers <= 0:
            raise ValueError("workers должен быть > 0")
        if channel_model is None:
            raise ValueError("channel_model не должен быть None")

        self.channel_model_spec = build_channel_model_spec(channel_model)
        self.workers = int(workers)
        self.timeout_s = float(timeout_s)
        self.seed = seed
        self.verbose = verbose

        self._ctx = mp.get_context()
        self._in_queues: List[mp.Queue] = []
        self._out_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._processes: List[mp.Process] = []
        self._started = False

        # step_idx -> worker_id -> snapshots
        self._pending_results: Dict[int, Dict[int, List[ChannelSnapshot]]] = {}
        self._last_error: Optional[Dict[str, str]] = None

        self.steps_processed = 0
        self.fallback_steps = 0
        self.ue_snapshots_processed = 0
        self.stale_results_discarded = 0
        self.pending_results_used = 0
        self.worker_errors = 0

    @property
    def started(self) -> bool:
        return self._started

    @property
    def last_error(self) -> Optional[Dict[str, str]]:
        return self._last_error

    def start(self) -> None:
        if self._started:
            return

        self._out_queue = self._ctx.Queue()
        self._stop_event = self._ctx.Event()
        self._in_queues = [self._ctx.Queue(maxsize=2) for _ in range(self.workers)]

        for worker_id in range(self.workers):
            proc = self._ctx.Process(
                target=_channel_worker_loop,
                args=(
                    worker_id,
                    self.channel_model_spec,
                    self._in_queues[worker_id],
                    self._out_queue,
                    self._stop_event,
                    self.seed,
                ),
                daemon=True,
                name=f"ChannelWorker-{worker_id}",
            )
            proc.start()
            self._processes.append(proc)

        self._started = True
        if self.verbose:
            print(f"[CHANNEL_PARALLEL] Started {self.workers} worker process(es)")

    def _build_inputs(self, users: Iterable[Any], step_idx: int) -> List[ChannelInput]:
        inputs: List[ChannelInput] = []
        for ue in users:
            inputs.append(
                ChannelInput(
                    ue_id=int(ue.UE_ID),
                    step_idx=int(step_idx),
                    position=tuple(ue.position),
                    velocity=float(getattr(ue, "velocity", 0.0)),
                    ue_class=str(getattr(ue, "ue_class", "pedestrian")),
                    is_indoor=bool(getattr(ue, "is_indoor", False)),
                    indoor_boundaries=tuple(getattr(ue, "indoor_boundaries", (0, 0, 0, 0))),
                    ue_height=float(getattr(ue, "UE_height", 0.0)),
                )
            )
        return inputs

    def _store_worker_payload(self, worker_id: int, returned_step: int, payload: List[ChannelSnapshot]) -> None:
        step_payloads = self._pending_results.setdefault(int(returned_step), {})
        step_payloads[int(worker_id)] = payload

    def _drain_completed_results(self, *, drop_before_step: Optional[int] = None) -> None:
        """Забрать все уже готовые worker-ответы без блокировки."""
        if self._out_queue is None:
            return

        while True:
            try:
                worker_id, returned_step, payload = self._out_queue.get_nowait()
            except queue.Empty:
                break

            if returned_step == "__error__":
                self._last_error = payload
                self.worker_errors += 1
                continue

            returned_step = int(returned_step)
            if drop_before_step is not None and returned_step < int(drop_before_step):
                self.stale_results_discarded += 1
                continue

            self._store_worker_payload(int(worker_id), returned_step, payload)

    def _collect_pending_for_step(
        self,
        step_idx: int,
        expected_workers: Set[int],
    ) -> Tuple[Dict[int, ChannelSnapshot], Set[int]]:
        """Собрать уже сохраненные payloads по step_idx."""
        results: Dict[int, ChannelSnapshot] = {}
        received_workers: Set[int] = set()
        step_payloads = self._pending_results.get(int(step_idx), {})

        for worker_id in list(expected_workers):
            payload = step_payloads.get(worker_id)
            if payload is None:
                continue
            for snap in payload:
                results[int(snap.ue_id)] = snap
            received_workers.add(worker_id)
            self.pending_results_used += 1

        if received_workers:
            remaining = {
                worker_id: payload
                for worker_id, payload in step_payloads.items()
                if worker_id not in received_workers
            }
            if remaining:
                self._pending_results[int(step_idx)] = remaining
            else:
                self._pending_results.pop(int(step_idx), None)

        return results, received_workers

    def calculate_batch(
        self,
        users: Iterable[Any],
        *,
        step_idx: int,
        channel_update_interval: int,
    ) -> Optional[Dict[int, ChannelSnapshot]]:
        """
        Вернуть batch snapshots для всех UE или None, если надо fallback.
        """
        if not self._started:
            return None
        if self._out_queue is None:
            return None

        step_idx = int(step_idx)
        inputs = self._build_inputs(users, step_idx)
        if not inputs:
            return {}

        # Забираем поздние ответы с прошлых вызовов. Всё, что старше текущего
        # step_idx, уже нельзя применять к live UE state.
        self._drain_completed_results(drop_before_step=step_idx)
        if self._last_error is not None:
            self.fallback_steps += 1
            if self.verbose:
                print("[CHANNEL_PARALLEL] Worker error:", self._last_error.get("error"))
                print(self._last_error.get("traceback"))
            return None

        # Стабильное закрепление UE за worker: состояние канала UE не прыгает между процессами.
        buckets: List[List[ChannelInput]] = [[] for _ in range(self.workers)]
        for item in inputs:
            buckets[item.ue_id % self.workers].append(item)

        expected_workers: Set[int] = {worker_id for worker_id, bucket in enumerate(buckets) if bucket}
        results, received_workers = self._collect_pending_for_step(step_idx, expected_workers)

        deadline = time.monotonic() + self.timeout_s
        try:
            for worker_id, bucket in enumerate(buckets):
                if not bucket or worker_id in received_workers:
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                self._in_queues[worker_id].put(
                    (step_idx, int(channel_update_interval), bucket),
                    timeout=remaining,
                )

            while received_workers < expected_workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timeout waiting for channel workers "
                        f"{sorted(expected_workers - received_workers)} on step {step_idx}"
                    )

                worker_id, returned_step, payload = self._out_queue.get(timeout=remaining)
                if returned_step == "__error__":
                    self._last_error = payload
                    self.worker_errors += 1
                    self.fallback_steps += 1
                    if self.verbose:
                        print("[CHANNEL_PARALLEL] Worker error:", payload.get("error"))
                        print(payload.get("traceback"))
                    return None

                worker_id = int(worker_id)
                returned_step = int(returned_step)

                if returned_step < step_idx:
                    self.stale_results_discarded += 1
                    continue

                if returned_step != step_idx or worker_id not in expected_workers:
                    # Future/out-of-order или ответ worker'а, который в этом step не нужен.
                    self._store_worker_payload(worker_id, returned_step, payload)
                    continue

                for snap in payload:
                    results[int(snap.ue_id)] = snap
                received_workers.add(worker_id)

            self.steps_processed += 1
            self.ue_snapshots_processed += len(results)
            return results

        except Exception as exc:
            self.fallback_steps += 1
            if self.verbose:
                print(f"[CHANNEL_PARALLEL] Fallback on step {step_idx}: {exc!r}")
            return None

    def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        for q in self._in_queues:
            try:
                q.put_nowait("__stop__")
            except Exception:
                pass

        for proc in self._processes:
            if proc.is_alive():
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)

        for q in self._in_queues:
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        if self._out_queue is not None:
            try:
                self._out_queue.close()
                self._out_queue.join_thread()
            except Exception:
                pass

        self._started = False
        self._processes.clear()
        self._in_queues.clear()
        self._out_queue = None
        self._stop_event = None
        self._pending_results.clear()

        if self.verbose:
            print(
                f"[CHANNEL_PARALLEL] Stopped. steps_processed={self.steps_processed}, "
                f"ue_snapshots_processed={self.ue_snapshots_processed}, "
                f"fallback_steps={self.fallback_steps}, "
                f"stale_results_discarded={self.stale_results_discarded}"
            )
