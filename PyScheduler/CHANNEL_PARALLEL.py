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
#     конкретного UE не прыгало между процессами.
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    channel_model_template: Any,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    stop_event: mp.Event,
    seed: Optional[int] = None,
) -> None:
    """Long-lived worker: держит свою копию channel_model и считает батчи."""
    try:
        if seed is not None:
            np.random.seed(seed + worker_id)

        channel_model = copy.deepcopy(channel_model_template)

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

        self.channel_model_template = copy.deepcopy(channel_model)
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

        self.steps_processed = 0
        self.fallback_steps = 0
        self.ue_snapshots_processed = 0

    @property
    def started(self) -> bool:
        return self._started

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
                    self.channel_model_template,
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

        inputs = self._build_inputs(users, step_idx)
        if not inputs:
            return {}

        # Стабильное закрепление UE за worker: состояние канала UE не прыгает между процессами.
        buckets: List[List[ChannelInput]] = [[] for _ in range(self.workers)]
        for item in inputs:
            buckets[item.ue_id % self.workers].append(item)

        active_workers = 0
        try:
            for worker_id, bucket in enumerate(buckets):
                if not bucket:
                    continue
                self._in_queues[worker_id].put(
                    (int(step_idx), int(channel_update_interval), bucket),
                    timeout=self.timeout_s,
                )
                active_workers += 1

            results: Dict[int, ChannelSnapshot] = {}
            received = 0
            while received < active_workers:
                worker_id, returned_step, payload = self._out_queue.get(timeout=self.timeout_s)
                if returned_step == "__error__":
                    self.fallback_steps += 1
                    if self.verbose:
                        print("[CHANNEL_PARALLEL] Worker error:", payload.get("error"))
                        print(payload.get("traceback"))
                    return None

                if int(returned_step) != int(step_idx):
                    # Старый/чужой ответ. Для простоты MVP пропускаем.
                    continue

                for snap in payload:
                    results[int(snap.ue_id)] = snap
                received += 1

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

        self._started = False
        self._processes.clear()
        self._in_queues.clear()

        if self.verbose:
            print(
                f"[CHANNEL_PARALLEL] Stopped. steps_processed={self.steps_processed}, "
                f"ue_snapshots_processed={self.ue_snapshots_processed}, "
                f"fallback_steps={self.fallback_steps}"
            )
