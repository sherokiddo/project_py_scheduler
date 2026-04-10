"""
Базовые контракты и первый прикладной мост между runtime PyScheduler и
DRL-компонентами.
"""

from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class DRLSimulationRuntime:
    """
    Набор runtime-объектов, которые PyScheduler предоставляет DRL-слою.

    Структура намеренно остается легковесной. Здесь пока нет описания
    observation/action/reward. На этом этапе она только фиксирует те объекты,
    которые понадобятся будущему DRL-стеку для построения этих сущностей.
    """

    simulation_manager: Any
    base_station: Any
    ue_collection: Any
    lte_grid: Any
    scheduler: Any


@dataclass(slots=True)
class DRLSchedulerStepContext:
    """
    Снимок одного вызова scheduler внутри одного TTI симуляции.
    """

    current_time: int
    users: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DRLUEState:
    """
    Нормализованное состояние одного UE для DRL-пайплайна.
    """

    ue_id: int
    wb_cqi: int
    sb_cqi: List[int]
    sinr_db: float
    average_throughput_bps: float
    current_dl_throughput_bps: float
    buffer_bytes: int
    is_active: bool

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать состояние UE в словарь.
        """

        return asdict(self)


@dataclass(slots=True)
class DRLRuntimePayload:
    """
    Снимок runtime-состояния симуляции в DRL-ориентированном формате.
    """

    current_time: int
    scheduler_algorithm: str
    traffic_mode: str
    stats_enabled: bool
    bandwidth_mhz: float
    rb_per_slot: int
    rbg_size: int
    num_rbg: int
    ue_states: List[DRLUEState]
    scheduler_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать payload в словарь.
        """

        data = asdict(self)
        data["ue_states"] = [state.to_dict() for state in self.ue_states]
        return data


class DRLSimulationBridge:
    """
    Базовый lifecycle-мост между PyScheduler и будущими DRL-модулями.

    Цели дизайна:
    - сохранить `SimulationManager` главным оркестратором
    - оставить scheduler-логику расширяемой
    - дать DRL-слою стабильную точку доступа к runtime-состоянию
    - подготовить основу для observation builder, action adapter,
      parity-метрик и анимации
    """

    def __init__(self) -> None:
        self.runtime: Optional[DRLSimulationRuntime] = None

    def bind_runtime(self, runtime: DRLSimulationRuntime) -> None:
        """
        Привязать мост к текущему runtime симуляции.
        """

        self.runtime = runtime

    def require_runtime(self) -> DRLSimulationRuntime:
        """
        Вернуть привязанный runtime или выбросить понятную ошибку,
        если привязка еще не выполнена.
        """

        if self.runtime is None:
            raise RuntimeError(
                "DRL bridge еще не привязан к runtime симуляции."
            )

        return self.runtime

    def on_simulation_start(self, runtime: DRLSimulationRuntime) -> None:
        """
        Lifecycle-hook, вызываемый один раз после подготовки runtime симуляции.
        """

    def before_scheduler_step(self, step_context: DRLSchedulerStepContext) -> None:
        """
        Lifecycle-hook, вызываемый непосредственно перед
        `scheduler.schedule(...)`.
        """

    def after_scheduler_step(
        self,
        step_context: DRLSchedulerStepContext,
        scheduler_result: Dict[str, Any],
    ) -> None:
        """
        Lifecycle-hook, вызываемый непосредственно после
        `scheduler.schedule(...)`.
        """

    def on_simulation_end(self) -> None:
        """
        Lifecycle-hook, вызываемый один раз при завершении симуляции.
        """


class PySchedulerDRLBridge(DRLSimulationBridge):
    """
    Первый прикладной DRL-мост для PyScheduler.

    На текущем этапе не управляет планировщиком и не вызывает модель.
    Его задача — собрать и сохранить последнюю DRL-ориентированную картину
    runtime-состояния симулятора вокруг scheduler-step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.latest_payload: Optional[DRLRuntimePayload] = None

    def on_simulation_start(self, runtime: DRLSimulationRuntime) -> None:
        """
        Сбросить последнее сохраненное состояние при старте новой симуляции.
        """

        self.latest_payload = None

    def before_scheduler_step(self, step_context: DRLSchedulerStepContext) -> None:
        """
        Собрать DRL payload до вызова scheduler.
        """

        self.latest_payload = self.build_runtime_payload(
            step_context=step_context,
            scheduler_result=None,
        )

    def after_scheduler_step(
        self,
        step_context: DRLSchedulerStepContext,
        scheduler_result: Dict[str, Any],
    ) -> None:
        """
        Обновить DRL payload после завершения scheduler-step.
        """

        self.latest_payload = self.build_runtime_payload(
            step_context=step_context,
            scheduler_result=scheduler_result,
        )

    def get_latest_runtime_payload(self) -> DRLRuntimePayload:
        """
        Вернуть последний собранный runtime payload.
        """

        if self.latest_payload is None:
            raise RuntimeError(
                "DRL payload еще не собран. Выполните хотя бы один scheduler-step."
            )

        return self.latest_payload

    def build_runtime_payload(
        self,
        step_context: DRLSchedulerStepContext,
        scheduler_result: Optional[Dict[str, Any]],
    ) -> DRLRuntimePayload:
        """
        Построить DRL-ориентированный payload из текущего состояния симуляции.
        """

        runtime = self.require_runtime()
        lte_grid = runtime.lte_grid
        rbg_size = self._get_rbg_size(lte_grid)
        rb_per_slot = getattr(lte_grid, "rb_per_slot", 0)
        num_rbg = ceil(rb_per_slot / rbg_size) if rbg_size > 0 else 0

        ue_states = [
            self._build_ue_state(runtime=runtime, user=user)
            for user in step_context.users
        ]

        return DRLRuntimePayload(
            current_time=step_context.current_time,
            scheduler_algorithm=str(
                step_context.metadata.get("scheduler_algorithm", "")
            ),
            traffic_mode=str(step_context.metadata.get("traffic_mode", "")),
            stats_enabled=bool(step_context.metadata.get("stats_enabled", False)),
            bandwidth_mhz=float(getattr(lte_grid, "bandwidth", 0.0) or 0.0),
            rb_per_slot=int(rb_per_slot),
            rbg_size=int(rbg_size),
            num_rbg=int(num_rbg),
            ue_states=ue_states,
            scheduler_result=scheduler_result,
        )

    def _build_ue_state(
        self,
        runtime: DRLSimulationRuntime,
        user: Dict[str, Any],
    ) -> DRLUEState:
        """
        Построить нормализованное состояние одного UE.
        """

        ue = user.get("ue")
        ue_id = int(user.get("UE_ID", -1))
        wb_cqi = int(user.get("cqi", getattr(ue, "cqi", 0)) or 0)
        sb_cqi = list(user.get("sbb_cqi", getattr(ue, "cqi_subband", [])) or [])
        sinr_db = float(getattr(ue, "SINR", 0.0) or 0.0)
        avg_tput = float(getattr(ue, "average_throughput", 0.0) or 0.0)
        current_dl_tput = float(getattr(ue, "current_dl_throughput", 0.0) or 0.0)
        buffer_bytes = self._get_ue_buffer_bytes(runtime, ue_id)
        is_active = buffer_bytes > 0 and 1 <= wb_cqi <= 15

        return DRLUEState(
            ue_id=ue_id,
            wb_cqi=wb_cqi,
            sb_cqi=sb_cqi,
            sinr_db=sinr_db,
            average_throughput_bps=avg_tput,
            current_dl_throughput_bps=current_dl_tput,
            buffer_bytes=buffer_bytes,
            is_active=is_active,
        )

    @staticmethod
    def _get_rbg_size(lte_grid: Any) -> int:
        """
        Получить размер RBG из resource grid.
        """

        if lte_grid is None:
            return 0

        get_rbg_size = getattr(lte_grid, "GET_RBG_SIZE", None)
        if callable(get_rbg_size):
            return int(get_rbg_size())

        return 0

    @staticmethod
    def _get_ue_buffer_bytes(runtime: DRLSimulationRuntime, ue_id: int) -> int:
        """
        Получить суммарный размер буфера UE в байтах.
        """

        buffer_manager = getattr(runtime.base_station, "buffer_manager", None)
        if buffer_manager is None:
            return 0

        ue_has_buffer = getattr(buffer_manager, "ue_has_buffer", None)
        if callable(ue_has_buffer) and not ue_has_buffer(ue_id):
            return 0

        get_buffer_status = getattr(buffer_manager, "get_buffer_status", None)
        if not callable(get_buffer_status):
            return 0

        try:
            buffer_status_list = get_buffer_status(ue_id)
        except Exception:
            return 0

        total_bytes = 0
        for buffer_status in buffer_status_list or []:
            total_bytes += int(getattr(buffer_status, "buffer_size", 0) or 0)

        return total_bytes
