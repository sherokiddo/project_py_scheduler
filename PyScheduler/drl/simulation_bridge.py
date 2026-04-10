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


@dataclass(slots=True)
class DRLPlaygroundSimulationConfig:
    """
    Снимок настроек симуляции в терминах среды `drl_playground`.
    """

    sim_duration_tti: Optional[int]
    update_interval_tti: int
    mobility_update_interval_tti: int
    channel_update_interval_tti: int
    traffic_mode: str
    traffic_generator: str
    buffer_mode: str
    scheduler_algorithm: str
    scheduler_max_dl_ue_tti: Optional[int]
    scheduler_window_size: int
    scheduler_window_enabled: bool
    stats_enabled: bool
    bandwidth_mhz: float
    frequency_ghz: float
    n_rb_dl: int
    rbg_size_rb: int
    n_rbg: int
    channel_model_type: str
    enable_tdl: bool

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать снимок настроек в словарь.
        """

        return asdict(self)


@dataclass(slots=True)
class DRLPlaygroundUEState:
    """
    Состояние одного UE в формате playground snapshot.
    """

    ue_id: int
    reported_wb_cqi: int
    true_wb_cqi: int
    wb_cqi_age_tti: Optional[int]
    active_flag: bool
    buffer_bytes: int
    average_throughput_bps: float
    current_dl_throughput_bps: float
    alloc_rbg_count_tti: int
    alloc_rbg_frac_tti: float
    sinr_db: float
    reported_sb_cqi: List[int]

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать состояние UE в словарь.
        """

        return asdict(self)


@dataclass(slots=True)
class DRLPlaygroundStepSnapshot:
    """
    Снимок шага среды в терминах `current_tti/current_rbg`.
    """

    current_time: int
    current_tti: int
    current_rbg_index: Optional[int]
    allocated_rbg_fraction_progress: Optional[float]
    allocated_rbg_fraction_final_tti: float
    n_rb_dl: int
    rbg_size_rb: int
    n_rbg: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать снимок шага в словарь.
        """

        return asdict(self)


@dataclass(slots=True)
class DRLPlaygroundCompatibilityReport:
    """
    Отчет о степени совместимости snapshot с семантикой `drl_playground`.
    """

    exact_per_rbg_step_supported: bool
    reported_vs_true_wb_cqi_supported: bool
    wb_cqi_age_supported: bool
    alloc_frac_this_tti_supported: bool
    current_rbg_index_supported: bool
    scheduler_eligibility_mask_supported: bool
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать отчет совместимости в словарь.
        """

        return asdict(self)


@dataclass(slots=True)
class DRLPlaygroundSnapshot:
    """
    Полный snapshot состояния для observation adapter и DRL scheduler.
    """

    simulation_config: DRLPlaygroundSimulationConfig
    step_snapshot: DRLPlaygroundStepSnapshot
    ue_states: List[DRLPlaygroundUEState]
    action_mask: List[int]
    compatibility: DRLPlaygroundCompatibilityReport
    scheduler_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать snapshot в словарь.
        """

        return {
            "simulation_config": self.simulation_config.to_dict(),
            "step_snapshot": self.step_snapshot.to_dict(),
            "ue_states": [state.to_dict() for state in self.ue_states],
            "action_mask": list(self.action_mask),
            "compatibility": self.compatibility.to_dict(),
            "scheduler_result": self.scheduler_result,
        }


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
        self.latest_playground_snapshot: Optional[DRLPlaygroundSnapshot] = None

    def on_simulation_start(self, runtime: DRLSimulationRuntime) -> None:
        """
        Сбросить последнее сохраненное состояние при старте новой симуляции.
        """

        self.latest_payload = None
        self.latest_playground_snapshot = None

    def before_scheduler_step(self, step_context: DRLSchedulerStepContext) -> None:
        """
        Собрать DRL payload до вызова scheduler.
        """

        self.latest_payload = self.build_runtime_payload(
            step_context=step_context,
            scheduler_result=None,
        )
        self.latest_playground_snapshot = self.build_playground_snapshot(
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
        self.latest_playground_snapshot = self.build_playground_snapshot(
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

    def get_latest_playground_snapshot(self) -> DRLPlaygroundSnapshot:
        """
        Вернуть последний собранный snapshot в терминах `drl_playground`.
        """

        if self.latest_playground_snapshot is None:
            raise RuntimeError(
                "DRL playground snapshot еще не собран. Выполните хотя бы один scheduler-step."
            )

        return self.latest_playground_snapshot

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

    def build_playground_snapshot(
        self,
        step_context: DRLSchedulerStepContext,
        scheduler_result: Optional[Dict[str, Any]],
    ) -> DRLPlaygroundSnapshot:
        """
        Построить snapshot состояния симуляции в терминах среды `drl_playground`.
        """

        runtime = self.require_runtime()
        manager = runtime.simulation_manager
        lte_grid = runtime.lte_grid
        base_station = runtime.base_station
        scheduler = runtime.scheduler

        rbg_size = self._get_rbg_size(lte_grid)
        n_rb_dl = int(getattr(lte_grid, "rb_per_slot", 0) or 0)
        n_rbg = self._get_num_rbg(lte_grid)
        allocation_map = self._extract_allocation_map(scheduler_result)
        alloc_rbg_counts = self._get_alloc_rbg_counts_tti(lte_grid, allocation_map)
        allocated_rbg_total = sum(alloc_rbg_counts.values())
        allocated_rbg_fraction_final = (
            allocated_rbg_total / n_rbg if n_rbg > 0 else 0.0
        )

        ue_states: List[DRLPlaygroundUEState] = []
        action_mask: List[int] = []
        for user in step_context.users:
            ue = user.get("ue")
            ue_id = int(user.get("UE_ID", -1))
            fallback_cqi = int(user.get("cqi", getattr(ue, "cqi", 0)) or 0)
            buffer_bytes = self._get_ue_buffer_bytes(runtime, ue_id)
            reported_wb_cqi, wb_cqi_age_tti = self._get_reported_wb_cqi_and_age(
                scheduler=scheduler,
                ue_id=ue_id,
                current_tti=step_context.current_time,
                fallback_cqi=fallback_cqi,
            )
            is_eligible = self._is_scheduler_eligible(
                runtime=runtime,
                scheduler=scheduler,
                ue_id=ue_id,
                reported_wb_cqi=reported_wb_cqi,
            )
            alloc_rbg_count = int(alloc_rbg_counts.get(ue_id, 0))

            ue_states.append(
                DRLPlaygroundUEState(
                    ue_id=ue_id,
                    reported_wb_cqi=reported_wb_cqi,
                    true_wb_cqi=fallback_cqi,
                    wb_cqi_age_tti=wb_cqi_age_tti,
                    active_flag=is_eligible,
                    buffer_bytes=buffer_bytes,
                    average_throughput_bps=float(
                        getattr(ue, "average_throughput", 0.0) or 0.0
                    ),
                    current_dl_throughput_bps=float(
                        getattr(ue, "current_dl_throughput", 0.0) or 0.0
                    ),
                    alloc_rbg_count_tti=alloc_rbg_count,
                    alloc_rbg_frac_tti=(
                        alloc_rbg_count / n_rbg if n_rbg > 0 else 0.0
                    ),
                    sinr_db=float(getattr(ue, "SINR", 0.0) or 0.0),
                    reported_sb_cqi=self._get_reported_sb_cqi(
                        scheduler=scheduler,
                        ue_id=ue_id,
                        fallback_sb_cqi=list(
                            user.get("sbb_cqi", getattr(ue, "cqi_subband", [])) or []
                        ),
                    ),
                )
            )
            action_mask.append(int(is_eligible))

        compatibility = DRLPlaygroundCompatibilityReport(
            exact_per_rbg_step_supported=False,
            reported_vs_true_wb_cqi_supported=True,
            wb_cqi_age_supported=True,
            alloc_frac_this_tti_supported=True,
            current_rbg_index_supported=False,
            scheduler_eligibility_mask_supported=True,
            notes=[
                "PyScheduler наружу отдает шаг на уровне TTI, а внутренний per-RBG цикл остается скрыт внутри scheduler.",
                "current_rbg_index и allocated_rbg_fraction_progress недоступны без отдельного per-RBG scheduler path.",
                "action_mask построен по тем же базовым условиям eligibility, что и scheduler: буфер UE и валидный reported WB CQI.",
            ],
        )

        traffic_mode = str(step_context.metadata.get("traffic_mode", ""))
        if not traffic_mode:
            traffic_mode = (
                "legacy_simple_buffer"
                if getattr(manager.sim_config, "use_legacy_traffic", False)
                else "layered_buffer"
            )

        return DRLPlaygroundSnapshot(
            simulation_config=DRLPlaygroundSimulationConfig(
                sim_duration_tti=getattr(manager.sim_config, "sim_duration", None),
                update_interval_tti=int(
                    getattr(manager.sim_config, "update_interval", 0) or 0
                ),
                mobility_update_interval_tti=int(
                    getattr(manager.sim_config, "mobility_update_interval", 0) or 0
                ),
                channel_update_interval_tti=int(
                    getattr(manager.sim_config, "channel_update_interval", 0) or 0
                ),
                traffic_mode=traffic_mode,
                traffic_generator=type(getattr(manager, "traffic_gen", None)).__name__,
                buffer_mode=(
                    "simple_buffer"
                    if bool(getattr(base_station, "use_simple_buffer", False))
                    else "layered_buffer"
                ),
                scheduler_algorithm=str(
                    step_context.metadata.get(
                        "scheduler_algorithm",
                        getattr(manager.sched_config, "algorithm", ""),
                    )
                ),
                scheduler_max_dl_ue_tti=getattr(
                    manager.sched_config, "max_dl_ue_tti", None
                ),
                scheduler_window_size=int(
                    getattr(manager.sched_config, "window_size", 0) or 0
                ),
                scheduler_window_enabled=bool(
                    getattr(manager.sched_config, "enable_window", False)
                ),
                stats_enabled=bool(
                    step_context.metadata.get(
                        "stats_enabled",
                        getattr(manager.stats_config, "enabled", False),
                    )
                ),
                bandwidth_mhz=float(
                    getattr(lte_grid, "bandwidth", getattr(base_station, "bandwidth", 0.0))
                    or 0.0
                ),
                frequency_ghz=float(
                    getattr(base_station, "frequency_GHz", 0.0) or 0.0
                ),
                n_rb_dl=n_rb_dl,
                rbg_size_rb=rbg_size,
                n_rbg=n_rbg,
                channel_model_type=str(
                    getattr(base_station, "ch_model_type", "") or ""
                ),
                enable_tdl=bool(getattr(base_station, "enable_tdl", False)),
            ),
            step_snapshot=DRLPlaygroundStepSnapshot(
                current_time=step_context.current_time,
                current_tti=step_context.current_time,
                current_rbg_index=None,
                allocated_rbg_fraction_progress=None,
                allocated_rbg_fraction_final_tti=allocated_rbg_fraction_final,
                n_rb_dl=n_rb_dl,
                rbg_size_rb=rbg_size,
                n_rbg=n_rbg,
            ),
            ue_states=ue_states,
            action_mask=action_mask,
            compatibility=compatibility,
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

    def _get_num_rbg(self, lte_grid: Any) -> int:
        """
        Получить общее число RBG в одном TTI.
        """

        rbg_size = self._get_rbg_size(lte_grid)
        rb_per_slot = int(getattr(lte_grid, "rb_per_slot", 0) or 0)
        if rbg_size <= 0:
            return 0
        return int(ceil(rb_per_slot / rbg_size))

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

    @staticmethod
    def _extract_allocation_map(
        scheduler_result: Optional[Dict[str, Any]],
    ) -> Dict[int, List[int]]:
        """
        Извлечь allocation map из результата scheduler.
        """

        if not scheduler_result:
            return {}

        raw_allocation = scheduler_result.get("allocation", {})
        if not isinstance(raw_allocation, dict):
            return {}

        allocation: Dict[int, List[int]] = {}
        for raw_ue_id, raw_indices in raw_allocation.items():
            try:
                ue_id = int(raw_ue_id)
            except (TypeError, ValueError):
                continue

            allocation[ue_id] = [int(index) for index in list(raw_indices or [])]

        return allocation

    def _get_alloc_rbg_counts_tti(
        self,
        lte_grid: Any,
        allocation_map: Dict[int, List[int]],
    ) -> Dict[int, int]:
        """
        Получить число уникально выделенных RBG на UE по итоговой allocation map.
        """

        return {
            ue_id: len(self._get_rbg_indices(lte_grid, rb_indices))
            for ue_id, rb_indices in allocation_map.items()
        }

    def _get_rbg_indices(self, lte_grid: Any, rb_indices: List[int]) -> List[int]:
        """
        Преобразовать список RB индексов в список уникальных RBG индексов.
        """

        rbg_size = self._get_rbg_size(lte_grid)
        if rbg_size <= 0:
            return []

        rbg_indices = {
            int(rb_idx) // rbg_size
            for rb_idx in list(rb_indices or [])
            if int(rb_idx) >= 0
        }
        return sorted(rbg_indices)

    @staticmethod
    def _get_reported_sb_cqi(
        *,
        scheduler: Any,
        ue_id: int,
        fallback_sb_cqi: List[int],
    ) -> List[int]:
        """
        Получить reported SB CQI из scheduler CQI map с fallback на runtime-данные UE.
        """

        cqi_map = getattr(scheduler, "cqi_map", {})
        cqi_entry = cqi_map.get(ue_id)
        if cqi_entry is None:
            return [int(cqi) for cqi in list(fallback_sb_cqi or [])]

        return [int(cqi) for cqi in list(getattr(cqi_entry, "sb_cqi", []) or [])]

    @staticmethod
    def _get_reported_wb_cqi_and_age(
        *,
        scheduler: Any,
        ue_id: int,
        current_tti: int,
        fallback_cqi: int,
    ) -> tuple[int, Optional[int]]:
        """
        Получить reported WB CQI и его возраст в TTI.
        """

        cqi_map = getattr(scheduler, "cqi_map", {})
        cqi_entry = cqi_map.get(ue_id)
        if cqi_entry is None:
            return int(fallback_cqi), None

        reported_wb_cqi = int(getattr(cqi_entry, "wb_cqi", fallback_cqi) or fallback_cqi)
        last_wb_update = getattr(cqi_entry, "last_wb_update", None)
        if last_wb_update is None:
            return reported_wb_cqi, None

        age_tti = max(0, current_tti - int(last_wb_update))
        return reported_wb_cqi, age_tti

    def _is_scheduler_eligible(
        self,
        *,
        runtime: DRLSimulationRuntime,
        scheduler: Any,
        ue_id: int,
        reported_wb_cqi: Optional[int] = None,
    ) -> bool:
        """
        Проверить eligibility UE по тем же базовым условиям, что и в scheduler.
        """

        buffer_bytes = self._get_ue_buffer_bytes(runtime, ue_id)
        if buffer_bytes <= 0:
            return False

        if reported_wb_cqi is None:
            reported_wb_cqi, _ = self._get_reported_wb_cqi_and_age(
                scheduler=scheduler,
                ue_id=ue_id,
                current_tti=0,
                fallback_cqi=0,
            )

        return 1 <= int(reported_wb_cqi) <= 15
