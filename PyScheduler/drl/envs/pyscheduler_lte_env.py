"""
Gymnasium-среда для обучения DRL-агентов на реальном runtime PyScheduler.

Среда не переписывает симулятор и не заменяет `SimulationManager`.
Она использует уже существующие точки интеграции:
- `SimulationManager.initialize_runtime()`
- `SimulationManager.prepare_tti_scheduler_input()`
- `PDSCHAllocationSession`

Таким образом, агент учится на той же механике буферов, CQI, PDCCH, AMC и
resource-grid, которая используется при обычном запуске симуляции.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drl.pdsch_allocation_session import PDSCHAllocationSession
from drl.playground_adapter import (
    DRLPlaygroundObservation,
    DRLPlaygroundObservationAdapter,
    MODE_CURRENT_STEP,
    PLAYGROUND_MAX_AVG_TPUT_BPS,
    PLAYGROUND_MAX_BUFFER_BYTES,
    PLAYGROUND_N_CONTEXT_FEATURES,
    PLAYGROUND_N_UE_FEATURES,
)
from drl.simulation_bridge import (
    DRLPlaygroundCompatibilityReport,
    DRLPlaygroundSimulationConfig,
    DRLPlaygroundSnapshot,
    DRLPlaygroundStepSnapshot,
    DRLPlaygroundUEState,
)


@dataclass(slots=True)
class PreparedSchedulerStepContext:
    """
    Подготовленный контекст одного TTI для simulation-backed DRL-среды.

    Этот dataclass живет в `drl`, а не в ядре scheduler, чтобы не менять
    production-пайплайн `SchedulerInterface.schedule()`.
    """

    eligible_ues: List[Dict[str, Any]]
    windowed_ues: List[Dict[str, Any]]
    prioritized_ues: List[Dict[str, Any]]
    priority_list: List[Dict[str, Any]]
    priority_list_filtered: List[Dict[str, Any]]
    ues_with_pdcch: List[Dict[str, Any]]
    priority_calc_time_us: float
    priority_sort_time_us: float


def _jain_index(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    total = float(np.sum(arr))
    total_sq = float(np.sum(arr**2))
    if total_sq < 1e-12:
        return 0.0
    return float((total**2) / (arr.size * total_sq))


def _jain_index_masked(values: Sequence[float], mask: Sequence[bool]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    active_mask = np.asarray(mask, dtype=bool)
    active = arr[active_mask]
    if active.size == 0:
        return 1.0
    total = float(np.sum(active))
    total_sq = float(np.sum(active**2))
    if total_sq < 1e-12:
        return 1.0
    return float((total**2) / (active.size * total_sq))


class PySchedulerLteEnv(gym.Env):
    """
    DRL-среда поверх реального runtime PyScheduler.

    Один `step()` соответствует одному решению по текущему RBG внутри
    активного TTI. Когда все RBG текущего TTI обработаны, среда:
    - завершает scheduler pipeline;
    - считает reward по реальным метрикам;
    - автоматически переходит к следующему decision-point.

    Если между decision-point встречаются TTI без eligible UE, они
    автоматически пропускаются, а их reward аккумулируется в следующий
    возвращаемый transition. Это semi-MDP поведение здесь осознанное:
    агент делает шаги только там, где у него действительно есть выбор.
    """

    metadata = {"render_modes": []}
    N_UE_FEATURES = PLAYGROUND_N_UE_FEATURES
    N_CONTEXT_FEATURES = PLAYGROUND_N_CONTEXT_FEATURES

    def __init__(
        self,
        simulation_factory: Callable[..., Any],
        *,
        max_n_ue: int,
        reward_mode: str = "per_tti",
        reward_window: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rate_scale_bps: float = 1e6,
        jfi_target: float = 0.70,
        lambda_jfi: float = 2.0,
        strict_observation: bool = True,
        ensure_nonempty_action_mask: bool = False,
        wb_cqi_report_period_tti: Optional[int] = None,
        max_buffer_bytes: float = PLAYGROUND_MAX_BUFFER_BYTES,
        max_avg_tput_bps: float = PLAYGROUND_MAX_AVG_TPUT_BPS,
    ) -> None:
        super().__init__()

        if not callable(simulation_factory):
            raise TypeError("simulation_factory должен быть вызываемым объектом.")
        if max_n_ue < 1:
            raise ValueError("max_n_ue должен быть >= 1.")
        if reward_mode not in ("per_tti", "delayed"):
            raise ValueError("reward_mode должен быть 'per_tti' или 'delayed'.")

        self.simulation_factory = simulation_factory
        self.max_n_ue = int(max_n_ue)
        self.n_ue = int(max_n_ue)
        self.ue_feature_dim = int(self.N_UE_FEATURES)
        self.context_dim = int(self.N_CONTEXT_FEATURES)
        self.n_rb_dl = 0
        self.rbg_size_rb = 0
        self.n_rbg = 0
        self.reward_mode = reward_mode
        self.reward_window = max(int(reward_window), 1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rate_scale_bps = max(float(rate_scale_bps), 1.0)
        self.jfi_target = float(jfi_target)
        self.lambda_jfi = float(lambda_jfi)
        self.strict_observation = bool(strict_observation)
        self.ensure_nonempty_action_mask = bool(ensure_nonempty_action_mask)
        self.wb_cqi_report_period_tti = wb_cqi_report_period_tti

        obs_dim = self.max_n_ue * self.N_UE_FEATURES + self.N_CONTEXT_FEATURES
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_n_ue)

        self.observation_adapter = DRLPlaygroundObservationAdapter(
            max_n_ue=self.max_n_ue,
            episode_len_tti=None,
            wb_cqi_report_period_tti=self.wb_cqi_report_period_tti,
            max_buffer_bytes=max_buffer_bytes,
            max_avg_tput_bps=max_avg_tput_bps,
            strict_mode=self.strict_observation,
            ensure_nonempty_action_mask=self.ensure_nonempty_action_mask,
        )

        self.manager = None
        self.scheduler = None
        self.current_session: Optional[PDSCHAllocationSession] = None
        self.current_pdsch_context = None
        self.current_users: List[Dict[str, Any]] = []
        self.current_observation_packet: Optional[DRLPlaygroundObservation] = None
        self.current_tti: int = 0
        self.episode_len_tti: int = 0
        self._reward_accum: float = 0.0
        self._window_count: int = 0
        self._latest_info: Dict[str, Any] = {}

        self.history: Dict[str, List[float]] = {
            "throughput": [],
            "se": [],
            "jfi": [],
            "jfi_all": [],
            "reward": [],
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.manager = self._build_manager(seed=seed, options=options)
        self._validate_manager(self.manager)
        self.manager.initialize_runtime()
        self.scheduler = self.manager.scheduler
        self.episode_len_tti = int(getattr(self.manager.sim_config, "sim_duration", 0) or 0)
        if self.episode_len_tti <= 0:
            raise ValueError("simulation_factory должен возвращать manager с sim_duration > 0.")

        if self.wb_cqi_report_period_tti is not None:
            report_period = max(int(self.wb_cqi_report_period_tti), 1)
            self.scheduler.wb_cqi_upd_interval = report_period
            self.scheduler.sb_cqi_upd_interval = report_period
            self.observation_adapter.wb_cqi_report_period_tti = report_period
        else:
            report_period = int(getattr(self.scheduler, "wb_cqi_upd_interval", 1) or 1)
            self.observation_adapter.wb_cqi_report_period_tti = max(report_period, 1)
        self.observation_adapter.episode_len_tti = self.episode_len_tti
        self.n_rb_dl = int(getattr(self.scheduler.lte_grid, "rb_per_slot", 0) or 0)
        self.rbg_size_rb = int(self.scheduler.lte_grid.GET_RBG_SIZE())
        if self.rbg_size_rb > 0:
            self.n_rbg = int((self.n_rb_dl + self.rbg_size_rb - 1) // self.rbg_size_rb)
        else:
            self.n_rbg = 0

        self.current_session = None
        self.current_pdsch_context = None
        self.current_users = []
        self.current_observation_packet = None
        self.current_tti = 0
        self._reward_accum = 0.0
        self._window_count = 0
        self._latest_info = {}
        self.history = {
            "throughput": [],
            "se": [],
            "jfi": [],
            "jfi_all": [],
            "reward": [],
        }

        _, terminated = self._advance_to_next_decision_point()
        if terminated:
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = self._build_terminal_info()
            return zero_obs, info

        info = self._build_info(invalid_action=False, padded_invalid_action=False)
        return self.current_observation_packet.observation.copy(), info

    def step(self, action: int):
        if self.current_observation_packet is None or self.current_session is None:
            raise RuntimeError("Среда не подготовлена к step(). Сначала вызовите reset().")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Некорректное действие вне action_space: {action}")

        invalid_action = False
        padded_invalid_action = False
        resolved_action = int(action)
        action_mask = self.current_observation_packet.action_mask
        actual_n_ue = int(self.current_observation_packet.actual_n_ue)

        if resolved_action >= actual_n_ue:
            invalid_action = True
            padded_invalid_action = True
        elif not bool(action_mask[resolved_action]):
            invalid_action = True

        chosen_ue_id = None
        if invalid_action:
            valid_actions = [
                idx
                for idx, is_valid in enumerate(action_mask[:actual_n_ue])
                if bool(is_valid)
            ]
            if valid_actions:
                resolved_action = int(valid_actions[0])
                chosen_ue_id = int(
                    self.current_observation_packet.ue_ids_in_order[resolved_action]
                )
            else:
                resolved_action = 0
        else:
            chosen_ue_id = int(self.current_observation_packet.ue_ids_in_order[resolved_action])

        self.current_session.record_raw_action(int(action), invalid_action)
        if chosen_ue_id is not None:
            self.current_session.apply_selected_ue(chosen_ue_id)

        reward = 0.0
        terminated = False

        if self.current_session.is_done() or not any(self.current_session.get_action_mask()):
            reward = self._complete_current_tti()
            skipped_reward, terminated = self._advance_to_next_decision_point()
            reward += skipped_reward
        else:
            self.current_observation_packet = self._build_current_observation_packet()

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = self._build_terminal_info(
                invalid_action=invalid_action,
                padded_invalid_action=padded_invalid_action,
                resolved_action=resolved_action,
            )
        else:
            obs = self.current_observation_packet.observation.copy()
            info = self._build_info(
                invalid_action=invalid_action,
                padded_invalid_action=padded_invalid_action,
                resolved_action=resolved_action,
            )

        return obs, float(reward), bool(terminated), False, info

    def _build_manager(self, seed: Optional[int], options: Optional[Dict]) -> Any:
        """
        Создать новый SimulationManager через пользовательскую factory-функцию.
        """
        attempts: List[Dict[str, Any]] = []
        if seed is not None and options is not None:
            attempts.append({"seed": seed, "options": options})
        if options is not None:
            attempts.append({"options": options})
        if seed is not None:
            attempts.append({"seed": seed})
        attempts.append({})

        last_error = None
        for kwargs in attempts:
            try:
                return self.simulation_factory(**kwargs)
            except TypeError as exc:
                last_error = exc
                continue

        raise TypeError(
            "Не удалось вызвать simulation_factory ни с одним из поддерживаемых наборов аргументов."
        ) from last_error

    @staticmethod
    def _validate_manager(manager: Any) -> None:
        required_attrs = (
            "initialize_runtime",
            "prepare_tti_scheduler_input",
            "sim_config",
            "ue_collection",
            "base_station",
        )
        missing = [name for name in required_attrs if not hasattr(manager, name)]
        if missing:
            raise TypeError(
                f"simulation_factory вернул объект без обязательных полей/методов: {missing}"
            )

    def _advance_to_next_decision_point(self) -> Tuple[float, bool]:
        """
        Продвинуть среду до следующего decision-point или завершения эпизода.

        Возвращает накопленный reward по TTI, которые завершились между
        decision-point.
        """
        accumulated_reward = 0.0

        while self.current_tti < self.episode_len_tti:
            prepared_users = self.manager.prepare_tti_scheduler_input(self.current_tti)
            prepared_active_ids = self._extract_active_ue_ids(prepared_users)
            prev_avg_tput = self._snapshot_average_throughput(prepared_users)

            self.current_users = prepared_users
            t_sch_start = time.perf_counter()
            pdsch_context = self._prepare_scheduler_step_context(
                self.current_tti,
                prepared_users,
            )

            if pdsch_context is None:
                self.scheduler._empty_result()
                accumulated_reward += self._collect_completed_tti_metrics(
                    tti=self.current_tti,
                    active_ue_ids=prepared_active_ids,
                    prev_avg_tput=prev_avg_tput,
                )
                self.current_tti += 1
                continue

            self.current_pdsch_context = pdsch_context
            self.current_session = self._create_pdsch_session(
                tti=self.current_tti,
                eligible_ues=pdsch_context.eligible_ues,
                ues_with_pdcch=pdsch_context.ues_with_pdcch,
            )
            self._current_tti_start_time = t_sch_start
            self._current_tti_prev_avg_tput = prev_avg_tput
            self._current_tti_active_ue_ids = prepared_active_ids
            self.current_observation_packet = self._build_current_observation_packet()
            return accumulated_reward, False

        self.current_session = None
        self.current_pdsch_context = None
        self.current_observation_packet = None
        return accumulated_reward, True

    def _create_pdsch_session(
        self,
        *,
        tti: int,
        eligible_ues: List[Dict[str, Any]],
        ues_with_pdcch: List[Dict[str, Any]],
    ) -> PDSCHAllocationSession:
        return PDSCHAllocationSession(
            tti=tti,
            eligible_ues=eligible_ues,
            ues_with_pdcch=ues_with_pdcch,
            lte_grid=self.scheduler.lte_grid,
            get_reported_wb_cqi=self.scheduler._get_wb_cqi,
            get_cqi_for_rbg=self._get_cqi_for_rbg,
            bits_per_rb_fn=self.scheduler.amc.GET_BITS_PER_RB,
            build_step_snapshot_fn=lambda **kwargs: self._build_current_step_snapshot(
                tti=tti,
                eligible_ues=eligible_ues,
                eligible_ue_ids=kwargs["eligible_ue_ids"],
                remaining_buffer_bits=kwargs["remaining_buffer_bits"],
                alloc_rbg_counts=kwargs["alloc_rbg_counts"],
                action_mask=kwargs["action_mask"],
                current_rbg_index=kwargs["current_rbg_index"],
                total_rbg=kwargs["total_rbg"],
            ),
        )

    def _build_current_observation_packet(self) -> DRLPlaygroundObservation:
        snapshot = self.current_session.build_step_snapshot(self.current_session.get_action_mask())
        return self.observation_adapter.build(
            snapshot=snapshot,
            mode=MODE_CURRENT_STEP,
            ue_ids=self.current_session.eligible_ue_ids,
        )

    def _build_current_step_snapshot(
        self,
        *,
        tti: int,
        eligible_ues: List[Dict[str, Any]],
        eligible_ue_ids: List[int],
        remaining_buffer_bits: Dict[int, int],
        alloc_rbg_counts: Dict[int, int],
        action_mask: List[int],
        current_rbg_index: int,
        total_rbg: int,
    ) -> DRLPlaygroundSnapshot:
        allocated_rbg_total = sum(alloc_rbg_counts.values())
        allocated_fraction_progress = (
            allocated_rbg_total / total_rbg if total_rbg > 0 else 0.0
        )

        ue_states: List[DRLPlaygroundUEState] = []
        for user in eligible_ues:
            ue_id = int(user["UE_ID"])
            ue = user["ue"]
            reported_wb_cqi, wb_cqi_age_tti = self._get_reported_wb_cqi_and_age(
                ue_id=ue_id,
                current_tti=tti,
                fallback_cqi=int(user.get("cqi", 0) or 0),
            )

            ue_states.append(
                DRLPlaygroundUEState(
                    ue_id=ue_id,
                    reported_wb_cqi=reported_wb_cqi,
                    true_wb_cqi=int(getattr(ue, "cqi", user.get("cqi", 0)) or 0),
                    wb_cqi_age_tti=wb_cqi_age_tti,
                    active_flag=remaining_buffer_bits.get(ue_id, 0) > 0,
                    buffer_bytes=int(remaining_buffer_bits.get(ue_id, 0) // 8),
                    average_throughput_bps=float(
                        getattr(ue, "average_throughput", 0.0) or 0.0
                    ),
                    current_dl_throughput_bps=float(
                        getattr(ue, "current_dl_throughput", 0.0) or 0.0
                    ),
                    alloc_rbg_count_tti=int(alloc_rbg_counts.get(ue_id, 0)),
                    alloc_rbg_frac_tti=(
                        alloc_rbg_counts.get(ue_id, 0) / total_rbg
                        if total_rbg > 0
                        else 0.0
                    ),
                    sinr_db=float(getattr(ue, "SINR", 0.0) or 0.0),
                    reported_sb_cqi=self._get_reported_sb_cqi(
                        ue_id=ue_id,
                        fallback_sb_cqi=list(user.get("sbb_cqi", []) or []),
                    ),
                )
            )

        compatibility = DRLPlaygroundCompatibilityReport(
            exact_per_rbg_step_supported=True,
            reported_vs_true_wb_cqi_supported=True,
            wb_cqi_age_supported=True,
            alloc_frac_this_tti_supported=True,
            current_rbg_index_supported=True,
            scheduler_eligibility_mask_supported=True,
            notes=[
                "current_rbg_index и allocated_rbg_fraction_progress формируются внутри simulation-backed env.",
                "action_mask учитывает eligibility текущего TTI, PDCCH и остаток буфера.",
            ],
        )

        return DRLPlaygroundSnapshot(
            simulation_config=DRLPlaygroundSimulationConfig(
                sim_duration_tti=self.episode_len_tti,
                update_interval_tti=int(
                    getattr(self.manager.sim_config, "update_interval", 0) or 0
                ),
                mobility_update_interval_tti=int(
                    getattr(self.manager.sim_config, "mobility_update_interval", 0) or 0
                ),
                channel_update_interval_tti=int(
                    getattr(self.manager.sim_config, "channel_update_interval", 0) or 0
                ),
                traffic_mode=(
                    "legacy_simple_buffer"
                    if bool(getattr(self.manager.sim_config, "use_legacy_traffic", False))
                    else "layered_buffer"
                ),
                traffic_generator=type(getattr(self.manager, "traffic_gen", None)).__name__,
                buffer_mode=(
                    "simple_buffer"
                    if bool(getattr(self.manager.base_station, "use_simple_buffer", False))
                    else "layered_buffer"
                ),
                scheduler_algorithm=str(
                    getattr(self.manager.sched_config, "algorithm", "") or ""
                ),
                scheduler_max_dl_ue_tti=getattr(
                    self.scheduler,
                    "max_dl_ue_tti",
                    None,
                ),
                scheduler_window_size=int(getattr(self.scheduler, "window_size", 0) or 0),
                scheduler_window_enabled=bool(
                    getattr(self.scheduler, "enable_window", False)
                ),
                stats_enabled=bool(getattr(self.manager.stats_config, "enabled", False)),
                bandwidth_mhz=float(
                    getattr(self.manager.base_station, "bandwidth", 0.0) or 0.0
                ),
                frequency_ghz=float(
                    getattr(self.manager.base_station, "frequency_GHz", 0.0) or 0.0
                ),
                n_rb_dl=int(getattr(self.scheduler.lte_grid, "rb_per_slot", 0) or 0),
                rbg_size_rb=int(self.scheduler.lte_grid.GET_RBG_SIZE()),
                n_rbg=int(total_rbg),
                channel_model_type=str(
                    getattr(self.manager.base_station, "ch_model_type", "") or ""
                ),
                enable_tdl=bool(getattr(self.manager.base_station, "enable_tdl", False)),
            ),
            step_snapshot=DRLPlaygroundStepSnapshot(
                current_time=tti,
                current_tti=tti,
                current_rbg_index=current_rbg_index,
                allocated_rbg_fraction_progress=allocated_fraction_progress,
                allocated_rbg_fraction_final_tti=allocated_fraction_progress,
                n_rb_dl=int(getattr(self.scheduler.lte_grid, "rb_per_slot", 0) or 0),
                rbg_size_rb=int(self.scheduler.lte_grid.GET_RBG_SIZE()),
                n_rbg=int(total_rbg),
            ),
            ue_states=ue_states,
            action_mask=[int(value) for value in action_mask],
            compatibility=compatibility,
            scheduler_result=None,
        )

    def _get_reported_wb_cqi_and_age(
        self,
        *,
        ue_id: int,
        current_tti: int,
        fallback_cqi: int,
    ) -> Tuple[int, Optional[int]]:
        if ue_id in self.scheduler.cqi_map:
            cqi_entry = self.scheduler.cqi_map[ue_id]
            age = max(
                0,
                current_tti - int(getattr(cqi_entry, "last_wb_update", 0) or 0),
            )
            return int(getattr(cqi_entry, "wb_cqi", 0) or 0), age
        return int(fallback_cqi), None

    def _get_reported_sb_cqi(
        self,
        *,
        ue_id: int,
        fallback_sb_cqi: List[int],
    ) -> List[int]:
        if ue_id in self.scheduler.cqi_map:
            return [
                int(cqi)
                for cqi in list(getattr(self.scheduler.cqi_map[ue_id], "sb_cqi", []) or [])
            ]
        return [int(cqi) for cqi in list(fallback_sb_cqi or [])]

    def _get_cqi_for_rbg(self, ue_id: int, rbg_idx: int) -> int:
        sb_cqi = self.scheduler._get_sb_cqi(ue_id)
        if sb_cqi and rbg_idx < len(sb_cqi) and sb_cqi[rbg_idx] > 0:
            return int(sb_cqi[rbg_idx])
        return int(self.scheduler._get_wb_cqi(ue_id))

    def _prepare_scheduler_step_context(
        self,
        tti: int,
        users: List[Dict[str, Any]],
    ) -> Optional[PreparedSchedulerStepContext]:
        """
        Подготовить scheduler state до этапа PDSCH allocation.

        Если конкретный scheduler предоставляет собственный служебный хук,
        среда использует его. Иначе оркестрация выполняется локально в DRL,
        чтобы не расширять `SCHEDULER.py`.
        """
        external_hook = getattr(self.scheduler, "_prepare_pdsch_context", None)
        if callable(external_hook):
            return external_hook(tti, users)

        self.scheduler._last_tti = tti

        if tti % self.scheduler.wb_cqi_upd_interval == 0:
            self.scheduler._refresh_cqi(tti, users)

        eligible_ues = self.scheduler._filter_eligible_ues(tti, users)
        if not eligible_ues:
            return None

        self.scheduler._update_active_window(eligible_ues)

        windowed_ues = self.scheduler.filter_by_window(eligible_ues)
        self.scheduler._last_windowed_users = list(windowed_ues)
        if not windowed_ues:
            return None

        t_priority_start = time.perf_counter()
        prioritized_ues = self.scheduler._calculate_priorities(windowed_ues, tti)
        self.scheduler._last_prioritized_users = prioritized_ues
        t_priority_end = time.perf_counter()
        priority_calc_time_us = (t_priority_end - t_priority_start) * 1_000_000

        t_sort_start = time.perf_counter()
        priority_list = self.scheduler._form_priority_list(prioritized_ues, tti)
        t_sort_end = time.perf_counter()
        priority_sort_time_us = (t_sort_end - t_sort_start) * 1_000_000

        self.scheduler._last_priority_list_size = len(priority_list)
        if priority_list:
            self.scheduler._last_avg_priority_value = (
                sum(user.get("priority", 0) for user in priority_list) / len(priority_list)
            )
        else:
            return None

        priority_list_filtered = self.scheduler._apply_pdsch_estimation(priority_list, tti)
        self.scheduler._last_priority_list = priority_list_filtered
        self.scheduler._last_priority_list_full = priority_list

        ues_with_pdcch = self.scheduler._allocate_pdcch(priority_list_filtered)
        self.scheduler._last_pdcch_blocked_count = (
            len(priority_list_filtered) - len(ues_with_pdcch)
        )
        if not ues_with_pdcch:
            return None

        return PreparedSchedulerStepContext(
            eligible_ues=eligible_ues,
            windowed_ues=windowed_ues,
            prioritized_ues=prioritized_ues,
            priority_list=priority_list,
            priority_list_filtered=priority_list_filtered,
            ues_with_pdcch=ues_with_pdcch,
            priority_calc_time_us=priority_calc_time_us,
            priority_sort_time_us=priority_sort_time_us,
        )

    def _finalize_scheduler_step(
        self,
        *,
        tti: int,
        users: List[Dict[str, Any]],
        eligible_ues: List[Dict[str, Any]],
        allocation: Dict[int, List[int]],
        t_sch_start: float,
        priority_calc_time_us: float,
        priority_sort_time_us: float,
    ) -> Dict[str, Any]:
        """
        Завершить scheduler-цикл после того, как DRL-среда собрала allocation.

        Логика находится в `drl`, чтобы сохранить базовый `schedule()` без
        дополнительных production-хуков.
        """
        external_hook = getattr(self.scheduler, "_finalize_schedule", None)
        if callable(external_hook):
            return external_hook(
                tti=tti,
                users=users,
                eligible_ues=eligible_ues,
                allocation=allocation,
                t_sch_start=t_sch_start,
                priority_calc_time_us=priority_calc_time_us,
                priority_sort_time_us=priority_sort_time_us,
            )

        self.scheduler._process_buffers(tti, users, allocation)

        allocated_rbs = sum(len(rbs) for rbs in allocation.values())
        active_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
        self.scheduler._update_stats(tti, len(eligible_ues), allocated_rbs, active_ues)
        self.scheduler._last_allocation = allocation.copy()
        self.scheduler._last_users = users

        t_sch_end = time.perf_counter()
        sch_time_us = (t_sch_end - t_sch_start) * 1_000_000
        self.scheduler._save_timing_stats(
            sch_time_us,
            priority_calc_time_us,
            priority_sort_time_us,
        )

        return self.scheduler._build_result(allocation, users, eligible_ues, tti)

    def _complete_current_tti(self) -> float:
        result = self._finalize_scheduler_step(
            tti=self.current_tti,
            users=self.current_users,
            eligible_ues=self.current_pdsch_context.eligible_ues,
            allocation=self.current_session.allocation,
            t_sch_start=self._current_tti_start_time,
            priority_calc_time_us=self.current_pdsch_context.priority_calc_time_us,
            priority_sort_time_us=self.current_pdsch_context.priority_sort_time_us,
        )
        reward = self._collect_completed_tti_metrics(
            tti=self.current_tti,
            active_ue_ids=self._current_tti_active_ue_ids,
            prev_avg_tput=self._current_tti_prev_avg_tput,
        )
        self._latest_info["scheduler_result"] = result
        self.current_tti += 1
        return reward

    def _collect_completed_tti_metrics(
        self,
        *,
        tti: int,
        active_ue_ids: set[int],
        prev_avg_tput: Dict[int, float],
    ) -> float:
        all_users = list(self.manager.ue_collection.GET_ALL_USERS())
        ue_ids = [int(ue.UE_ID) for ue in all_users]
        curr_avg = np.asarray(
            [float(getattr(ue, "average_throughput", 0.0) or 0.0) for ue in all_users],
            dtype=np.float64,
        )
        prev_avg = np.asarray(
            [float(prev_avg_tput.get(ue_id, 0.0)) for ue_id in ue_ids],
            dtype=np.float64,
        )
        active_mask = np.asarray([ue_id in active_ue_ids for ue_id in ue_ids], dtype=bool)

        amc_stats = self.scheduler.amc.get_stats()
        throughput_bps = float(amc_stats.get("dl_throughput_sum_kbps", 0.0) or 0.0) * 1000.0
        bandwidth_hz = float(getattr(self.manager.base_station, "bandwidth", 0.0) or 0.0) * 1e6
        se_bps_hz = throughput_bps / bandwidth_hz if bandwidth_hz > 0 else 0.0
        jfi_all = _jain_index(curr_avg)
        jfi_active = _jain_index_masked(curr_avg, active_mask)

        reward = self._compute_reward(
            se_bps_hz=se_bps_hz,
            jfi=jfi_active,
            prev_avg_tput=prev_avg,
            curr_avg_tput=curr_avg,
        )

        self.history["throughput"].append(throughput_bps / 1e6)
        self.history["se"].append(se_bps_hz)
        self.history["jfi"].append(jfi_active)
        self.history["jfi_all"].append(jfi_all)

        emitted_reward = self._accumulate_reward(reward)
        self._latest_info = {
            "last_completed_tti": int(tti),
            "throughput_mbps": float(throughput_bps / 1e6),
            "se_bps_hz": float(se_bps_hz),
            "jfi_active": float(jfi_active),
            "jfi_all": float(jfi_all),
        }
        return emitted_reward

    def _compute_reward(
        self,
        *,
        se_bps_hz: float,
        jfi: float,
        prev_avg_tput: np.ndarray,
        curr_avg_tput: np.ndarray,
    ) -> float:
        prev_u = np.mean(np.log1p(prev_avg_tput / self.rate_scale_bps)) if prev_avg_tput.size else 0.0
        curr_u = np.mean(np.log1p(curr_avg_tput / self.rate_scale_bps)) if curr_avg_tput.size else 0.0
        pf_delta = curr_u - prev_u
        se_term = np.log1p(max(se_bps_hz, 0.0))
        jfi_penalty = self.lambda_jfi * max(0.0, self.jfi_target - jfi) ** 2
        return float(self.alpha * se_term + self.beta * pf_delta - jfi_penalty)

    def _accumulate_reward(self, tti_reward: float) -> float:
        if self.reward_mode == "per_tti":
            self.history["reward"].append(float(tti_reward))
            return float(tti_reward)

        self._reward_accum += float(tti_reward)
        self._window_count += 1
        if self._window_count >= self.reward_window:
            avg_reward = self._reward_accum / self.reward_window
            self.history["reward"].append(float(avg_reward))
            self._reward_accum = 0.0
            self._window_count = 0
            return float(avg_reward)

        return 0.0

    def _extract_active_ue_ids(self, users: List[Dict[str, Any]]) -> set[int]:
        buffer_manager = self.manager.base_station.buffer_manager
        active_ids: set[int] = set()

        for user in users:
            ue_id = int(user["UE_ID"])
            if not buffer_manager.ue_has_buffer(ue_id):
                continue
            buffer_status = buffer_manager.get_buffer_status(ue_id)
            buffer_bytes = sum(int(status.buffer_size) for status in buffer_status)
            if buffer_bytes > 0:
                active_ids.add(ue_id)
        return active_ids

    @staticmethod
    def _snapshot_average_throughput(users: List[Dict[str, Any]]) -> Dict[int, float]:
        snapshot: Dict[int, float] = {}
        for user in users:
            ue = user.get("ue")
            ue_id = int(user.get("UE_ID"))
            snapshot[ue_id] = float(getattr(ue, "average_throughput", 0.0) or 0.0)
        return snapshot

    def _build_info(
        self,
        *,
        invalid_action: bool,
        padded_invalid_action: bool,
        resolved_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        packet = self.current_observation_packet
        info = dict(self._latest_info)
        info.update(
            {
                "tti": int(self.current_tti),
                "rbg_step": int(self.current_session.current_rbg_index) if self.current_session else 0,
                "actual_n_ue": int(packet.actual_n_ue),
                "max_n_ue": int(packet.max_n_ue),
                "action_mask": packet.action_mask.copy(),
                "ue_ids_in_order": list(packet.ue_ids_in_order),
                "invalid_action": bool(invalid_action),
                "padded_invalid_action": bool(padded_invalid_action),
                "resolved_action": None if resolved_action is None else int(resolved_action),
                "exact_env_match": bool(packet.exact_env_match),
                "adapter_notes": list(packet.notes),
            }
        )
        return info

    def _build_terminal_info(
        self,
        *,
        invalid_action: bool = False,
        padded_invalid_action: bool = False,
        resolved_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        info = dict(self._latest_info)
        info.update(
            {
                "tti": int(self.episode_len_tti),
                "rbg_step": 0,
                "actual_n_ue": 0,
                "max_n_ue": int(self.max_n_ue),
                "action_mask": np.zeros(self.max_n_ue, dtype=bool),
                "ue_ids_in_order": [],
                "invalid_action": bool(invalid_action),
                "padded_invalid_action": bool(padded_invalid_action),
                "resolved_action": None if resolved_action is None else int(resolved_action),
                "exact_env_match": True,
                "adapter_notes": [],
            }
        )
        return info

    def get_episode_summary(self) -> Dict[str, float]:
        if not self.history["se"]:
            return {}

        throughput = np.asarray(self.history["throughput"], dtype=np.float64)
        se = np.asarray(self.history["se"], dtype=np.float64)
        jfi_active = np.asarray(self.history["jfi"], dtype=np.float64)
        jfi_all = np.asarray(self.history["jfi_all"], dtype=np.float64)
        reward = np.asarray(self.history["reward"], dtype=np.float64)

        return {
            "mean_throughput_mbps": float(np.mean(throughput)) if throughput.size else 0.0,
            "mean_se_bps_hz": float(np.mean(se)) if se.size else 0.0,
            "mean_jfi": float(np.mean(jfi_active)) if jfi_active.size else 0.0,
            "mean_jfi_active": float(np.mean(jfi_active)) if jfi_active.size else 0.0,
            "mean_jfi_all": float(np.mean(jfi_all)) if jfi_all.size else 0.0,
            "mean_reward": float(np.mean(reward)) if reward.size else 0.0,
            "min_jfi": float(np.min(jfi_active)) if jfi_active.size else 0.0,
            "min_jfi_active": float(np.min(jfi_active)) if jfi_active.size else 0.0,
            "min_jfi_all": float(np.min(jfi_all)) if jfi_all.size else 0.0,
            "max_se_bps_hz": float(np.max(se)) if se.size else 0.0,
        }

    def close(self) -> None:
        """
        Освободить ссылки на runtime-объекты текущего эпизода.
        """
        self.current_session = None
        self.current_pdsch_context = None
        self.current_observation_packet = None
        self.current_users = []
        self.scheduler = None
        self.manager = None
