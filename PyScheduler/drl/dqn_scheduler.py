"""
Inference-only DQN scheduler для PyScheduler.
"""

from typing import Any, Dict, List, Optional

from SCHEDULER import FDxFairGreedyScheduler
from drl.dqn_model_runner import DQNModelRunner
from drl.playground_adapter import DRLPlaygroundObservationAdapter, MODE_CURRENT_STEP
from drl.simulation_bridge import (
    DRLPlaygroundCompatibilityReport,
    DRLPlaygroundSimulationConfig,
    DRLPlaygroundSnapshot,
    DRLPlaygroundStepSnapshot,
    DRLPlaygroundUEState,
)


class DqnScheduler(FDxFairGreedyScheduler):
    """
    DQN-backed scheduler с per-RBG inference внутри одного TTI.

    Базовый pipeline PyScheduler сохраняется:
    - CQI refresh
    - eligibility
    - active window
    - PDCCH allocation
    - buffer processing
    - result formation

    Модель управляет только этапом `_allocate_pdsch()`.
    """

    def __init__(self, lte_grid, bs, **kwargs):
        dqn_model_path = kwargs.pop("dqn_model_path", None)
        dqn_max_n_ue = kwargs.pop("dqn_max_n_ue", None)
        dqn_wb_cqi_report_period_tti = kwargs.pop(
            "dqn_wb_cqi_report_period_tti", 5
        )
        dqn_episode_len_tti = kwargs.pop("dqn_episode_len_tti", None)
        dqn_strict_observation = kwargs.pop("dqn_strict_observation", False)
        dqn_deterministic = kwargs.pop("dqn_deterministic", True)
        dqn_policy_runner = kwargs.pop("dqn_policy_runner", None)

        super().__init__(lte_grid, bs, **kwargs)

        self.policy_runner = self._build_policy_runner(
            model_path=dqn_model_path,
            provided_runner=dqn_policy_runner,
            deterministic=dqn_deterministic,
        )
        resolved_max_n_ue = self._resolve_max_n_ue(
            configured_max_n_ue=dqn_max_n_ue,
            policy_runner=self.policy_runner,
        )
        self.observation_adapter = DRLPlaygroundObservationAdapter(
            max_n_ue=resolved_max_n_ue,
            episode_len_tti=dqn_episode_len_tti,
            wb_cqi_report_period_tti=dqn_wb_cqi_report_period_tti,
            strict_mode=bool(dqn_strict_observation),
            ensure_nonempty_action_mask=False,
        )

        self.dqn_model_path = dqn_model_path
        self.dqn_max_n_ue = resolved_max_n_ue
        self.dqn_wb_cqi_report_period_tti = int(
            max(dqn_wb_cqi_report_period_tti, 1)
        )
        self.dqn_episode_len_tti = dqn_episode_len_tti
        self.dqn_strict_observation = bool(dqn_strict_observation)
        self.dqn_deterministic = bool(dqn_deterministic)

        self._last_dqn_invalid_action_count = 0
        self._last_dqn_selected_ue_ids: List[int] = []
        self._last_dqn_raw_actions: List[int] = []
        self._last_dqn_step_count = 0

    def _allocate_pdsch(
        self,
        tti: int,
        ues_with_pdcch: List[Dict],
        eligible_ues: List[Dict],
    ) -> Dict[int, List[int]]:
        """
        Выполнить per-RBG распределение через DQN policy.
        """

        allocation = {user["UE_ID"]: [] for user in eligible_ues}
        if not ues_with_pdcch:
            self._reset_dqn_step_trace()
            return allocation

        total_rbg = self._get_total_rbg()
        if total_rbg <= 0:
            self._reset_dqn_step_trace()
            return allocation

        eligible_ue_ids = [int(user["UE_ID"]) for user in eligible_ues]
        pdcch_ue_ids = {int(user["UE_ID"]) for user in ues_with_pdcch}
        remaining_buffer_bits = {
            int(user["UE_ID"]): int(user.get("bs_buffer_size", 0) * 8)
            for user in eligible_ues
        }
        alloc_rbg_counts = {ue_id: 0 for ue_id in eligible_ue_ids}
        invalid_action_count = 0
        selected_ue_ids: List[int] = []
        raw_actions: List[int] = []
        last_served_idx = -1

        for rbg_idx in range(total_rbg):
            action_mask = self._build_action_mask(
                eligible_ue_ids=eligible_ue_ids,
                pdcch_ue_ids=pdcch_ue_ids,
                remaining_buffer_bits=remaining_buffer_bits,
            )
            if not any(action_mask):
                break

            snapshot = self._build_current_step_snapshot(
                tti=tti,
                eligible_ues=eligible_ues,
                eligible_ue_ids=eligible_ue_ids,
                remaining_buffer_bits=remaining_buffer_bits,
                alloc_rbg_counts=alloc_rbg_counts,
                action_mask=action_mask,
                current_rbg_index=rbg_idx,
                total_rbg=total_rbg,
            )
            adapted = self.observation_adapter.build(
                snapshot=snapshot,
                mode=MODE_CURRENT_STEP,
                ue_ids=eligible_ue_ids,
            )

            raw_action = int(
                self.policy_runner.predict(
                    adapted.observation,
                    adapted.action_mask,
                    deterministic=self.dqn_deterministic,
                )
            )
            raw_actions.append(raw_action)
            chosen_ue_id, invalid_action = self._resolve_model_action(
                raw_action=raw_action,
                adapted_action_mask=adapted.action_mask,
                ue_ids_in_order=adapted.ue_ids_in_order,
            )
            invalid_action_count += int(invalid_action)

            cqi = self._get_cqi_for_rbg(chosen_ue_id, rbg_idx)
            if cqi <= 0:
                continue

            if not self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, chosen_ue_id):
                continue

            rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
            allocation[chosen_ue_id].extend(rb_indices)

            bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)
            rbg_capacity_bits = len(rb_indices) * bits_per_rb
            remaining_buffer_bits[chosen_ue_id] = max(
                0,
                remaining_buffer_bits[chosen_ue_id] - rbg_capacity_bits,
            )
            alloc_rbg_counts[chosen_ue_id] += 1
            selected_ue_ids.append(chosen_ue_id)

            try:
                last_served_idx = eligible_ue_ids.index(chosen_ue_id)
            except ValueError:
                last_served_idx = last_served_idx

        if last_served_idx >= 0 and eligible_ue_ids:
            self.rr_ue_offset = (last_served_idx + 1) % len(eligible_ue_ids)

        self._last_dqn_invalid_action_count = invalid_action_count
        self._last_dqn_selected_ue_ids = selected_ue_ids
        self._last_dqn_raw_actions = raw_actions
        self._last_dqn_step_count = len(raw_actions)

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(
                f"[SCHEDULER.DqnScheduler TTI {tti}] "
                f"PDSCH: {allocated_ues} UE, {total_rb} RB total, "
                f"invalid_actions={invalid_action_count}"
            )

        return allocation

    def get_stats(self) -> Dict:
        """
        Расширить базовую статистику диагностикой DQN inference.
        """

        stats = super().get_stats()
        stats.update(
            {
                "dqn_invalid_action_count": self._last_dqn_invalid_action_count,
                "dqn_step_count": self._last_dqn_step_count,
                "dqn_invalid_action_rate": (
                    self._last_dqn_invalid_action_count / self._last_dqn_step_count
                    if self._last_dqn_step_count > 0
                    else 0.0
                ),
                "dqn_selected_ue_ids": list(self._last_dqn_selected_ue_ids),
                "dqn_raw_actions": list(self._last_dqn_raw_actions),
                "dqn_model_path": self.dqn_model_path,
            }
        )
        return stats

    def _build_current_step_snapshot(
        self,
        *,
        tti: int,
        eligible_ues: List[Dict],
        eligible_ue_ids: List[int],
        remaining_buffer_bits: Dict[int, int],
        alloc_rbg_counts: Dict[int, int],
        action_mask: List[int],
        current_rbg_index: int,
        total_rbg: int,
    ) -> DRLPlaygroundSnapshot:
        """
        Построить точный current-step snapshot для текущего RBG.
        """

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
                "current_rbg_index и allocated_rbg_fraction_progress формируются прямо внутри DqnScheduler per-RBG loop.",
                "action_mask дополнительно учитывает PDCCH доступность текущего TTI.",
            ],
        )

        return DRLPlaygroundSnapshot(
            simulation_config=DRLPlaygroundSimulationConfig(
                sim_duration_tti=self.dqn_episode_len_tti,
                update_interval_tti=1,
                mobility_update_interval_tti=0,
                channel_update_interval_tti=self.wb_cqi_upd_interval,
                traffic_mode=(
                    "legacy_simple_buffer"
                    if getattr(self.lte_grid.bs, "use_simple_buffer", True)
                    else "layered_buffer"
                ),
                traffic_generator="",
                buffer_mode=(
                    "simple_buffer"
                    if getattr(self.lte_grid.bs, "use_simple_buffer", True)
                    else "layered_buffer"
                ),
                scheduler_algorithm="DqnScheduler",
                scheduler_max_dl_ue_tti=self.max_dl_ue_tti,
                scheduler_window_size=self.window_size,
                scheduler_window_enabled=self.enable_window,
                stats_enabled=True,
                bandwidth_mhz=float(getattr(self.lte_grid, "bandwidth", 0.0) or 0.0),
                frequency_ghz=float(
                    getattr(self.lte_grid.bs, "frequency_GHz", 0.0) or 0.0
                ),
                n_rb_dl=int(getattr(self.lte_grid, "rb_per_slot", 0) or 0),
                rbg_size_rb=int(self.lte_grid.GET_RBG_SIZE()),
                n_rbg=int(total_rbg),
                channel_model_type=str(
                    getattr(self.lte_grid.bs, "ch_model_type", "") or ""
                ),
                enable_tdl=bool(getattr(self.lte_grid.bs, "enable_tdl", False)),
            ),
            step_snapshot=DRLPlaygroundStepSnapshot(
                current_time=tti,
                current_tti=tti,
                current_rbg_index=current_rbg_index,
                allocated_rbg_fraction_progress=allocated_fraction_progress,
                allocated_rbg_fraction_final_tti=allocated_fraction_progress,
                n_rb_dl=int(getattr(self.lte_grid, "rb_per_slot", 0) or 0),
                rbg_size_rb=int(self.lte_grid.GET_RBG_SIZE()),
                n_rbg=int(total_rbg),
            ),
            ue_states=ue_states,
            action_mask=[int(value) for value in action_mask],
            compatibility=compatibility,
            scheduler_result=None,
        )

    def _build_action_mask(
        self,
        *,
        eligible_ue_ids: List[int],
        pdcch_ue_ids: set[int],
        remaining_buffer_bits: Dict[int, int],
    ) -> List[int]:
        """
        Сформировать mask действий для текущего RBG.
        """

        mask: List[int] = []
        for ue_id in eligible_ue_ids:
            reported_cqi = self._get_wb_cqi(ue_id)
            is_valid = (
                ue_id in pdcch_ue_ids
                and remaining_buffer_bits.get(ue_id, 0) > 0
                and 1 <= reported_cqi <= 15
            )
            mask.append(int(is_valid))
        return mask

    def _resolve_model_action(
        self,
        *,
        raw_action: int,
        adapted_action_mask,
        ue_ids_in_order: List[int],
    ) -> tuple[int, bool]:
        """
        Преобразовать action модели в реальный UE_ID.
        """

        if 0 <= raw_action < len(ue_ids_in_order) and adapted_action_mask[raw_action]:
            return int(ue_ids_in_order[raw_action]), False

        valid_actions = [
            idx for idx, is_valid in enumerate(adapted_action_mask[: len(ue_ids_in_order)])
            if bool(is_valid)
        ]
        if not valid_actions:
            raise RuntimeError(
                "DqnScheduler получил invalid action при пустом наборе valid actions."
            )

        fallback_action = valid_actions[0]
        return int(ue_ids_in_order[fallback_action]), True

    def _get_total_rbg(self) -> int:
        """
        Получить количество RBG в одном TTI.
        """

        rbg_size = int(self.lte_grid.GET_RBG_SIZE())
        if rbg_size <= 0:
            return 0
        return int((self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size)

    def _get_cqi_for_rbg(self, ue_id: int, rbg_idx: int) -> int:
        """
        Получить CQI для конкретного RBG с fallback на WB CQI.
        """

        sb_cqi = self._get_sb_cqi(ue_id)
        if sb_cqi and rbg_idx < len(sb_cqi) and sb_cqi[rbg_idx] > 0:
            return int(sb_cqi[rbg_idx])
        return int(self._get_wb_cqi(ue_id))

    def _get_reported_wb_cqi_and_age(
        self,
        *,
        ue_id: int,
        current_tti: int,
        fallback_cqi: int,
    ) -> tuple[int, Optional[int]]:
        """
        Получить reported WB CQI и его возраст.
        """

        if ue_id in self.cqi_map:
            cqi_entry = self.cqi_map[ue_id]
            age = max(0, current_tti - int(getattr(cqi_entry, "last_wb_update", 0) or 0))
            return int(getattr(cqi_entry, "wb_cqi", 0) or 0), age
        return int(fallback_cqi), None

    def _get_reported_sb_cqi(
        self,
        *,
        ue_id: int,
        fallback_sb_cqi: List[int],
    ) -> List[int]:
        """
        Получить reported SB CQI из CQI map.
        """

        if ue_id in self.cqi_map:
            return [int(cqi) for cqi in list(getattr(self.cqi_map[ue_id], "sb_cqi", []) or [])]
        return [int(cqi) for cqi in list(fallback_sb_cqi or [])]

    def _reset_dqn_step_trace(self) -> None:
        """
        Сбросить внутренний trace последнего inference TTI.
        """

        self._last_dqn_invalid_action_count = 0
        self._last_dqn_selected_ue_ids = []
        self._last_dqn_raw_actions = []
        self._last_dqn_step_count = 0

    @staticmethod
    def _build_policy_runner(
        *,
        model_path: Optional[str],
        provided_runner: Optional[Any],
        deterministic: bool,
    ) -> Any:
        """
        Построить объект runner для DQN policy.
        """

        if provided_runner is not None:
            return provided_runner

        if not model_path:
            raise ValueError(
                "Для DqnScheduler необходимо указать dqn_model_path или dqn_policy_runner."
            )

        return DQNModelRunner(
            model_path=str(model_path),
            deterministic=deterministic,
        )

    @staticmethod
    def _resolve_max_n_ue(
        *,
        configured_max_n_ue: Optional[int],
        policy_runner: Any,
    ) -> Optional[int]:
        """
        Определить max_n_ue для observation adapter.
        """

        if configured_max_n_ue is not None:
            return int(configured_max_n_ue)

        runner_max_n_ue = getattr(policy_runner, "max_n_ue", None)
        if runner_max_n_ue is not None:
            return int(runner_max_n_ue)

        return None
