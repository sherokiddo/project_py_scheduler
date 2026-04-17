"""
Inference-only PPO ranker scheduler для PyScheduler.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from SCHEDULER import SchedulerInterface
from drl.model_runners import PPORankerModelRunner
from drl.pdsch_allocation_session import PDSCHAllocationSession
from drl.playground_adapter import DRLPlaygroundObservationAdapter, MODE_PROXY_START_TTI
from drl.simulation_bridge import (
    DRLPlaygroundCompatibilityReport,
    DRLPlaygroundSimulationConfig,
    DRLPlaygroundSnapshot,
    DRLPlaygroundStepSnapshot,
    DRLPlaygroundUEState,
)


class PpoRankerScheduler(SchedulerInterface):
    """
    TTI-level PPO ranker для runtime PyScheduler.

    Планировщик делает один inference на весь TTI, получает score-вектор по UE,
    переводит его в bounded rank-weight и затем завершает PDSCH allocation через
    hybrid FD/PF метрику:

        combined_metric(u, k) = rank_weight(u) * fd_pf_metric(u, k)
    """

    def __init__(self, lte_grid, bs, **kwargs):
        ppo_ranker_model_path = kwargs.pop("ppo_ranker_model_path", None)
        ppo_ranker_max_n_ue = kwargs.pop("ppo_ranker_max_n_ue", None)
        ppo_ranker_wb_cqi_report_period_tti = kwargs.pop(
            "ppo_ranker_wb_cqi_report_period_tti", 5
        )
        ppo_ranker_episode_len_tti = kwargs.pop("ppo_ranker_episode_len_tti", None)
        ppo_ranker_deterministic = kwargs.pop("ppo_ranker_deterministic", True)
        ppo_ranker_inference_device = kwargs.pop(
            "ppo_ranker_inference_device", "cpu"
        )
        ppo_ranker_policy_runner = kwargs.pop("ppo_ranker_policy_runner", None)
        ppo_ranker_rank_weight_beta = kwargs.pop("ppo_ranker_rank_weight_beta", 0.3)
        ppo_ranker_pf_epsilon_bps = kwargs.pop("ppo_ranker_pf_epsilon_bps", 1e-6)

        resolved_inference_device = self._normalize_inference_device(
            ppo_ranker_inference_device
        )
        super().__init__(lte_grid, bs, **kwargs)

        resolved_episode_len_tti = ppo_ranker_episode_len_tti
        if resolved_episode_len_tti is None:
            resolved_episode_len_tti = self.simulation_context.get("sim_duration_tti")

        self.policy_runner = self._build_policy_runner(
            model_path=ppo_ranker_model_path,
            provided_runner=ppo_ranker_policy_runner,
            deterministic=ppo_ranker_deterministic,
            device=resolved_inference_device,
        )
        self.max_n_ue = self._resolve_max_n_ue(
            configured_max_n_ue=ppo_ranker_max_n_ue,
            policy_runner=self.policy_runner,
        )
        self.observation_adapter = DRLPlaygroundObservationAdapter(
            max_n_ue=self.max_n_ue,
            episode_len_tti=resolved_episode_len_tti,
            wb_cqi_report_period_tti=ppo_ranker_wb_cqi_report_period_tti,
            strict_mode=False,
            ensure_nonempty_action_mask=False,
        )

        self.ppo_ranker_model_path = ppo_ranker_model_path
        self.ppo_ranker_wb_cqi_report_period_tti = int(
            max(ppo_ranker_wb_cqi_report_period_tti, 1)
        )
        self.ppo_ranker_episode_len_tti = resolved_episode_len_tti
        self.ppo_ranker_deterministic = bool(ppo_ranker_deterministic)
        self.ppo_ranker_inference_device = self._describe_policy_device(
            policy_runner=self.policy_runner,
            fallback_device=resolved_inference_device,
        )
        self.rank_weight_beta = max(float(ppo_ranker_rank_weight_beta), 0.0)
        self.pf_epsilon_bps = max(float(ppo_ranker_pf_epsilon_bps), 1e-9)

        self._priority_rotation_offset = 0
        self._last_invalid_action_count = 0
        self._last_step_count = 0
        self._last_selected_ue_ids: List[int] = []
        self._last_ranked_ue_ids: List[int] = []
        self._last_score_vector: Optional[np.ndarray] = None
        self._last_rank_weight_vector: Optional[np.ndarray] = None

    def _calculate_priorities(
        self,
        windowed_ues: List[Dict],
        tti: int,
    ) -> List[Dict]:
        num_ues = len(windowed_ues)
        if num_ues == 0:
            return windowed_ues

        if self.max_dl_ue_tti and self.max_dl_ue_tti < num_ues:
            ues_to_plan = self.max_dl_ue_tti
        else:
            ues_to_plan = num_ues

        start_idx = self._priority_rotation_offset % num_ues
        rotated_ues = windowed_ues[start_idx:] + windowed_ues[:start_idx]

        for idx, user in enumerate(rotated_ues):
            user["priority"] = ues_to_plan - idx

        if self.verbose:
            top_ue = rotated_ues[0]["UE_ID"]
            print(
                f"[SCHEDULER.{self.__class__.__name__} TTI {tti}] Priority order: "
                f"rotated start UE {top_ue}, offset={self._priority_rotation_offset}"
            )

        return rotated_ues

    def _apply_pdsch_estimation(
        self,
        priority_list: List[Dict],
        tti: int,
    ) -> List[Dict]:
        if self.verbose:
            print(
                f"[SCHEDULER.{self.__class__.__name__} TTI {tti}] "
                f"PDSCH estimation: SKIPPED (TTI-level ranker)"
            )
        return priority_list

    def _allocate_pdsch(
        self,
        tti: int,
        ues_with_pdcch: List[Dict],
        eligible_ues: List[Dict],
    ) -> Dict[int, List[int]]:
        allocation = {int(user["UE_ID"]): [] for user in eligible_ues}
        self._reset_step_trace()

        if not ues_with_pdcch:
            return allocation

        total_rbg = self._get_total_rbg()
        if total_rbg <= 0:
            return allocation

        session = PDSCHAllocationSession(
            tti=tti,
            eligible_ues=eligible_ues,
            ues_with_pdcch=ues_with_pdcch,
            lte_grid=self.lte_grid,
            get_reported_wb_cqi=self._get_wb_cqi,
            get_cqi_for_rbg=self._get_cqi_for_rbg,
            bits_per_rb_fn=self.amc.GET_BITS_PER_RB,
            build_step_snapshot_fn=lambda **_: None,
        )

        snapshot = self._build_tti_start_snapshot(
            tti=tti,
            eligible_ues=eligible_ues,
            session=session,
        )
        adapted = self.observation_adapter.build(
            snapshot=snapshot,
            mode=MODE_PROXY_START_TTI,
            ue_ids=session.eligible_ue_ids,
        )

        score_vector, invalid_action = self._coerce_action_scores(
            self.policy_runner.predict(
                adapted.observation,
                adapted.action_mask,
                deterministic=self.ppo_ranker_deterministic,
            )
        )
        rank_weight_vector = self._build_rank_weight_vector(score_vector)

        self._last_invalid_action_count = int(invalid_action)
        self._last_step_count = 1
        self._last_score_vector = score_vector.copy()
        self._last_rank_weight_vector = rank_weight_vector.copy()
        self._last_ranked_ue_ids = self._build_ranked_ue_ids(
            score_vector=score_vector,
            action_mask=np.asarray(adapted.action_mask[: adapted.actual_n_ue], dtype=bool),
            ue_ids_in_order=list(adapted.ue_ids_in_order),
        )

        ue_id_to_user = {
            int(user["UE_ID"]): user for user in eligible_ues
        }

        while not session.is_done():
            current_mask = np.asarray(session.get_action_mask(), dtype=bool)
            if not np.any(current_mask):
                break

            chosen_idx = self._select_best_idx_for_current_rbg(
                action_mask=current_mask,
                rank_weight_vector=rank_weight_vector,
                session=session,
                ue_id_to_user=ue_id_to_user,
            )
            if chosen_idx is None:
                break

            chosen_ue_id = int(adapted.ue_ids_in_order[chosen_idx])
            session.apply_selected_ue(chosen_ue_id)

        if (
            session.last_allocated_ue_id is not None
            and session.last_allocated_ue_id in session.eligible_ue_ids
        ):
            last_served_idx = session.eligible_ue_ids.index(session.last_allocated_ue_id)
            self._priority_rotation_offset = (last_served_idx + 1) % len(
                session.eligible_ue_ids
            )

        self._last_selected_ue_ids = list(session.selected_ue_ids)
        allocation = session.allocation

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(
                f"[SCHEDULER.{self.__class__.__name__} TTI {tti}] "
                f"PDSCH: {allocated_ues} UE, {total_rb} RB total, "
                f"ranker_call=1, invalid_vector={invalid_action}"
            )

        return allocation

    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats.update(
            {
                "ppo_ranker_invalid_action_count": self._last_invalid_action_count,
                "ppo_ranker_step_count": self._last_step_count,
                "ppo_ranker_invalid_action_rate": (
                    self._last_invalid_action_count / self._last_step_count
                    if self._last_step_count > 0
                    else 0.0
                ),
                "ppo_ranker_selected_ue_ids": list(self._last_selected_ue_ids),
                "ppo_ranker_ranked_ue_ids": list(self._last_ranked_ue_ids),
                "ppo_ranker_model_path": self.ppo_ranker_model_path,
                "ppo_ranker_inference_device": self.ppo_ranker_inference_device,
                "ppo_ranker_rank_weight_beta": self.rank_weight_beta,
                "ppo_ranker_pf_epsilon_bps": self.pf_epsilon_bps,
                "ppo_ranker_score_vector": (
                    None
                    if self._last_score_vector is None
                    else self._last_score_vector.tolist()
                ),
                "ppo_ranker_rank_weight_vector": (
                    None
                    if self._last_rank_weight_vector is None
                    else self._last_rank_weight_vector.tolist()
                ),
            }
        )
        return stats

    def _build_tti_start_snapshot(
        self,
        *,
        tti: int,
        eligible_ues: List[Dict],
        session: PDSCHAllocationSession,
    ) -> DRLPlaygroundSnapshot:
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
                    active_flag=session.remaining_buffer_bits.get(ue_id, 0) > 0,
                    buffer_bytes=int(session.remaining_buffer_bits.get(ue_id, 0) // 8),
                    average_throughput_bps=float(
                        getattr(ue, "average_throughput", 0.0) or 0.0
                    ),
                    current_dl_throughput_bps=float(
                        getattr(ue, "current_dl_throughput", 0.0) or 0.0
                    ),
                    alloc_rbg_count_tti=0,
                    alloc_rbg_frac_tti=0.0,
                    sinr_db=float(getattr(ue, "SINR", 0.0) or 0.0),
                    reported_sb_cqi=self._get_reported_sb_cqi(
                        ue_id=ue_id,
                        fallback_sb_cqi=list(user.get("sbb_cqi", []) or []),
                    ),
                )
            )

        compatibility = DRLPlaygroundCompatibilityReport(
            exact_per_rbg_step_supported=False,
            reported_vs_true_wb_cqi_supported=True,
            wb_cqi_age_supported=True,
            alloc_frac_this_tti_supported=True,
            current_rbg_index_supported=True,
            scheduler_eligibility_mask_supported=True,
            notes=[
                "TTI-level PPO ranker делает один inference на старт TTI, а не per-RBG loop.",
                "PDSCH allocation завершается hybrid FD/PF метрикой поверх rank-weight.",
            ],
        )

        action_mask = session.get_action_mask()
        total_rbg = session.total_rbg
        return DRLPlaygroundSnapshot(
            simulation_config=DRLPlaygroundSimulationConfig(
                sim_duration_tti=self.ppo_ranker_episode_len_tti,
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
                scheduler_algorithm=self.__class__.__name__,
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
                current_rbg_index=0,
                allocated_rbg_fraction_progress=0.0,
                allocated_rbg_fraction_final_tti=0.0,
                n_rb_dl=int(getattr(self.lte_grid, "rb_per_slot", 0) or 0),
                rbg_size_rb=int(self.lte_grid.GET_RBG_SIZE()),
                n_rbg=int(total_rbg),
            ),
            ue_states=ue_states,
            action_mask=[int(value) for value in action_mask],
            compatibility=compatibility,
            scheduler_result=None,
        )

    def _select_best_idx_for_current_rbg(
        self,
        *,
        action_mask: np.ndarray,
        rank_weight_vector: np.ndarray,
        session: PDSCHAllocationSession,
        ue_id_to_user: Dict[int, Dict],
    ) -> Optional[int]:
        rbg_idx = int(session.current_rbg_index)
        best_idx: Optional[int] = None
        best_metric = -1.0

        for idx, is_valid in enumerate(action_mask):
            if not bool(is_valid):
                continue

            ue_id = int(session.eligible_ue_ids[idx])
            fd_pf_metric = self._compute_fd_pf_metric_for_rbg(
                ue_id=ue_id,
                rbg_idx=rbg_idx,
                session=session,
                user=ue_id_to_user[ue_id],
            )
            combined_metric = float(rank_weight_vector[idx]) * fd_pf_metric

            if combined_metric > best_metric:
                best_metric = combined_metric
                best_idx = int(idx)

        return best_idx

    def _compute_fd_pf_metric_for_rbg(
        self,
        *,
        ue_id: int,
        rbg_idx: int,
        session: PDSCHAllocationSession,
        user: Dict,
    ) -> float:
        cqi = int(session.get_cqi_for_rbg(ue_id, rbg_idx))
        if cqi <= 0:
            return 0.0

        bits_per_rb = int(self.amc.GET_BITS_PER_RB(cqi))
        rbg_width = len(self.lte_grid.GET_RBG_INDICES(rbg_idx))
        r_j_k = float(rbg_width * bits_per_rb)

        avg_tput_bps = float(getattr(user["ue"], "average_throughput", 0.0) or 0.0)
        denom = 1.0 if avg_tput_bps <= self.pf_epsilon_bps else (avg_tput_bps / 1000.0)
        return r_j_k / denom

    def _coerce_action_scores(self, action: Any) -> tuple[np.ndarray, bool]:
        invalid_action = False
        score_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if score_vector.shape != (self.max_n_ue,):
            raise ValueError(
                f"Invalid PPO ranker action shape: expected {(self.max_n_ue,)}, got {score_vector.shape}."
            )

        finite_mask = np.isfinite(score_vector)
        if not np.all(finite_mask):
            invalid_action = True
            score_vector = score_vector.copy()
            score_vector[~finite_mask] = -1e9

        return score_vector, invalid_action

    def _build_rank_weight_vector(self, score_vector: np.ndarray) -> np.ndarray:
        weight_vector = 1.0 + self.rank_weight_beta * np.tanh(score_vector)
        return weight_vector.astype(np.float32, copy=False)

    @staticmethod
    def _build_ranked_ue_ids(
        *,
        score_vector: np.ndarray,
        action_mask: np.ndarray,
        ue_ids_in_order: List[int],
    ) -> List[int]:
        valid_indices = np.flatnonzero(action_mask)
        if valid_indices.size == 0:
            return []

        ranked = sorted(
            (int(idx) for idx in valid_indices),
            key=lambda idx: (-float(score_vector[idx]), idx),
        )
        return [int(ue_ids_in_order[idx]) for idx in ranked]

    def _get_total_rbg(self) -> int:
        rbg_size = int(self.lte_grid.GET_RBG_SIZE())
        if rbg_size <= 0:
            return 0
        return int((self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size)

    def _get_cqi_for_rbg(self, ue_id: int, rbg_idx: int) -> int:
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
        if ue_id in self.cqi_map:
            cqi_entry = self.cqi_map[ue_id]
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
        if ue_id in self.cqi_map:
            return [
                int(cqi) for cqi in list(getattr(self.cqi_map[ue_id], "sb_cqi", []) or [])
            ]
        return [int(cqi) for cqi in list(fallback_sb_cqi or [])]

    def _reset_step_trace(self) -> None:
        self._last_invalid_action_count = 0
        self._last_step_count = 0
        self._last_selected_ue_ids = []
        self._last_ranked_ue_ids = []
        self._last_score_vector = None
        self._last_rank_weight_vector = None

    @staticmethod
    def _build_policy_runner(
        *,
        model_path: Optional[str],
        provided_runner: Optional[Any],
        deterministic: bool,
        device: Optional[Any],
    ) -> Any:
        if provided_runner is not None:
            return provided_runner

        if not model_path:
            raise ValueError(
                "Для PpoRankerScheduler необходимо указать "
                "ppo_ranker_model_path или ppo_ranker_policy_runner."
            )

        return PPORankerModelRunner(
            model_path=str(model_path),
            deterministic=deterministic,
            device=device,
        )

    @staticmethod
    def _normalize_inference_device(device: Optional[Any]) -> Optional[Any]:
        if device is None:
            return "cpu"

        if isinstance(device, str):
            normalized = device.strip().lower()
            if normalized in {"", "default"}:
                return "cpu"
            if normalized == "auto":
                return None
            return normalized

        return device

    @staticmethod
    def _describe_policy_device(
        *,
        policy_runner: Any,
        fallback_device: Optional[Any],
    ) -> str:
        runner_device = getattr(policy_runner, "inference_device", None)
        if runner_device is not None:
            return str(runner_device)

        if fallback_device is None:
            return "auto"

        return str(fallback_device)

    @staticmethod
    def _resolve_max_n_ue(
        *,
        configured_max_n_ue: Optional[int],
        policy_runner: Any,
    ) -> Optional[int]:
        if configured_max_n_ue is not None:
            return int(configured_max_n_ue)

        runner_max_n_ue = getattr(policy_runner, "max_n_ue", None)
        if runner_max_n_ue is not None:
            return int(runner_max_n_ue)

        return None
