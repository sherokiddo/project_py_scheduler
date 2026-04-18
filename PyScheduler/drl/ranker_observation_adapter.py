"""
Облегченный observation-адаптер для TTI-level ranker-политик.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from drl.playground_adapter import (
    DRLPlaygroundObservation,
    DRLPlaygroundObservationAdapter,
    MODE_PROXY_START_TTI,
    PLAYGROUND_MAX_AVG_TPUT_BPS,
    PLAYGROUND_MAX_BUFFER_BYTES,
    PLAYGROUND_MAX_N_RBG,
    PLAYGROUND_MAX_WB_CQI,
)
from drl.simulation_bridge import DRLPlaygroundSnapshot


RANKER_OBSERVATION_N_UE_FEATURES = 4
RANKER_OBSERVATION_N_CONTEXT_FEATURES = 2


class RankerObservationAdapter(DRLPlaygroundObservationAdapter):
    """
    Compact observation builder для ranker-моделей.
    """

    def __init__(
        self,
        *,
        max_n_ue: Optional[int] = None,
        episode_len_tti: Optional[int] = None,
        wb_cqi_report_period_tti: Optional[int] = None,
        max_buffer_bytes: float = PLAYGROUND_MAX_BUFFER_BYTES,
        max_avg_tput_bps: float = PLAYGROUND_MAX_AVG_TPUT_BPS,
        strict_mode: bool = False,
        ensure_nonempty_action_mask: bool = False,
    ) -> None:
        super().__init__(
            max_n_ue=max_n_ue,
            episode_len_tti=episode_len_tti,
            wb_cqi_report_period_tti=wb_cqi_report_period_tti,
            max_buffer_bytes=max_buffer_bytes,
            max_avg_tput_bps=max_avg_tput_bps,
            strict_mode=strict_mode,
            ensure_nonempty_action_mask=ensure_nonempty_action_mask,
        )
        self.ue_feature_dim = RANKER_OBSERVATION_N_UE_FEATURES
        self.context_dim = RANKER_OBSERVATION_N_CONTEXT_FEATURES

    def build(
        self,
        snapshot: DRLPlaygroundSnapshot,
        *,
        mode: str = MODE_PROXY_START_TTI,
        ue_ids: Optional[Sequence[int]] = None,
    ) -> DRLPlaygroundObservation:
        if mode != MODE_PROXY_START_TTI:
            raise ValueError(
                "RankerObservationAdapter поддерживает только mode='proxy_start_tti'."
            )

        selected_states, selected_mask = self._select_ue_states(
            snapshot=snapshot,
            ue_ids=ue_ids,
        )
        actual_n_ue = len(selected_states)
        max_n_ue = self._resolve_max_n_ue(actual_n_ue)
        age_denom, age_exact = self._resolve_wb_cqi_age_denominator(selected_states)

        observation = np.zeros(
            max_n_ue * self.ue_feature_dim + self.context_dim,
            dtype=np.float32,
        )
        action_mask = np.zeros(max_n_ue, dtype=bool)
        action_mask[:actual_n_ue] = np.asarray(selected_mask, dtype=bool)

        if self.ensure_nonempty_action_mask and not np.any(action_mask) and len(action_mask) > 0:
            action_mask[0] = True

        if actual_n_ue > 0:
            ue_obs = observation[: max_n_ue * self.ue_feature_dim].reshape(
                max_n_ue,
                self.ue_feature_dim,
            )[:actual_n_ue]

            reported_wb_cqi = np.fromiter(
                (float(state.reported_wb_cqi) for state in selected_states),
                dtype=np.float32,
                count=actual_n_ue,
            )
            wb_cqi_age = np.fromiter(
                (float(int(state.wb_cqi_age_tti or 0)) for state in selected_states),
                dtype=np.float32,
                count=actual_n_ue,
            )
            buffer_bytes = np.fromiter(
                (float(state.buffer_bytes) for state in selected_states),
                dtype=np.float32,
                count=actual_n_ue,
            )
            average_throughput_bps = np.fromiter(
                (float(state.average_throughput_bps) for state in selected_states),
                dtype=np.float32,
                count=actual_n_ue,
            )

            age_denom_value = max(age_denom, 1)
            ue_obs[:, 0] = np.clip(reported_wb_cqi / PLAYGROUND_MAX_WB_CQI, 0.0, 1.0)
            ue_obs[:, 1] = np.clip(wb_cqi_age / age_denom_value, 0.0, 1.0)
            ue_obs[:, 2] = np.clip(buffer_bytes / self.max_buffer_bytes, 0.0, 1.0)
            ue_obs[:, 3] = np.clip(
                average_throughput_bps / self.max_avg_tput_bps,
                0.0,
                1.0,
            )

        ctx = observation[max_n_ue * self.ue_feature_dim :]
        active_ue_count = sum(1 for state in selected_states if float(state.buffer_bytes) > 0.0)
        n_rbg = int(snapshot.step_snapshot.n_rbg or 0)
        ctx[0] = float(min(max(active_ue_count / max(max_n_ue, 1), 0.0), 1.0))
        ctx[1] = float(min(max(n_rbg / PLAYGROUND_MAX_N_RBG, 0.0), 1.0))

        notes: List[str] = []
        if not age_exact:
            notes.append(
                "Нормировка wb_cqi_age использует выведенный знаменатель, а не явный период report."
            )
        notes.append(
            "ranker_observation использует compact TTI-level contract без признаков внутренней alloc-фазы."
        )

        return DRLPlaygroundObservation(
            observation=observation,
            action_mask=action_mask,
            ue_ids_in_order=[int(state.ue_id) for state in selected_states],
            actual_n_ue=actual_n_ue,
            max_n_ue=max_n_ue,
            ue_feature_dim=self.ue_feature_dim,
            context_dim=self.context_dim,
            mode=MODE_PROXY_START_TTI,
            exact_env_match=True,
            notes=notes,
        )
