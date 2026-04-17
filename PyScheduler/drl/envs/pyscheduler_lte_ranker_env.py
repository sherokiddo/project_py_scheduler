"""
Simulation-backed Gymnasium environment for TTI-level PPO ranker training.

The environment reuses the runtime-backed mechanics from `PySchedulerLteEnv`,
but changes the action semantics:
- the base env makes one decision per RBG;
- this ranker env receives one score vector per TTI;
- allocation is then completed through a hybrid metric that combines
  PPO-derived UE weights with the existing FD/PF per-RBG logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from drl.envs.pyscheduler_lte_env import PySchedulerLteEnv


class PySchedulerLteRankerEnv(PySchedulerLteEnv):
    """
    TTI-level ranker environment over the real PyScheduler runtime.

    The agent emits one raw score per UE. The environment turns these scores
    into bounded UE weights and then uses them to modulate the existing
    FD/PF per-RBG metric:

        combined_metric(u, k) = rank_weight(u) * fd_pf_metric(u, k)
    """

    def __init__(
        self,
        *args,
        rank_weight_beta: float = 0.3,
        pf_epsilon_bps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_n_ue,),
            dtype=np.float32,
        )
        self.rank_weight_beta = max(float(rank_weight_beta), 0.0)
        self.pf_epsilon_bps = max(float(pf_epsilon_bps), 1e-9)
        self.last_ranked_ue_ids: List[int] = []
        self.last_score_vector: Optional[np.ndarray] = None
        self.last_rank_weight_vector: Optional[np.ndarray] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        observation, info = super().reset(seed=seed, options=options)
        self.last_ranked_ue_ids = []
        self.last_score_vector = None
        self.last_rank_weight_vector = None
        return observation, info

    def step(self, action):
        if self.current_observation_packet is None or self.current_session is None:
            raise RuntimeError("Environment is not ready for step(). Call reset() first.")

        score_vector, invalid_action = self._coerce_action_scores(action)
        ranked_indices = self._build_ranked_action_indices(score_vector)
        rank_weight_vector = self._build_rank_weight_vector(score_vector)

        self.last_score_vector = score_vector.copy()
        self.last_rank_weight_vector = rank_weight_vector.copy()
        self.last_ranked_ue_ids = [
            int(self.current_observation_packet.ue_ids_in_order[idx])
            for idx in ranked_indices
        ]

        self._allocate_current_tti_with_hybrid_metric(rank_weight_vector)

        reward = self._complete_current_tti()
        skipped_reward, terminated = self._advance_to_next_decision_point()
        reward += skipped_reward

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = self._build_terminal_info(invalid_action=invalid_action)
        else:
            obs = self.current_observation_packet.observation.copy()
            info = self._build_info(invalid_action=invalid_action)

        return obs, float(reward), bool(terminated), False, info

    def _coerce_action_scores(self, action: Any) -> Tuple[np.ndarray, bool]:
        """
        Normalize the raw agent action to a fixed-width score vector.
        """
        invalid_action = False
        score_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if score_vector.shape != (self.max_n_ue,):
            raise ValueError(
                f"Invalid action shape: expected {(self.max_n_ue,)}, got {score_vector.shape}."
            )

        finite_mask = np.isfinite(score_vector)
        if not np.all(finite_mask):
            invalid_action = True
            score_vector = score_vector.copy()
            score_vector[~finite_mask] = -1e9

        return score_vector, invalid_action

    def _build_ranked_action_indices(self, score_vector: np.ndarray) -> List[int]:
        """
        Build the raw score ranking in observation order for debugging/analysis.
        """
        packet = self.current_observation_packet
        actual_n_ue = int(packet.actual_n_ue)
        action_mask = np.asarray(packet.action_mask[:actual_n_ue], dtype=bool)

        valid_indices = np.flatnonzero(action_mask)
        if valid_indices.size == 0:
            return []

        ranked = sorted(
            (int(idx) for idx in valid_indices),
            key=lambda idx: (-float(score_vector[idx]), idx),
        )
        return ranked

    def _build_rank_weight_vector(self, score_vector: np.ndarray) -> np.ndarray:
        """
        Convert raw PPO score to a bounded multiplicative UE weight.

        rank_weight(u) = 1 + beta * tanh(rank_score(u))
        """
        weight_vector = 1.0 + self.rank_weight_beta * np.tanh(score_vector)
        return weight_vector.astype(np.float32, copy=False)

    def _allocate_current_tti_with_hybrid_metric(
        self,
        rank_weight_vector: np.ndarray,
    ) -> None:
        """
        Finish the current TTI allocation via the hybrid FD/PF metric.
        """
        session = self.current_session
        while not session.is_done():
            current_mask = np.asarray(session.get_action_mask(), dtype=bool)
            if not np.any(current_mask):
                break

            chosen_idx = self._select_best_idx_for_current_rbg(
                action_mask=current_mask,
                rank_weight_vector=rank_weight_vector,
            )
            if chosen_idx is None:
                break

            chosen_ue_id = int(self.current_observation_packet.ue_ids_in_order[chosen_idx])
            session.apply_selected_ue(chosen_ue_id)

    def _select_best_idx_for_current_rbg(
        self,
        *,
        action_mask: np.ndarray,
        rank_weight_vector: np.ndarray,
    ) -> Optional[int]:
        """
        Select the best UE for the current RBG using hybrid metric.
        """
        packet = self.current_observation_packet
        session = self.current_session
        rbg_idx = int(session.current_rbg_index)

        best_idx: Optional[int] = None
        best_metric = -1.0

        for idx, is_valid in enumerate(action_mask):
            if not bool(is_valid):
                continue

            ue_id = int(packet.ue_ids_in_order[idx])
            fd_pf_metric = self._compute_fd_pf_metric_for_rbg(ue_id=ue_id, rbg_idx=rbg_idx)
            combined_metric = float(rank_weight_vector[idx]) * fd_pf_metric

            if combined_metric > best_metric:
                best_metric = combined_metric
                best_idx = int(idx)

        return best_idx

    def _compute_fd_pf_metric_for_rbg(self, *, ue_id: int, rbg_idx: int) -> float:
        """
        Reproduce the existing FD/PF per-RBG metric from the scheduler baseline:

            metric = r_j(k) / avg_tput_per_tti
        """
        cqi = int(self.current_session.get_cqi_for_rbg(ue_id, rbg_idx))
        if cqi <= 0:
            return 0.0

        bits_per_rb = int(self.scheduler.amc.GET_BITS_PER_RB(cqi))
        rbg_width = len(self.scheduler.lte_grid.GET_RBG_INDICES(rbg_idx))
        r_j_k = float(rbg_width * bits_per_rb)

        user = self._find_user_by_ue_id(ue_id)
        avg_tput_bps = float(getattr(user["ue"], "average_throughput", 0.0) or 0.0)
        denom = 1.0 if avg_tput_bps <= self.pf_epsilon_bps else (avg_tput_bps / 1000.0)
        return r_j_k / denom

    def _find_user_by_ue_id(self, ue_id: int) -> Dict[str, Any]:
        """
        Find the current-TTI user dictionary by UE id.
        """
        for user in self.current_pdsch_context.eligible_ues:
            if int(user["UE_ID"]) == int(ue_id):
                return user
        raise KeyError(f"UE {ue_id} not found in current PDSCH context.")

    def _build_info(
        self,
        *,
        invalid_action: bool,
        padded_invalid_action: bool = False,
        resolved_action: Optional[int] = None,
    ) -> Dict[str, object]:
        info = super()._build_info(
            invalid_action=invalid_action,
            padded_invalid_action=padded_invalid_action,
            resolved_action=resolved_action,
        )
        info.update(
            {
                "ranked_ue_ids": list(self.last_ranked_ue_ids),
                "score_vector": (
                    None if self.last_score_vector is None else self.last_score_vector.copy()
                ),
                "rank_weight_vector": (
                    None
                    if self.last_rank_weight_vector is None
                    else self.last_rank_weight_vector.copy()
                ),
            }
        )
        return info

    def _build_terminal_info(
        self,
        *,
        invalid_action: bool = False,
        padded_invalid_action: bool = False,
        resolved_action: Optional[int] = None,
    ) -> Dict[str, object]:
        info = super()._build_terminal_info(
            invalid_action=invalid_action,
            padded_invalid_action=padded_invalid_action,
            resolved_action=resolved_action,
        )
        info.update(
            {
                "ranked_ue_ids": list(self.last_ranked_ue_ids),
                "score_vector": (
                    None if self.last_score_vector is None else self.last_score_vector.copy()
                ),
                "rank_weight_vector": (
                    None
                    if self.last_rank_weight_vector is None
                    else self.last_rank_weight_vector.copy()
                ),
            }
        )
        return info
