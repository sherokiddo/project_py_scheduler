"""
Адаптер observation/action-mask между snapshot из PyScheduler и средой
из `drl_playground`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from drl.simulation_bridge import DRLPlaygroundSnapshot, DRLPlaygroundUEState


PLAYGROUND_MAX_BUFFER_BYTES = 1_000_000.0
PLAYGROUND_MAX_AVG_TPUT_BPS = 100e6
PLAYGROUND_MAX_WB_CQI = 15.0
PLAYGROUND_N_UE_FEATURES = 6
PLAYGROUND_N_CONTEXT_FEATURES = 3

MODE_CURRENT_STEP = "current_step"
MODE_SNAPSHOT = "snapshot"
MODE_PROXY_START_TTI = "proxy_start_tti"
SUPPORTED_ADAPTER_MODES = (
    MODE_CURRENT_STEP,
    MODE_SNAPSHOT,
    MODE_PROXY_START_TTI,
)


@dataclass(slots=True)
class DRLPlaygroundObservation:
    """
    Готовый observation-пакет для модели/среды из drl_playground.
    """

    observation: np.ndarray
    action_mask: np.ndarray
    ue_ids_in_order: List[int]
    actual_n_ue: int
    max_n_ue: int
    ue_feature_dim: int = PLAYGROUND_N_UE_FEATURES
    context_dim: int = PLAYGROUND_N_CONTEXT_FEATURES
    mode: str = MODE_SNAPSHOT
    exact_env_match: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразовать observation-пакет в словарь для отладки и сериализации.
        """

        return {
            "observation": self.observation.tolist(),
            "action_mask": self.action_mask.astype(bool).tolist(),
            "ue_ids_in_order": list(self.ue_ids_in_order),
            "actual_n_ue": int(self.actual_n_ue),
            "max_n_ue": int(self.max_n_ue),
            "ue_feature_dim": int(self.ue_feature_dim),
            "context_dim": int(self.context_dim),
            "mode": self.mode,
            "exact_env_match": bool(self.exact_env_match),
            "notes": list(self.notes),
        }


class DRLPlaygroundObservationAdapter:
    """
    Построитель observation/action-mask в формате `drl_playground`.

    Адаптер решает две задачи:
    - нормализует snapshot PyScheduler до того же порядка признаков, что и env;
    - при необходимости паддит observation и action-mask до `max_n_ue`.

    Важное ограничение:
    PyScheduler пока не раскрывает внешний per-RBG step так же, как toy env.
    Поэтому адаптер поддерживает как "строгий" режим, так и "proxy" режим
    для старта TTI.
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
        self.max_n_ue = max_n_ue
        self.episode_len_tti = episode_len_tti
        self.wb_cqi_report_period_tti = wb_cqi_report_period_tti
        self.max_buffer_bytes = float(max(max_buffer_bytes, 1.0))
        self.max_avg_tput_bps = float(max(max_avg_tput_bps, 1.0))
        self.strict_mode = bool(strict_mode)
        self.ensure_nonempty_action_mask = bool(ensure_nonempty_action_mask)

    def build(
        self,
        snapshot: DRLPlaygroundSnapshot,
        *,
        mode: str = MODE_SNAPSHOT,
        ue_ids: Optional[Sequence[int]] = None,
    ) -> DRLPlaygroundObservation:
        """
        Построить observation/action-mask из snapshot.
        """

        self._validate_mode(mode)

        selected_states, selected_mask = self._select_ue_states(
            snapshot=snapshot,
            ue_ids=ue_ids,
        )
        actual_n_ue = len(selected_states)
        max_n_ue = self._resolve_max_n_ue(actual_n_ue)
        episode_len_tti, episode_len_exact = self._resolve_episode_len_tti(snapshot)
        age_denom, age_exact = self._resolve_wb_cqi_age_denominator(selected_states)

        ue_observation, notes, exact_env_match = self._build_ue_observation(
            selected_states=selected_states,
            age_denom=age_denom,
            mode=mode,
        )
        context_observation, context_notes, context_exact = self._build_context_observation(
            snapshot=snapshot,
            mode=mode,
            episode_len_tti=episode_len_tti,
        )
        notes.extend(context_notes)
        exact_env_match = exact_env_match and context_exact and age_exact and episode_len_exact

        if not episode_len_exact:
            notes.append(
                "Нормировка current_tti использует fallback, потому что длина эпизода не задана явно."
            )

        if not age_exact:
            notes.append(
                "Нормировка wb_cqi_age использует выведенный знаменатель, а не явный период report."
            )

        base_obs = np.concatenate([ue_observation, context_observation], dtype=np.float32)
        base_mask = np.asarray(selected_mask, dtype=bool)

        if mode == MODE_PROXY_START_TTI:
            exact_env_match = False
            notes.append(
                "proxy_start_tti синтетически обнуляет progress-контекст и alloc_frac_this_tti."
            )

        if self.strict_mode and not exact_env_match:
            raise ValueError(
                "Невозможно построить строгое env-совместимое observation из текущего snapshot: "
                + "; ".join(notes)
            )

        observation = self._pad_observation(base_obs, actual_n_ue, max_n_ue)
        action_mask = self._pad_action_mask(base_mask, actual_n_ue, max_n_ue)

        return DRLPlaygroundObservation(
            observation=observation,
            action_mask=action_mask,
            ue_ids_in_order=[state.ue_id for state in selected_states],
            actual_n_ue=actual_n_ue,
            max_n_ue=max_n_ue,
            mode=mode,
            exact_env_match=exact_env_match,
            notes=notes,
        )

    def _build_ue_observation(
        self,
        *,
        selected_states: Sequence[DRLPlaygroundUEState],
        age_denom: int,
        mode: str,
    ) -> tuple[np.ndarray, List[str], bool]:
        """
        Построить per-UE часть observation.
        """

        obs = np.zeros(
            len(selected_states) * PLAYGROUND_N_UE_FEATURES,
            dtype=np.float32,
        )
        notes: List[str] = []
        exact_env_match = True

        for slot, state in enumerate(selected_states):
            alloc_frac_this_tti = float(state.alloc_rbg_frac_tti)
            if mode == MODE_PROXY_START_TTI:
                alloc_frac_this_tti = 0.0

            age_value = int(state.wb_cqi_age_tti or 0)
            age_norm = float(np.clip(age_value / max(age_denom, 1), 0.0, 1.0))

            base = slot * PLAYGROUND_N_UE_FEATURES
            obs[base + 0] = float(np.clip(state.reported_wb_cqi / PLAYGROUND_MAX_WB_CQI, 0.0, 1.0))
            obs[base + 1] = age_norm
            obs[base + 2] = float(bool(state.active_flag))
            obs[base + 3] = float(np.clip(state.buffer_bytes / self.max_buffer_bytes, 0.0, 1.0))
            obs[base + 4] = float(
                np.clip(state.average_throughput_bps / self.max_avg_tput_bps, 0.0, 1.0)
            )
            obs[base + 5] = float(np.clip(alloc_frac_this_tti, 0.0, 1.0))

        if mode == MODE_SNAPSHOT:
            notes.append(
                "snapshot mode использует alloc_rbg_frac_tti из snapshot как есть; для PyScheduler это может быть постфактум TTI-оценка."
            )
            exact_env_match = False

        return obs, notes, exact_env_match

    def _build_context_observation(
        self,
        *,
        snapshot: DRLPlaygroundSnapshot,
        mode: str,
        episode_len_tti: int,
    ) -> tuple[np.ndarray, List[str], bool]:
        """
        Построить глобальный контекст observation.
        """

        ctx = np.zeros(PLAYGROUND_N_CONTEXT_FEATURES, dtype=np.float32)
        notes: List[str] = []
        exact_env_match = True

        n_rbg = max(int(snapshot.step_snapshot.n_rbg), 1)
        current_tti = int(snapshot.step_snapshot.current_tti)

        if mode == MODE_CURRENT_STEP:
            current_rbg_index = snapshot.step_snapshot.current_rbg_index
            allocated_fraction = snapshot.step_snapshot.allocated_rbg_fraction_progress

            if current_rbg_index is None:
                notes.append("В snapshot отсутствует current_rbg_index для точного current_step режима.")
                exact_env_match = False
                current_rbg_index = 0

            if allocated_fraction is None:
                notes.append("В snapshot отсутствует allocated_rbg_fraction_progress для точного current_step режима.")
                exact_env_match = False
                allocated_fraction = 0.0

            if not snapshot.compatibility.exact_per_rbg_step_supported:
                notes.append("PyScheduler пока не раскрывает точную per-RBG step семантику наружу.")
                exact_env_match = False

        elif mode == MODE_PROXY_START_TTI:
            current_rbg_index = 0
            allocated_fraction = 0.0

        else:
            current_rbg_index = snapshot.step_snapshot.current_rbg_index
            allocated_fraction = snapshot.step_snapshot.allocated_rbg_fraction_progress

            if current_rbg_index is None:
                current_rbg_index = 0
                notes.append("В snapshot нет current_rbg_index, поэтому используется fallback 0.")
                exact_env_match = False

            if allocated_fraction is None:
                allocated_fraction = snapshot.step_snapshot.allocated_rbg_fraction_final_tti
                notes.append(
                    "В snapshot нет allocated_rbg_fraction_progress, используется final_tti значение."
                )
                exact_env_match = False

        ctx[0] = float(np.clip(current_rbg_index / n_rbg, 0.0, 1.0))
        ctx[1] = float(np.clip(current_tti / max(episode_len_tti, 1), 0.0, 1.0))
        ctx[2] = float(np.clip(float(allocated_fraction), 0.0, 1.0))
        return ctx, notes, exact_env_match

    def _select_ue_states(
        self,
        *,
        snapshot: DRLPlaygroundSnapshot,
        ue_ids: Optional[Sequence[int]],
    ) -> tuple[List[DRLPlaygroundUEState], List[bool]]:
        """
        Выбрать UE и их порядок в observation.
        """

        states = list(snapshot.ue_states)
        if len(snapshot.action_mask) != len(states):
            raise ValueError(
                "Длина action_mask не совпадает с числом UE в snapshot."
            )

        seen_ids: set[int] = set()
        state_by_id: Dict[int, DRLPlaygroundUEState] = {}
        mask_by_id: Dict[int, bool] = {}
        original_order: List[int] = []

        for idx, state in enumerate(states):
            ue_id = int(state.ue_id)
            if ue_id in seen_ids:
                raise ValueError(f"Повторяющийся UE_ID в snapshot: {ue_id}")
            seen_ids.add(ue_id)
            state_by_id[ue_id] = state
            mask_by_id[ue_id] = bool(snapshot.action_mask[idx])
            original_order.append(ue_id)

        selected_ids = original_order if ue_ids is None else [int(ue_id) for ue_id in ue_ids]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Список ue_ids содержит дубликаты.")

        missing_ids = [ue_id for ue_id in selected_ids if ue_id not in state_by_id]
        if missing_ids:
            raise ValueError(
                f"В snapshot отсутствуют UE из запрошенного порядка: {missing_ids}"
            )

        selected_states = [state_by_id[ue_id] for ue_id in selected_ids]
        selected_mask = [mask_by_id[ue_id] for ue_id in selected_ids]
        return selected_states, selected_mask

    def _resolve_max_n_ue(self, actual_n_ue: int) -> int:
        """
        Определить итоговую ширину action/observation пространства.
        """

        max_n_ue = actual_n_ue if self.max_n_ue is None else int(self.max_n_ue)
        if max_n_ue < actual_n_ue:
            raise ValueError(
                f"max_n_ue={max_n_ue} меньше числа выбранных UE={actual_n_ue}."
            )
        return max_n_ue

    def _resolve_episode_len_tti(
        self,
        snapshot: DRLPlaygroundSnapshot,
    ) -> tuple[int, bool]:
        """
        Определить знаменатель для нормировки current_tti.
        """

        if self.episode_len_tti is not None:
            return max(int(self.episode_len_tti), 1), True

        sim_duration_tti = snapshot.simulation_config.sim_duration_tti
        if sim_duration_tti is not None and int(sim_duration_tti) > 0:
            return int(sim_duration_tti), True

        return 1, False

    def _resolve_wb_cqi_age_denominator(
        self,
        states: Sequence[DRLPlaygroundUEState],
    ) -> tuple[int, bool]:
        """
        Определить знаменатель для нормировки wb_cqi_age.
        """

        if self.wb_cqi_report_period_tti is not None:
            return max(int(self.wb_cqi_report_period_tti) - 1, 1), True

        observed_ages = [
            int(state.wb_cqi_age_tti)
            for state in states
            if state.wb_cqi_age_tti is not None
        ]
        if not observed_ages:
            return 1, False

        return max(max(observed_ages), 1), False

    def _pad_observation(
        self,
        obs: np.ndarray,
        actual_n_ue: int,
        max_n_ue: int,
    ) -> np.ndarray:
        """
        Паддить observation по той же схеме, что и PaddedLTESchedulerEnv.
        """

        obs = np.asarray(obs, dtype=np.float32)
        base_ue_dim = actual_n_ue * PLAYGROUND_N_UE_FEATURES
        full_obs_dim = max_n_ue * PLAYGROUND_N_UE_FEATURES + PLAYGROUND_N_CONTEXT_FEATURES

        if max_n_ue == actual_n_ue:
            return np.clip(obs, 0.0, 1.0)

        padded = np.zeros(full_obs_dim, dtype=np.float32)
        padded[:base_ue_dim] = obs[:base_ue_dim]
        padded[max_n_ue * PLAYGROUND_N_UE_FEATURES :] = obs[base_ue_dim:]
        return np.clip(padded, 0.0, 1.0)

    def _pad_action_mask(
        self,
        action_mask: np.ndarray,
        actual_n_ue: int,
        max_n_ue: int,
    ) -> np.ndarray:
        """
        Паддить action-mask до `max_n_ue`.
        """

        mask = np.asarray(action_mask, dtype=bool)
        if max_n_ue == actual_n_ue:
            if self.ensure_nonempty_action_mask and not np.any(mask) and len(mask) > 0:
                mask = mask.copy()
                mask[0] = True
            return mask

        padded = np.zeros(max_n_ue, dtype=bool)
        padded[:actual_n_ue] = mask[:actual_n_ue]

        if self.ensure_nonempty_action_mask and not np.any(padded) and len(padded) > 0:
            padded[0] = True
        return padded

    @staticmethod
    def _validate_mode(mode: str) -> None:
        """
        Проверить поддерживаемость режима адаптации.
        """

        if mode not in SUPPORTED_ADAPTER_MODES:
            raise ValueError(
                f"Неподдерживаемый режим адаптера: {mode}. "
                f"Ожидается один из {SUPPORTED_ADAPTER_MODES}."
            )
