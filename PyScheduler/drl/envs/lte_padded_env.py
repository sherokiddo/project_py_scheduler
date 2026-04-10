"""
Обертка фиксированной размерности над LTE-средой.

Базовая среда моделирует фактическое число UE, а эта обертка дополняет
observation и action-mask до `max_n_ue`, чтобы сеть работала со стабильной
размерностью входа и выхода.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PaddedLTESchedulerEnv(gym.Wrapper):
    """
    Обертка fixed-shape над `LTESchedulerEnv`.
    """

    def __init__(self, env: gym.Env, max_n_ue: int):
        super().__init__(env)
        if max_n_ue < env.n_ue:
            raise ValueError(
                f"max_n_ue={max_n_ue} must be >= base env.n_ue={env.n_ue}"
            )

        self.max_n_ue = int(max_n_ue)
        self.ue_feature_dim = int(env.N_UE_FEATURES)
        self.context_dim = int(env.N_CONTEXT_FEATURES)
        obs_dim = self.max_n_ue * self.ue_feature_dim + self.context_dim

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_n_ue)
        self._last_padded_action_mask = None

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        base_ue_dim = self.env.n_ue * self.ue_feature_dim
        padded = np.zeros(self.observation_space.shape, dtype=np.float32)
        padded[:base_ue_dim] = obs[:base_ue_dim]
        padded[self.max_n_ue * self.ue_feature_dim :] = obs[base_ue_dim:]
        return padded

    def _pad_action_mask(self, action_mask: np.ndarray) -> np.ndarray:
        action_mask = np.asarray(action_mask, dtype=bool)
        padded = np.zeros(self.max_n_ue, dtype=bool)
        padded[: self.env.n_ue] = action_mask[: self.env.n_ue]

        if not np.any(padded):
            padded[0] = True
        return padded

    def _augment_info(self, info: dict, padded_invalid_action: bool = False) -> dict:
        info = dict(info)
        info["actual_n_ue"] = int(self.env.n_ue)
        info["max_n_ue"] = self.max_n_ue
        info["action_mask"] = self._pad_action_mask(info["action_mask"])
        info["padded_invalid_action"] = bool(padded_invalid_action)
        self._last_padded_action_mask = info["action_mask"].copy()
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = self._augment_info(info, padded_invalid_action=False)
        return self._pad_obs(obs), info

    def step(self, action: int):
        padded_invalid_action = False
        base_action = int(action)

        if base_action >= self.env.n_ue:
            padded_invalid_action = True
        elif (
            self._last_padded_action_mask is not None
            and not self._last_padded_action_mask[base_action]
        ):
            padded_invalid_action = True

        if padded_invalid_action:
            valid_base_actions = np.flatnonzero(
                self._get_current_action_mask()[: self.env.n_ue]
            )
            base_action = int(valid_base_actions[0]) if len(valid_base_actions) > 0 else 0

        obs, reward, terminated, truncated, info = self.env.step(base_action)
        info = self._augment_info(info, padded_invalid_action=padded_invalid_action)
        return self._pad_obs(obs), reward, terminated, truncated, info

    def _get_current_action_mask(self) -> np.ndarray:
        if self._last_padded_action_mask is None:
            return np.ones(self.max_n_ue, dtype=bool)
        return self._last_padded_action_mask.copy()
