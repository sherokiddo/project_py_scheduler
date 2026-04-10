"""
Минимальный DQN-агент для inference в формате, совместимом с моделями,
обученными в `drl_playground`.
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SharedUEQNetwork(nn.Module):
    """
    Общая сеть-скоратор для множества UE.

    Каждый UE кодируется одним и тем же MLP, затем объединяется с pooled
    представлением множества активных UE и глобальным контекстом observation.
    """

    def __init__(
        self,
        max_n_ue: int,
        ue_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.max_n_ue = int(max_n_ue)
        self.ue_feature_dim = int(ue_feature_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)

        self.ue_encoder = nn.Sequential(
            nn.Linear(self.ue_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _split_obs(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ue_flat_dim = self.max_n_ue * self.ue_feature_dim
        ue_obs = obs[:, :ue_flat_dim].reshape(-1, self.max_n_ue, self.ue_feature_dim)
        context_obs = obs[:, ue_flat_dim:]
        return ue_obs, context_obs

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        ue_obs, context_obs = self._split_obs(obs)
        ue_hidden = self.ue_encoder(ue_obs)
        context_hidden = self.context_encoder(context_obs)

        mask = action_mask.float().unsqueeze(-1)
        masked_sum = torch.sum(ue_hidden * mask, dim=1)
        masked_count = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled_hidden = masked_sum / masked_count

        pooled_expanded = pooled_hidden.unsqueeze(1).expand(-1, self.max_n_ue, -1)
        context_expanded = context_hidden.unsqueeze(1).expand(-1, self.max_n_ue, -1)
        q_input = torch.cat([ue_hidden, pooled_expanded, context_expanded], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)
        return q_values


class LTEDQNAgent:
    """
    Легковесный inference-агент для LTE DQN.

    Сохраняет тот же формат весов и тот же способ выбора действия,
    что и агент из `drl_playground`.
    """

    def __init__(
        self,
        max_n_ue: int,
        ue_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        device: Optional[torch.device | str] = None,
        epsilon_start: float = 0.0,
        epsilon_end: float = 0.0,
        epsilon_decay_steps: int = 1,
    ) -> None:
        self.max_n_ue = int(max_n_ue)
        self.ue_feature_dim = int(ue_feature_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = SharedUEQNetwork(
            max_n_ue=max_n_ue,
            ue_feature_dim=ue_feature_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.q_net.eval()

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(max(epsilon_decay_steps, 1))
        self.total_steps = 0

    def _get_epsilon(self) -> float:
        fraction = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    @staticmethod
    def _mask_q_values(
        q_values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        return q_values.masked_fill(~action_mask.bool(), -1e9)

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        return self.select_action(
            obs=obs,
            action_mask=action_mask,
            deterministic=deterministic,
        )

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        action_mask = np.asarray(action_mask, dtype=bool)
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            valid_actions = np.arange(self.max_n_ue, dtype=np.int64)

        epsilon = 0.0 if deterministic else self._get_epsilon()
        self.total_steps += 1

        if not deterministic and random.random() < epsilon:
            return int(random.choice(valid_actions))

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor, mask_tensor)
            masked_q = self._mask_q_values(q_values, mask_tensor)
        return int(torch.argmax(masked_q, dim=1).item())

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device | str] = None,
    ) -> "LTEDQNAgent":
        payload = torch.load(path, map_location=device or "cpu")
        config = payload["config"]
        agent = cls(device=device, **config)
        agent.q_net.load_state_dict(payload["state_dict"])
        agent.q_net.eval()
        return agent
