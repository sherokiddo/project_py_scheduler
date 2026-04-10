"""
Полноценный LTE DQN-агент для обучения и инференса.

Модуль перенесен из `drl_playground` и оставлен внутри репозитория PyScheduler,
чтобы обучение, сохранение весов и инференс жили в одном месте.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class MaskedTransitionBatch:
    """
    Батч переходов с action-mask для обучения DQN.
    """

    state: tuple[np.ndarray, ...]
    action_mask: tuple[np.ndarray, ...]
    action: tuple[int, ...]
    reward: tuple[float, ...]
    next_state: tuple[np.ndarray, ...]
    next_action_mask: tuple[np.ndarray, ...]
    done: tuple[bool, ...]


class MaskedReplayBuffer:
    """
    Replay buffer, сохраняющий action-mask вместе с переходами.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                state,
                action_mask,
                action,
                reward,
                next_state,
                next_action_mask,
                done,
            )
        )

    def sample(self, batch_size: int) -> MaskedTransitionBatch:
        batch = random.sample(self.buffer, batch_size)
        return MaskedTransitionBatch(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class SharedUEQNetwork(nn.Module):
    """
    Общая сеть-скоратор для множества UE.

    Каждый UE кодируется одним и тем же MLP, после чего признаки объединяются
    с pooled-представлением активного набора UE и глобальным контекстом.
    """

    def __init__(
        self,
        max_n_ue: int,
        ue_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
    ):
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

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ue_flat_dim = self.max_n_ue * self.ue_feature_dim
        ue_obs = obs[:, :ue_flat_dim].reshape(-1, self.max_n_ue, self.ue_feature_dim)
        context_obs = obs[:, ue_flat_dim:]
        return ue_obs, context_obs

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
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
    LTE DQN-агент с поддержкой обучения, сохранения и инференса.
    """

    def __init__(
        self,
        max_n_ue: int,
        ue_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 150_000,
        gamma: float = 0.995,
        lr: float = 3e-4,
    ):
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
        self.target_q_net = SharedUEQNetwork(
            max_n_ue=max_n_ue,
            ue_feature_dim=ue_feature_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    def _get_epsilon(self) -> float:
        fraction = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    @staticmethod
    def _mask_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        return q_values.masked_fill(~action_mask.bool(), -1e9)

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        return self.select_action(obs, action_mask, deterministic=deterministic)

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
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

    def train_step(self, batch: MaskedTransitionBatch) -> float:
        states = torch.as_tensor(
            np.asarray(batch.state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        action_masks = torch.as_tensor(
            np.asarray(batch.action_mask, dtype=bool),
            dtype=torch.bool,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(batch.action, dtype=np.int64),
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(-1)
        rewards = torch.as_tensor(
            np.asarray(batch.reward, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)
        next_states = torch.as_tensor(
            np.asarray(batch.next_state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        next_action_masks = torch.as_tensor(
            np.asarray(batch.next_action_mask, dtype=bool),
            dtype=torch.bool,
            device=self.device,
        )
        dones = torch.as_tensor(
            np.asarray(batch.done, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        q_values = self.q_net(states, action_masks).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states, next_action_masks)
            masked_next_q = self._mask_q_values(next_q_values, next_action_masks)
            max_next_q = masked_next_q.max(dim=1, keepdim=True).values
            target_q = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = torch.nn.functional.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.q_net.state_dict(),
            "config": {
                "max_n_ue": self.max_n_ue,
                "ue_feature_dim": self.ue_feature_dim,
                "context_dim": self.context_dim,
                "hidden_dim": self.hidden_dim,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        payload = torch.load(path, map_location=device or "cpu")
        config = payload["config"]
        agent = cls(device=device, **config)
        agent.q_net.load_state_dict(payload["state_dict"])
        agent.target_q_net.load_state_dict(agent.q_net.state_dict())
        agent.target_q_net.eval()
        return agent
