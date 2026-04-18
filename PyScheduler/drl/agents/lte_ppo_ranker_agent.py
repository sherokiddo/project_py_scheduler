"""
PPO-агент для TTI-level ранжирования UE в задачах LTE-планирования.

Этот агент отличается от per-RBG PPO тем, что за один шаг политики
выдает не одного выбранного UE, а вектор score по всем UE в окне
наблюдения. Во время runtime эти score могут напрямую использоваться
как приоритеты для scheduler-а, а во время обучения PPO работает с
непрерывным action-space поверх тех же per-UE представлений.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class UEPPOScoreRolloutBatch:
    """
    Rollout-батч для PPO ranker-а.

    В action хранится полный вектор score по всем UE, который был
    сэмплирован политикой на конкретном шаге.
    """

    state:          tuple[np.ndarray, ...]
    action_mask:    tuple[np.ndarray, ...]
    action:         tuple[np.ndarray, ...]
    reward:         tuple[float, ...]
    done:           tuple[bool, ...]
    log_prob:       tuple[float, ...]
    value:          tuple[float, ...]


class UEPPOScoreRolloutBuffer:
    """
    Последовательный rollout-buffer для PPO ranker-а.

    Как и обычный PPO, ranker обновляется по свежему rollout, а не по
    replay-buffer. Поэтому буфер просто аккумулирует шаги очередной
    траектории и очищается после update.
    """

    def __init__(self) -> None:
        self.clear()

    def push(
        self,
        state:          np.ndarray,
        action_mask:    np.ndarray,
        action:         np.ndarray,
        reward:         float,
        done:           bool,
        log_prob:       float,
        value:          float,
    ) -> None:
        self._state.append(np.asarray(state, dtype=np.float32).copy())
        self._action_mask.append(np.asarray(action_mask, dtype=bool).copy())
        self._action.append(np.asarray(action, dtype=np.float32).copy())
        self._reward.append(float(reward))
        self._done.append(bool(done))
        self._log_prob.append(float(log_prob))
        self._value.append(float(value))

    def as_batch(self) -> UEPPOScoreRolloutBatch:
        return UEPPOScoreRolloutBatch(
            state       =tuple(self._state),
            action_mask =tuple(self._action_mask),
            action      =tuple(self._action),
            reward      =tuple(self._reward),
            done        =tuple(self._done),
            log_prob    =tuple(self._log_prob),
            value       =tuple(self._value),
        )

    def get_tensors(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch = self.as_batch()
        return (
            torch.as_tensor(
                np.asarray(batch.state, dtype=np.float32),
                dtype   =torch.float32,
                device  =device,
            ),
            torch.as_tensor(
                np.asarray(batch.action_mask, dtype=bool),
                dtype   =torch.bool,
                device  =device,
            ),
            torch.as_tensor(
                np.asarray(batch.action, dtype=np.float32),
                dtype   =torch.float32,
                device  =device,
            ),
            torch.as_tensor(
                np.asarray(batch.reward, dtype=np.float32),
                dtype   =torch.float32,
                device  =device,
            ),
            torch.as_tensor(
                np.asarray(batch.done, dtype=np.float32),
                dtype   =torch.float32,
                device  =device,
            ),
            torch.as_tensor(
                np.asarray(batch.log_prob, dtype=np.float32),
                dtype   =torch.float32,
                device  =device,
            ),
            torch.as_tensor(
                np.asarray(batch.value, dtype=np.float32),
                dtype   =torch.float32,
                device  =device,
            ),
        )

    def clear(self) -> None:
        self._state:        list[np.ndarray] = []
        self._action_mask:  list[np.ndarray] = []
        self._action:       list[np.ndarray] = []
        self._reward:       list[float] = []
        self._done:         list[bool] = []
        self._log_prob:     list[float] = []
        self._value:        list[float] = []

    def __len__(self) -> int:
        return len(self._action)


class SharedUERankerActorCritic(nn.Module):
    """
    Actor-critic сеть для ranker-задачи.

    Для каждого UE строится скрытое представление общим энкодером.
    Actor выдает скалярный score на UE, critic оценивает value всего
    TTI-состояния.
    """

    def __init__(
        self,
        max_n_ue: int,
        ue_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.max_n_ue       = int(max_n_ue)
        self.ue_feature_dim = int(ue_feature_dim)
        self.context_dim    = int(context_dim)
        self.hidden_dim     = int(hidden_dim)

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
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((self.max_n_ue,), -0.5))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        final_score = self.score_head[-1]
        final_value = self.value_head[-1]
        nn.init.orthogonal_(final_score.weight, gain=0.01)
        nn.init.zeros_(final_score.bias)
        nn.init.orthogonal_(final_value.weight, gain=1.0)
        nn.init.zeros_(final_value.bias)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ue_flat_dim = self.max_n_ue * self.ue_feature_dim
        ue_obs      = obs[:, :ue_flat_dim].reshape(-1, self.max_n_ue, self.ue_feature_dim)
        context_obs = obs[:, ue_flat_dim:]
        return ue_obs, context_obs

    def _encode_features(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ue_obs, context_obs = self._split_obs(obs)
        ue_hidden           = self.ue_encoder(ue_obs)
        context_hidden      = self.context_encoder(context_obs)

        mask            = action_mask.float().unsqueeze(-1)
        masked_sum      = torch.sum(ue_hidden * mask, dim=1)
        masked_count    = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled_hidden   = masked_sum / masked_count
        return ue_hidden, context_hidden, pooled_hidden

    def _compute_score_mean(
        self,
        ue_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        pooled_hidden: torch.Tensor,
    ) -> torch.Tensor:
        pooled_expanded  = pooled_hidden.unsqueeze(1).expand(-1, self.max_n_ue, -1)
        context_expanded = context_hidden.unsqueeze(1).expand(-1, self.max_n_ue, -1)

        score_input = torch.cat(
            [ue_hidden, pooled_expanded, context_expanded],
            dim=-1,
        )
        return self.score_head(score_input).squeeze(-1)

    def forward_actor(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ue_hidden, context_hidden, pooled_hidden = self._encode_features(
            obs,
            action_mask,
        )
        score_mean = self._compute_score_mean(
            ue_hidden,
            context_hidden,
            pooled_hidden,
        )
        log_std = self.log_std.unsqueeze(0).expand_as(score_mean)
        return score_mean, log_std

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ue_hidden, context_hidden, pooled_hidden = self._encode_features(
            obs,
            action_mask,
        )
        score_mean = self._compute_score_mean(
            ue_hidden,
            context_hidden,
            pooled_hidden,
        )
        value_input = torch.cat([pooled_hidden, context_hidden], dim=-1)
        value = self.value_head(value_input).squeeze(-1)
        log_std = self.log_std.unsqueeze(0).expand_as(score_mean)
        return score_mean, log_std, value


class LTEPPORankerAgent:
    """
    PPO-агент для UE-ranking в одном TTI.

    Во время обучения policy сэмплирует непрерывный score-вектор по всем
    UE. Во время инференса deterministic-режим возвращает средние score,
    которые можно напрямую использовать как priority для scheduler-а.
    """

    def __init__(
        self,
        max_n_ue:       int,
        ue_feature_dim: int,
        context_dim:    int,
        hidden_dim:     int   = 64,
        device:         Optional[torch.device] = None,
        lr:             float = 3e-4,
        gamma:          float = 0.995,
        gae_lambda:     float = 0.95,
        clip_epsilon:   float = 0.2,
        value_coef:     float = 0.5,
        entropy_coef:   float = 0.01,
        n_epochs:       int = 4,
        batch_size:     int = 256,
        min_log_std:    float = -4.0,
        max_log_std:    float = 1.0,
    ) -> None:
        self.max_n_ue       = int(max_n_ue)
        self.ue_feature_dim = int(ue_feature_dim)
        self.context_dim    = int(context_dim)
        self.hidden_dim     = int(hidden_dim)
        self.device         = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.gamma          = float(gamma)
        self.gae_lambda     = float(gae_lambda)
        self.clip_epsilon   = float(clip_epsilon)
        self.value_coef     = float(value_coef)
        self.entropy_coef   = float(entropy_coef)
        self.n_epochs       = int(n_epochs)
        self.batch_size     = int(batch_size)
        self.lr             = float(lr)
        self.min_log_std    = float(min_log_std)
        self.max_log_std    = float(max_log_std)
        self.total_steps    = 0

        self.policy = SharedUERankerActorCritic(
            max_n_ue        =max_n_ue,
            ue_feature_dim  =ue_feature_dim,
            context_dim     =context_dim,
            hidden_dim      =hidden_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)

    @staticmethod
    def _sanitize_action_mask_np(action_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.size == 0:
            raise ValueError("action_mask не должен быть пустым.")
        if np.any(mask):
            return mask
        return np.ones_like(mask, dtype=bool)

    @staticmethod
    def _sanitize_action_mask_tensor(action_mask: torch.Tensor) -> torch.Tensor:
        mask            = action_mask.bool()
        valid_counts    = mask.sum(dim=1, keepdim=True)
        all_valid       = torch.ones_like(mask, dtype=torch.bool)
        return torch.where(valid_counts > 0, mask, all_valid)

    def _policy_outputs(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sanitized_mask = self._sanitize_action_mask_tensor(action_mask)
        score_mean, log_std, value = self.policy(obs, sanitized_mask)
        log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
        std = torch.exp(log_std)
        return score_mean, std, value

    def _policy_actor_outputs(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sanitized_mask = self._sanitize_action_mask_tensor(action_mask)
        score_mean, log_std = self.policy.forward_actor(obs, sanitized_mask)
        log_std = torch.clamp(log_std, min=self.min_log_std, max=self.max_log_std)
        std = torch.exp(log_std)
        return score_mean, std

    def _masked_scores(
        self,
        score_mean: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        sanitized_mask = self._sanitize_action_mask_tensor(action_mask)
        return score_mean.masked_fill(~sanitized_mask, -1e9)

    def _log_prob_and_entropy(
        self,
        actions: torch.Tensor,
        score_mean: torch.Tensor,
        std: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sanitized_mask = self._sanitize_action_mask_tensor(action_mask)
        mask = sanitized_mask.float()
        valid_counts = torch.clamp(mask.sum(dim=1), min=1.0)

        dist = torch.distributions.Normal(score_mean, std)
        log_prob = (dist.log_prob(actions) * mask).sum(dim=1)
        entropy = (dist.entropy() * mask).sum(dim=1) / valid_counts
        return log_prob, entropy

    def predict_scores(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        mask = self._sanitize_action_mask_np(action_mask)
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            score_mean, std = self._policy_actor_outputs(obs_tensor, mask_tensor)
            if deterministic:
                action_tensor = score_mean
            else:
                dist = torch.distributions.Normal(score_mean, std)
                action_tensor = dist.sample()

            action_tensor = torch.where(
                mask_tensor,
                action_tensor,
                torch.full_like(action_tensor, -1e9),
            )

        return (
            action_tensor.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def predict_ranking(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        mask = self._sanitize_action_mask_np(action_mask)
        scores = self.predict_scores(
            obs=obs,
            action_mask=mask,
            deterministic=deterministic,
        )
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size == 0:
            return np.arange(self.max_n_ue, dtype=np.int64)

        order = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        return order.astype(np.int64, copy=False)

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        return self.predict_scores(
            obs=obs,
            action_mask=action_mask,
            deterministic=deterministic,
        )

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        self.total_steps += 1

        sanitized_mask = self._sanitize_action_mask_np(action_mask)
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            sanitized_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            score_mean, std, value = self._policy_outputs(obs_tensor, mask_tensor)
            if deterministic:
                action_tensor = score_mean
            else:
                dist = torch.distributions.Normal(score_mean, std)
                action_tensor = dist.sample()
            action_tensor = torch.where(
                mask_tensor,
                action_tensor,
                torch.zeros_like(action_tensor),
            )
            log_prob, _ = self._log_prob_and_entropy(
                action_tensor,
                score_mean,
                std,
                mask_tensor,
            )

        return (
            action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.squeeze(0).item()),
        )

    def estimate_value(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
    ) -> float:
        sanitized_mask = self._sanitize_action_mask_np(action_mask)
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            sanitized_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            _, _, value = self._policy_outputs(obs_tensor, mask_tensor)
        return float(value.squeeze(0).item())

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_steps = len(rewards)
        advantages = torch.zeros(total_steps, dtype=torch.float32, device=self.device)
        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_value = last_value.reshape(())

        for step_idx in reversed(range(total_steps)):
            next_non_terminal = 1.0 - dones[step_idx]
            delta = (
                rewards[step_idx]
                + self.gamma * next_value * next_non_terminal
                - values[step_idx]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step_idx] = gae
            next_value = values[step_idx]

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        buffer: UEPPOScoreRolloutBuffer,
        last_value: float | torch.Tensor = 0.0,
    ) -> dict[str, float]:
        if len(buffer) == 0:
            raise ValueError("Невозможно выполнить PPO update по пустому rollout buffer.")

        (
            states,
            action_masks,
            actions,
            rewards,
            dones,
            old_log_probs,
            values,
        ) = buffer.get_tensors(self.device)

        action_masks = self._sanitize_action_mask_tensor(action_masks)
        last_value_tensor = torch.as_tensor(
            last_value,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            advantages, returns = self.compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                last_value=last_value_tensor,
            )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_size          = len(states)
        mean_loss           = 0.0
        mean_actor_loss     = 0.0
        mean_critic_loss    = 0.0
        mean_entropy        = 0.0
        num_updates         = 0

        for _ in range(self.n_epochs):
            permutation = torch.randperm(total_size, device=self.device)
            for start_idx in range(0, total_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_idx = permutation[start_idx:end_idx]

                mb_states       = states[batch_idx]
                mb_masks        = action_masks[batch_idx]
                mb_actions      = actions[batch_idx]
                mb_advantages   = advantages[batch_idx]
                mb_returns      = returns[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]

                score_mean, std, values_new = self._policy_outputs(mb_states, mb_masks)
                new_log_probs, entropy = self._log_prob_and_entropy(
                    mb_actions,
                    score_mean,
                    std,
                    mb_masks,
                )
                entropy_mean = entropy.mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surrogate_1 = ratio * mb_advantages
                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * mb_advantages

                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                critic_loss = F.mse_loss(values_new, mb_returns)
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                mean_loss       += float(loss.item())
                mean_actor_loss += float(actor_loss.item())
                mean_critic_loss += float(critic_loss.item())
                mean_entropy    += float(entropy_mean.item())
                num_updates     += 1

        normalizer = max(num_updates, 1)
        return {
            "loss":         mean_loss / normalizer,
            "actor_loss":   mean_actor_loss / normalizer,
            "critic_loss":  mean_critic_loss / normalizer,
            "entropy":      mean_entropy / normalizer,
            "num_updates":  float(num_updates),
            "rollout_size": float(total_size),
        }

    def save(self, path: str | BinaryIO) -> None:
        payload = {
            "state_dict": self.policy.state_dict(),
            "config": {
                "max_n_ue":         self.max_n_ue,
                "ue_feature_dim":   self.ue_feature_dim,
                "context_dim":      self.context_dim,
                "hidden_dim":       self.hidden_dim,
                "lr":               self.lr,
                "gamma":            self.gamma,
                "gae_lambda":       self.gae_lambda,
                "clip_epsilon":     self.clip_epsilon,
                "value_coef":       self.value_coef,
                "entropy_coef":     self.entropy_coef,
                "n_epochs":         self.n_epochs,
                "batch_size":       self.batch_size,
                "min_log_std":      self.min_log_std,
                "max_log_std":      self.max_log_std,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | BinaryIO,
        device: Optional[torch.device] = None,
    ) -> "LTEPPORankerAgent":
        payload = torch.load(path, map_location=device or "cpu")
        config = dict(payload["config"])
        agent = cls(device=device, **config)
        agent.policy.load_state_dict(payload["state_dict"])
        return agent
