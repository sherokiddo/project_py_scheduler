"""
PPO-агент для будущих экспериментов с LTE-средой.

Модуль перенесен из `drl_playground` и хранится рядом с DQN-реализацией,
чтобы внутри PyScheduler можно было развивать несколько DRL-алгоритмов.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    """
    Единая сеть с общим backbone и двумя головами:
    - actor -> логиты распределения действий
    - critic -> скалярная оценка V(s)
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, state: torch.Tensor):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """
    PPO-агент: сбор траекторий и обновление политики по PPO-loss.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device=None,
        lr: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 1.0,
        entropy_coef: float = 0.01,
        n_epochs: int = 10,
        batch_size: int = 64,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state: np.ndarray):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.ac.get_action(state_tensor)[:3]
        return action, log_prob.squeeze(), value.squeeze()

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ):
        total_steps = len(rewards)
        advantages = torch.zeros(total_steps, device=self.device)
        last_gae = 0.0

        for t in reversed(range(total_steps)):
            next_value = last_value if t == total_steps - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + values
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return advantages, returns

    def update(self, buffer, last_value: torch.Tensor):
        states, actions, rewards, dones, old_log_probs, values = buffer.get_tensors(
            self.device
        )
        values = values.squeeze(-1)
        old_log_probs = old_log_probs.squeeze(-1)

        with torch.no_grad():
            advantages, returns = self.compute_gae(
                rewards,
                values,
                dones,
                last_value.squeeze(),
            )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_size = len(states)

        for _ in range(self.n_epochs):
            indices = torch.randperm(total_size, device=self.device)

            for start in range(0, total_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                mb_old_lp = old_log_probs[idx]

                logits, values_new = self.ac(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(
                    values_new.squeeze(-1),
                    mb_returns,
                )
                entropy_loss = -entropy.mean()
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

        return loss.item()
