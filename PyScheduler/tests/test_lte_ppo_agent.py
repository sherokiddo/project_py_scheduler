from io import BytesIO
import numpy as np
import pytest
import torch

from drl.agents.lte_ppo_agent import LTEPPOAgent, MaskedPPORolloutBuffer


def _make_agent() -> LTEPPOAgent:
    return LTEPPOAgent(
        max_n_ue=4,
        ue_feature_dim=3,
        context_dim=2,
        hidden_dim=16,
        device=torch.device("cpu"),
        n_epochs=2,
        batch_size=4,
    )


def _make_observation(agent: LTEPPOAgent, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs_dim = agent.max_n_ue * agent.ue_feature_dim + agent.context_dim
    return rng.normal(size=obs_dim).astype(np.float32)


def test_lte_ppo_agent_select_action_respects_mask() -> None:
    agent = _make_agent()
    obs = _make_observation(agent, seed=1)
    action_mask = np.array([False, True, False, True], dtype=bool)

    for _ in range(16):
        action, log_prob, value = agent.select_action(
            obs,
            action_mask,
            deterministic=False,
        )
        assert action in (1, 3)
        assert np.isfinite(log_prob)
        assert np.isfinite(value)


def test_lte_ppo_agent_update_runs_on_rollout_buffer() -> None:
    agent = _make_agent()
    buffer = MaskedPPORolloutBuffer()

    for step_idx in range(8):
        obs = _make_observation(agent, seed=10 + step_idx)
        action_mask = np.array([True, step_idx % 2 == 0, True, False], dtype=bool)
        action, log_prob, value = agent.select_action(
            obs,
            action_mask,
            deterministic=False,
        )
        reward = 1.0 if action in (0, 2) else -0.25
        done = step_idx == 7
        buffer.push(
            state=obs,
            action_mask=action_mask,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

    metrics = agent.update(buffer, last_value=0.0)

    assert metrics["rollout_size"] == pytest.approx(8.0)
    assert metrics["num_updates"] >= 2.0
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["entropy"])


def test_lte_ppo_agent_save_load_roundtrip() -> None:
    agent = _make_agent()
    buffer = BytesIO()

    agent.save(buffer)
    buffer.seek(0)
    loaded = LTEPPOAgent.load(buffer, device=torch.device("cpu"))

    assert loaded.max_n_ue == agent.max_n_ue
    assert loaded.ue_feature_dim == agent.ue_feature_dim
    assert loaded.context_dim == agent.context_dim
    assert loaded.hidden_dim == agent.hidden_dim

    obs = _make_observation(agent, seed=99)
    action_mask = np.array([True, False, True, True], dtype=bool)
    action = loaded.predict(obs, action_mask, deterministic=True)
    assert action in (0, 2, 3)
