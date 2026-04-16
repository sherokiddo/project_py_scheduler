import os
import sys
from io import BytesIO

import numpy as np
import pytest
import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drl.agents.lte_ppo_ranker_agent import (
    LTEPPORankerAgent,
    UEPPOScoreRolloutBuffer,
)


def _make_agent() -> LTEPPORankerAgent:
    return LTEPPORankerAgent(
        max_n_ue=4,
        ue_feature_dim=3,
        context_dim=2,
        hidden_dim=16,
        device=torch.device("cpu"),
        n_epochs=2,
        batch_size=4,
    )


def _make_observation(agent: LTEPPORankerAgent, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs_dim = agent.max_n_ue * agent.ue_feature_dim + agent.context_dim
    return rng.normal(size=obs_dim).astype(np.float32)


def test_lte_ppo_ranker_agent_predict_ranking_respects_mask() -> None:
    agent = _make_agent()
    obs = _make_observation(agent, seed=1)
    action_mask = np.array([False, True, False, True], dtype=bool)

    scores = agent.predict_scores(obs, action_mask, deterministic=True)
    ranking = agent.predict_ranking(obs, action_mask, deterministic=True)

    assert scores.shape == (agent.max_n_ue,)
    assert scores[0] < -1e8
    assert scores[2] < -1e8
    assert set(ranking.tolist()) == {1, 3}


def test_lte_ppo_ranker_agent_select_action_returns_score_vector() -> None:
    agent = _make_agent()
    obs = _make_observation(agent, seed=5)
    action_mask = np.array([True, False, True, False], dtype=bool)

    action_scores, log_prob, value = agent.select_action(
        obs,
        action_mask,
        deterministic=False,
    )

    assert action_scores.shape == (agent.max_n_ue,)
    assert action_scores[1] == pytest.approx(0.0)
    assert action_scores[3] == pytest.approx(0.0)
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


def test_lte_ppo_ranker_agent_update_runs_on_rollout_buffer() -> None:
    agent = _make_agent()
    buffer = UEPPOScoreRolloutBuffer()

    for step_idx in range(8):
        obs = _make_observation(agent, seed=10 + step_idx)
        action_mask = np.array([True, step_idx % 2 == 0, True, False], dtype=bool)
        action_scores, log_prob, value = agent.select_action(
            obs,
            action_mask,
            deterministic=False,
        )
        reward = float(action_scores[0] - 0.25 * abs(action_scores[2]))
        done = step_idx == 7
        buffer.push(
            state=obs,
            action_mask=action_mask,
            action=action_scores,
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


def test_lte_ppo_ranker_agent_save_load_roundtrip() -> None:
    agent = _make_agent()
    buffer = BytesIO()

    agent.save(buffer)
    buffer.seek(0)
    loaded = LTEPPORankerAgent.load(buffer, device=torch.device("cpu"))

    assert loaded.max_n_ue == agent.max_n_ue
    assert loaded.ue_feature_dim == agent.ue_feature_dim
    assert loaded.context_dim == agent.context_dim
    assert loaded.hidden_dim == agent.hidden_dim

    obs = _make_observation(agent, seed=99)
    action_mask = np.array([True, False, True, True], dtype=bool)
    scores = loaded.predict(obs, action_mask, deterministic=True)
    ranking = loaded.predict_ranking(obs, action_mask, deterministic=True)

    assert scores.shape == (agent.max_n_ue,)
    assert scores[1] < -1e8
    assert set(ranking.tolist()) == {0, 2, 3}
