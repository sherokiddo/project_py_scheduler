import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_lte_env_stack_builds_when_gymnasium_is_available():
    pytest.importorskip("gymnasium")

    from drl.envs.lte_padded_env import PaddedLTESchedulerEnv
    from drl.envs.lte_scheduler_env import LTESchedulerEnv

    env = PaddedLTESchedulerEnv(
        LTESchedulerEnv(
            n_ue=3,
            n_rb_dl=50,
            episode_len=10,
            reward_mode="per_tti",
            reward_window=1,
            seed=123,
        ),
        max_n_ue=8,
    )

    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert info["actual_n_ue"] == 3
    assert info["max_n_ue"] == 8
    assert len(info["action_mask"]) == 8

    env.close()


def test_lte_dqn_agent_builds_when_torch_is_available():
    pytest.importorskip("torch")

    from drl.agents.lte_dqn_agent import LTEDQNAgent, MaskedReplayBuffer

    agent = LTEDQNAgent(
        max_n_ue=8,
        ue_feature_dim=6,
        context_dim=3,
    )
    replay = MaskedReplayBuffer(capacity=16)

    assert agent.max_n_ue == 8
    assert replay.buffer.maxlen == 16
