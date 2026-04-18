import os
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


pytest.importorskip("gymnasium")
pytest.importorskip("torch")


import SIMULATION_MANAGER as simulation_manager_module

from drl.agents.lte_dqn_agent import LTEDQNAgent, MaskedReplayBuffer
from drl.paths import PYSCHEDULER_DQN_RUN_DIR
from drl.runtime_scenario_factory import SCENARIO_CONFIGS, create_inference_manager
from drl.scripts.train_lte_dqn_pyscheduler import make_env


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        self.description = kwargs.get("desc", "")
        self.updated = 0

    def update(self, value):
        self.updated += value

    def set_description_str(self, value):
        self.description = value

    def refresh(self):
        return None

    def close(self):
        return None


def _build_smoke_checkpoint(weights_path: Path) -> None:
    scenario = replace(
        SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
        sim_duration_tti=6,
    )
    env = make_env(scenario, max_n_ue=8, seed=123)
    agent = LTEDQNAgent(
        max_n_ue=env.max_n_ue,
        ue_feature_dim=env.ue_feature_dim,
        context_dim=env.context_dim,
        hidden_dim=32,
        epsilon_decay_steps=32,
    )
    replay_buffer = MaskedReplayBuffer(capacity=128)

    obs, info = env.reset(seed=123)
    for step_idx in range(24):
        action = agent.select_action(obs, info["action_mask"], deterministic=False)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        replay_buffer.push(
            obs,
            info["action_mask"],
            action,
            float(reward),
            next_obs,
            next_info["action_mask"],
            bool(terminated or truncated),
        )

        if len(replay_buffer) >= 4 and next_info["rbg_step"] == 0:
            agent.train_step(replay_buffer.sample(4))

        obs, info = next_obs, next_info
        if terminated or truncated:
            obs, info = env.reset(seed=124 + step_idx)

    agent.update_target()
    agent.save(str(weights_path))
    env.close()


def test_runtime_dqn_train_to_inference_end_to_end(monkeypatch):
    weights_path = PYSCHEDULER_DQN_RUN_DIR / f"smoke_lte_dqn_{uuid4().hex}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    _build_smoke_checkpoint(weights_path)

    try:
        monkeypatch.setattr(simulation_manager_module, "tqdm", DummyTqdm)

        inference_scenario = replace(
            SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
            sim_duration_tti=4,
        )
        manager = create_inference_manager(
            inference_scenario,
            model_path=weights_path,
            max_n_ue=8,
            seed=123,
            deterministic=True,
        )

        manager.start_simulation()

        assert manager.scheduler.__class__.__name__ == "DqnScheduler"
        stats = manager.scheduler.get_stats()
        assert stats["dqn_step_count"] > 0
        assert Path(stats["dqn_model_path"]) == weights_path
        assert stats["dqn_inference_device"] == "cpu"
    finally:
        try:
            weights_path.unlink(missing_ok=True)
        except PermissionError:
            pass
