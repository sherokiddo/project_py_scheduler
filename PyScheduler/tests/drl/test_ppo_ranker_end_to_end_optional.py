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

from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent, UEPPOScoreRolloutBuffer
from drl.paths import PYSCHEDULER_PPO_RANKER_RUN_DIR
from drl.runtime_scenario_factory import (
    SCENARIO_CONFIGS,
    create_ppo_ranker_inference_manager,
)
from drl.scripts.train_lte_ppo_ranker import make_env


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
    env = make_env(
        scenario,
        max_n_ue=8,
        seed=123,
        rank_weight_beta=0.3,
        pf_epsilon_bps=1e-6,
    )
    agent = LTEPPORankerAgent(
        max_n_ue=env.max_n_ue,
        ue_feature_dim=env.ue_feature_dim,
        context_dim=env.context_dim,
        hidden_dim=32,
        batch_size=8,
        n_epochs=2,
    )
    rollout_buffer = UEPPOScoreRolloutBuffer()

    obs, info = env.reset(seed=123)
    last_done = False

    for step_idx in range(24):
        action, log_prob, value = agent.select_action(
            obs,
            info["action_mask"],
            deterministic=False,
        )
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        last_done = bool(terminated or truncated)
        rollout_buffer.push(
            state=obs,
            action_mask=info["action_mask"],
            action=action,
            reward=float(reward),
            done=last_done,
            log_prob=log_prob,
            value=value,
        )

        obs, info = next_obs, next_info
        if last_done:
            obs, info = env.reset(seed=124 + step_idx)

    if last_done:
        last_value = 0.0
    else:
        last_value = agent.estimate_value(obs, info["action_mask"])

    agent.update(rollout_buffer, last_value=last_value)
    agent.save(str(weights_path))
    env.close()


def test_runtime_ppo_ranker_train_to_inference_end_to_end(monkeypatch):
    weights_path = (
        PYSCHEDULER_PPO_RANKER_RUN_DIR / f"smoke_lte_ppo_ranker_{uuid4().hex}.pt"
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    _build_smoke_checkpoint(weights_path)

    try:
        monkeypatch.setattr(simulation_manager_module, "tqdm", DummyTqdm)

        inference_scenario = replace(
            SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
            sim_duration_tti=4,
        )
        manager = create_ppo_ranker_inference_manager(
            inference_scenario,
            model_path=weights_path,
            max_n_ue=8,
            seed=123,
            deterministic=True,
            rank_weight_beta=0.3,
            pf_epsilon_bps=1e-6,
        )

        manager.start_simulation()

        assert manager.scheduler.__class__.__name__ == "PpoRankerScheduler"
        stats = manager.scheduler.get_stats()
        assert stats["ppo_ranker_step_count"] > 0
        assert Path(stats["ppo_ranker_model_path"]) == weights_path
        assert stats["ppo_ranker_inference_device"] == "cpu"
        assert len(stats["ppo_ranker_ranked_ue_ids"]) > 0
    finally:
        try:
            weights_path.unlink(missing_ok=True)
        except PermissionError:
            pass
