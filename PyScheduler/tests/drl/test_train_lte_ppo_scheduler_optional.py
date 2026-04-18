import os
import sys
from dataclasses import replace

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


pytest.importorskip("gymnasium")
pytest.importorskip("torch")


from drl.agents.lte_ppo_agent import LTEPPOAgent, MaskedPPORolloutBuffer
from drl.scripts.train_lte_ppo_scheduler import (
    SCENARIO_CONFIGS,
    make_env,
    resolve_eval_scenario_keys,
)


def test_runtime_ppo_train_script_builds_pyscheduler_env():
    scenario = SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"]
    env = make_env(scenario, max_n_ue=8, seed=123)

    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert 1 <= info["actual_n_ue"] <= scenario.n_ue
    assert info["max_n_ue"] == 8
    assert env.ue_feature_dim > 0
    assert env.context_dim > 0
    assert env.n_rbg > 0
    assert env.scheduler.__class__.__name__ == "DqnScheduler"

    env.close()


def test_runtime_ppo_agent_can_update_on_pyscheduler_env():
    scenario = replace(
        SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
        sim_duration_tti=6,
    )
    env = make_env(scenario, max_n_ue=8, seed=123)
    agent = LTEPPOAgent(
        max_n_ue=env.max_n_ue,
        ue_feature_dim=env.ue_feature_dim,
        context_dim=env.context_dim,
        hidden_dim=32,
        batch_size=8,
        n_epochs=2,
    )
    rollout_buffer = MaskedPPORolloutBuffer()

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

    metrics = agent.update(rollout_buffer, last_value=last_value)

    assert metrics["rollout_size"] == pytest.approx(24.0)
    assert metrics["num_updates"] >= 2.0
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["entropy"])

    env.close()


def test_runtime_ppo_eval_scenario_limit_helper():
    assert resolve_eval_scenario_keys(None) == (
        "anchor_5ue_10mhz_wb5_umi_fb",
        "anchor_5ue_10mhz_wb5_umi_onoff",
        "anchor_5ue_10mhz_wb5_uma_fb",
        "mid_8ue_10mhz_wb5_umi_fb",
        "mid_16ue_10mhz_wb5_umi_fb",
        "target_40ue_10mhz_wb5_umi_fb",
        "bw_16ue_5mhz_wb5_umi_fb",
        "bw_16ue_20mhz_wb5_umi_fb",
        "cqi_16ue_10mhz_wb1_umi_fb",
        "cqi_16ue_10mhz_wb10_umi_fb",
    )
    assert resolve_eval_scenario_keys(3) == (
        "anchor_5ue_10mhz_wb5_umi_fb",
        "anchor_5ue_10mhz_wb5_umi_onoff",
        "anchor_5ue_10mhz_wb5_uma_fb",
    )
    assert resolve_eval_scenario_keys(0) == ()
