import os
import sys
from dataclasses import replace

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


pytest.importorskip("gymnasium")
pytest.importorskip("torch")


from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent, UEPPOScoreRolloutBuffer
from drl.ranker_observation_adapter import (
    RANKER_OBSERVATION_N_CONTEXT_FEATURES,
    RANKER_OBSERVATION_N_UE_FEATURES,
)
from drl.scripts.train_lte_ppo_ranker import (
    SCENARIO_CONFIGS,
    build_ranker_observation_contract,
    evaluate_agent,
    make_env,
    resolve_eval_scenario_keys,
    run_probe_evaluation,
)


def test_runtime_ppo_ranker_train_script_builds_pyscheduler_env():
    scenario = SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"]
    env = make_env(scenario, max_n_ue=8, seed=123)

    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert 1 <= info["actual_n_ue"] <= scenario.n_ue
    assert info["max_n_ue"] == 8
    assert env.ue_feature_dim == RANKER_OBSERVATION_N_UE_FEATURES
    assert env.context_dim == RANKER_OBSERVATION_N_CONTEXT_FEATURES
    assert env.n_rbg > 0
    assert env.scheduler.__class__.__name__ == "DqnScheduler"

    env.close()


def test_runtime_ppo_ranker_train_script_reports_compact_observation_contract():
    scenario = SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"]
    env = make_env(scenario, max_n_ue=8, seed=123)

    contract = build_ranker_observation_contract(env)

    assert contract["variant"] == "ranker_compact_v1"
    assert contract["adapter_mode"] == "proxy_start_tti"
    assert contract["ue_feature_dim"] == RANKER_OBSERVATION_N_UE_FEATURES
    assert contract["context_dim"] == RANKER_OBSERVATION_N_CONTEXT_FEATURES
    assert contract["ue_feature_names"] == [
        "reported_wb_cqi",
        "wb_cqi_age_tti",
        "buffer_bytes",
        "average_throughput_bps",
    ]
    assert contract["context_feature_names"] == [
        "active_ue_count",
        "n_rbg",
    ]
    assert contract["obs_dim"] == (
        env.max_n_ue * env.ue_feature_dim + env.context_dim
    )

    env.close()


def test_runtime_ppo_ranker_agent_can_update_on_ranker_env():
    scenario = replace(
        SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
        sim_duration_tti=6,
    )
    env = make_env(scenario, max_n_ue=8, seed=123)
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

    metrics = agent.update(rollout_buffer, last_value=last_value)

    assert metrics["rollout_size"] == pytest.approx(24.0)
    assert metrics["num_updates"] >= 2.0
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["entropy"])

    env.close()


def test_runtime_ppo_ranker_eval_scenario_limit_helper():
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


def test_runtime_ppo_ranker_evaluate_agent_supports_both_eval_modes():
    scenario = replace(
        SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
        sim_duration_tti=6,
    )
    env = make_env(scenario, max_n_ue=8, seed=123)
    agent = LTEPPORankerAgent(
        max_n_ue=env.max_n_ue,
        ue_feature_dim=env.ue_feature_dim,
        context_dim=env.context_dim,
        hidden_dim=32,
        batch_size=8,
        n_epochs=2,
    )

    deterministic_summary = evaluate_agent(
        agent,
        env,
        seed=123,
        deterministic=True,
    )
    stochastic_summary = evaluate_agent(
        agent,
        env,
        seed=124,
        deterministic=False,
    )

    assert np.isfinite(deterministic_summary["mean_reward"])
    assert np.isfinite(deterministic_summary["mean_jfi_all"])
    assert np.isfinite(stochastic_summary["mean_reward"])
    assert np.isfinite(stochastic_summary["mean_jfi_all"])

    env.close()


def test_runtime_ppo_ranker_probe_eval_returns_finite_metrics():
    scenario = replace(
        SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
        sim_duration_tti=6,
    )
    env = make_env(scenario, max_n_ue=8, seed=123)
    agent = LTEPPORankerAgent(
        max_n_ue=env.max_n_ue,
        ue_feature_dim=env.ue_feature_dim,
        context_dim=env.context_dim,
        hidden_dim=32,
        batch_size=8,
        n_epochs=2,
    )

    summary = run_probe_evaluation(
        agent,
        "anchor_5ue_10mhz_wb5_umi_fb",
        max_n_ue=8,
        seed=123,
        rank_weight_beta=0.3,
        pf_epsilon_bps=1e-6,
        eval_device=agent.device,
    )

    assert summary["probe_scenario"] == "anchor_5ue_10mhz_wb5_umi_fb"
    assert summary["probe_seed"] == 123
    assert np.isfinite(summary["mean_reward"])
    assert np.isfinite(summary["mean_se_bps_hz"])
    assert np.isfinite(summary["mean_jfi_all"])

    env.close()
