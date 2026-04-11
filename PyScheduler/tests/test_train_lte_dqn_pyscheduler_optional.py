import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


pytest.importorskip("gymnasium")


from drl.scripts.train_lte_dqn_pyscheduler import SCENARIO_CONFIGS, make_env


def test_runtime_train_script_builds_pyscheduler_env():
    scenario = SCENARIO_CONFIGS["train_3ue_10mhz_wb5"]
    env = make_env(scenario, max_n_ue=8, seed=123)

    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert info["actual_n_ue"] == 3
    assert info["max_n_ue"] == 8
    assert env.ue_feature_dim > 0
    assert env.context_dim > 0
    assert env.n_rbg > 0
    assert env.scheduler.__class__.__name__ == "DqnScheduler"
    assert env.scheduler.wb_cqi_upd_interval == scenario.wb_cqi_report_period_tti
    assert env.scheduler.sb_cqi_upd_interval == scenario.wb_cqi_report_period_tti

    action = next(idx for idx, is_valid in enumerate(info["action_mask"]) if is_valid)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert truncated is False
    assert next_info["max_n_ue"] == 8
    assert terminated in (True, False)

    env.close()
