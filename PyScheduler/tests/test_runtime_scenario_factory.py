import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drl.runtime_scenario_factory import (
    SCENARIO_CONFIGS,
    build_simulation_manager_factory,
    create_training_manager,
)


def test_create_training_manager_builds_runtime_configuration():
    scenario = SCENARIO_CONFIGS["train_3ue_10mhz_wb5"]

    manager = create_training_manager(
        scenario,
        max_n_ue=8,
        seed=123,
    )

    assert manager.sched_config.algorithm == "DqnScheduler"
    assert manager.sched_config.algorithm_kwargs["dqn_max_n_ue"] == 8
    assert manager.sched_config.algorithm_kwargs["dqn_episode_len_tti"] == 200
    assert manager.sched_config.algorithm_kwargs["dqn_wb_cqi_report_period_tti"] == 5
    assert manager.base_station.bandwidth == 10
    assert len(manager.ue_collection.GET_ALL_USERS()) == 3
    assert manager.sim_config.sim_duration == 200
    assert manager.sim_config.channel_update_interval == 10
    assert manager.stats_config.enabled is False


def test_build_simulation_manager_factory_applies_runtime_overrides():
    scenario = SCENARIO_CONFIGS["train_3ue_10mhz_wb5"]
    factory = build_simulation_manager_factory(
        scenario,
        max_n_ue=8,
        default_seed=123,
    )

    manager = factory(
        options={
            "bandwidth_mhz": 20,
            "n_ue": 5,
            "wb_cqi_report_period_tti": 10,
            "traffic_packet_rate": 7000,
        }
    )

    assert manager.base_station.bandwidth == 20
    assert len(manager.ue_collection.GET_ALL_USERS()) == 5
    assert manager.sched_config.algorithm_kwargs["dqn_wb_cqi_report_period_tti"] == 10
    assert len(manager.traffic_gen.models) == 5
    assert manager.traffic_gen.models[1].packet_rate == 7000
