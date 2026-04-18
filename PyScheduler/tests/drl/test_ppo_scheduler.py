import os
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SIMULATION_MANAGER import SimulationManager
from SCHEDULER import SchedulerInterface
from drl.dqn_scheduler import DqnScheduler
from drl.ppo_scheduler import PpoScheduler


class FakeGrid:
    def __init__(self):
        self.bandwidth = 10.0
        self.rb_per_slot = 6
        self.bs = None
        self.allocations = {}

    def SET_BS(self, bs):
        self.bs = bs

    def GET_RBG_SIZE(self):
        return 2

    def GET_RBG_INDICES(self, rbg_idx):
        start = rbg_idx * 2
        return [start, start + 1]

    def ALLOCATE_RBG(self, tti, rbg_idx, ue_id):
        self.allocations[(tti, rbg_idx)] = ue_id
        return True


class FakeBufferManager:
    def ue_has_buffer(self, ue_id):
        return True


class FakeBaseStation:
    def __init__(self):
        self.buffer_manager = FakeBufferManager()
        self.use_simple_buffer = True
        self.frequency_GHz = 3.5
        self.ch_model_type = "UMa"
        self.enable_tdl = True


class FakeUE:
    def __init__(self, ue_id, cqi, avg_tput, sinr):
        self.UE_ID = ue_id
        self.cqi = cqi
        self.average_throughput = avg_tput
        self.current_dl_throughput = 0.0
        self.SINR = sinr


class FakeRunner:
    def __init__(self, actions, max_n_ue):
        self.actions = list(actions)
        self.max_n_ue = max_n_ue
        self.calls = []

    def predict(self, obs, action_mask, deterministic=None):
        self.calls.append(
            {
                "obs": obs.copy(),
                "mask": action_mask.copy(),
                "deterministic": deterministic,
            }
        )
        return int(self.actions.pop(0))


def _build_eligible_users():
    ue_1 = FakeUE(ue_id=1, cqi=10, avg_tput=10e6, sinr=12.0)
    ue_2 = FakeUE(ue_id=2, cqi=7, avg_tput=5e6, sinr=8.0)
    return [
        {
            "UE_ID": 1,
            "ue": ue_1,
            "cqi": 10,
            "sbb_cqi": [10, 10, 10],
            "bs_buffer_size": 100_000,
        },
        {
            "UE_ID": 2,
            "ue": ue_2,
            "cqi": 7,
            "sbb_cqi": [7, 7, 7],
            "bs_buffer_size": 100_000,
        },
    ]


def _prime_cqi_map(scheduler):
    scheduler.cqi_map = {
        1: SimpleNamespace(wb_cqi=10, last_wb_update=3, sb_cqi=[10, 10, 10]),
        2: SimpleNamespace(wb_cqi=7, last_wb_update=4, sb_cqi=[7, 7, 7]),
    }


def test_scheduler_factory_registers_ppo_scheduler():
    scheduler = SchedulerInterface.create(
        algorithm="PpoScheduler",
        lte_grid=FakeGrid(),
        bs=FakeBaseStation(),
        ppo_policy_runner=FakeRunner(actions=[0], max_n_ue=4),
        ppo_max_n_ue=4,
        ppo_wb_cqi_report_period_tti=5,
        ppo_episode_len_tti=10,
    )

    assert isinstance(scheduler, PpoScheduler)
    assert not isinstance(scheduler, DqnScheduler)
    assert "PpoScheduler" in SchedulerInterface.available_algorithms()


def test_ppo_scheduler_allocates_per_rbg_and_tracks_runtime_stats():
    grid = FakeGrid()
    runner = FakeRunner(actions=[1, 3, 0], max_n_ue=4)
    scheduler = PpoScheduler(
        lte_grid=grid,
        bs=FakeBaseStation(),
        ppo_policy_runner=runner,
        ppo_max_n_ue=4,
        ppo_wb_cqi_report_period_tti=5,
        ppo_episode_len_tti=10,
    )
    _prime_cqi_map(scheduler)

    eligible_ues = _build_eligible_users()
    allocation = scheduler._allocate_pdsch(
        tti=4,
        ues_with_pdcch=eligible_ues,
        eligible_ues=eligible_ues,
    )

    assert allocation[1] == [2, 3, 4, 5]
    assert allocation[2] == [0, 1]

    stats = scheduler.get_stats()
    assert stats["ppo_invalid_action_count"] == 1
    assert stats["ppo_raw_actions"] == [1, 3, 0]
    assert stats["ppo_selected_ue_ids"] == [2, 1, 1]
    assert stats["ppo_step_count"] == 3
    assert stats["ppo_inference_device"] == "cpu"

    assert len(runner.calls) == 3
    assert runner.calls[0]["mask"].tolist() == [True, True, False, False]


def test_simulation_manager_accepts_ppo_scheduler_config():
    manager = SimulationManager()
    fake_runner = FakeRunner(actions=[0], max_n_ue=4)

    manager.set_scheduler(
        algorithm="PpoScheduler",
        algorithm_kwargs={
            "ppo_policy_runner": fake_runner,
            "ppo_max_n_ue": 4,
            "ppo_wb_cqi_report_period_tti": 5,
            "ppo_episode_len_tti": 100,
            "ppo_strict_observation": True,
            "ppo_deterministic": True,
            "ppo_inference_device": "cpu",
        },
    )

    assert manager.sched_config.algorithm == "PpoScheduler"
    assert manager.sched_config.algorithm_kwargs["ppo_policy_runner"] is fake_runner
    assert manager.sched_config.algorithm_kwargs["ppo_max_n_ue"] == 4
    assert manager.sched_config.algorithm_kwargs["ppo_wb_cqi_report_period_tti"] == 5
    assert manager.sched_config.algorithm_kwargs["ppo_episode_len_tti"] == 100
    assert manager.sched_config.algorithm_kwargs["ppo_strict_observation"] is True
    assert manager.sched_config.algorithm_kwargs["ppo_deterministic"] is True
    assert manager.sched_config.algorithm_kwargs["ppo_inference_device"] == "cpu"
