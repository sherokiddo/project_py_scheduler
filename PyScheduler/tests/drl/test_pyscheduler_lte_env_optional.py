import math
import os
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class FakeBufferStatus:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size


class FakeBufferManager:
    def __init__(self, buffer_by_ue):
        self.buffer_by_ue = dict(buffer_by_ue)

    def ue_has_buffer(self, ue_id):
        return ue_id in self.buffer_by_ue

    def get_buffer_status(self, ue_id):
        return [FakeBufferStatus(self.buffer_by_ue.get(ue_id, 0))]


class FakeBaseStation:
    def __init__(self):
        self.buffer_manager = FakeBufferManager({1: 50_000, 2: 40_000})
        self.use_simple_buffer = True
        self.bandwidth = 10.0
        self.frequency_GHz = 3.5
        self.ch_model_type = "UMa"
        self.enable_tdl = True


class FakeUE:
    def __init__(self, ue_id, cqi):
        self.UE_ID = ue_id
        self.cqi = cqi
        self.cqi_subband = [cqi, cqi]
        self.SINR = 10.0 + ue_id
        self.average_throughput = 0.0
        self.current_dl_throughput = 0.0


class FakeUECollection:
    def __init__(self, users):
        self._users = list(users)

    def GET_ALL_USERS(self):
        return list(self._users)


class FakeGrid:
    def __init__(self):
        self.bandwidth = 10.0
        self.rb_per_slot = 4
        self.allocations = {}

    def GET_RBG_SIZE(self):
        return 2

    def GET_RBG_INDICES(self, rbg_idx):
        start = rbg_idx * 2
        return [start, start + 1]

    def ALLOCATE_RBG(self, tti, rbg_idx, ue_id):
        self.allocations[(tti, rbg_idx)] = ue_id
        return True


class FakeAMC:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def GET_BITS_PER_RB(self, cqi):
        return int(cqi) * 100

    def get_stats(self):
        users = self.scheduler._last_users or []
        total_bps = sum(float(user["ue"].current_dl_throughput) for user in users)
        return {
            "dl_throughput_sum_kbps": total_bps / 1000.0,
        }


class FakeScheduler:
    def __init__(self, lte_grid, users):
        self.lte_grid = lte_grid
        self.window_size = 4
        self.enable_window = False
        self.max_dl_ue_tti = None
        self.wb_cqi_upd_interval = 5
        self.cqi_map = {
            1: SimpleNamespace(wb_cqi=10, sb_cqi=[10, 10], last_wb_update=0),
            2: SimpleNamespace(wb_cqi=8, sb_cqi=[8, 8], last_wb_update=0),
        }
        self.amc = FakeAMC(self)
        self._users = list(users)
        self._last_users = []
        self._last_allocation = {}

    def _get_wb_cqi(self, ue_id):
        return int(self.cqi_map[ue_id].wb_cqi)

    def _get_sb_cqi(self, ue_id):
        return list(self.cqi_map[ue_id].sb_cqi)

    def _prepare_pdsch_context(self, tti, users):
        eligible_ues = list(users)
        return SimpleNamespace(
            eligible_ues=eligible_ues,
            windowed_ues=eligible_ues,
            prioritized_ues=eligible_ues,
            priority_list=eligible_ues,
            priority_list_filtered=eligible_ues,
            ues_with_pdcch=eligible_ues,
            priority_calc_time_us=10.0,
            priority_sort_time_us=5.0,
        )

    def _empty_result(self):
        self._last_users = []
        self._last_allocation = {}
        return {
            "allocation": {},
            "statistics": {},
            "bitmap": {},
            "pdcch_stats": {},
        }

    def _finalize_schedule(
        self,
        *,
        tti,
        users,
        eligible_ues,
        allocation,
        t_sch_start,
        priority_calc_time_us,
        priority_sort_time_us,
    ):
        for user in users:
            ue_id = int(user["UE_ID"])
            throughput = len(allocation.get(ue_id, [])) * 1000.0
            user["ue"].current_dl_throughput = throughput
            user["ue"].average_throughput = 0.5 * user["ue"].average_throughput + 0.5 * throughput

        self._last_users = list(users)
        self._last_allocation = {int(k): list(v) for k, v in allocation.items()}
        return {
            "allocation": self._last_allocation,
            "statistics": {},
            "bitmap": {},
            "pdcch_stats": {},
        }


class FakeManager:
    def __init__(self):
        self.sim_config = SimpleNamespace(
            sim_duration=2,
            update_interval=1,
            mobility_update_interval=10,
            channel_update_interval=5,
            use_legacy_traffic=True,
            verbose=False,
        )
        self.sched_config = SimpleNamespace(algorithm="RoundRobin")
        self.stats_config = SimpleNamespace(enabled=False)
        self.base_station = FakeBaseStation()
        self.users = [FakeUE(1, 10), FakeUE(2, 8)]
        self.ue_collection = FakeUECollection(self.users)
        self.traffic_gen = SimpleNamespace()
        self.scheduler = None

    def initialize_runtime(self):
        self.lte_grid = FakeGrid()
        self.scheduler = FakeScheduler(self.lte_grid, self.users)

    def prepare_tti_scheduler_input(self, current_time):
        return [
            {
                "UE_ID": user.UE_ID,
                "ue": user,
                "cqi": user.cqi,
                "sbb_cqi": list(user.cqi_subband),
                "bs_buffer_size": 10_000,
            }
            for user in self.users
        ]


def _build_env():
    pytest.importorskip("gymnasium")
    from drl.envs.pyscheduler_lte_env import PySchedulerLteEnv

    return PySchedulerLteEnv(
        simulation_factory=FakeManager,
        max_n_ue=4,
        reward_mode="per_tti",
        reward_window=1,
        strict_observation=True,
    )


def test_pyscheduler_lte_env_reset_and_full_episode():
    env = _build_env()

    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert info["actual_n_ue"] == 2
    assert info["max_n_ue"] == 4
    assert info["rbg_step"] == 0
    assert info["exact_env_match"] is True

    action = next(idx for idx, value in enumerate(info["action_mask"]) if value)
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert reward == 0.0
    assert info["rbg_step"] == 1

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert math.isfinite(reward)
    assert info["tti"] == 1
    assert info["rbg_step"] == 0

    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is False
    assert truncated is False

    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert truncated is False
    assert obs.shape == env.observation_space.shape

    summary = env.get_episode_summary()
    assert summary["mean_se_bps_hz"] >= 0.0
    assert summary["mean_jfi_active"] >= 0.0
    env.close()


def test_pyscheduler_lte_env_invalid_padded_action_falls_back():
    env = _build_env()

    _, info = env.reset(seed=123)
    obs, reward, terminated, truncated, next_info = env.step(3)

    assert terminated is False
    assert truncated is False
    assert reward == 0.0
    assert next_info["invalid_action"] is True
    assert next_info["padded_invalid_action"] is True
    assert next_info["resolved_action"] in (0, 1)
    assert obs.shape == env.observation_space.shape
    env.close()
