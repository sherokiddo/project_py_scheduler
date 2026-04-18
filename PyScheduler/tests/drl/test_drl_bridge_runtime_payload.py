import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SIMULATION_MANAGER import SimulationManager
from drl import DRLSimulationRuntime, PySchedulerDRLBridge


class FakeBufferStatus:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size


class FakeBufferManager:
    def upd_buffers_all(self):
        return None

    def add_packet(self, ue_id, packet):
        return None

    def ue_has_buffer(self, ue_id):
        return ue_id == 1

    def get_buffer_status(self, ue_id):
        if ue_id == 1:
            return [FakeBufferStatus(2048)]
        return []


class FakeBaseStation:
    def __init__(self):
        self.buffer_manager = FakeBufferManager()
        self.use_simple_buffer = True
        self.bandwidth = 10.0
        self.frequency_GHz = 3.5
        self.ch_model_type = "UMa"
        self.enable_tdl = True


class FakeUE:
    def __init__(self, ue_id, cqi, cqi_subband, sinr, average_throughput, current_dl_throughput):
        self.UE_ID = ue_id
        self.cqi = cqi
        self.cqi_subband = cqi_subband
        self.SINR = sinr
        self.average_throughput = average_throughput
        self.current_dl_throughput = current_dl_throughput


class FakeUECollection:
    def __init__(self, ues):
        self.ues = ues

    def UPDATE_ALL_USERS(
        self,
        current_time,
        update_interval,
        mobility_update_interval,
        channel_update_interval,
    ):
        return None

    def GET_ALL_USERS(self):
        return list(self.ues)

    def GET_USERS_FOR_SCHEDULER(self):
        return [
            {
                "UE_ID": ue.UE_ID,
                "cqi": ue.cqi,
                "sbb_cqi": ue.cqi_subband,
                "ue": ue,
            }
            for ue in self.ues
        ]


class FakeTrafficGenerator:
    def __init__(self):
        self.models = {}


class FakeGrid:
    bandwidth = 10.0
    rb_per_slot = 50

    def GET_RBG_SIZE(self):
        return 3

    def GET_RBG_INDICES(self, rbg_idx):
        start = rbg_idx * self.GET_RBG_SIZE()
        end = min(start + self.GET_RBG_SIZE(), self.rb_per_slot)
        return list(range(start, end))


class FakeCQIEntry:
    def __init__(self, wb_cqi, last_wb_update, sb_cqi):
        self.wb_cqi = wb_cqi
        self.last_wb_update = last_wb_update
        self.sb_cqi = sb_cqi
        self.last_sb_update = last_wb_update


class FakeScheduler:
    def __init__(self):
        self.cqi_map = {
            1: FakeCQIEntry(wb_cqi=11, last_wb_update=1, sb_cqi=[15] + [10] * 16),
            2: FakeCQIEntry(wb_cqi=9, last_wb_update=2, sb_cqi=[9] * 17),
        }

    def schedule(self, current_time, users):
        return {
            "allocation": {
                1: [0, 1, 2],
                2: [],
            },
            "statistics": {},
            "bitmap": {},
            "pdcch_stats": {},
        }


def test_drl_bridge_collects_runtime_payload_and_playground_snapshot():
    manager = SimulationManager()
    bridge = PySchedulerDRLBridge()
    ue_1 = FakeUE(
        ue_id=1,
        cqi=10,
        cqi_subband=[10, 9, 8],
        sinr=12.5,
        average_throughput=3210.0,
        current_dl_throughput=777.0,
    )
    ue_2 = FakeUE(
        ue_id=2,
        cqi=9,
        cqi_subband=[9, 8, 7],
        sinr=8.0,
        average_throughput=1111.0,
        current_dl_throughput=0.0,
    )

    manager.base_station = FakeBaseStation()
    manager.ue_collection = FakeUECollection([ue_1, ue_2])
    manager.traffic_gen = FakeTrafficGenerator()
    manager.scheduler = FakeScheduler()
    manager.lte_grid = FakeGrid()
    manager.sim_config.use_legacy_traffic = True
    manager.sched_config.algorithm = "RoundRobin"
    manager.set_drl_bridge(bridge)

    bridge.bind_runtime(
        DRLSimulationRuntime(
            simulation_manager=manager,
            base_station=manager.base_station,
            ue_collection=manager.ue_collection,
            lte_grid=manager.lte_grid,
            scheduler=manager.scheduler,
        )
    )

    result = manager.run_tti(current_time=4)
    payload = bridge.get_latest_runtime_payload()
    snapshot = bridge.get_latest_playground_snapshot()

    assert result["allocation"][1] == [0, 1, 2]

    assert payload.current_time == 4
    assert payload.scheduler_algorithm == "RoundRobin"
    assert payload.traffic_mode == "legacy_simple_buffer"
    assert payload.stats_enabled is True
    assert payload.bandwidth_mhz == 10.0
    assert payload.rb_per_slot == 50
    assert payload.rbg_size == 3
    assert payload.num_rbg == 17
    assert payload.scheduler_result == result
    assert len(payload.ue_states) == 2

    payload_states = {state.ue_id: state for state in payload.ue_states}
    assert payload_states[1].buffer_bytes == 2048
    assert payload_states[1].is_active is True
    assert payload_states[2].buffer_bytes == 0
    assert payload_states[2].is_active is False

    sim_cfg = snapshot.simulation_config
    assert sim_cfg.traffic_mode == "legacy_simple_buffer"
    assert sim_cfg.traffic_generator == "FakeTrafficGenerator"
    assert sim_cfg.buffer_mode == "simple_buffer"
    assert sim_cfg.scheduler_algorithm == "RoundRobin"
    assert sim_cfg.bandwidth_mhz == 10.0
    assert sim_cfg.frequency_ghz == 3.5
    assert sim_cfg.n_rb_dl == 50
    assert sim_cfg.rbg_size_rb == 3
    assert sim_cfg.n_rbg == 17
    assert sim_cfg.channel_model_type == "UMa"
    assert sim_cfg.enable_tdl is True

    step = snapshot.step_snapshot
    assert step.current_time == 4
    assert step.current_tti == 4
    assert step.current_rbg_index is None
    assert step.allocated_rbg_fraction_progress is None
    assert step.allocated_rbg_fraction_final_tti == pytest.approx(1 / 17)

    ue_states = {state.ue_id: state for state in snapshot.ue_states}
    assert ue_states[1].reported_wb_cqi == 11
    assert ue_states[1].true_wb_cqi == 10
    assert ue_states[1].wb_cqi_age_tti == 3
    assert ue_states[1].active_flag is True
    assert ue_states[1].buffer_bytes == 2048
    assert ue_states[1].average_throughput_bps == 3210.0
    assert ue_states[1].current_dl_throughput_bps == 777.0
    assert ue_states[1].alloc_rbg_count_tti == 1
    assert ue_states[1].alloc_rbg_frac_tti == pytest.approx(1 / 17)
    assert ue_states[1].sinr_db == 12.5
    assert ue_states[1].reported_sb_cqi[0] == 15

    assert ue_states[2].reported_wb_cqi == 9
    assert ue_states[2].true_wb_cqi == 9
    assert ue_states[2].wb_cqi_age_tti == 2
    assert ue_states[2].active_flag is False
    assert ue_states[2].buffer_bytes == 0
    assert ue_states[2].alloc_rbg_count_tti == 0
    assert ue_states[2].alloc_rbg_frac_tti == 0.0

    assert snapshot.action_mask == [1, 0]
    assert snapshot.compatibility.exact_per_rbg_step_supported is False
    assert snapshot.compatibility.reported_vs_true_wb_cqi_supported is True
    assert snapshot.compatibility.wb_cqi_age_supported is True
    assert snapshot.compatibility.alloc_frac_this_tti_supported is True
    assert snapshot.compatibility.current_rbg_index_supported is False
    assert snapshot.compatibility.scheduler_eligibility_mask_supported is True
    assert snapshot.scheduler_result == result
