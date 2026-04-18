import os
import sys
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drl.ppo_ranker_scheduler import PpoRankerScheduler
from drl.playground_adapter import (
    DRLPlaygroundObservationAdapter,
    MODE_PROXY_START_TTI,
    PLAYGROUND_N_CONTEXT_FEATURES,
    PLAYGROUND_N_UE_FEATURES,
)
from drl.ranker_observation_adapter import (
    RANKER_OBSERVATION_N_CONTEXT_FEATURES,
    RANKER_OBSERVATION_N_UE_FEATURES,
    RankerObservationAdapter,
)
from drl.simulation_bridge import (
    DRLPlaygroundCompatibilityReport,
    DRLPlaygroundSimulationConfig,
    DRLPlaygroundSnapshot,
    DRLPlaygroundStepSnapshot,
    DRLPlaygroundUEState,
)


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

    def get_buffer_status(self, ue_id):
        return []

    def get_packets(self, grants):
        return [], 0


class FakeBaseStation:
    def __init__(self):
        self.buffer_manager = FakeBufferManager()
        self.use_simple_buffer = True
        self.frequency_GHz = 3.5
        self.ch_model_type = "UMa"
        self.enable_tdl = False


class FakeUE:
    def __init__(self, ue_id, cqi, avg_tput, sinr):
        self.UE_ID = ue_id
        self.cqi = cqi
        self.average_throughput = avg_tput
        self.current_dl_throughput = 0.0
        self.SINR = sinr

    def UPD_DL_THROUGHPUT_BPS(self, transmitted_bits, time_interval_ms):
        self.current_dl_throughput = float(transmitted_bits)


class FakeRankerRunner:
    def __init__(self, score_vector, max_n_ue):
        self.score_vector = np.asarray(score_vector, dtype=np.float32)
        self.max_n_ue = max_n_ue
        self.calls = []
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def predict(self, obs, action_mask, deterministic=None):
        self.calls.append(
            {
                "obs": obs.copy(),
                "mask": action_mask.copy(),
                "deterministic": deterministic,
            }
        )
        return self.score_vector.copy()


def _build_users():
    ue_1 = FakeUE(ue_id=1, cqi=10, avg_tput=10e6, sinr=12.0)
    ue_2 = FakeUE(ue_id=2, cqi=10, avg_tput=10e6, sinr=12.0)
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
            "cqi": 10,
            "sbb_cqi": [10, 10, 10],
            "bs_buffer_size": 100_000,
        },
    ]


def _prime_cqi_map(scheduler):
    scheduler.cqi_map = {
        1: SimpleNamespace(wb_cqi=10, last_wb_update=3, sb_cqi=[10, 10, 10]),
        2: SimpleNamespace(wb_cqi=10, last_wb_update=4, sb_cqi=[10, 10, 10]),
    }


def test_ppo_ranker_scheduler_runs_ml_in_priority_stage_only():
    grid = FakeGrid()
    runner = FakeRankerRunner(score_vector=[0.1, 0.9, -1e9, -1e9], max_n_ue=4)
    scheduler = PpoRankerScheduler(
        lte_grid=grid,
        bs=FakeBaseStation(),
        ppo_ranker_policy_runner=runner,
        ppo_ranker_max_n_ue=4,
        ppo_ranker_wb_cqi_report_period_tti=5,
        ppo_ranker_episode_len_tti=10,
        ppo_ranker_deterministic=True,
        ppo_ranker_rank_weight_beta=0.3,
        ppo_ranker_pf_epsilon_bps=1e-6,
    )
    _prime_cqi_map(scheduler)

    eligible_ues = _build_users()
    prioritized = scheduler._calculate_priorities(eligible_ues, tti=4)
    priority_list = scheduler._form_priority_list(prioritized, tti=4)
    allocation = scheduler._allocate_pdsch(
        tti=4,
        ues_with_pdcch=priority_list,
        eligible_ues=eligible_ues,
    )

    stats = scheduler.get_stats()

    assert len(runner.calls) == 1
    assert isinstance(scheduler.observation_adapter, RankerObservationAdapter)
    assert priority_list[0]["UE_ID"] == 2
    assert stats["ppo_ranker_ranked_ue_ids"][:2] == [2, 1]
    assert allocation[2] == [0, 1, 2, 3, 4, 5]
    assert allocation[1] == []


def test_playground_adapter_proxy_start_tti_fast_path_keeps_layout():
    adapter = DRLPlaygroundObservationAdapter(
        max_n_ue=4,
        episode_len_tti=100,
        wb_cqi_report_period_tti=5,
        strict_mode=False,
    )

    snapshot = DRLPlaygroundSnapshot(
        simulation_config=DRLPlaygroundSimulationConfig(
            sim_duration_tti=100,
            update_interval_tti=1,
            mobility_update_interval_tti=0,
            channel_update_interval_tti=5,
            traffic_mode="simple_buffer",
            traffic_generator="",
            buffer_mode="simple_buffer",
            scheduler_algorithm="PpoRankerScheduler",
            scheduler_max_dl_ue_tti=None,
            scheduler_window_size=4,
            scheduler_window_enabled=False,
            stats_enabled=True,
            bandwidth_mhz=10.0,
            frequency_ghz=3.5,
            n_rb_dl=50,
            rbg_size_rb=3,
            n_rbg=17,
            channel_model_type="UMi",
            enable_tdl=False,
        ),
        step_snapshot=DRLPlaygroundStepSnapshot(
            current_time=10,
            current_tti=10,
            current_rbg_index=0,
            allocated_rbg_fraction_progress=0.0,
            allocated_rbg_fraction_final_tti=0.0,
            n_rb_dl=50,
            rbg_size_rb=3,
            n_rbg=17,
        ),
        ue_states=[
            DRLPlaygroundUEState(
                ue_id=1,
                reported_wb_cqi=10,
                true_wb_cqi=10,
                wb_cqi_age_tti=2,
                active_flag=True,
                buffer_bytes=100_000,
                average_throughput_bps=20_000_000.0,
                current_dl_throughput_bps=0.0,
                alloc_rbg_count_tti=0,
                alloc_rbg_frac_tti=0.4,
                sinr_db=12.0,
                reported_sb_cqi=[10, 10, 10],
            ),
            DRLPlaygroundUEState(
                ue_id=2,
                reported_wb_cqi=8,
                true_wb_cqi=8,
                wb_cqi_age_tti=1,
                active_flag=False,
                buffer_bytes=0,
                average_throughput_bps=5_000_000.0,
                current_dl_throughput_bps=0.0,
                alloc_rbg_count_tti=0,
                alloc_rbg_frac_tti=0.2,
                sinr_db=9.0,
                reported_sb_cqi=[8, 8, 8],
            ),
        ],
        action_mask=[1, 0],
        compatibility=DRLPlaygroundCompatibilityReport(
            exact_per_rbg_step_supported=False,
            reported_vs_true_wb_cqi_supported=True,
            wb_cqi_age_supported=True,
            alloc_frac_this_tti_supported=True,
            current_rbg_index_supported=True,
            scheduler_eligibility_mask_supported=True,
            notes=[],
        ),
        scheduler_result=None,
    )

    packet = adapter.build(
        snapshot,
        mode=MODE_PROXY_START_TTI,
        ue_ids=[1, 2],
    )

    assert packet.actual_n_ue == 2
    assert packet.max_n_ue == 4
    assert packet.ue_ids_in_order == [1, 2]
    assert packet.action_mask.tolist() == [True, False, False, False]
    assert packet.observation.shape == (
        4 * PLAYGROUND_N_UE_FEATURES + PLAYGROUND_N_CONTEXT_FEATURES,
    )

    obs = packet.observation
    assert np.isclose(obs[0], 10.0 / 15.0)
    assert np.isclose(obs[1], 2.0 / 4.0)
    assert np.isclose(obs[2], 1.0)
    assert np.isclose(obs[3], 100_000.0 / 1_000_000.0)
    assert np.isclose(obs[4], 20_000_000.0 / 100_000_000.0)
    assert np.isclose(obs[5], 0.0)

    ctx_base = 4 * PLAYGROUND_N_UE_FEATURES
    assert np.isclose(obs[ctx_base + 0], 0.0)
    assert np.isclose(obs[ctx_base + 1], 10.0 / 100.0)
    assert np.isclose(obs[ctx_base + 2], 0.0)


def test_ppo_ranker_scheduler_initialize_policy_runtime_runs_warmup_predict():
    grid = FakeGrid()
    runner = FakeRankerRunner(score_vector=[0.1, 0.9, -1e9, -1e9], max_n_ue=4)
    scheduler = PpoRankerScheduler(
        lte_grid=grid,
        bs=FakeBaseStation(),
        ppo_ranker_policy_runner=runner,
        ppo_ranker_max_n_ue=4,
        ppo_ranker_wb_cqi_report_period_tti=5,
        ppo_ranker_episode_len_tti=10,
        ppo_ranker_deterministic=True,
    )

    scheduler.initialize_policy_runtime()

    assert runner.initialized is True
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["obs"].shape == (
        4 * RANKER_OBSERVATION_N_UE_FEATURES + RANKER_OBSERVATION_N_CONTEXT_FEATURES,
    )
    assert call["mask"].tolist() == [True, False, False, False]
    assert call["deterministic"] is True
