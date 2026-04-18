import os
import sys

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def test_ranker_observation_adapter_builds_compact_tti_level_observation():
    adapter = RankerObservationAdapter(
        max_n_ue=4,
        wb_cqi_report_period_tti=5,
        max_buffer_bytes=1_000_000.0,
        max_avg_tput_bps=100_000_000.0,
    )

    snapshot = DRLPlaygroundSnapshot(
        simulation_config=DRLPlaygroundSimulationConfig(
            sim_duration_tti=100,
            update_interval_tti=1,
            mobility_update_interval_tti=50,
            channel_update_interval_tti=10,
            traffic_mode="poisson",
            traffic_generator="Poisson",
            buffer_mode="layered_buffer",
            scheduler_algorithm="PpoRankerScheduler",
            scheduler_max_dl_ue_tti=None,
            scheduler_window_size=8,
            scheduler_window_enabled=True,
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
                active_flag=True,
                buffer_bytes=250_000,
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

    packet = adapter.build(snapshot, ue_ids=[1, 2])

    assert packet.actual_n_ue == 2
    assert packet.max_n_ue == 4
    assert packet.ue_ids_in_order == [1, 2]
    assert packet.action_mask.tolist() == [True, False, False, False]
    assert packet.ue_feature_dim == RANKER_OBSERVATION_N_UE_FEATURES
    assert packet.context_dim == RANKER_OBSERVATION_N_CONTEXT_FEATURES
    assert packet.observation.shape == (
        4 * RANKER_OBSERVATION_N_UE_FEATURES + RANKER_OBSERVATION_N_CONTEXT_FEATURES,
    )

    obs = packet.observation
    assert np.isclose(obs[0], 10.0 / 15.0)
    assert np.isclose(obs[1], 2.0 / 4.0)
    assert np.isclose(obs[2], 100_000.0 / 1_000_000.0)
    assert np.isclose(obs[3], 20_000_000.0 / 100_000_000.0)

    assert np.isclose(obs[4], 8.0 / 15.0)
    assert np.isclose(obs[5], 1.0 / 4.0)
    assert np.isclose(obs[6], 250_000.0 / 1_000_000.0)
    assert np.isclose(obs[7], 5_000_000.0 / 100_000_000.0)

    ctx_base = 4 * RANKER_OBSERVATION_N_UE_FEATURES
    assert np.isclose(obs[ctx_base + 0], 2.0 / 4.0)
    assert np.isclose(obs[ctx_base + 1], 17.0 / 25.0)
