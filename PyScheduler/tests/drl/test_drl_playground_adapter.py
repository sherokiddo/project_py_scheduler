import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drl import (
    DRLPlaygroundCompatibilityReport,
    DRLPlaygroundObservationAdapter,
    DRLPlaygroundSimulationConfig,
    DRLPlaygroundSnapshot,
    DRLPlaygroundStepSnapshot,
    DRLPlaygroundUEState,
    MODE_CURRENT_STEP,
    MODE_PROXY_START_TTI,
    MODE_SNAPSHOT,
)


def _build_snapshot(*, exact_per_rbg_step_supported, current_rbg_index, allocated_rbg_fraction_progress):
    return DRLPlaygroundSnapshot(
        simulation_config=DRLPlaygroundSimulationConfig(
            sim_duration_tti=100,
            update_interval_tti=1,
            mobility_update_interval_tti=500,
            channel_update_interval_tti=1,
            traffic_mode="legacy_simple_buffer",
            traffic_generator="SimpleGenerator",
            buffer_mode="simple_buffer",
            scheduler_algorithm="RoundRobin",
            scheduler_max_dl_ue_tti=None,
            scheduler_window_size=100,
            scheduler_window_enabled=True,
            stats_enabled=True,
            bandwidth_mhz=10.0,
            frequency_ghz=3.5,
            n_rb_dl=50,
            rbg_size_rb=3,
            n_rbg=16,
            channel_model_type="UMa",
            enable_tdl=True,
        ),
        step_snapshot=DRLPlaygroundStepSnapshot(
            current_time=7,
            current_tti=7,
            current_rbg_index=current_rbg_index,
            allocated_rbg_fraction_progress=allocated_rbg_fraction_progress,
            allocated_rbg_fraction_final_tti=0.5,
            n_rb_dl=50,
            rbg_size_rb=3,
            n_rbg=16,
        ),
        ue_states=[
            DRLPlaygroundUEState(
                ue_id=101,
                reported_wb_cqi=15,
                true_wb_cqi=14,
                wb_cqi_age_tti=2,
                active_flag=True,
                buffer_bytes=500_000,
                average_throughput_bps=50e6,
                current_dl_throughput_bps=3e6,
                alloc_rbg_count_tti=4,
                alloc_rbg_frac_tti=0.25,
                sinr_db=12.0,
                reported_sb_cqi=[15] * 16,
            ),
            DRLPlaygroundUEState(
                ue_id=202,
                reported_wb_cqi=3,
                true_wb_cqi=4,
                wb_cqi_age_tti=1,
                active_flag=False,
                buffer_bytes=0,
                average_throughput_bps=10e6,
                current_dl_throughput_bps=0.0,
                alloc_rbg_count_tti=0,
                alloc_rbg_frac_tti=0.0,
                sinr_db=5.0,
                reported_sb_cqi=[3] * 16,
            ),
        ],
        action_mask=[1, 0],
        compatibility=DRLPlaygroundCompatibilityReport(
            exact_per_rbg_step_supported=exact_per_rbg_step_supported,
            reported_vs_true_wb_cqi_supported=True,
            wb_cqi_age_supported=True,
            alloc_frac_this_tti_supported=True,
            current_rbg_index_supported=exact_per_rbg_step_supported,
            scheduler_eligibility_mask_supported=True,
            notes=[],
        ),
        scheduler_result={"allocation": {101: [0, 1, 2]}},
    )


def test_adapter_builds_exact_current_step_observation():
    snapshot = _build_snapshot(
        exact_per_rbg_step_supported=True,
        current_rbg_index=3,
        allocated_rbg_fraction_progress=0.25,
    )
    adapter = DRLPlaygroundObservationAdapter(
        wb_cqi_report_period_tti=5,
    )

    adapted = adapter.build(snapshot, mode=MODE_CURRENT_STEP)

    expected = np.array(
        [
            1.0,
            0.5,
            1.0,
            0.5,
            0.5,
            0.25,
            0.2,
            0.25,
            0.0,
            0.0,
            0.1,
            0.0,
            3 / 16,
            7 / 100,
            0.25,
        ],
        dtype=np.float32,
    )

    assert adapted.observation.shape == (15,)
    assert np.allclose(adapted.observation, expected)
    assert adapted.action_mask.dtype == np.bool_
    assert adapted.action_mask.tolist() == [True, False]
    assert adapted.ue_ids_in_order == [101, 202]
    assert adapted.actual_n_ue == 2
    assert adapted.max_n_ue == 2
    assert adapted.exact_env_match is True


def test_adapter_pads_observation_like_padded_env():
    snapshot = _build_snapshot(
        exact_per_rbg_step_supported=True,
        current_rbg_index=3,
        allocated_rbg_fraction_progress=0.25,
    )
    adapter = DRLPlaygroundObservationAdapter(
        max_n_ue=4,
        wb_cqi_report_period_tti=5,
    )

    adapted = adapter.build(snapshot, mode=MODE_CURRENT_STEP)

    assert adapted.observation.shape == (27,)
    assert np.allclose(adapted.observation[:12], np.array(
        [
            1.0,
            0.5,
            1.0,
            0.5,
            0.5,
            0.25,
            0.2,
            0.25,
            0.0,
            0.0,
            0.1,
            0.0,
        ],
        dtype=np.float32,
    ))
    assert np.allclose(adapted.observation[12:24], 0.0)
    assert np.allclose(
        adapted.observation[24:],
        np.array([3 / 16, 7 / 100, 0.25], dtype=np.float32),
    )
    assert adapted.action_mask.tolist() == [True, False, False, False]


def test_adapter_proxy_start_tti_builds_zero_progress_context():
    snapshot = _build_snapshot(
        exact_per_rbg_step_supported=False,
        current_rbg_index=None,
        allocated_rbg_fraction_progress=None,
    )
    adapter = DRLPlaygroundObservationAdapter(
        wb_cqi_report_period_tti=5,
    )

    adapted = adapter.build(snapshot, mode=MODE_PROXY_START_TTI)

    assert adapted.observation.shape == (15,)
    assert adapted.observation[5] == pytest.approx(0.0)
    assert adapted.observation[11] == pytest.approx(0.0)
    assert adapted.observation[12] == pytest.approx(0.0)
    assert adapted.observation[13] == pytest.approx(7 / 100)
    assert adapted.observation[14] == pytest.approx(0.0)
    assert adapted.action_mask.tolist() == [True, False]
    assert adapted.exact_env_match is False
    assert any("proxy_start_tti" in note for note in adapted.notes)


def test_adapter_snapshot_mode_uses_final_tti_fallback_when_progress_is_missing():
    snapshot = _build_snapshot(
        exact_per_rbg_step_supported=False,
        current_rbg_index=None,
        allocated_rbg_fraction_progress=None,
    )
    adapter = DRLPlaygroundObservationAdapter(
        wb_cqi_report_period_tti=5,
    )

    adapted = adapter.build(snapshot, mode=MODE_SNAPSHOT)

    assert adapted.observation[5] == pytest.approx(0.25)
    assert adapted.observation[11] == pytest.approx(0.0)
    assert adapted.observation[12] == pytest.approx(0.0)
    assert adapted.observation[13] == pytest.approx(7 / 100)
    assert adapted.observation[14] == pytest.approx(0.5)
    assert adapted.exact_env_match is False
    assert any("final_tti" in note for note in adapted.notes)


def test_adapter_strict_mode_rejects_incomplete_current_step_snapshot():
    snapshot = _build_snapshot(
        exact_per_rbg_step_supported=False,
        current_rbg_index=None,
        allocated_rbg_fraction_progress=None,
    )
    adapter = DRLPlaygroundObservationAdapter(
        wb_cqi_report_period_tti=5,
        strict_mode=True,
    )

    with pytest.raises(ValueError, match="current_step"):
        adapter.build(snapshot, mode=MODE_CURRENT_STEP)
