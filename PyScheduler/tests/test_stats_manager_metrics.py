import os
import sys
from types import SimpleNamespace


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SIMULATION_MANAGER import LevelsConfig, MetricLevel, StatisticsConfig, StatsManager


class _FakeAMC:
    def __init__(self, stats):
        self._stats = stats

    def get_stats(self):
        return dict(self._stats)


class _FakePDCCH:
    def get_stats(self):
        return {
            "pdcch_cce_total_count": 25,
            "pdcch_cce_allocated_count": 8,
        }


class _FakeGrid:
    def __init__(self):
        self.bandwidth = 10
        self.rb_per_slot = 50


class _FakeScheduler:
    def __init__(self):
        self.lte_grid = _FakeGrid()
        self.pdcch_manager = _FakePDCCH()
        self._last_users = [
            {"UE_ID": 1, "ue": SimpleNamespace(average_throughput=10_000_000.0)},
            {"UE_ID": 2, "ue": SimpleNamespace(average_throughput=5_000_000.0)},
            {"UE_ID": 3, "ue": SimpleNamespace(average_throughput=1_000_000.0)},
        ]
        self._last_windowed_users = self._last_users[:2]
        self.amc = _FakeAMC(
            {
                "dl_capacity_bits_sum_tti": 40_000,
                "dl_transmitted_bits_sum_tti": 35_000,
                "dl_throughput_sum_kbps": 39_900.0,
                "dl_bits_per_rb_avg": 700.0,
                "dl_cqi_wb_avg_idx": 10.0,
                "dl_sinr_avg": 15.0,
                "dl_ue_throughputs": {
                    1: 20_000_000.0,
                    2: 19_900_000.0,
                    3: 0.0,
                },
                "ue_cqi": {1: 12, 2: 11, 3: 8},
                "ue_sinr": {1: 15.0, 2: 14.0, 3: 7.0},
                "ue_rb_allocated": {1: 25, 2: 25, 3: 0},
            }
        )

    def get_stats(self):
        return {
            "tti": 100,
            "sch_eligible_ue_count": 3,
            "sch_active_ue_count": 2,
            "dl_rb_allocated_count": 50,
            "dl_rb_per_ue_avg": 25.0,
            "buffer_size_sum_bytes": 2048,
            "dl_prb_utilization_pct": 100.0,
            "sch_total_time_us": 12_000.0,
            "sch_priority_list": [],
            "sch_priority_calc_time_us": 100.0,
            "sch_priority_sort_time_us": 50.0,
            "sch_priority_list_size": 2,
            "sch_window_ue_count": 2,
            "sch_pdcch_blocked_count": 0,
            "sch_avg_priority_value": 1.5,
            "ue_buffer_sizes": {1: 1024, 2: 1024, 3: 0},
            "ue_transmitted_bits": {1: 20_000, 2: 15_000, 3: 0},
        }


def test_stats_manager_exposes_explicit_fairness_metrics():
    scheduler = _FakeScheduler()
    stats_manager = StatsManager(
        scheduler,
        StatisticsConfig(
            collect_interval=1,
            levels=LevelsConfig(
                scheduler=MetricLevel.ADVANCED,
                amc=MetricLevel.ADVANCED,
                pdcch=MetricLevel.BASIC,
            ),
        ),
    )

    stats_manager.collect(100)

    snapshot = stats_manager.history[-1]
    assert snapshot["dl_throughput_sum_kbps"] == 39_900.0
    assert snapshot["dl_throughput_sum_kbps_instant"] == 39_900.0
    assert snapshot["sch_window_ue_count"] == 2
    assert snapshot["dl_fairness_jain_index_active_tti"] == 1.0
    assert snapshot["dl_fairness_jain_index_active_window_long"] == 0.9
    assert snapshot["dl_fairness_jain_index_all_ue_long"] == 0.6772
