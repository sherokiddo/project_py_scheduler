import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PyScheduler"))

from SIMULATION_MANAGER import LevelsConfig, MetricLevel, StatisticsConfig, StatsManager  # noqa: E402


class _FakeAMC:
    def get_stats(self):
        return {
            "dl_bits_per_rb_avg": 700.0,
            "dl_capacity_bits_sum_tti": 40_000,
            "dl_transmitted_bits_sum_tti": 35_000,
            "dl_throughput_sum_kbps": 39_900.0,
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


class _FakePDCCH:
    def get_stats(self):
        return {
            "pdcch_cce_total_count": 25,
            "pdcch_cce_allocated_count": 8,
        }


class _FakeGrid:
    bandwidth = 10
    rb_per_slot = 50


class _FakeScheduler:
    def __init__(self):
        self.lte_grid = _FakeGrid()
        self.amc = _FakeAMC()
        self.pdcch_manager = _FakePDCCH()
        self._last_windowed_users = [
            {"UE_ID": 1, "ue": SimpleNamespace(average_throughput=10_000_000.0)},
            {"UE_ID": 2, "ue": SimpleNamespace(average_throughput=5_000_000.0)},
        ]

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
            "sch_pdcch_blocked_count": 0,
            "sch_avg_priority_value": 1.5,
        }


class StatsManagerFairnessExportTests(unittest.TestCase):
    def test_active_window_long_fairness_exported_to_csv_and_json(self):
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

        self.assertEqual(snapshot["dl_fairness_jain_index_active_window_long"], 0.9)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "stats.csv"
            json_path = Path(tmp_dir) / "stats.json"

            stats_manager.export_csv(str(csv_path), locale="en")
            stats_manager.export_json(str(json_path))

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter=","))
            self.assertTrue(rows)
            self.assertIn("dl_fairness_jain_index_active_window_long", rows[0])
            self.assertEqual(rows[0]["dl_fairness_jain_index_active_window_long"], "0.9")

            with json_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            self.assertTrue(payload)
            self.assertEqual(payload[0]["dl_fairness_jain_index_active_window_long"], 0.9)


if __name__ == "__main__":
    unittest.main()
