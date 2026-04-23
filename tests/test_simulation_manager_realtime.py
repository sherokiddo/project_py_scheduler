import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PyScheduler"))

from SIMULATION_MANAGER import SimulationManager  # noqa: E402


class _FakeUE:
    def __init__(self):
        self.UE_ID = 1
        self.position = (10.0, 20.0)
        self.SINR = 7.5
        self.cqi = 9
        self.current_dl_throughput = 1234.5
        self.average_throughput = 678.9
        self.last_transmitted_bits = 2048
        self.cqi_subband = [8, 9, 10]


class _FakeUECollection:
    def __init__(self):
        self._ues = [_FakeUE()]

    def GET_ALL_USERS(self):
        return self._ues


class _FakeBuffer:
    def __init__(self, sizes):
        self.sizes = sizes


class _FakeBaseStation:
    def __init__(self):
        self.ue_buffers = {1: _FakeBuffer({1: 4096})}


class _FakeWithStats:
    def __init__(self, payload):
        self._payload = payload

    def get_stats(self):
        return self._payload


class _FakeScheduler:
    def __init__(self):
        self.amc = _FakeWithStats({"amc": "ok"})
        self.pdcch_manager = _FakeWithStats({"pdcch": "ok"})

    def get_stats(self):
        return {"scheduler": "ok"}


class SimulationManagerRealtimeTests(unittest.TestCase):
    def setUp(self):
        self.manager = SimulationManager()
        self.manager.ue_collection = _FakeUECollection()
        self.manager.base_station = _FakeBaseStation()
        self.scheduler = _FakeScheduler()
        self.sched_result = {
            "allocation": {1: [0, 1, 2]},
            "bitmap": {1: [1, 0, 1]},
        }

    def test_set_realtime_payload_mode_rejects_invalid_value(self):
        with self.assertRaises(ValueError):
            self.manager.set_realtime_payload_mode("debug")

    def test_build_snapshot_light_mode(self):
        self.manager.set_realtime_payload_mode("light")
        snapshot = self.manager._build_tti_snapshot(tti=5, scheduler=self.scheduler, sched_result=self.sched_result)

        self.assertEqual(snapshot["type"], "tti_snapshot")
        self.assertEqual(snapshot["payload_mode"], "light")
        self.assertEqual(snapshot["tti"], 5)
        self.assertIn("ues", snapshot)
        self.assertNotIn("allocation", snapshot)
        self.assertNotIn("bitmap", snapshot)

        ue = snapshot["ues"][0]
        self.assertEqual(ue["ue_id"], 1)
        self.assertEqual(ue["allocated_rb"], 3)
        self.assertEqual(ue["buffer_bytes"], 4096)
        self.assertNotIn("cqi_subband", ue)

    def test_build_snapshot_full_mode(self):
        self.manager.set_realtime_payload_mode("full")
        snapshot = self.manager._build_tti_snapshot(tti=7, scheduler=self.scheduler, sched_result=self.sched_result)

        self.assertEqual(snapshot["payload_mode"], "full")
        self.assertIn("allocation", snapshot)
        self.assertIn("bitmap", snapshot)
        self.assertEqual(snapshot["allocation"], {1: [0, 1, 2]})
        self.assertEqual(snapshot["bitmap"], {1: [1, 0, 1]})
        self.assertEqual(snapshot["ues"][0]["cqi_subband"], [8, 9, 10])


if __name__ == "__main__":
    unittest.main()

