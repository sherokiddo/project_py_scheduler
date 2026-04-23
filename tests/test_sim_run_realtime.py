import asyncio
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim-sched-api"))
sys.path.insert(0, str(ROOT / "PyScheduler"))

from core import sim_run  # noqa: E402


class _FakeWS:
    def __init__(self, fail=False):
        self.fail = fail
        self.messages = []

    async def send_json(self, payload):
        if self.fail:
            raise RuntimeError("send failed")
        self.messages.append(payload)

    async def close(self):
        return None


class _FakeUECollection:
    def __init__(self):
        self.users = []

    def ADD_USER(self, ue):
        self.users.append(ue)
        return True

    def SET_MOBILITY_MODEL(self, *_args, **_kwargs):
        return None

    def SET_TRAFFIC_MODEL(self, _traffic_model):
        return None

    def REG_USERS_TO_BS(self, _bs):
        return None


class _FakeSimulationManager:
    def __init__(self):
        self._progress_cb = None
        self.stats_manager = None
        self.stats_config = SimpleNamespace(enabled=False)

    def set_sim_duration(self, _value):
        return None

    def set_upd_interval(self, _value):
        return None

    def set_realtime_emit_interval(self, _value):
        return None

    def set_realtime_payload_mode(self, _value):
        return None

    def enable_verbose_log(self, to_file=False):
        return None

    def set_stats_manager(self, **_kwargs):
        return None

    def set_base_station(self, _bs):
        return None

    def set_ue_collection(self, _ue_collection):
        return None

    def set_scheduler(self, algorithm, **_kwargs):
        return None

    def set_progress_callback(self, cb):
        self._progress_cb = cb

    def start_simulation(self):
        if self._progress_cb:
            self._progress_cb({"type": "tti_snapshot", "tti": 0, "sim_time_ms": 0, "payload_mode": "light", "ues": []})


class _FakeSimulationManagerFail(_FakeSimulationManager):
    def start_simulation(self):
        raise RuntimeError("boom")


class SimRunRealtimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._simulations_backup = sim_run.simulations.copy()
        sim_run.simulations.clear()

    def tearDown(self):
        sim_run.simulations.clear()
        sim_run.simulations.update(self._simulations_backup)

    def test_append_event_respects_bounded_queue(self):
        sim_run.simulations["run-1"] = {
            "events": deque(maxlen=2),
            "latest_event": None,
        }
        sim_run.append_event("run-1", {"n": 1})
        sim_run.append_event("run-1", {"n": 2})
        sim_run.append_event("run-1", {"n": 3})

        events = list(sim_run.simulations["run-1"]["events"])
        self.assertEqual(events, [{"n": 2}, {"n": 3}])
        self.assertEqual(sim_run.simulations["run-1"]["latest_event"], {"n": 3})

    async def test_send_progress_removes_dead_clients(self):
        good = _FakeWS()
        bad = _FakeWS(fail=True)
        clients = [good, bad]

        await sim_run.send_progress(clients, {"type": "tti_snapshot", "tti": 1})

        self.assertEqual(len(clients), 1)
        self.assertIs(clients[0], good)
        self.assertEqual(good.messages[0]["type"], "tti_snapshot")

    async def test_run_simulation_success_updates_status_and_events(self):
        run_id = "run-success"
        ws = _FakeWS()
        sim_run.simulations[run_id] = {
            "status": "starting",
            "ws_clients": [ws],
            "events": deque(maxlen=256),
            "latest_event": None,
        }

        config = {
            "sim_duration": 2,
            "update_interval": 1,
            "realtime_batch_size": 1,
            "bs_coords": {"x": 0.0, "y": 0.0},
            "bs_bw_mhz": 10,
            "bs_scheduler": "RoundRobin",
            "ue_ids": [1],
            "ue_coords": {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
            "ue_move_pattern": "RandomWalk",
            "ue_traffic_pattern": "PoissonModel",
            "sim_packet_rate": 100,
        }

        loop = asyncio.get_running_loop()

        async def _run_sync(_executor, func, *args):
            return func(*args)

        with patch.object(loop, "run_in_executor", side_effect=_run_sync), \
             patch.object(sim_run, "SimulationManager", _FakeSimulationManager), \
             patch.object(sim_run, "UECollection", _FakeUECollection), \
             patch.object(sim_run, "UserEquipment", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "BaseStation", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "PoissonModel", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "OnOffModel", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "MMPPModel", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "cleanup_simulation", new=AsyncMock(return_value=None)):
            await sim_run.run_simulation(run_id, config)

        self.assertEqual(sim_run.simulations[run_id]["status"], "completed")
        self.assertEqual(sim_run.simulations[run_id]["latest_event"]["type"], "completed")
        events = list(sim_run.simulations[run_id]["events"])
        self.assertTrue(any(evt.get("type") == "tti_batch" for evt in events))

    async def test_run_simulation_failure_sends_failed_event(self):
        run_id = "run-fail"
        ws = _FakeWS()
        sim_run.simulations[run_id] = {
            "status": "starting",
            "ws_clients": [ws],
            "events": deque(maxlen=256),
            "latest_event": None,
        }

        config = {
            "sim_duration": 1,
            "update_interval": 1,
            "realtime_batch_size": 1,
            "bs_coords": {"x": 0.0, "y": 0.0},
            "bs_bw_mhz": 10,
            "bs_scheduler": "RoundRobin",
            "ue_ids": [1],
            "ue_coords": {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
            "ue_move_pattern": "RandomWalk",
            "ue_traffic_pattern": "PoissonModel",
            "sim_packet_rate": 100,
        }

        loop = asyncio.get_running_loop()

        async def _run_sync(_executor, func, *args):
            return func(*args)

        with patch.object(loop, "run_in_executor", side_effect=_run_sync), \
             patch.object(sim_run, "SimulationManager", _FakeSimulationManagerFail), \
             patch.object(sim_run, "UECollection", _FakeUECollection), \
             patch.object(sim_run, "UserEquipment", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "BaseStation", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "PoissonModel", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "OnOffModel", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "MMPPModel", lambda **kwargs: SimpleNamespace(**kwargs)), \
             patch.object(sim_run, "cleanup_simulation", new=AsyncMock(return_value=None)):
            await sim_run.run_simulation(run_id, config)

        self.assertEqual(sim_run.simulations[run_id]["status"], "failed")
        self.assertEqual(sim_run.simulations[run_id]["latest_event"]["type"], "failed")
        self.assertTrue(any(msg.get("type") == "failed" for msg in ws.messages))


if __name__ == "__main__":
    unittest.main()
