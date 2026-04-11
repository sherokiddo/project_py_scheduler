import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import SIMULATION_MANAGER as simulation_manager_module
from BS_MODULE import BaseStation
from SIMULATION_MANAGER import SimulationManager
from UE_MODULE import UECollection


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        self.description = kwargs.get("desc", "")
        self.updated = 0

    def update(self, value):
        self.updated += value

    def set_description_str(self, value):
        self.description = value

    def refresh(self):
        return None

    def close(self):
        return None


def _build_manager(sim_duration=20, stats_enabled=True):
    manager = SimulationManager()
    manager.set_base_station(BaseStation(bandwidth=10, use_simple_buffer=True))
    manager.set_ue_collection(UECollection())
    manager.set_scheduler(
        algorithm="RoundRobin",
        pcfich=2,
        enable_window=False,
        window_size=4,
    )
    manager.set_sim_duration(sim_duration)
    manager.set_stats_manager(
        enabled=stats_enabled,
        collect_interval=5,
        scheduler_level="advanced",
        amc_level="basic",
        pdcch_level="none",
        export_detailed_format="json",
    )
    return manager


def test_initialize_runtime_creates_grid_scheduler_and_stats_manager():
    manager = _build_manager(sim_duration=20, stats_enabled=True)

    manager.initialize_runtime()

    assert manager.lte_grid is not None
    assert manager.lte_grid.bandwidth == 10
    assert manager.lte_grid.rb_per_slot == 50
    assert manager.scheduler is not None
    assert manager.scheduler.simulation_context["sim_duration_tti"] == 20
    assert manager.stats_manager is not None
    assert manager.stats_manager.scheduler is manager.scheduler
    assert manager.stats_manager.config.collect_interval == 5
    assert manager.stats_manager.config.export_detailed_format == "json"


def test_start_simulation_initializes_runtime_without_file_logging(monkeypatch):
    manager = _build_manager(sim_duration=1, stats_enabled=False)
    initialize_called = {"value": False}
    real_initialize_runtime = SimulationManager.initialize_runtime

    def tracked_initialize_runtime(self):
        initialize_called["value"] = True
        return real_initialize_runtime(self)

    monkeypatch.setattr(SimulationManager, "initialize_runtime", tracked_initialize_runtime)
    monkeypatch.setattr(simulation_manager_module, "tqdm", DummyTqdm)

    manager.start_simulation()

    assert initialize_called["value"] is True
    assert manager.lte_grid is not None
    assert manager.scheduler is not None
    assert manager.stats_manager is None
    assert manager._log_file is None
