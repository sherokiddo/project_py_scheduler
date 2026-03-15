import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import GLOBALS
from BS_MODULE import LayeredBufferManager
from TRAFFIC_MODEL import Packet, QCI
from SCHEDULER import SchedulingGrant, SchedulerInterface

@pytest.fixture(autouse=True)
def reset_time():
    GLOBALS.CURRENT_TIME = 0
    yield
    GLOBALS.CURRENT_TIME = 0

@pytest.fixture
def manager():
    return LayeredBufferManager(
        global_max=10_000, 
        per_ue_max=5_000
    )

@pytest.fixture
def single_bearer():
    return {
        "bearers": {
            1: {"bearer_id": 1, "qci": 9}
        }
    }

@pytest.fixture
def two_bearers():
    return {
        "bearers": {
            1: {"bearer_id": 1, "qci": 9},
            2: {"bearer_id": 2, "qci": 1},
        }
    }

@pytest.fixture
def packet():
    def make(size=100, ue_id=1, bearer_id=1, creation_time=0, qci=QCI.DEFAULT):
        return Packet(
            size=size,
            ue_id=ue_id,
            bearer_id=bearer_id,
            creation_time=creation_time,
            qci=qci
        )

    return make

@pytest.fixture
def qci():
    return QCI

@pytest.fixture
def grant():
    def make(ue_id=1, num_bytes=100, lcid=3):
        return SchedulingGrant(
            ue_id=ue_id, 
            num_bytes=num_bytes, 
            lcid=lcid
        )
    
    return make

@pytest.fixture
def scheduler():
    sched = SchedulerInterface.__new__(SchedulerInterface)

    sched.lte_grid = MagicMock()
    sched.lte_grid.bs = MagicMock()

    return sched
