import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import GLOBALS
from BS_MODULE import LayeredBufferManager
from TRAFFIC_MODEL import Packet, QCI, BearerInfo, UeBearersInfo
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

def make_bearer(bearer_id: int, qci: int) -> BearerInfo:
    return BearerInfo(
        bearer_id=bearer_id,
        model="TestModel",
        qci=qci,
        gbr=None,
        mbr=None,
        enabled=True,
    )

@pytest.fixture
def single_bearer():
    return UeBearersInfo(
        ue_id=1,
        num_bearers=1,
        active_bearers=1,
        bearers={
            1: make_bearer(1, 9),
        },
    )

@pytest.fixture
def two_bearers():
    return UeBearersInfo(
        ue_id=1,
        num_bearers=2,
        active_bearers=2,
        bearers={
            1: make_bearer(1, 9),
            2: make_bearer(2, 1),
        },
    )

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
