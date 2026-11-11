import pytest
from unittest.mock import MagicMock
from typing import List, Dict

BANDWIDTH   = 5
RB_PER_SLOT = 25
RBG_SIZE    = 2
TOTAL_RBG   = (RB_PER_SLOT + RBG_SIZE - 1) // RBG_SIZE  # 13

CQI_BITS_TABLE = {
    0: 0, 1: 152, 5: 616, 7: 1032, 10: 1736, 12: 2344, 15: 3752
}


@pytest.fixture
def lte_grid():
    print("\n  [FIXTURE] lte_grid: создание mock...", end=" ")
    grid = MagicMock()
    grid.bandwidth   = BANDWIDTH
    grid.rb_per_slot = RB_PER_SLOT
    grid.GET_RBG_SIZE.return_value = RBG_SIZE

    def get_rbg_indices(rbg_idx):
        start = rbg_idx * RBG_SIZE
        end   = min(start + RBG_SIZE, RB_PER_SLOT)
        return list(range(start, end))

    grid.GET_RBG_INDICES.side_effect = get_rbg_indices
    grid.ALLOCATE_RBG.return_value   = True
    grid.GENERATE_BITMAP.return_value = {}
    print(f"[OK] rb_per_slot={RB_PER_SLOT}, rbg_size={RBG_SIZE}, total_rbg={TOTAL_RBG}")
    return grid


@pytest.fixture
def amc():
    print("\n  [FIXTURE] amc: создание mock...", end=" ")
    mock_amc = MagicMock()

    def get_bits_per_rb(cqi: int) -> int:
        if cqi <= 0:
            return 0
        for k in sorted(CQI_BITS_TABLE.keys(), reverse=True):
            if cqi >= k:
                return CQI_BITS_TABLE[k]
        return 0

    mock_amc.GET_BITS_PER_RB.side_effect = get_bits_per_rb
    print(f"[OK] CQI table: {len(CQI_BITS_TABLE)} entries")
    return mock_amc


@pytest.fixture
def buffermanager():
    print("\n  [FIXTURE] buffermanager: создание mock...", end=" ")
    bm = MagicMock()
    bm.ue_has_buffer.return_value = True

    buffer_status          = MagicMock()
    buffer_status.buffersize = 10_000
    bm.get_buffer_status.return_value = [buffer_status]
    bm.get_packets.return_value       = ([], 5_000)
    print("[OK] default buffer=10000 bytes")
    return bm


@pytest.fixture
def bs(buffermanager):
    print("\n  [FIXTURE] bs: создание mock...", end=" ")
    station = MagicMock()
    station.buffermanager   = buffermanager
    station.usesimplebuffer = True
    print("[OK]")
    return station


def make_user(
    ue_id:          int,
    wb_cqi:         int       = 10,
    sb_cqi:         List[int] = None,
    avg_throughput: float     = 0.0,
    buffer_bytes:   int       = 10_000,
    priority:       float     = 0.0,
) -> Dict:
    ue_mock = MagicMock()
    ue_mock.average_throughput  = avg_throughput
    ue_mock.currentdlthroughput = 0.0
    ue_mock.UPDDLTHROUGHPUTBPS  = MagicMock()

    return {
        'UE_ID':          ue_id,
        'ue':             ue_mock,
        'bs_buffer_size': buffer_bytes,
        'priority':       priority,
        '_wb_cqi':        wb_cqi,
        '_sb_cqi':        sb_cqi or [],
    }


@pytest.fixture
def three_users():
    print("\n  [FIXTURE] three_users: создание 3 UE...", end=" ")
    users = [
        make_user(ue_id=1, wb_cqi=15, avg_throughput=0.0,      buffer_bytes=10_000),
        make_user(ue_id=2, wb_cqi=10, avg_throughput=50_000.0,  buffer_bytes=10_000),
        make_user(ue_id=3, wb_cqi=5,  avg_throughput=10_000.0,  buffer_bytes=10_000),
    ]
    print(f"[OK] UE IDs={[u['UE_ID'] for u in users]}, "
          f"CQIs={[u['_wb_cqi'] for u in users]}")
    return users


def patch_cqi(scheduler, users: List[Dict]):
    cqi_wb = {u['UE_ID']: u['_wb_cqi'] for u in users}
    cqi_sb = {u['UE_ID']: u['_sb_cqi'] for u in users}
    scheduler._get_wb_cqi = lambda ue_id: cqi_wb.get(ue_id, 0)
    scheduler._get_sb_cqi = lambda ue_id: cqi_sb.get(ue_id, [])
