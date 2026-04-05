"""
Автономная проверка фикстур conftest.py.
Запуск: python check_fixtures.py
Лог:    logs/check_fixtures.log
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
from log_utils import Logger

BANDWIDTH   = 5
RB_PER_SLOT = 25
RBG_SIZE    = 2
TOTAL_RBG   = (RB_PER_SLOT + RBG_SIZE - 1) // RBG_SIZE

CQI_BITS_TABLE = {
    0: 0, 1: 152, 5: 616, 7: 1032, 10: 1736, 12: 2344, 15: 3752
}

log = Logger("check_fixtures.log")


# ── Сборка объектов ────────────────────────────────────────

def build_lte_grid():
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
    return grid


def build_amc():
    mock_amc = MagicMock()

    def get_bits_per_rb(cqi: int) -> int:
        if cqi <= 0:
            return 0
        for k in sorted(CQI_BITS_TABLE.keys(), reverse=True):
            if cqi >= k:
                return CQI_BITS_TABLE[k]
        return 0

    mock_amc.GET_BITS_PER_RB.side_effect = get_bits_per_rb
    return mock_amc


def make_user(ue_id, wb_cqi=10, sb_cqi=None,
              avg_throughput=0.0, buffer_bytes=10_000):
    ue_mock = MagicMock()
    ue_mock.average_throughput = avg_throughput
    return {
        'UE_ID':          ue_id,
        'ue':             ue_mock,
        'bs_buffer_size': buffer_bytes,
        'priority':       0.0,
        '_wb_cqi':        wb_cqi,
        '_sb_cqi':        sb_cqi or [],
    }


# ══════════════════════════════════════════════════════════

log.section("lte_grid")
grid = build_lte_grid()

log.check("rb_per_slot == 25",
          lambda: grid.rb_per_slot == 25)
log.check("GET_RBG_SIZE() == 2",
          lambda: grid.GET_RBG_SIZE() == 2)
log.check("GET_RBG_INDICES(0) == [0, 1]",
          lambda: grid.GET_RBG_INDICES(0) == [0, 1])
log.check("GET_RBG_INDICES(12) == [24] (последний неполный RBG)",
          lambda: grid.GET_RBG_INDICES(12) == [24])
log.check(f"total_rbg == {TOTAL_RBG}",
          lambda: (RB_PER_SLOT + RBG_SIZE - 1) // RBG_SIZE == TOTAL_RBG)
log.check("ALLOCATE_RBG() == True по умолчанию",
          lambda: grid.ALLOCATE_RBG(0, 0, 1) is True)

log.info(f"GET_RBG_INDICES(0)  = {grid.GET_RBG_INDICES(0)}")
log.info(f"GET_RBG_INDICES(12) = {grid.GET_RBG_INDICES(12)}")

# ──────────────────────────────────────────────────────────
log.section("amc")
amc = build_amc()

log.check("CQI=0  → 0 bits/RB",   lambda: amc.GET_BITS_PER_RB(0)  == 0)
log.check("CQI=1  → 152 bits/RB", lambda: amc.GET_BITS_PER_RB(1)  == 152)
log.check("CQI=10 → 1736 bits/RB",lambda: amc.GET_BITS_PER_RB(10) == 1736)
log.check("CQI=15 → 3752 bits/RB",lambda: amc.GET_BITS_PER_RB(15) == 3752)
log.check("CQI=-1 → 0 bits/RB",   lambda: amc.GET_BITS_PER_RB(-1) == 0)

log.info("CQI table: " + ", ".join(
    f"CQI{k}→{v}b" for k, v in sorted(CQI_BITS_TABLE.items()) if k > 0
))

# ──────────────────────────────────────────────────────────
log.section("make_user")
u = make_user(ue_id=1, wb_cqi=12, avg_throughput=5000.0, buffer_bytes=8_000)

log.check("UE_ID == 1",                    lambda: u['UE_ID'] == 1)
log.check("_wb_cqi == 12",                 lambda: u['_wb_cqi'] == 12)
log.check("bs_buffer_size == 8000",        lambda: u['bs_buffer_size'] == 8_000)
log.check("average_throughput == 5000.0",  lambda: u['ue'].average_throughput == 5000.0)
log.check("_sb_cqi == [] по умолчанию",    lambda: u['_sb_cqi'] == [])

log.info(f"user = UE_ID={u['UE_ID']}, CQI={u['_wb_cqi']}, "
         f"buf={u['bs_buffer_size']}B, avg={u['ue'].average_throughput}")

# ──────────────────────────────────────────────────────────
log.section("patch_cqi")

scheduler = MagicMock()
users = [
    make_user(ue_id=1, wb_cqi=10, sb_cqi=[5, 7, 12]),
    make_user(ue_id=2, wb_cqi=7),
]
cqi_wb = {u['UE_ID']: u['_wb_cqi'] for u in users}
cqi_sb = {u['UE_ID']: u['_sb_cqi'] for u in users}
scheduler._get_wb_cqi = lambda ue_id: cqi_wb.get(ue_id, 0)
scheduler._get_sb_cqi = lambda ue_id: cqi_sb.get(ue_id, [])

log.check("_get_wb_cqi(1) == 10",           lambda: scheduler._get_wb_cqi(1) == 10)
log.check("_get_wb_cqi(2) == 7",            lambda: scheduler._get_wb_cqi(2) == 7)
log.check("_get_sb_cqi(1) == [5, 7, 12]",   lambda: scheduler._get_sb_cqi(1) == [5, 7, 12])
log.check("_get_sb_cqi(2) == []",            lambda: scheduler._get_sb_cqi(2) == [])
log.check("_get_wb_cqi(99) == 0 (unknown)",  lambda: scheduler._get_wb_cqi(99) == 0)

log.info(f"WB CQI map: {cqi_wb}")
log.info(f"SB CQI map: {cqi_sb}")

# ══════════════════════════════════════════════════════════
log.summary()
