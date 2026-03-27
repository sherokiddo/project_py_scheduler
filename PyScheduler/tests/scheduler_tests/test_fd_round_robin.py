"""
Этап 4: Юнит-тесты FDxRoundRobin планировщика.

Проверяет уникальную логику: честная ротация UE по RBG,
сохранение offset между TTI, сброс раунда.

Запуск: python tests/test_fd_round_robin.py
Лог:    logs/test_fd_round_robin.log
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
from log_utils import Logger
from conftest import make_user, patch_cqi, RB_PER_SLOT, TOTAL_RBG, RBG_SIZE, CQI_BITS_TABLE
from SCHEDULER import SchedulerInterface

log  = Logger("test_fd_round_robin.log")
ALGO = "FD_RR"


# ══════════════════════════════════════════════════════════
# Фабрика
# ══════════════════════════════════════════════════════════

def build_lte_grid():
    grid = MagicMock()
    grid.bandwidth   = 5
    grid.rb_per_slot = RB_PER_SLOT
    grid.GET_RBG_SIZE.return_value = RBG_SIZE

    def get_rbg_indices(rbg_idx):
        start = rbg_idx * RBG_SIZE
        end   = min(start + RBG_SIZE, RB_PER_SLOT)
        return list(range(start, end))

    grid.GET_RBG_INDICES.side_effect  = get_rbg_indices
    grid.ALLOCATE_RBG.return_value    = True
    grid.GENERATE_BITMAP.return_value = {}
    return grid


def build_bs():
    bm = MagicMock()
    bm.ue_has_buffer.return_value     = True
    buf                               = MagicMock()
    buf.buffersize                    = 10_000
    bm.get_buffer_status.return_value = [buf]
    bm.get_packets.return_value       = ([], 5_000)
    station                           = MagicMock()
    station.buffermanager             = bm
    station.usesimplebuffer           = True
    return station


def make_scheduler():
    sched = SchedulerInterface.create(
        ALGO, build_lte_grid(), build_bs(),
        verbose=False, enable_window=False,
    )
    mock_amc = MagicMock()

    def get_bits_per_rb(cqi: int) -> int:
        if cqi <= 0:
            return 0
        for k in sorted(CQI_BITS_TABLE.keys(), reverse=True):
            if cqi >= k:
                return CQI_BITS_TABLE[k]
        return 0

    sched.amc = mock_amc
    sched.amc.GET_BITS_PER_RB.side_effect = get_bits_per_rb
    return sched


# ══════════════════════════════════════════════════════════
# ТЕСТ 1: Честное распределение RBG между UE
# ══════════════════════════════════════════════════════════

def test_fair_distribution():
    """
    4 UE, одинаковый CQI, одинаковый буфер, TOTAL_RBG=13.
    Ожидание: каждый UE получает ~3-4 RBG (разброс не более 1 RBG).

    RR не гарантирует точное равенство из-за целочисленного деления,
    но разброс обязан быть минимальным.
    """
    log.section("ТЕСТ 1: честное распределение RBG между UE")

    users = [
        make_user(ue_id=i, wb_cqi=10, buffer_bytes=500_000)
        for i in range(1, 5)   # 4 UE
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb_counts = {uid: len(rbs) for uid, rbs in allocation.items()}
    total_rb  = sum(rb_counts.values())
    min_rb    = min(rb_counts.values())
    max_rb    = max(rb_counts.values())

    log.check(f"total_rb == {RB_PER_SLOT} (все RB распределены)",
              lambda: total_rb == RB_PER_SLOT)
    log.check("разброс RB между UE не более 1 RBG (справедливость)",
              lambda: (max_rb - min_rb) <= RBG_SIZE)
    log.check("каждый UE получил хотя бы 1 RBG",
              lambda: min_rb >= RBG_SIZE)

    log.info(f"RB по UE: {rb_counts}")
    log.info(f"min={min_rb}, max={max_rb}, total={total_rb}")


# ══════════════════════════════════════════════════════════
# ТЕСТ 2: rr_ue_offset сохраняется между TTI
# ══════════════════════════════════════════════════════════

def test_rr_offset_persists_between_ttis():
    """
    Запускаем два TTI подряд на одном планировщике.
    Ожидание: rr_ue_offset после TTI 0 ≠ начальному значению (0),
    и TTI 1 начинает ротацию с нового места.

    Проверяем что состояние планировщика меняется между TTI — 
    это гарантия что ротация реально продвигается.
    """
    log.section("ТЕСТ 2: rr_ue_offset сохраняется между TTI")

    users = [
        make_user(ue_id=i, wb_cqi=10, buffer_bytes=500_000)
        for i in range(1, 4)   # 3 UE
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    offset_before_tti0 = sched.rr_ue_offset

    sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)
    offset_after_tti0 = sched.rr_ue_offset

    alloc_tti1 = sched._allocate_pdsch(tti=1, ues_with_pdcch=users, eligible_ues=users)
    offset_after_tti1 = sched.rr_ue_offset

    log.check("rr_ue_offset изменился после TTI 0",
              lambda: offset_after_tti0 != offset_before_tti0)
    log.check("rr_ue_offset изменился после TTI 1",
              lambda: offset_after_tti1 != offset_after_tti0)
    log.check("TTI 1 распределил все RB",
              lambda: sum(len(rbs) for rbs in alloc_tti1.values()) == RB_PER_SLOT)

    log.info(f"offset: before={offset_before_tti0} → "
             f"after TTI0={offset_after_tti0} → after TTI1={offset_after_tti1}")


# ══════════════════════════════════════════════════════════
# ТЕСТ 3: Разные TTI → разные "первые" UE
# ══════════════════════════════════════════════════════════

def test_different_ttis_give_different_first_winner():
    """
    3 UE с одинаковым CQI.
    TTI 0: первый победитель = UE1 (offset=0).
    TTI 1: первый победитель должен смениться (offset сдвинулся).

    Проверяем реальную ротацию, а не только факт изменения offset.
    """
    log.section("ТЕСТ 3: разные TTI → разные победители первого RBG")

    users = [
        make_user(ue_id=1, wb_cqi=10, buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, buffer_bytes=500_000),
        make_user(ue_id=3, wb_cqi=10, buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    # Нулевой RBG → rb_indices = [0, 1]
    first_rbg_rbs = build_lte_grid().GET_RBG_INDICES(0)

    alloc0 = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)
    alloc1 = sched._allocate_pdsch(tti=1, ues_with_pdcch=users, eligible_ues=users)

    # Кто получил первый RBG в каждом TTI
    winner_tti0 = next(
        (uid for uid, rbs in alloc0.items() if first_rbg_rbs[0] in rbs), None
    )
    winner_tti1 = next(
        (uid for uid, rbs in alloc1.items() if first_rbg_rbs[0] in rbs), None
    )

    log.check("TTI 0 и TTI 1 имеют разных победителей первого RBG",
              lambda: winner_tti0 != winner_tti1)
    log.check("TTI 0: победитель первого RBG найден",
              lambda: winner_tti0 is not None)
    log.check("TTI 1: победитель первого RBG найден",
              lambda: winner_tti1 is not None)

    log.info(f"first_rbg_rbs = {first_rbg_rbs}")
    log.info(f"TTI 0: первый RBG → UE {winner_tti0}")
    log.info(f"TTI 1: первый RBG → UE {winner_tti1}")


# ══════════════════════════════════════════════════════════
# ТЕСТ 4: served_in_round — каждый UE max 1 раз за раунд
# ══════════════════════════════════════════════════════════

def test_each_ue_served_once_per_round():
    """
    2 UE, TOTAL_RBG=13 RBG.
    Каждый UE должен получить не более ceil(TOTAL_RBG/2) RBG за один TTI,
    и сумма = TOTAL_RBG (все RBG распределены).

    Если served_in_round не работает — один UE может занять все RBG.
    """
    log.section("ТЕСТ 4: каждый UE обслуживается не чаще одного раза за раунд")

    users = [
        make_user(ue_id=1, wb_cqi=10, buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1      = len(allocation[1])
    rb2      = len(allocation[2])
    total_rb = rb1 + rb2

    # При 2 UE и 13 RBG: один получит 7 RB, другой 6 RB (ceil/floor)
    max_expected = (TOTAL_RBG * RBG_SIZE + 1) // 2 + RBG_SIZE

    log.check(f"total_rb == {RB_PER_SLOT} (все RB распределены)",
              lambda: total_rb == RB_PER_SLOT)
    log.check(f"UE1 получил не более {max_expected} RB (ограничение раунда)",
              lambda: rb1 <= max_expected)
    log.check(f"UE2 получил не более {max_expected} RB (ограничение раунда)",
              lambda: rb2 <= max_expected)
    log.check("оба UE получили RB (никто не монополизировал)",
              lambda: rb1 > 0 and rb2 > 0)

    log.info(f"UE1: {rb1} RB | UE2: {rb2} RB | total: {total_rb}")
    log.info(f"max_expected per UE: {max_expected} RB")


# ══════════════════════════════════════════════════════════
# ТЕСТ 5: CQI tie-break в пользу лучшего SB CQI
# ══════════════════════════════════════════════════════════

def test_sb_cqi_tiebreak_within_rr():
    """
    В RR при выборе победителя для конкретного RBG среди "не обслуженных"
    побеждает тот у кого выше SB CQI — это CQI-aware тай-брейк.

    UE1 и UE2 оба не обслужены в раунде.
    На RBG 0: UE1 SB=5, UE2 SB=14.
    Ожидание: RBG 0 → UE2 (лучший SB CQI на этом RBG).
    """
    log.section("ТЕСТ 5: SB CQI tie-break внутри RR-раунда")

    sb_ue1    = [5]  + [10] * (TOTAL_RBG - 1)   # слабый на RBG 0
    sb_ue2    = [14] + [3]  * (TOTAL_RBG - 1)   # сильный только на RBG 0

    users = [
        make_user(ue_id=1, wb_cqi=10, sb_cqi=sb_ue1, buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, sb_cqi=sb_ue2, buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    first_rbg_rbs = build_lte_grid().GET_RBG_INDICES(0)
    ue2_has_rbg0  = all(rb in allocation[2] for rb in first_rbg_rbs)

    log.check("RBG 0 → UE2 (SB CQI=14 > UE1 SB CQI=5)",
              lambda: ue2_has_rbg0)
    log.info(f"first_rbg_rbs = {first_rbg_rbs}")
    log.info(f"UE1 SB[0]={sb_ue1[0]}, UE2 SB[0]={sb_ue2[0]}")
    log.info(f"UE1: {len(allocation[1])} RB | UE2: {len(allocation[2])} RB")
    log.info(f"UE2 RBs: {sorted(allocation[2])}")


# ══════════════════════════════════════════════════════════
# ТЕСТ 6: UE с пустым буфером пропускается в ротации
# ══════════════════════════════════════════════════════════

def test_empty_buffer_ue_skipped_in_rotation():
    """
    3 UE: UE2 имеет buffer=0.
    Ожидание: UE2 не получает ни одного RB,
    а RB делятся только между UE1 и UE3.
    Ротация продолжается корректно без UE2.
    """
    log.section("ТЕСТ 6: UE с пустым буфером пропускается в ротации")

    users = [
        make_user(ue_id=1, wb_cqi=10, buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, buffer_bytes=0),        # ← пустой буфер
        make_user(ue_id=3, wb_cqi=10, buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1      = len(allocation[1])
    rb2      = len(allocation[2])
    rb3      = len(allocation[3])
    total_rb = rb1 + rb2 + rb3

    log.check("UE2 (buffer=0) не получил ни одного RB",
              lambda: rb2 == 0)
    log.check(f"total_rb == {RB_PER_SLOT} (UE1 и UE3 получили все RB)",
              lambda: total_rb == RB_PER_SLOT)
    log.check("UE1 и UE3 получили RB примерно поровну (разброс ≤ RBG_SIZE)",
              lambda: abs(rb1 - rb3) <= RBG_SIZE)

    log.info(f"UE1: {rb1} RB | UE2 (buf=0): {rb2} RB | UE3: {rb3} RB")
    log.info(f"total: {total_rb}")


# ══════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_fair_distribution,
        test_rr_offset_persists_between_ttis,
        test_different_ttis_give_different_first_winner,
        test_each_ue_served_once_per_round,
        test_sb_cqi_tiebreak_within_rr,
        test_empty_buffer_ue_skipped_in_rotation,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            log.info(f"[FATAL] {test_fn.__name__} → {e}")

    log.summary()
