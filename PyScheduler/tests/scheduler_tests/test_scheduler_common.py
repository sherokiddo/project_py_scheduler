"""
Этап 2: Инвариантные тесты всех планировщиков.

Проверяет контракт SchedulerInterface._allocate_pdsch,
который обязан соблюдать каждый из трёх планировщиков.

Запуск: python tests/test_scheduler_common.py
Лог:    logs/test_common.log
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
from scheduler_tests.log_utils import Logger
from conftest import (
    make_user, patch_cqi,
    RB_PER_SLOT, TOTAL_RBG, CQI_BITS_TABLE
)
from SCHEDULER import SchedulerInterface

log = Logger("test_common.log")

ALGORITHMS = ["FD_BCQI", "FD_FGS", "FD_PF"]


# ══════════════════════════════════════════════════════════
# Фабрика планировщиков
# ══════════════════════════════════════════════════════════

def make_scheduler(algorithm: str, lte_grid, bs):
    sched = SchedulerInterface.create(
        algorithm, lte_grid, bs,
        verbose=False,
        enable_window=False,
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


def build_lte_grid():
    from conftest import BANDWIDTH, RB_PER_SLOT, RBG_SIZE
    grid = MagicMock()
    grid.bandwidth   = BANDWIDTH
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

    station                = MagicMock()
    station.buffermanager  = bm
    station.usesimplebuffer = True
    return station


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 1: Ключи allocation == UE_IDs всех eligible_ues
# ══════════════════════════════════════════════════════════

def test_keys_equal_eligible_ues():
    log.section("ИНВАРИАНТ 1: ключи allocation == eligible_ues UE_IDs")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        users    = [
            make_user(ue_id=1, wb_cqi=10, buffer_bytes=10_000),
            make_user(ue_id=2, wb_cqi=8,  buffer_bytes=10_000),
            make_user(ue_id=3, wb_cqi=5,  buffer_bytes=10_000),
        ]
        sched = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, users)

        allocation    = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)
        expected_keys = {u['UE_ID'] for u in users}
        actual_keys   = set(allocation.keys())

        log.check(
            f"[{algo}] ключи {sorted(expected_keys)} == {sorted(actual_keys)}",
            lambda e=expected_keys, a=actual_keys: e == a
        )
        log.info(f"[{algo}] allocation RB counts: "
                 f"{ {uid: len(rbs) for uid, rbs in allocation.items()} }")


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 2: Значения — List[int]
# ══════════════════════════════════════════════════════════

def test_values_are_lists_of_int():
    log.section("ИНВАРИАНТ 2: значения allocation = List[int]")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        users    = [make_user(ue_id=1, wb_cqi=10, buffer_bytes=5_000)]
        sched    = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, users)

        allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

        type_ok = all(
            isinstance(rbs, list) and all(isinstance(rb, int) for rb in rbs)
            for rbs in allocation.values()
        )
        log.check(f"[{algo}] все значения List[int]", lambda ok=type_ok: ok)
        log.info(f"[{algo}] UE 1 получил {len(allocation[1])} RB")


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 3: Пустой ues_with_pdcch → пустые списки
# ══════════════════════════════════════════════════════════

def test_empty_ues_with_pdcch():
    log.section("ИНВАРИАНТ 3: ues_with_pdcch=[] → все списки пустые")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        eligible = [make_user(ue_id=1, wb_cqi=10)]
        sched    = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, eligible)

        allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=[], eligible_ues=eligible)

        all_empty = all(rbs == [] for rbs in allocation.values())
        log.check(f"[{algo}] все списки RB пустые при ues_with_pdcch=[]",
                  lambda ok=all_empty: ok)
        log.info(f"[{algo}] allocation = {allocation}")


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 4: RB индексы в пределах [0, rb_per_slot)
# ══════════════════════════════════════════════════════════

def test_rb_indices_in_range():
    log.section(f"ИНВАРИАНТ 4: все RB индексы в [0, {RB_PER_SLOT})")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        users    = [
            make_user(ue_id=1, wb_cqi=15, buffer_bytes=100_000),
            make_user(ue_id=2, wb_cqi=12, buffer_bytes=100_000),
        ]
        sched = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, users)

        allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

        out_of_range = [
            (uid, rb)
            for uid, rbs in allocation.items()
            for rb in rbs
            if not (0 <= rb < RB_PER_SLOT)
        ]
        total_rb = sum(len(rbs) for rbs in allocation.values())

        log.check(f"[{algo}] нет RB за пределами [0, {RB_PER_SLOT})",
                  lambda oob=out_of_range: len(oob) == 0)
        log.info(f"[{algo}] всего выделено: {total_rb} RB "
                 f"| out_of_range: {out_of_range if out_of_range else 'нет'}")


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 5: Нет дублирования RB между UE
# ══════════════════════════════════════════════════════════

def test_no_duplicate_rb():
    log.section("ИНВАРИАНТ 5: нет дублирования RB между UE")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        users    = [
            make_user(ue_id=1, wb_cqi=15, buffer_bytes=100_000),
            make_user(ue_id=2, wb_cqi=12, buffer_bytes=100_000),
            make_user(ue_id=3, wb_cqi=10, buffer_bytes=100_000),
        ]
        sched = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, users)

        allocation   = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)
        all_rbs      = [rb for rbs in allocation.values() for rb in rbs]
        duplicates   = [rb for rb in set(all_rbs) if all_rbs.count(rb) > 1]

        log.check(f"[{algo}] дублирующихся RB нет",
                  lambda d=duplicates: len(d) == 0)
        log.info(f"[{algo}] всего RB: {len(all_rbs)}, уникальных: {len(set(all_rbs))}"
                 f" | дубли: {duplicates if duplicates else 'нет'}")


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 6: UE с buffer=0 не получает RB
# ══════════════════════════════════════════════════════════

def test_zero_buffer_gets_no_rb():
    log.section("ИНВАРИАНТ 6: UE с bs_buffer_size=0 не получает RB")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        users    = [
            make_user(ue_id=1, wb_cqi=15, buffer_bytes=0),       # ← пустой буфер
            make_user(ue_id=2, wb_cqi=5,  buffer_bytes=10_000),
        ]
        sched = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, users)

        allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

        log.check(f"[{algo}] UE 1 (buffer=0) не получил RB",
                  lambda a=allocation: a[1] == [])
        log.info(f"[{algo}] UE 1: {len(allocation[1])} RB "
                 f"| UE 2: {len(allocation[2])} RB")


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 7: UE с CQI=0 не получает RB
# ══════════════════════════════════════════════════════════

def test_zero_cqi_gets_no_rb():
    log.section("ИНВАРИАНТ 7: UE с CQI=0 не получает RB")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        users    = [
            make_user(ue_id=1, wb_cqi=0,  buffer_bytes=10_000),  # ← нет CQI
            make_user(ue_id=2, wb_cqi=10, buffer_bytes=10_000),
        ]
        sched = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, users)

        allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

        log.check(f"[{algo}] UE 1 (CQI=0) не получил RB",
                  lambda a=allocation: a[1] == [])
        log.info(f"[{algo}] UE 1 (CQI=0): {len(allocation[1])} RB "
                 f"| UE 2 (CQI=10): {len(allocation[2])} RB")


# ══════════════════════════════════════════════════════════
# ИНВАРИАНТ 8: ALLOCATE_RBG=False → 0 RB в allocation
# ══════════════════════════════════════════════════════════

def test_allocate_rbg_false():
    log.section("ИНВАРИАНТ 8: ALLOCATE_RBG=False → ни один RB не выдаётся")

    for algo in ALGORITHMS:
        lte_grid = build_lte_grid()
        bs       = build_bs()
        lte_grid.ALLOCATE_RBG.return_value = False   # сетка отклоняет всё

        users = [make_user(ue_id=1, wb_cqi=10, buffer_bytes=100_000)]
        sched = make_scheduler(algo, lte_grid, bs)
        patch_cqi(sched, users)

        allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)
        total_rb   = sum(len(rbs) for rbs in allocation.values())

        log.check(f"[{algo}] total_rb == 0 при ALLOCATE_RBG=False",
                  lambda t=total_rb: t == 0)
        log.info(f"[{algo}] total_rb = {total_rb}")


# ══════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_keys_equal_eligible_ues,
        test_values_are_lists_of_int,
        test_empty_ues_with_pdcch,
        test_rb_indices_in_range,
        test_no_duplicate_rb,
        test_zero_buffer_gets_no_rb,
        test_zero_cqi_gets_no_rb,
        test_allocate_rbg_false,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            log.info(f"[FATAL] {test_fn.__name__} упал с исключением: {e}")

    log.summary()
