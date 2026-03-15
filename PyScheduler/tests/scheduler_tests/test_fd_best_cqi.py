"""
Этап 3: Юнит-тесты FDBestCQI планировщика.

Проверяет уникальную логику: победитель per-RBG = UE с максимальным CQI.

Запуск: python tests/test_fd_best_cqi.py
Лог:    logs/test_fd_best_cqi.log
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
from log_utils import Logger
from conftest import make_user, patch_cqi, RB_PER_SLOT, TOTAL_RBG, RBG_SIZE, CQI_BITS_TABLE
from SCHEDULER import SchedulerInterface

log = Logger("test_fd_best_cqi.log")
ALGO = "FD_BCQI"


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
# ТЕСТ 1: Победитель — UE с максимальным WB CQI
# ══════════════════════════════════════════════════════════

def test_best_wb_cqi_wins():
    """
    UE1 CQI=15, UE2 CQI=5, одинаковый буфер, нет SB CQI.
    Ожидание: UE1 получает ВСЕ RBG — его WB CQI выше на каждом RBG.
    """
    log.section("ТЕСТ 1: победитель — UE с максимальным WB CQI")

    lte_grid = build_lte_grid()
    sched    = SchedulerInterface.create(
        ALGO, lte_grid, build_bs(), verbose=False, enable_window=False
    )
    sched.amc = MagicMock()
    sched.amc.GET_BITS_PER_RB.side_effect = lambda cqi: (
        next((CQI_BITS_TABLE[k] for k in sorted(CQI_BITS_TABLE, reverse=True) if cqi >= k), 0)
        if cqi > 0 else 0
    )

    users = [
        make_user(ue_id=1, wb_cqi=15, buffer_bytes=500_000),  # лучший CQI
        make_user(ue_id=2, wb_cqi=5,  buffer_bytes=500_000),
    ]
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1 = len(allocation[1])
    rb2 = len(allocation[2])

    log.check("UE1 (CQI=15) получил больше RB чем UE2 (CQI=5)",
              lambda: rb1 > rb2)
    log.check("UE1 (CQI=15) получил все RBG",
              lambda: rb1 == RB_PER_SLOT and rb2 == 0)
    log.info(f"UE1 (CQI=15): {rb1} RB | UE2 (CQI=5): {rb2} RB")


# ══════════════════════════════════════════════════════════
# ТЕСТ 2: SB CQI перекрывает WB CQI на конкретном RBG
# ══════════════════════════════════════════════════════════

def test_sb_cqi_overrides_wb_on_specific_rbg():
    """
    UE1: WB CQI=12, SB CQI все по умолчанию (пусто → WB).
    UE2: WB CQI=5, SB CQI[rbg=2]=15 (резкий пик на RBG 2).

    Ожидание:
      - RBG 2 → UE2 (его SB CQI=15 > UE1 WB CQI=12)
      - остальные RBG → UE1 (его WB CQI=12 > UE2 WB CQI=5)
    """
    log.section("ТЕСТ 2: SB CQI перекрывает WB CQI на конкретном RBG")

    TARGET_RBG = 2

    # Строим SB CQI для UE2: все 0 (→ fallback WB=5), кроме rbg=2 → 15
    sb_ue2 = [0] * TOTAL_RBG
    sb_ue2[TARGET_RBG] = 15

    lte_grid = build_lte_grid()
    sched    = SchedulerInterface.create(
        ALGO, lte_grid, build_bs(), verbose=False, enable_window=False
    )
    sched.amc = MagicMock()
    sched.amc.GET_BITS_PER_RB.side_effect = lambda cqi: (
        next((CQI_BITS_TABLE[k] for k in sorted(CQI_BITS_TABLE, reverse=True) if cqi >= k), 0)
        if cqi > 0 else 0
    )

    users = [
        make_user(ue_id=1, wb_cqi=12, sb_cqi=[],     buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=5,  sb_cqi=sb_ue2, buffer_bytes=500_000),
    ]
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    # RBG 2 → rb_indices = [4, 5] при RBG_SIZE=2
    target_rb_indices = lte_grid.GET_RBG_INDICES(TARGET_RBG)
    ue2_has_target    = all(rb in allocation[2] for rb in target_rb_indices)
    ue1_has_target    = all(rb in allocation[1] for rb in target_rb_indices)

    log.check(f"RBG {TARGET_RBG} (SB CQI=15) → UE2",
              lambda: ue2_has_target)
    log.check(f"RBG {TARGET_RBG} НЕ достался UE1",
              lambda: not ue1_has_target)
    log.check("Остальные RBG → UE1 (WB CQI=12 > UE2 WB=5)",
              lambda: len(allocation[1]) > len(allocation[2]))

    log.info(f"target_rb_indices (RBG={TARGET_RBG}): {target_rb_indices}")
    log.info(f"UE1: {len(allocation[1])} RB | UE2: {len(allocation[2])} RB")
    log.info(f"UE2 RBs: {sorted(allocation[2])}")


# ══════════════════════════════════════════════════════════
# ТЕСТ 3: Все RB распределены при достаточных буферах
# ══════════════════════════════════════════════════════════

def test_all_rbs_allocated_with_sufficient_buffers():
    """
    Все UE имеют большой буфер и валидный CQI.
    Ожидание: сумма всех выделенных RB == rb_per_slot (ни один RB не пропущен).
    """
    log.section("ТЕСТ 3: все RB распределены при достаточных буферах")

    users = [
        make_user(ue_id=1, wb_cqi=12, buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, buffer_bytes=500_000),
        make_user(ue_id=3, wb_cqi=8,  buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)
    total_rb   = sum(len(rbs) for rbs in allocation.values())

    log.check(f"total_rb == rb_per_slot ({RB_PER_SLOT})",
              lambda: total_rb == RB_PER_SLOT)
    log.info(f"Распределено: {total_rb}/{RB_PER_SLOT} RB")
    log.info(f"По UE: { {uid: len(rbs) for uid, rbs in allocation.items()} }")


# ══════════════════════════════════════════════════════════
# ТЕСТ 4: Единственный UE с валидным CQI забирает все RBG
# ══════════════════════════════════════════════════════════

def test_single_valid_cqi_ue_gets_all():
    """
    UE1 CQI=0 (невалидный), UE2 CQI=10.
    Ожидание: UE2 получает все RB, UE1 не получает ничего.
    """
    log.section("ТЕСТ 4: единственный UE с валидным CQI забирает все RBG")

    users = [
        make_user(ue_id=1, wb_cqi=0,  buffer_bytes=500_000),  # невалидный CQI
        make_user(ue_id=2, wb_cqi=10, buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1 = len(allocation[1])
    rb2 = len(allocation[2])

    log.check("UE1 (CQI=0) не получил ни одного RB",  lambda: rb1 == 0)
    log.check(f"UE2 (CQI=10) получил все {RB_PER_SLOT} RB", lambda: rb2 == RB_PER_SLOT)
    log.info(f"UE1: {rb1} RB | UE2: {rb2} RB")


# ══════════════════════════════════════════════════════════
# ТЕСТ 5: Буфер ограничивает количество получаемых RB
# ══════════════════════════════════════════════════════════

def test_small_buffer_limits_rb_allocation():
    """
    UE имеет маленький буфер — меньше чем capacity всех RBG.
    Ожидание: UE получает только столько RBG, сколько нужно для исчерпания буфера,
    а не все доступные RBG.

    bits_per_rb (CQI=10) = 1736, RBG_SIZE=2 → capacity per RBG = 3472 bits = 434 bytes.
    buffer = 1000 bytes = 8000 bits → хватит на ~2 RBG (6944 bits), 3-й уже не нужен.
    """
    log.section("ТЕСТ 5: маленький буфер ограничивает количество RB")

    SMALL_BUFFER = 1_000   # bytes
    CQI          = 10
    bits_per_rb  = CQI_BITS_TABLE[CQI]                       # 1736 bits
    rbg_capacity = RBG_SIZE * bits_per_rb                    # 3472 bits per RBG
    buffer_bits  = SMALL_BUFFER * 8                          # 8000 bits
    expected_max_rbg = -(-buffer_bits // rbg_capacity)       # ceiling division = 3

    users = [
        make_user(ue_id=1, wb_cqi=CQI, buffer_bytes=SMALL_BUFFER),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1       = len(allocation[1])
    max_rb    = expected_max_rbg * RBG_SIZE

    log.check(f"UE получил не больше {max_rb} RB (buffer={SMALL_BUFFER}B)",
              lambda: rb1 <= max_rb)
    log.check(f"UE получил хотя бы 1 RB (буфер не пуст)",
              lambda: rb1 >= 1)
    log.check(f"UE получил меньше всех RB ({RB_PER_SLOT})",
              lambda: rb1 < RB_PER_SLOT)

    log.info(f"buffer={SMALL_BUFFER}B = {buffer_bits} bits")
    log.info(f"capacity per RBG = {rbg_capacity} bits ({RBG_SIZE} RB × {bits_per_rb} bits/RB)")
    log.info(f"ожидаемый max_rb = {max_rb} | фактически получено: {rb1} RB")


# ══════════════════════════════════════════════════════════
# ТЕСТ 6: SB CQI=0 → fallback на WB CQI
# ══════════════════════════════════════════════════════════

def test_sb_cqi_zero_falls_back_to_wb():
    """
    UE1: WB CQI=10, SB CQI все = 0 (нет отчёта по подполосам).
    UE2: WB CQI=5, нет SB CQI.

    Ожидание: UE1 побеждает на всех RBG через WB CQI=10,
    потому что SB CQI=0 → fallback на WB=10, а не на 0.
    Если fallback не работает — UE1 проигрывает UE2 (0 < 5), что неверно.
    """
    log.section("ТЕСТ 6: SB CQI=0 → fallback на WB CQI")

    # UE1: все SB CQI = 0 → должен использовать WB=10
    sb_ue1 = [0] * TOTAL_RBG

    users = [
        make_user(ue_id=1, wb_cqi=10, sb_cqi=sb_ue1, buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=5,  sb_cqi=[],      buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1 = len(allocation[1])
    rb2 = len(allocation[2])

    log.check("UE1 (WB=10, SB все=0) победил UE2 (WB=5) → fallback работает",
              lambda: rb1 > rb2)
    log.check(f"UE1 получил все {RB_PER_SLOT} RB",
              lambda: rb1 == RB_PER_SLOT)
    log.info(f"UE1 (WB=10, SB=0): {rb1} RB | UE2 (WB=5): {rb2} RB")
    log.info("Если UE1=0 и UE2=25 → fallback сломан (SB=0 не заменяется на WB)")


# ══════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_best_wb_cqi_wins,
        test_sb_cqi_overrides_wb_on_specific_rbg,
        test_all_rbs_allocated_with_sufficient_buffers,
        test_single_valid_cqi_ue_gets_all,
        test_small_buffer_limits_rb_allocation,
        test_sb_cqi_zero_falls_back_to_wb,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            log.info(f"[FATAL] {test_fn.__name__} → {e}")

    log.summary()
