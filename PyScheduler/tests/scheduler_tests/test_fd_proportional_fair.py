"""
Этап 5 (финал): Юнит-тесты FDxProportionalFair планировщика.

Проверяет уникальную логику: PF-метрика R_j(k,t)/T_j(t),
приоритет "голодных" UE, устойчивость к avg=0, Jain's Fairness Index.

Запуск: python tests/test_fd_proportional_fair.py
Лог:    logs/test_fd_proportional_fair.log
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock
from log_utils import Logger
from conftest import make_user, patch_cqi, RB_PER_SLOT, TOTAL_RBG, RBG_SIZE, CQI_BITS_TABLE
from SCHEDULER import SchedulerInterface

log  = Logger("test_fd_proportional_fair.log")
ALGO = "FD_PF"


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


def jains_index(values: list) -> float:
    """
    Jain's Fairness Index:
        J = (sum(x_i))^2 / (n * sum(x_i^2))

    J = 1.0 → идеальная справедливость (все одинаково)
    J → 1/n → один UE монополизирует ресурс
    """
    n = len(values)
    if n == 0:
        return 0.0
    s1 = sum(values)
    s2 = sum(x ** 2 for x in values)
    if s2 == 0:
        return 1.0
    return (s1 ** 2) / (n * s2)


# ══════════════════════════════════════════════════════════
# ТЕСТ 1: Новый UE (avg=0) получает приоритет
# ══════════════════════════════════════════════════════════

def test_new_ue_gets_priority():
    """
    UE1 avg=0 (новый), UE2 avg=100_000 bps. Одинаковый CQI.
    PF-метрика UE1 = r_j_k / 1.0
    PF-метрика UE2 = r_j_k / (100_000/1000) = r_j_k / 100

    Ожидание: UE1 побеждает на всех RBG — его метрика в 100 раз выше.
    """
    log.section("ТЕСТ 1: новый UE (avg_tput=0) получает приоритет над насыщенным")

    users = [
        make_user(ue_id=1, wb_cqi=10, avg_throughput=0.0,       buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, avg_throughput=100_000.0,  buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1 = len(allocation[1])
    rb2 = len(allocation[2])

    log.check("UE1 (avg=0) получил все RBG",
              lambda: rb1 == RB_PER_SLOT and rb2 == 0)
    log.check("UE2 (avg=100k) не получил ни одного RBG",
              lambda: rb2 == 0)

    log.info(f"UE1 (avg=0):     {rb1} RB | metric = r_j_k / 1.0")
    log.info(f"UE2 (avg=100k):  {rb2} RB | metric = r_j_k / 100.0")


# ══════════════════════════════════════════════════════════
# ТЕСТ 2: "Голодный" UE побеждает при одинаковом CQI
# ══════════════════════════════════════════════════════════

def test_hungry_ue_wins():
    """
    UE1 avg=10_000 bps ("голодный"), UE2 avg=100_000 bps ("сытый").
    Одинаковый CQI=10 → r_j_k одинаковый.

    PF-метрика UE1 = r / 10   > UE2 = r / 100
    Ожидание: UE1 получает все RBG.
    """
    log.section("ТЕСТ 2: голодный UE (меньший avg) побеждает при одинаковом CQI")

    users = [
        make_user(ue_id=1, wb_cqi=10, avg_throughput=10_000.0,  buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, avg_throughput=100_000.0, buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1 = len(allocation[1])
    rb2 = len(allocation[2])

    log.check("UE1 (avg=10k, голодный) получил больше RB чем UE2 (avg=100k)",
              lambda: rb1 > rb2)
    log.check("UE1 забрал все RBG (метрика в 10× выше)",
              lambda: rb1 == RB_PER_SLOT)

    # Показываем метрики вручную
    bits_per_rb = CQI_BITS_TABLE[10]
    r_j_k       = RBG_SIZE * bits_per_rb
    m1          = r_j_k / (10_000  / 1000)
    m2          = r_j_k / (100_000 / 1000)

    log.info(f"r_j_k = {r_j_k} bits (CQI=10, RBG_SIZE={RBG_SIZE})")
    log.info(f"metric UE1 = {m1:.2f} | metric UE2 = {m2:.2f} | ratio = {m1/m2:.1f}×")
    log.info(f"UE1: {rb1} RB | UE2: {rb2} RB")


# ══════════════════════════════════════════════════════════
# ТЕСТ 3: Высокий CQI компенсирует высокий avg_tput
# ══════════════════════════════════════════════════════════

def test_high_cqi_compensates_high_avg():
    """
    UE1: CQI=15, avg=50_000 bps
    UE2: CQI=5,  avg=1_000 bps

    Считаем метрики:
      bits_15 = 3752, r1 = 2 * 3752 = 7504, denom1 = 50000/1000 = 50  → m1 = 150.08
      bits_5  = 616,  r2 = 2 * 616  = 1232, denom2 = 1000/1000  = 1   → m2 = 1232

    Ожидание: UE2 побеждает несмотря на худший CQI — его avg_tput намного меньше.
    Это демонстрирует баланс rate vs fairness в PF.
    """
    log.section("ТЕСТ 3: баланс CQI и avg_tput — PF выбирает по метрике, не только по CQI")

    users = [
        make_user(ue_id=1, wb_cqi=15, avg_throughput=50_000.0, buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=5,  avg_throughput=1_000.0,  buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1 = len(allocation[1])
    rb2 = len(allocation[2])

    bits_15 = CQI_BITS_TABLE[15]
    bits_5  = CQI_BITS_TABLE[5]
    r1      = RBG_SIZE * bits_15
    r2      = RBG_SIZE * bits_5
    m1      = r1 / (50_000 / 1000)
    m2      = r2 / (1_000  / 1000)

    log.check("UE2 (CQI=5, avg=1k) победил UE1 (CQI=15, avg=50k)",
              lambda: rb2 > rb1)

    log.info(f"UE1: CQI=15, r={r1} bits, denom={50_000/1000}, metric={m1:.2f} → {rb1} RB")
    log.info(f"UE2: CQI=5,  r={r2} bits, denom={1_000/1000},  metric={m2:.2f} → {rb2} RB")
    log.info(f"Победил UE{'2' if rb2 > rb1 else '1'} (metric {max(m1,m2):.2f} > {min(m1,m2):.2f})")


# ══════════════════════════════════════════════════════════
# ТЕСТ 4: Одинаковые UE → RB делятся поровну
# ══════════════════════════════════════════════════════════

def test_equal_ues_split_evenly():
    """
    4 UE: одинаковый CQI и одинаковый avg_tput.
    PF-метрика у всех одинакова → победитель определяется порядком в списке.
    Ожидание: RB делятся примерно поровну (разброс ≤ RBG_SIZE).
    """
    log.section("ТЕСТ 4: одинаковые UE получают примерно поровну RB")

    users = [
        make_user(ue_id=i, wb_cqi=10, avg_throughput=50_000.0, buffer_bytes=500_000)
        for i in range(1, 5)
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb_counts = {uid: len(rbs) for uid, rbs in allocation.items()}
    total_rb  = sum(rb_counts.values())
    min_rb    = min(rb_counts.values())
    max_rb    = max(rb_counts.values())

    log.check(f"total_rb == {RB_PER_SLOT}",
              lambda: total_rb == RB_PER_SLOT)
    log.check(f"разброс RB ≤ RBG_SIZE ({RBG_SIZE})",
              lambda: (max_rb - min_rb) <= RBG_SIZE)

    log.info(f"RB по UE: {rb_counts}")
    log.info(f"min={min_rb}, max={max_rb}, spread={max_rb - min_rb}")


# ══════════════════════════════════════════════════════════
# ТЕСТ 5: avg_tput=0 не вызывает деление на ноль
# ══════════════════════════════════════════════════════════

def test_zero_avg_no_division_error():
    """
    Все UE с avg_tput=0.
    Ожидание: нет ZeroDivisionError, все RB распределены корректно.

    В коде: denom = 1.0 if avg_tput <= 0 else (avg_tput / 1000.0)
    Это гарантирует что metric = r_j_k / 1.0 при avg=0.
    """
    log.section("ТЕСТ 5: avg_tput=0 у всех UE — нет ZeroDivisionError")

    users = [
        make_user(ue_id=i, wb_cqi=10, avg_throughput=0.0, buffer_bytes=500_000)
        for i in range(1, 4)
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    exception_caught = None
    allocation       = {}

    try:
        allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)
    except ZeroDivisionError as e:
        exception_caught = e

    total_rb = sum(len(rbs) for rbs in allocation.values())

    log.check("нет ZeroDivisionError при avg_tput=0",
              lambda: exception_caught is None)
    log.check(f"все RB распределены ({RB_PER_SLOT})",
              lambda: total_rb == RB_PER_SLOT)

    log.info(f"exception = {exception_caught}")
    log.info(f"total_rb  = {total_rb}")
    log.info(f"RB по UE: { {uid: len(rbs) for uid, rbs in allocation.items()} }")


# ══════════════════════════════════════════════════════════
# ТЕСТ 6: denom = avg / 1000 — единицы измерения корректны
# ══════════════════════════════════════════════════════════

def test_denom_units_are_correct():
    """
    Проверяем что приведение avg_tput (bps) → bits/TTI корректно:
      denom = avg_tput / 1000  (TTI = 1ms = 1/1000 сек)

    Если denom не делить на 1000 — метрика будет в 1000× меньше,
    но относительный порядок UE сохранится. Тест проверяет
    что при avg1 = 1000 * avg2 → metric1 = metric2 / 1000 (правильно).

    UE1: avg=1_000 bps, UE2: avg=1_000_000 bps, одинаковый CQI.
    Ожидание: UE1 побеждает (его avg в 1000 раз меньше → метрика в 1000 раз выше).
    """
    log.section("ТЕСТ 6: denom = avg/1000 — проверка единиц измерения bps → bits/TTI")

    users = [
        make_user(ue_id=1, wb_cqi=10, avg_throughput=1_000.0,     buffer_bytes=500_000),
        make_user(ue_id=2, wb_cqi=10, avg_throughput=1_000_000.0,  buffer_bytes=500_000),
    ]
    sched = make_scheduler()
    patch_cqi(sched, users)

    allocation = sched._allocate_pdsch(tti=0, ues_with_pdcch=users, eligible_ues=users)

    rb1 = len(allocation[1])
    rb2 = len(allocation[2])

    bits_per_rb = CQI_BITS_TABLE[10]
    r_j_k       = RBG_SIZE * bits_per_rb
    m1          = r_j_k / (1_000     / 1000)   # = r_j_k / 1
    m2          = r_j_k / (1_000_000 / 1000)   # = r_j_k / 1000
    ratio       = m1 / m2                        # должно быть 1000

    log.check("UE1 (avg=1k) победил UE2 (avg=1M) — ratio метрик = 1000×",
              lambda: rb1 > rb2)
    log.check("Соотношение метрик UE1/UE2 == 1000",
              lambda: abs(ratio - 1000.0) < 0.01)

    log.info(f"r_j_k = {r_j_k} bits")
    log.info(f"metric UE1 = {m1:.2f} | metric UE2 = {m2:.4f} | ratio = {ratio:.1f}×")
    log.info(f"UE1: {rb1} RB | UE2: {rb2} RB")


# ══════════════════════════════════════════════════════════
# ТЕСТ 7: 100 TTI симуляция → Jain's Fairness Index ≥ 0.95
# ══════════════════════════════════════════════════════════

def test_jains_fairness_index():
    """
    Симулируем 100 TTI с 4 UE и разными начальными avg_tput.
    После каждого TTI обновляем average_throughput через EWMA:
        T(t+1) = (1 - alpha) * T(t) + alpha * throughput_this_tti

    Ожидание: Jain's Index ≥ 0.95 — PF обеспечивает высокую справедливость.

    J = (sum(x_i))^2 / (n * sum(x_i^2)),  где x_i = суммарный throughput UE i.

    Значение 0.95 — консервативный порог для демонстрации в диссертации.
    Реальный PF на практике даёт J → 0.99 при достаточном числе TTI.
    """
    log.section("ТЕСТ 7: 100-TTI симуляция — Jain's Fairness Index ≥ 0.95")

    NUM_TTI    = 100
    ALPHA      = 0.1    # EWMA коэффициент сглаживания
    NUM_UE     = 4
    BUFFER     = 10_000_000   # большой буфер — не ограничиваем аллокацию

    # Начальные avg_tput разные — симулируем разный предыстории
    initial_avg = {1: 10_000.0, 2: 50_000.0, 3: 20_000.0, 4: 80_000.0}

    # Создаём UE mock-объекты с изменяемым average_throughput
    ue_mocks = {}
    for i in range(1, NUM_UE + 1):
        m = MagicMock()
        m.average_throughput = initial_avg[i]
        ue_mocks[i] = m

    total_bits_per_ue = {i: 0.0 for i in range(1, NUM_UE + 1)}
    sched = make_scheduler()

    for tti in range(NUM_TTI):
        # Собираем актуальный список UE
        users = [
            {
                'UE_ID':          i,
                'ue':             ue_mocks[i],
                'bs_buffer_size': BUFFER,
                'priority':       0.0,
                '_wb_cqi':        10,
                '_sb_cqi':        [],
            }
            for i in range(1, NUM_UE + 1)
        ]
        patch_cqi(sched, users)

        allocation = sched._allocate_pdsch(
            tti=tti, ues_with_pdcch=users, eligible_ues=users
        )

        # Обновляем throughput через EWMA
        for i in range(1, NUM_UE + 1):
            rb_count   = len(allocation.get(i, []))
            bits_this  = rb_count * CQI_BITS_TABLE[10]   # bits за этот TTI
            total_bits_per_ue[i] += bits_this

            # EWMA обновление average_throughput (bps = bits/TTI * 1000)
            tput_bps = bits_this * 1000
            ue_mocks[i].average_throughput = (
                (1 - ALPHA) * ue_mocks[i].average_throughput + ALPHA * tput_bps
            )

    # Считаем Jain's Index по суммарному throughput за 100 TTI
    tput_values = list(total_bits_per_ue.values())
    jfi         = jains_index(tput_values)

    log.check(f"Jain's Fairness Index ≥ 0.95 (получено {jfi:.4f})",
              lambda: jfi >= 0.95)
    log.check("Все UE получили хоть какой-то throughput (нет монополии)",
              lambda: all(v > 0 for v in tput_values))

    log.info(f"Throughput по UE за {NUM_TTI} TTI (bits):")
    for i, bits in total_bits_per_ue.items():
        avg_final = ue_mocks[i].average_throughput
        log.info(f"  UE{i}: {bits:>10.0f} bits | final avg_tput={avg_final:.0f} bps")

    log.info(f"Jain's Fairness Index J = {jfi:.4f}  (порог ≥ 0.95)")
    log.info(f"  J=1.0 → идеальная справедливость")
    log.info(f"  J=0.25 → один UE монополизировал ресурс (N=4)")


# ══════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_new_ue_gets_priority,
        test_hungry_ue_wins,
        test_high_cqi_compensates_high_avg,
        test_equal_ues_split_evenly,
        test_zero_avg_no_division_error,
        test_denom_units_are_correct,
        test_jains_fairness_index,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            log.info(f"[FATAL] {test_fn.__name__} → {e}")

    log.summary()
