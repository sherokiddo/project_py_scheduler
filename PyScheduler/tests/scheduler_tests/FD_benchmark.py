"""
FD Scheduler Benchmark — Вариант A (Mock-based).

Прогоняет три планировщика (FD_BCQI, FD_FGS, FD_PF) на одинаковых данных
за NUM_TTI итераций и сравнивает ключевые метрики.

Запуск: python FD_benchmark.py
Лог:    logs/FD_benchmark.log
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from unittest.mock import MagicMock
from scheduler_tests.log_utils import Logger
from conftest import CQI_BITS_TABLE, RB_PER_SLOT, RBG_SIZE, TOTAL_RBG
from SCHEDULER import SchedulerInterface

log = Logger("FD_benchmark.log")

# ══════════════════════════════════════════════════════════
# Конфигурация симуляции
# ══════════════════════════════════════════════════════════

NUM_TTI    = 5_0000       # количество TTI
ALPHA      = 0.1         # EWMA коэффициент
BUFFER     = 10_000_000  # байт — не ограничиваем UE

ALGORITHMS = ["FD_BCQI", "FD_FGS", "FD_PF"]

# Профили UE: (ue_id, wb_cqi, начальный avg_tput bps, sb_cqi_profile)
# sb_cqi_profile: None = нет SB CQI, list = per-RBG CQI
UE_PROFILES = [
    # id  wb   avg_init     описание
    (1,   15,  10_000.0),   # хороший канал, новый UE
    (2,   10,  50_000.0),   # средний канал, средняя история
    (3,   5,   80_000.0),   # плохой канал, большая история
    (4,   12,  20_000.0),   # хороший канал, мало истории
    (5,   7,   30_000.0),   # ниже среднего
    (6,   14,  60_000.0),   # хороший канал, насыщенный
]

NUM_UE = len(UE_PROFILES)


# ══════════════════════════════════════════════════════════
# Инфраструктура
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
    bm                                = MagicMock()
    bm.ue_has_buffer.return_value     = True
    buf                               = MagicMock()
    buf.buffersize                    = BUFFER
    bm.get_buffer_status.return_value = [buf]
    bm.get_packets.return_value       = ([], BUFFER // 2)
    station                           = MagicMock()
    station.buffermanager             = bm
    station.usesimplebuffer           = True
    return station


def build_scheduler(algo: str):
    sched = SchedulerInterface.create(
        algo, build_lte_grid(), build_bs(),
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
    n  = len(values)
    s1 = sum(values)
    s2 = sum(x ** 2 for x in values)
    if s2 == 0 or n == 0:
        return 1.0
    return (s1 ** 2) / (n * s2)


# ══════════════════════════════════════════════════════════
# Ядро симуляции
# ══════════════════════════════════════════════════════════

def run_simulation(algo: str) -> dict:
    """
    Прогоняет NUM_TTI итераций для одного алгоритма.
    Возвращает dict с метриками.
    """
    sched = build_scheduler(algo)

    # Инициализируем UE mock-объекты
    ue_mocks = {}
    for ue_id, wb_cqi, avg_init in UE_PROFILES:
        m = MagicMock()
        m.average_throughput = avg_init
        ue_mocks[ue_id] = (m, wb_cqi)

    # Метрики
    total_bits     = {p[0]: 0.0  for p in UE_PROFILES}
    rb_per_tti     = []                                  # загрузка RB per TTI
    jfi_per_tti    = []                                  # Jain's Index per TTI
    tput_history   = {p[0]: []   for p in UE_PROFILES}  # throughput per TTI

    t_start = time.perf_counter()

    for tti in range(NUM_TTI):
        users = []
        for ue_id, wb_cqi, _ in UE_PROFILES:
            ue_mock, _ = ue_mocks[ue_id]
            users.append({
                'UE_ID':          ue_id,
                'ue':             ue_mock,
                'bs_buffer_size': BUFFER,
                'priority':       0.0,
                '_wb_cqi':        wb_cqi,
                '_sb_cqi':        [],
            })

        # patch CQI
        cqi_wb = {u['UE_ID']: u['_wb_cqi'] for u in users}
        cqi_sb = {u['UE_ID']: u['_sb_cqi'] for u in users}
        sched._get_wb_cqi = lambda ue_id, m=cqi_wb: m.get(ue_id, 0)
        sched._get_sb_cqi = lambda ue_id, m=cqi_sb: m.get(ue_id, [])

        allocation = sched._allocate_pdsch(
            tti=tti, ues_with_pdcch=users, eligible_ues=users
        )

        # Обновляем состояние
        tti_bits = {}
        for ue_id, wb_cqi, _ in UE_PROFILES:
            rb_count      = len(allocation.get(ue_id, []))
            bits_per_rb   = next(
                (CQI_BITS_TABLE[k] for k in sorted(CQI_BITS_TABLE, reverse=True)
                 if wb_cqi >= k), 0
            )
            bits_this_tti = rb_count * bits_per_rb
            total_bits[ue_id]     += bits_this_tti
            tti_bits[ue_id]        = bits_this_tti
            tput_history[ue_id].append(bits_this_tti)

            # EWMA обновление average_throughput
            tput_bps = bits_this_tti * 1000
            ue_mock, _ = ue_mocks[ue_id]
            ue_mock.average_throughput = (
                (1 - ALPHA) * ue_mock.average_throughput + ALPHA * tput_bps
            )

        # Per-TTI метрики
        total_rb_this_tti = sum(len(allocation.get(uid, [])) for uid in total_bits)
        rb_per_tti.append(total_rb_this_tti)

        cumulative_bits = [total_bits[p[0]] for p in UE_PROFILES]
        jfi_per_tti.append(jains_index(cumulative_bits))

    elapsed = time.perf_counter() - t_start

    # Итоговые метрики
    final_tput       = {uid: total_bits[uid] / NUM_TTI * 1000  # bps
                        for uid in total_bits}
    avg_rb_util      = sum(rb_per_tti) / (NUM_TTI * RB_PER_SLOT) * 100
    final_jfi        = jains_index(list(total_bits.values()))
    min_tput         = min(final_tput.values())
    max_tput         = max(final_tput.values())
    weakest_ue       = min(final_tput, key=final_tput.get)
    strongest_ue     = max(final_tput, key=final_tput.get)

    return {
        "algo":          algo,
        "total_bits":    total_bits,
        "final_tput":    final_tput,
        "avg_rb_util":   avg_rb_util,
        "final_jfi":     final_jfi,
        "min_tput":      min_tput,
        "max_tput":      max_tput,
        "weakest_ue":    weakest_ue,
        "strongest_ue":  strongest_ue,
        "jfi_per_tti":   jfi_per_tti,
        "tput_history":  tput_history,
        "elapsed_ms":    elapsed * 1000,
        "ue_mocks":      ue_mocks,
    }


# ══════════════════════════════════════════════════════════
# Вывод результатов
# ══════════════════════════════════════════════════════════

def print_results(results: list):

    log.section("КОНФИГУРАЦИЯ СИМУЛЯЦИИ")
    log.info(f"TTI:          {NUM_TTI}")
    log.info(f"UE:           {NUM_UE}")
    log.info(f"RB per slot:  {RB_PER_SLOT}")
    log.info(f"RBG size:     {RBG_SIZE}")
    log.info(f"EWMA alpha:   {ALPHA}")
    log.info(f"Buffer:       {BUFFER // 1_000_000} MB (не ограничиваем)")

    log.section("ПРОФИЛИ UE")
    for ue_id, wb_cqi, avg_init in UE_PROFILES:
        log.info(f"UE{ue_id}: WB_CQI={wb_cqi:2d} | avg_init={avg_init/1000:.0f}k bps")

    # ── Per-алгоритм детали ──────────────────────────────
    for r in results:
        algo = r["algo"]
        log.section(f"РЕЗУЛЬТАТЫ: {algo}  ({r['elapsed_ms']:.1f} ms)")

        log.info(f"Jain's Fairness Index : {r['final_jfi']:.4f}")
        log.info(f"Средняя загрузка RB   : {r['avg_rb_util']:.1f}%")
        log.info(f"Min throughput        : UE{r['weakest_ue']}  = "
                 f"{r['min_tput']/1000:.1f} kbps")
        log.info(f"Max throughput        : UE{r['strongest_ue']} = "
                 f"{r['max_tput']/1000:.1f} kbps")
        spread = (r['max_tput'] / r['min_tput']) if r['min_tput'] > 0 else float('inf')
        spread_str = f"{spread:.1f}×" if spread != float('inf') else "∞ (есть UE с 0 kbps)"
        log.info(f"Spread max/min        : {spread_str}")
        log.info("─" * 45)
        log.info("  Throughput по UE:")
        for ue_id, wb_cqi, _ in UE_PROFILES:
            tput = r['final_tput'][ue_id]
            bar  = "█" * int(tput / r['max_tput'] * 20)
            log.info(f"  UE{ue_id} (CQI={wb_cqi:2d}): "
                     f"{tput/1000:7.1f} kbps  {bar}")

    # ── Сравнительная таблица ────────────────────────────
    log.section("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")

    header = f"{'Метрика':<30} " + " ".join(f"{r['algo']:>12}" for r in results)
    log.info(header)
    log.info("─" * (30 + 13 * len(results)))

    # Jain's Index
    row = f"{'Jain Fairness Index':<30} "
    row += " ".join(f"{r['final_jfi']:>12.4f}" for r in results)
    log.info(row)

    # RB utilization
    row = f"{'RB utilization (%)':<30} "
    row += " ".join(f"{r['avg_rb_util']:>11.1f}%" for r in results)
    log.info(row)

    # Total throughput
    total_tput = {r['algo']: sum(r['final_tput'].values()) for r in results}
    row = f"{'Total throughput (kbps)':<30} "
    row += " ".join(f"{total_tput[r['algo']]/1000:>11.1f} " for r in results)
    log.info(row)

    # Min UE throughput
    row = f"{'Min UE throughput (kbps)':<30} "
    row += " ".join(f"{r['min_tput']/1000:>11.1f} " for r in results)
    log.info(row)

    # Max UE throughput
    row = f"{'Max UE throughput (kbps)':<30} "
    row += " ".join(f"{r['max_tput']/1000:>11.1f} " for r in results)
    log.info(row)

    # Spread
    row = f"{'Spread max/min':<30} "
    vals = []
    for r in results:
        if r['min_tput'] > 0:
            vals.append(f"{r['max_tput'] / r['min_tput']:>10.1f}×")
        else:
            vals.append(f"{'∞':>11}")
    row += " ".join(vals)
    log.info(row)

    # Execution time
    row = f"{'Execution time (ms)':<30} "
    row += " ".join(f"{r['elapsed_ms']:>11.1f} " for r in results)
    log.info(row)

    log.info("─" * (30 + 13 * len(results)))

    # ── Выводы ──────────────────────────────────────────
    log.section("ВЫВОДЫ")

    best_jfi   = max(results, key=lambda r: r['final_jfi'])
    best_tput  = max(results, key=lambda r: sum(r['final_tput'].values()))
    best_min   = max(results, key=lambda r: r['min_tput'])

    log.info(f"Наилучшая справедливость (Jain's Index):  {best_jfi['algo']} "
             f"({best_jfi['final_jfi']:.4f})")
    log.info(f"Наибольший суммарный throughput:          {best_tput['algo']} "
             f"({sum(best_tput['final_tput'].values())/1000:.1f} kbps)")
    log.info(f"Лучший throughput для слабых UE:          {best_min['algo']} "
             f"({best_min['min_tput']/1000:.1f} kbps)")

    # Jain по TTI — смотрим сходимость
    log.info("")
    log.info("Сходимость Jain's Index по TTI:")
    checkpoints = [10, 50, 100, 250, 500, 1000, 5000, 10000, NUM_TTI]
    header_row  = f"  {'TTI':<8} " + " ".join(f"{r['algo']:>10}" for r in results)
    log.info(header_row)
    for cp in checkpoints:
        if cp <= NUM_TTI:
            idx = cp - 1
            row = f"  {cp:<8} "
            row += " ".join(f"{r['jfi_per_tti'][idx]:>10.4f}" for r in results)
            log.info(row)


# ══════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.section(f"FD SCHEDULER BENCHMARK — {NUM_TTI} TTI × {NUM_UE} UE")

    all_results = []

    for algo in ALGORITHMS:
        log.info(f"Запуск {algo}...")
        try:
            result = run_simulation(algo)
            all_results.append(result)
            log.info(f"  [OK] {algo} завершён за {result['elapsed_ms']:.1f} ms")
        except Exception as e:
            log.info(f"  [ERROR] {algo} → {e}")
            import traceback
            log.info(traceback.format_exc())

    if all_results:
        print_results(all_results)

    log.summary()
