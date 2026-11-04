"""M1: проверки view-адаптеров стратегий (SPEC-FEAT-SCH-008 §5.7).

Что проверяется
---------------
1. View-адаптеры делегируют вызовы существующим компонентам и возвращают
   ожидаемые значения.
2. Чтение через CQI-view не позволяет изменить внутренний список SB CQI.
3. Legacy-алгоритмы пока не используют новые протоколы и адаптеры (R1/R2).
4. Отдельные тесты проверяют важные побочные эффекты legacy-аллокаторов.

Совместимость адаптеров с Protocol следует проверять статической проверкой типов,
а не isinstance() и не тестами на совпадение имён параметров.
Основные регрессионные снимки алгоритмов находятся в tests/baseline/.
"""
import ast
import inspect
from typing import List, Set

import baseline_scenarios as bsc
import pytest
import SCHEDULER
from arch_support import run_tti, wb_ctx_for
from BS_MODULE import BufferStatus
from SCHEDULER import (
    AmcView,
    BuffersView,
    CqiView,
    GridView,
    LegacyAmcView,
    LegacyBuffersView,
    LegacyCqiView,
    LegacyGridView,
    LegacyStateView,
    LegacyStatsView,
    StateView,
    StatsView,
)
from TRAFFIC_MODEL import QCI

# Идентификаторы, которых заведомо нет в контексте: проверяется fallback,
# а не конкретный номер, поэтому любое значение вне профиля подходит.
ABSENT_UE_ID = 999
ABSENT_QCI = 99

# Значения-метки. Важна не величина, а то, что адаптер отдаёт поле объекта
# (не копию и не кэш) и что запись проходит сквозь него в поля планировщика.
THROUGHPUT_PROBE = 1234.5
OFFSET_PROBE = (2, 3)

# Алгоритмы, у которых есть поля ротации rr_*_offset (SPEC-FEAT-SCH-008 §4.2);
# у остальных шести полей нет, и LegacyStateView читает их как 0 (AS-IS).
RR_OFFSET_ALGORITHMS = ("RoundRobin", "FD_FGS")

# Число RBG, на которых сверяется allocate_rbg: достаточно, чтобы покрыть
# конфликт «тот же RBG — другой UE», и заведомо меньше total_rbg любой полосы.
TWIN_RBG_COUNT = 4

# Полный набор QCI (значения 1..9 из TRAFFIC_MODEL.QCI).
ALL_QCI = tuple(range(1, 10))
# Допустимый диапазон CQI; 0 означает «отчёта нет» и AMC для него бросает
# ValueError (см. test_amc_view_propagates_invalid_cqi_error).
VALID_CQI = tuple(range(1, 16))
# Нижний предел числа проверенных потоков: защищает от «тест прошёл на пустом
# профиле» (QOS_MULTIFLOW — 5 потоков на трёх UE).
MIN_FLOWS_CHECKED = 5

# Два алгоритма с метрикой Proportional Fair (:2143 WB-вариант, :2614
# FD-вариант) — единственные, кто пишет отладочное поле `instant_rate`.
PF_ALGORITHMS = ("ProportionalFair", "FD_PF")

# Все алгоритмы матрицы, кроме RoundRobin (единственного писателя
# `bs_buffer_size`) и двух QoS (у них поток несколько, оценка считается
# по-другому). Выводится из ALL_ALGORITHMS — см. docstring теста.
NON_RR_NON_QOS_ALGORITHMS = tuple(
    a for a in bsc.ALL_ALGORITHMS
    if a != "RoundRobin" and "QosAware" not in a)

# Пары «протокол -> legacy-адаптер» (SPEC §5.7, таблица интерфейсов M1).
VIEW_PAIRS = [
    (CqiView, LegacyCqiView),
    (GridView, LegacyGridView),
    (AmcView, LegacyAmcView),
    (BuffersView, LegacyBuffersView),
    (StatsView, LegacyStatsView),
    (StateView, LegacyStateView),
]

# ── 1. CqiView: делегирование legacy-геттерам ──────────────────────────────

@pytest.mark.parametrize("algorithm", bsc.ALL_ALGORITHMS)
def test_cqi_view_delegates_wb_cqi(algorithm):
    """get_wb == _get_wb_cqi для всех восьми алгоритмов и всех UE профиля."""
    ctx = wb_ctx_for(algorithm)
    sched, view = ctx["sched"], LegacyCqiView(ctx["sched"])
    for ue in ctx["ues"]:
        ue_id = ue.UE_ID
        assert view.get_wb(ue_id) == sched._get_wb_cqi(ue_id) == ue.cqi, (
            f"{algorithm}: UE {ue_id} — get_wb расходится с legacy")

def test_cqi_view_wb_fallback_for_unknown_ue(wb_ctx):
    """UE вне cqi_map -> 0 (AS-IS :1078–1082): адаптер fallback не меняет."""
    view = LegacyCqiView(wb_ctx["sched"])
    assert view.get_wb(ABSENT_UE_ID) == 0
    assert view.get_sb(ABSENT_UE_ID) == []

def test_cqi_view_delegates_sb_cqi(fd_ctx):
    """get_sb == _get_sb_cqi == ue.cqi_subband для FD-профиля (SB_FULL)."""
    sched, view = fd_ctx["sched"], LegacyCqiView(fd_ctx["sched"])
    for ue in fd_ctx["ues"]:
        assert view.get_sb(ue.UE_ID) == sched._get_sb_cqi(ue.UE_ID) == ue.cqi_subband
        assert view.get_sb(ue.UE_ID), "SB-список пуст: FD-сценарий не собрался"

def test_cqi_view_sb_empty_in_wb_mode(wb_ctx):
    """WB-профиль: get_sb возвращает пустой список (non-FD — норма, :1092)."""
    view = LegacyCqiView(wb_ctx["sched"])
    for ue in wb_ctx["ues"]:
        assert view.get_sb(ue.UE_ID) == []

def test_cqi_view_get_sb_returns_copy_not_live_list(wb_ctx):
    """View не отдаёт живой список `sb_cqi`: мутация результата не попадает в карту.

    Контракт read-only view (SPEC-FEAT-SCH-008 §2.1, антипаттерн A3): стратегия,
    отсортировавшая или перезаписавшая SB-список на месте, не должна портить
    `CQIMap` для последующих FD-аллокаций. AS-IS `_get_sb_cqi` (:1084) возвращает
    живой список — копию делает адаптер (`list(...)`), legacy-метод и его
    потребители не тронуты (R2). Замечание внешнего ревью PR #44 (Codex, P2)
    закрыто этой проверкой: до фикса тест падает на `is not`, после — зелёный.
    """
    sched = wb_ctx["sched"]
    ue_id = 1
    sched.cqi_map[ue_id].sb_cqi = [7, 8, 9]
    view = LegacyCqiView(sched)
    sb = view.get_sb(ue_id)
    assert sb == [7, 8, 9]
    assert sb is not sched.cqi_map[ue_id].sb_cqi
    sb[0] = 15
    sb.sort(reverse=True)
    assert sched.cqi_map[ue_id].sb_cqi == [7, 8, 9]
    assert view.get_sb(ue_id) == [7, 8, 9]

# ── 2. GridView: делегирование RES_GRID_LTE ────────────────────────────────

def test_grid_view_matches_legacy_grid_api(wb_ctx):
    """Основные операции view совпадают с существующим API сетки."""
    grid, view = wb_ctx["grid"], LegacyGridView(wb_ctx["grid"])
    assert view.rbg_size() == grid.GET_RBG_SIZE()
    assert view.prb_count() == grid.rb_per_slot
    for rbg_idx in range(TWIN_RBG_COUNT):
        assert view.rbg_indices(rbg_idx) == grid.GET_RBG_INDICES(rbg_idx)

def test_grid_view_allocate_rbg_delegates_and_mutates_grid():
    """allocate_rbg == ALLOCATE_RBG, включая побочный эффект на сетке.

    ALLOCATE_RBG меняет владельца RB, поэтому сверка идёт на двух одинаково
    построенных сетках (близнецах): последовательность вызовов и результат
    совпадают, итоговое состояние сеток идентично.

    Фаза 2 — отказ на пересечении с чужой аллокацией: view повторяет семантику
    отказа `ALLOCATE_RBG` включая откат, который очищает всю группу без
    проверки владельца (BUG-SCH-009). Контракт view формулирует это явно и не
    обещает owner-safe rollback: чужой RB освобождается вместе с группой —
    это AS-IS семантика сетки, зафиксированная здесь, а не исправление
    (исправление — задача владельца RES_GRID, FEAT-SCH-001).
    """
    twin_a, twin_b = bsc.new_ctx("BestCQI"), bsc.new_ctx("BestCQI")
    view, grid = LegacyGridView(twin_a["grid"]), twin_b["grid"]
    results = []
    for rbg_idx in range(TWIN_RBG_COUNT):
        results.append((view.allocate_rbg(5, rbg_idx, 1),
                        grid.ALLOCATE_RBG(5, rbg_idx, 1)))
        # повторное выделение того же RBG тому же UE — как в FD-цикле
        results.append((view.allocate_rbg(5, rbg_idx, 2),
                        grid.ALLOCATE_RBG(5, rbg_idx, 2)))
    assert all(a == b for a, b in results), f"расхождение allocate_rbg: {results}"
    assert bsc.grid_state(twin_a["grid"], 5) == bsc.grid_state(twin_b["grid"], 5)

    # Фаза 2: чужой RB внутри группы -> отказ, и откат освобождает чужой RB.
    tti = 7
    grids = (twin_a["grid"], twin_b["grid"])
    victim = grids[0].GET_RBG_INDICES(0)[0]
    for g in grids:
        for slot in (0, 1):
            assert g.ALLOCATE_RB(tti, f"sub_{tti % 10}_slot_{slot}", victim, 7)
    rejected = (view.allocate_rbg(tti, 0, 8), grid.ALLOCATE_RBG(tti, 0, 8))
    assert rejected == (False, False), f"отказ не повторён: {rejected}"
    for g in grids:
        # RB жертвы освобождён вместе с группой (откат не owner-safe):
        # оба слота снова отдаются другому UE — иначе ALLOCATE_RB вернул бы False
        assert g.ALLOCATE_RB(tti, f"sub_{tti % 10}_slot_0", victim, 9)
        assert g.ALLOCATE_RB(tti, f"sub_{tti % 10}_slot_1", victim, 9)
    assert bsc.grid_state(twin_a["grid"], tti) == bsc.grid_state(twin_b["grid"], tti)

# ── 3. AmcView: делегирование GET_BITS_PER_RB ─────────────────────────────

def test_amc_view_matches_get_bits_per_rb(wb_ctx):
    """bits_per_rb(cqi) == amc.GET_BITS_PER_RB(cqi) для CQI 1..15 (:1730)."""
    amc, view = wb_ctx["sched"].amc, LegacyAmcView(wb_ctx["sched"].amc)
    for cqi in VALID_CQI:
        assert view.bits_per_rb(cqi) == amc.GET_BITS_PER_RB(cqi) > 0
    assert view.bits_per_rb(15) > view.bits_per_rb(1), "AMC немонотонна"

def test_amc_view_propagates_invalid_cqi_error(wb_ctx):
    """CQI=0 -> ValueError (:1735) и через адаптер, и напрямую (AS-IS).

    `_get_wb_cqi` возвращает 0 для UE вне cqi_map (:1078-1082), а AMC для 0
    бросает ValueError, поэтому корректность зависит от этапа 1
    (`_filter_eligible_ues`, :559). Адаптер ошибку не гасит: маскировать
    дефект запрещено (R6), целевая обработка CQI=0 — CQI-трек C1.
    """
    amc, view = wb_ctx["sched"].amc, LegacyAmcView(wb_ctx["sched"].amc)
    assert LegacyCqiView(wb_ctx["sched"]).get_wb(ABSENT_UE_ID) == 0
    with pytest.raises(ValueError, match="Invalid CQI 0"):
        view.bits_per_rb(0)
    with pytest.raises(ValueError, match="Invalid CQI 0"):
        amc.GET_BITS_PER_RB(0)

# ── 4. BuffersView: делегирование IBufferManager ──────────────────────────

def test_buffers_view_matches_manager_simple(wb_ctx):
    """Simple-режим: has_buffer/buffer_status равны прямым вызовам менеджера."""
    bm = wb_ctx["bs"].buffer_manager
    view = LegacyBuffersView(bm)
    for ue_id in (1, 2, 3):
        assert bm.ue_has_buffer(ue_id) is True
        assert view.has_buffer(ue_id) is True
        assert view.buffer_status(ue_id) == bm.get_buffer_status(ue_id)
        assert all(isinstance(s, BufferStatus) for s in view.buffer_status(ue_id))
    assert view.has_buffer(ABSENT_UE_ID) == bm.ue_has_buffer(ABSENT_UE_ID)

def test_buffers_view_matches_manager_layered(layered_ctx):
    """Layered-режим: статусы per-QCI совпадают с менеджером (несколько bearer)."""
    bm = layered_ctx["bs"].buffer_manager
    view = LegacyBuffersView(bm)
    for ue_id, (bearers, _cqi) in bsc.QOS_MULTIFLOW.items():
        expected = bm.get_buffer_status(ue_id)
        assert view.buffer_status(ue_id) == expected
        assert len(expected) == len(bearers), (
            f"UE {ue_id}: bearer {len(bearers)}, статусов {len(expected)}")
        assert view.has_buffer(ue_id) == bm.ue_has_buffer(ue_id)

def test_buffers_view_hol_delay_matches_status_layered(layered_ctx):
    """hol_delay(ue, qci) == hol_delay первого статуса с этим QCI (:3029)."""
    bm = layered_ctx["bs"].buffer_manager
    view = LegacyBuffersView(bm)
    checked = 0
    for ue_id, (bearers, _cqi) in bsc.QOS_MULTIFLOW.items():
        statuses = bm.get_buffer_status(ue_id)
        for _bid, qci, _bytes in bearers:
            first = next((s for s in statuses if s.qci == qci), None)
            assert view.hol_delay(ue_id, qci) == (first.hol_delay if first else None)
            checked += 1
        # QCI, которого у UE нет, — None (AS-IS: метрика такой поток не видит)
        assert view.hol_delay(ue_id, ABSENT_QCI) is None
    assert checked >= MIN_FLOWS_CHECKED, f"проверено всего {checked} потоков"

# ── 5. StatsView: статистика ───────────────────────────────────────────────

def test_stats_view_ue_throughput_matches_ue_field(wb_ctx):
    """ue_throughput == user['ue'].average_throughput (:2133); нет UE -> 0.0."""
    users_by_id = {u["UE_ID"]: u["ue"] for u in wb_ctx["users"]}
    view = LegacyStatsView(users_by_id, wb_ctx["bs"].buffer_manager)
    for ue_id, ue in users_by_id.items():
        ue.average_throughput = THROUGHPUT_PROBE + ue_id
        assert view.ue_throughput(ue_id) == ue.average_throughput
    assert view.ue_throughput(ABSENT_UE_ID) == 0.0
    assert LegacyStatsView().ue_throughput(1) == 0.0

def test_stats_view_qci_throughput_reads_tracker(layered_ctx):
    """Layered-режим: пара (current, average) берётся из QciThroughputTracker."""
    bm = layered_ctx["bs"].buffer_manager
    tracker = bm.qci_tput_tracker
    view = LegacyStatsView({}, bm)
    for ue_id, (bearers, _cqi) in bsc.QOS_MULTIFLOW.items():
        for _bid, qci, _bytes in bearers:
            assert view.qci_throughput(ue_id, qci) == (
                tracker.get_current_throughput(ue_id, qci),
                tracker.get_average_throughput(ue_id, qci))

def test_stats_view_qci_throughput_zero_without_tracker(wb_ctx):
    """Simple-режим: трекера нет -> (0.0, 0.0), без AttributeError (AS-IS)."""
    bm = wb_ctx["bs"].buffer_manager
    assert getattr(bm, "qci_tput_tracker", None) is None
    view = LegacyStatsView({}, bm)
    assert view.qci_throughput(1, ALL_QCI[-1]) == (0.0, 0.0)
    assert LegacyStatsView().qci_throughput(1, ALL_QCI[-1]) == (0.0, 0.0)

@pytest.mark.parametrize("qci", ALL_QCI)
def test_stats_view_pdb_matches_qci(qci):
    """PDB берётся из профиля QCI и совпадает с текущим расчётом."""
    view = LegacyStatsView()
    assert view.pdb_of(qci) == QCI(qci).get_delay_budget()
    assert view.pdb_of(qci) > 0

# ── 6. StateView: сквозная запись в поля планировщика ─────────────────────

def test_state_view_reads_and_writes_scheduler_fields():
    """RR/FD_FGS: get/set работают сквозь адаптер с полями экземпляра."""
    for algorithm in RR_OFFSET_ALGORITHMS:
        ctx = wb_ctx_for(algorithm)
        sched, view = ctx["sched"], LegacyStateView(ctx["sched"])
        assert view.get_rr_ue_offset() == sched.rr_ue_offset == 0
        assert view.get_rr_rbg_offset() == sched.rr_rbg_offset == 0
        view.set_rr_ue_offset(OFFSET_PROBE[0])
        view.set_rr_rbg_offset(OFFSET_PROBE[1])
        assert (sched.rr_ue_offset, sched.rr_rbg_offset) == OFFSET_PROBE, (
            f"{algorithm}: адаптер не пишет в поля планировщика")
        assert (view.get_rr_ue_offset(),
                view.get_rr_rbg_offset()) == OFFSET_PROBE

def test_state_view_absent_fields_read_as_zero():
    """У остальных шести алгоритмов полей нет — чтение даёт 0 (AS-IS, R2)."""
    for algorithm in bsc.ALL_ALGORITHMS:
        if algorithm in RR_OFFSET_ALGORITHMS:
            continue
        ctx = wb_ctx_for(algorithm)
        sched, view = ctx["sched"], LegacyStateView(ctx["sched"])
        assert not hasattr(sched, "rr_ue_offset"), f"{algorithm}: поле появилось"
        assert view.get_rr_ue_offset() == 0
        assert view.get_rr_rbg_offset() == 0

# ── 7. R1: legacy-код не знает о views ────────────────────────────────────

def _module_source() -> str:
    """Исходник SCHEDULER.py из checkout (единственный источник для проверок)."""
    return inspect.getsource(SCHEDULER)

def _view_names() -> Set[str]:
    """Имена протоколов и адаптеров M1: то, чего legacy-код знать не должен."""
    return ({adapter.__name__ for _proto, adapter in VIEW_PAIRS}
            | {proto.__name__ for proto, _adapter in VIEW_PAIRS})

def _m1_nodes(tree: ast.Module) -> List[ast.AST]:
    """Узлы раздела M1: view-классы и guard импорта аннотаций.

    Опознаются по составу VIEW_PAIRS и по `if TYPE_CHECKING:`, а не по номеру
    строки: состав раздела — контракт, позиция раздела в файле — нет.
    """
    view_names = _view_names()
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in view_names:
            nodes.append(node)
        elif (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            nodes.append(node)
    return nodes

def _legacy_top_level_nodes() -> List[ast.AST]:
    """Узлы модуля верхнего уровня, не входящие в раздел M1."""
    tree = ast.parse(_module_source())
    m1 = {id(node) for node in _m1_nodes(tree)}
    return [node for node in tree.body if id(node) not in m1]

def test_legacy_code_does_not_reference_views(wb_ctx):
    """Ни класс, ни функция legacy-части не знают о протоколах и адаптерах (R1).

    Если бы оркестратор или один из восьми алгоритмов уже вызывал адаптеры,
    это означало бы смену поведения в ветке M1 (запрещено R2) и ломало бы
    baseline-снимки. Проверка по AST, а не по исходникам методов: у
    dataclass-классов часть методов (`SchedulingGrant.__eq__`) синтезирована и
    исходника не имеет — `inspect.getsource` на них падает с OSError.

    Полнота контроля проверяется содержательно (все восемь реализаций обязаны
    попасть в проверенный набор), а не счётчиком классов: добавление нового
    класса владельцем модуля не должно ломать этот тест.
    """
    nodes = _legacy_top_level_nodes()
    view_names = _view_names()
    class_names = {n.name for n in nodes if isinstance(n, ast.ClassDef)}
    for algorithm in bsc.ALL_ALGORITHMS:
        impl = type(SCHEDULER.SchedulerInterface.create(
            algorithm, wb_ctx["grid"], wb_ctx["bs"]))
        assert impl.__name__ in class_names, (
            f"{algorithm} ({impl.__name__}) вне контроля: класс не найден "
            f"среди legacy-узлов модуля")
    for node in nodes:
        used = ({n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)})
        hits = view_names & used
        assert not hits, (
            f"{getattr(node, 'name', type(node).__name__)} "
            f"(строка {getattr(node, 'lineno', '?')}) ссылается на "
            f"{sorted(hits)} — потребитель переключён на views раньше срока")

# ── 8. Побочные эффекты legacy-аллокаторов ─────────────────────────────────

def test_rr_allocator_mutates_shared_bs_buffer_size():
    """RR-аллокатор уменьшает общую оценку `user['bs_buffer_size']` (:2086).

    RoundRobin — единственный из восьми, кто пишет в словарь пользователя,
    подготовленный этапом 1 (:593); остальные работают с локальной копией
    `remaining_buffer`. Значение после schedule() читают `get_stats()` (:470)
    и оценка PDSCH (:739), то есть побочный эффект пересекает этапы —
    зафиксирован как AS-IS (R2).

    Поле входит в baseline-снимки (схема 2, `ues[].bs_buffer_size`) и
    сравнивается там на каждом TTI всех 23 сценариев; этот тест дополняет
    снимок тем, чего сравнение дать не может, — проверкой НАПРАВЛЕНИЯ
    изменения (оценка не имеет права вырасти) на живом прогоне.
    """
    ctx, pre, result = run_tti("RoundRobin")
    assert any(result["allocation"].values()), "RR ничего не выделил"
    decreased = 0
    for user in ctx["users"]:
        ue_id = user["UE_ID"]
        assert user["bs_buffer_size"] <= pre[ue_id], (
            f"UE {ue_id}: оценка выросла {pre[ue_id]} -> {user['bs_buffer_size']}")
        decreased += user["bs_buffer_size"] < pre[ue_id]
    assert decreased, "ни одна оценка bs_buffer_size не уменьшилась"

@pytest.mark.parametrize("algorithm", NON_RR_NON_QOS_ALGORITHMS)
def test_other_allocators_keep_bs_buffer_size_intact(algorithm):
    """Остальные non-QoS алгоритмы оценку этапа 1 не трогают.

    Список выводится из `ALL_ALGORITHMS`, а не перечисляется вручную: новый
    алгоритм в матрице автоматически попадёт под контроль (и упадёт, если
    начнёт мутировать общее поле — тогда это надо обсуждать, а не молча
    принимать).
    """
    ctx, pre, _result = run_tti(algorithm)
    for user in ctx["users"]:
        assert user["bs_buffer_size"] == pre[user["UE_ID"]], (
            f"{algorithm}: UE {user['UE_ID']} — оценка изменена аллокатором")

@pytest.mark.parametrize("algorithm", PF_ALGORITHMS)
def test_pf_writes_debug_field_instant_rate(algorithm):
    """PF и FD_PF добавляют отладочное поле `instant_rate` (:2143, :2614).

    Поле — часть наблюдаемого контракта `user`-словаря (его читает внешняя
    диагностика), поэтому при переносе метрики в UserRanker (M4b) оно должно
    остаться на месте (R2). Значение входит в baseline-снимки (схема 2,
    `ues[].instant_rate`); здесь оно дополнительно пересчитывается независимо
    от снимка — по формуле `rb_per_slot * GET_BITS_PER_RB(wb_cqi)`, поэтому
    тест ловит и подмену значения, и потерю поля.
    """
    ctx, _pre, _result = run_tti(algorithm)
    sched = ctx["sched"]
    ranked = [u for u in ctx["users"] if "priority" in u]
    assert ranked, "PF не рассчитал приоритеты"
    for user in ranked:
        expected = ctx["grid"].rb_per_slot * sched.amc.GET_BITS_PER_RB(
            sched._get_wb_cqi(user["UE_ID"]))
        assert user["instant_rate"] == expected

    if algorithm == "ProportionalFair":
        ctx_best, _pre2, _result2 = run_tti("BestCQI")
        assert all("instant_rate" not in u for u in ctx_best["users"]), (
            "BestCQI начал писать instant_rate — контракт user-словаря изменён")
