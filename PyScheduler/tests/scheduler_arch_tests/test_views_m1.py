"""M1: протоколы стратегий и view-адаптеры (SPEC-FEAT-SCH-008 §5.7, шаг M1).

Что проверяется
---------------
1. Протоколы `runtime_checkable` и структурны: адаптер подходит под протокол
   без наследования, объект без нужного метода — не подходит.
2. Сигнатуры адаптера совпадают с сигнатурами протокола (имена методов и
   параметров) — опечатка в имени параметра ломает структурную типизацию.
3. Адаптеры ДЕЛЕГИРУЮТ legacy-коду и возвращают те же значения (R5): сверка
   идёт на тех же экземплярах grid/bs/sched, что и baseline-снимки.
4. Потребители НЕ переключены (R1): все обращения к view-API в SCHEDULER.py
   лежат внутри нового раздела, номера строк legacy-кода не изменились.
5. AS-IS дефект BUG-SCH-012 воспроизводится через `BuffersView.hol_delay`
   (Simple-буфер не заполняет `qci`/`hol_delay`, BS_MODULE.py :453–457).

Набор намеренно не проверяет поведение восьми алгоритмов — это работа
`tests/baseline/` (47 characterization-тестов, снимки dev 6c0aae2).
"""
import inspect
import re

import baseline_scenarios as bsc
import GLOBALS
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

SECTION_MARKER = "# ИНТЕРФЕЙСЫ СТРАТЕГИЙ И VIEW-АДАПТЕРЫ"

# Пары «протокол -> legacy-адаптер» (SPEC §5.7, таблица интерфейсов M1).
VIEW_PAIRS = [
    (CqiView, LegacyCqiView),
    (GridView, LegacyGridView),
    (AmcView, LegacyAmcView),
    (BuffersView, LegacyBuffersView),
    (StatsView, LegacyStatsView),
    (StateView, LegacyStateView),
]

# API view-адаптеров: имена классов и вызовы их методов. Совпадение с этим
# шаблоном вне нового раздела означало бы, что потребитель уже переключён (R1).
VIEW_API_RE = re.compile(
    r"\bLegacy(?:Cqi|Grid|Amc|Buffers|Stats|State)View\b"
    r"|\.(?:get_wb|get_sb|allocate_rbg|rbg_size|rbg_indices|total_rbg|prb_count"
    r"|bits_per_rb|has_buffer|buffer_status|hol_delay|ue_throughput"
    r"|qci_throughput|gbr_of|pdb_of|get_rr_(?:ue|rbg)_offset"
    r"|set_rr_(?:ue|rbg)_offset)\s*\("
)


# ── 1. Протоколы структурны и runtime_checkable ─────────────────────────────


def test_protocols_are_runtime_checkable():
    """Все шесть протоколов помечены @runtime_checkable (SPEC §5.7)."""
    for protocol, _adapter in VIEW_PAIRS:
        assert getattr(protocol, "_is_runtime_protocol", False), (
            f"{protocol.__name__}: нет @runtime_checkable — isinstance() "
            f"недоступен, контроль соответствия в тестах невозможен")


def test_adapters_satisfy_protocols(wb_ctx, layered_ctx):
    """Адаптеры подходят под протоколы без наследования (структурная типизация)."""
    sched = wb_ctx["sched"]
    users_by_id = {u["UE_ID"]: u["ue"] for u in wb_ctx["users"]}
    instances = {
        CqiView: LegacyCqiView(sched),
        GridView: LegacyGridView(wb_ctx["grid"]),
        AmcView: LegacyAmcView(sched.amc),
        # планировщик не хранит `bs`: доступ к буферам идёт через сетку
        # (`self.lte_grid.bs.buffer_manager`, :574/:900/:2776/:2829/:3015/:3065)
        BuffersView: LegacyBuffersView(wb_ctx["bs"].buffer_manager),
        StatsView: LegacyStatsView(users_by_id, wb_ctx["bs"].buffer_manager),
        StateView: LegacyStateView(sched),
    }
    for protocol, instance in instances.items():
        assert isinstance(instance, protocol), (
            f"{type(instance).__name__} не соответствует {protocol.__name__}")

    # Layered-режим: тот же адаптер буферов обязан проходить проверку
    assert isinstance(LegacyBuffersView(layered_ctx["bs"].buffer_manager), BuffersView)


def test_protocol_rejects_incomplete_object():
    """Протокол действительно ограничивает: объект без метода не проходит.

    Контроль «заглушки»: если бы isinstance() всегда возвращал True, тест
    test_adapters_satisfy_protocols ничего бы не гарантировал.
    """
    class NoSb:
        def get_wb(self, ue_id):
            return 10

    assert isinstance(NoSb(), CqiView) is False
    assert not isinstance(object(), GridView)
    assert not isinstance(object(), AmcView)


def test_adapter_signatures_match_protocol():
    """Имена методов и параметров адаптера совпадают с протоколом.

    Структурная типизация проверяет только наличие метода, поэтому опечатка в
    имени параметра (`ue` вместо `ue_id`) прошла бы isinstance(), но сломала бы
    вызовы по имени — ловим здесь.
    """
    for protocol, adapter in VIEW_PAIRS:
        # у Protocol в члены попадают служебные __init__/__subclasshook__
        proto_methods = {name: fn for name, fn in inspect.getmembers(
            protocol, predicate=inspect.isfunction)
            if not name.startswith("__")}
        adapter_methods = {name: fn for name, fn in inspect.getmembers(
            adapter, predicate=inspect.isfunction)
            if not name.startswith("__")}
        assert set(proto_methods) == set(adapter_methods), (
            f"{adapter.__name__}: методы {sorted(set(proto_methods) ^ set(adapter_methods))} "
            f"расходятся с {protocol.__name__}")
        for name in sorted(proto_methods):
            p_params = list(inspect.signature(proto_methods[name]).parameters)[1:]
            a_params = list(inspect.signature(adapter_methods[name]).parameters)[1:]
            assert p_params == a_params, (
                f"{adapter.__name__}.{name}: параметры {a_params}, "
                f"протокол требует {p_params}")


# ── 2. CqiView: делегирование legacy-геттерам (:1064 / :1084) ───────────────


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
    assert view.get_wb(999) == 0
    assert view.get_sb(999) == []


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


def test_cqi_view_returns_same_list_object(wb_ctx):
    """Адаптер не копирует SB-список: мутация результата видна в cqi_map.

    Фиксирует AS-IS (`return self.cqi_map[ueid].sb_cqi`, :1098). Копирование
    изменило бы семантику для потребителей — в M1 это запрещено (R2).
    """
    sched = wb_ctx["sched"]
    ue_id = 1
    sched.cqi_map[ue_id].sb_cqi = [7, 8, 9]
    assert LegacyCqiView(sched).get_sb(ue_id) is sched.cqi_map[ue_id].sb_cqi


# ── 3. GridView: делегирование RES_GRID_LTE ─────────────────────────────────


def test_grid_view_matches_legacy_grid_api(wb_ctx):
    """rbg_size/rbg_indices/total_rbg/prb_count равны прямым вызовам grid."""
    grid, view = wb_ctx["grid"], LegacyGridView(wb_ctx["grid"])
    assert view.rbg_size() == grid.GET_RBG_SIZE()
    assert view.prb_count() == grid.rb_per_slot
    for rbg_idx in range(view.total_rbg()):
        assert view.rbg_indices(rbg_idx) == grid.GET_RBG_INDICES(rbg_idx)


def test_grid_view_total_rbg_matches_qos_formula():
    """total_rbg повторяет формулу QoS-аллокатора (:3088–3089).

    Общее FD-выделение на M2 не должно менять число просматриваемых RBG (R2),
    поэтому формула зафиксирована тестом, а не переписана «аккуратнее».
    """
    for bandwidth in (5, 10, 20):
        ctx = bsc._new_ctx("QosAware", bandwidth=bandwidth)
        grid = ctx["grid"]
        rbg_size = grid.GET_RBG_SIZE()
        expected = (grid.rb_per_slot + rbg_size - 1) // rbg_size
        assert LegacyGridView(grid).total_rbg() == expected
        assert sum(len(LegacyGridView(grid).rbg_indices(i)) for i in range(expected)) \
            == grid.rb_per_slot


def test_grid_view_allocate_rbg_delegates_and_mutates_grid():
    """allocate_rbg == ALLOCATE_RBG, включая побочный эффект на сетке.

    ALLOCATE_RBG меняет владельца RB, поэтому сверка идёт на двух одинаково
    построенных сетках (близнецах): последовательность вызовов и результат
    совпадают, итоговое состояние сеток идентично.
    """
    twin_a, twin_b = bsc._new_ctx("BestCQI"), bsc._new_ctx("BestCQI")
    view, grid = LegacyGridView(twin_a["grid"]), twin_b["grid"]
    results = []
    for rbg_idx in range(4):
        results.append((view.allocate_rbg(5, rbg_idx, 1),
                        grid.ALLOCATE_RBG(5, rbg_idx, 1)))
        # повторное выделение того же RBG тому же UE — как в FD-цикле
        results.append((view.allocate_rbg(5, rbg_idx, 2),
                        grid.ALLOCATE_RBG(5, rbg_idx, 2)))
    assert all(a == b for a, b in results), f"расхождение allocate_rbg: {results}"
    assert bsc._grid_state(twin_a["grid"], 5) == bsc._grid_state(twin_b["grid"], 5)


# ── 4. AmcView: делегирование GET_BITS_PER_RB ───────────────────────────────


def test_amc_view_matches_get_bits_per_rb(wb_ctx):
    """bits_per_rb(cqi) == amc.GET_BITS_PER_RB(cqi) для CQI 1..15 (:1730)."""
    amc, view = wb_ctx["sched"].amc, LegacyAmcView(wb_ctx["sched"].amc)
    for cqi in range(1, 16):
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
    assert LegacyCqiView(wb_ctx["sched"]).get_wb(999) == 0
    with pytest.raises(ValueError, match="Invalid CQI 0"):
        view.bits_per_rb(0)
    with pytest.raises(ValueError, match="Invalid CQI 0"):
        amc.GET_BITS_PER_RB(0)


# ── 5. BuffersView: делегирование IBufferManager ────────────────────────────


def test_buffers_view_matches_manager_simple(wb_ctx):
    """Simple-режим: has_buffer/buffer_status равны прямым вызовам менеджера."""
    bm = wb_ctx["bs"].buffer_manager
    view = LegacyBuffersView(bm)
    for ue_id in (1, 2, 3):
        assert bm.ue_has_buffer(ue_id) is True
        assert view.has_buffer(ue_id) is True
        assert view.buffer_status(ue_id) == bm.get_buffer_status(ue_id)
        assert all(isinstance(s, BufferStatus) for s in view.buffer_status(ue_id))
    assert view.has_buffer(999) == bm.ue_has_buffer(999)


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
        assert view.hol_delay(ue_id, 99) is None
    assert checked >= 5, f"проверено всего {checked} потоков"


def test_buffers_view_hol_delay_none_on_simple_reproduces_bug_sch_012(wb_ctx):
    """BUG-SCH-012 (AS-IS): Simple-буфер не заполняет qci/hol_delay.

    `SimpleBuffer.get_buffer_status()` (BS_MODULE.py :453–457) создаёт
    BufferStatus только с ue_id/buffer_size/timestamp, поэтому qci и hol_delay
    остаются None. QoS-метрика (:3029 `pdb = QCI(qci).get_delay_budget()`)
    на таком статусе падает. Тест фиксирует дефект и НЕ лечит его (R2):
    после исправления BS_MODULE этот тест должен быть обновлён вместе с
    карточкой BUG-SCH-012 и baseline-снимками.
    """
    bm = wb_ctx["bs"].buffer_manager
    view = LegacyBuffersView(bm)
    ue_id = 1
    statuses = view.buffer_status(ue_id)
    assert statuses and statuses[0].buffer_size > 0, "буфер UE 1 пуст — сценарий не собрался"
    assert all(s.qci is None for s in statuses), (
        "SimpleBuffer начал заполнять qci — BUG-SCH-012 закрыт, обновите тест")
    assert all(s.hol_delay is None for s in statuses)
    assert view.hol_delay(ue_id, 9) is None
    with pytest.raises(ValueError, match="None is not a valid QCI"):
        QCI(statuses[0].qci)


# ── 6. StatsView: EWMA, per-QCI трекер, GBR/PDB ─────────────────────────────


def test_stats_view_ue_throughput_matches_ue_field(wb_ctx):
    """ue_throughput == user['ue'].average_throughput (:2133); нет UE -> 0.0."""
    users_by_id = {u["UE_ID"]: u["ue"] for u in wb_ctx["users"]}
    view = LegacyStatsView(users_by_id, wb_ctx["bs"].buffer_manager)
    for ue_id, ue in users_by_id.items():
        ue.average_throughput = 1234.5 + ue_id
        assert view.ue_throughput(ue_id) == ue.average_throughput
    assert view.ue_throughput(999) == 0.0
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
    assert view.qci_throughput(1, 9) == (0.0, 0.0)
    assert LegacyStatsView().qci_throughput(1, 9) == (0.0, 0.0)


@pytest.mark.parametrize("qci", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_stats_view_gbr_and_pdb_match_globals(qci):
    """gbr_of/pdb_of равны GLOBALS.is_gbr + QCI_PROFILES / QCI.get_delay_budget."""
    view = LegacyStatsView()
    expected_gbr = GLOBALS.QCI_PROFILES[qci].get("gbr") if GLOBALS.is_gbr(qci) else None
    assert view.gbr_of(qci) == expected_gbr
    assert view.pdb_of(qci) == QCI(qci).get_delay_budget()
    assert view.pdb_of(qci) > 0


def test_stats_view_gbr_present_for_gbr_qci():
    """Хотя бы у одного GBR-класса профиль содержит gbr (иначе view бессмысленна)."""
    view = LegacyStatsView()
    gbr_qcis = [q for q in range(1, 10) if GLOBALS.is_gbr(q)]
    assert gbr_qcis, "GLOBALS.is_gbr не выделяет ни одного GBR-класса"
    assert any(view.gbr_of(q) is not None for q in gbr_qcis)


# ── 7. StateView: сквозная запись в поля планировщика ───────────────────────


def test_state_view_reads_and_writes_scheduler_fields():
    """RR/FD_FGS: get/set работают сквозь адаптер с полями экземпляра."""
    for algorithm in ("RoundRobin", "FD_FGS"):
        ctx = wb_ctx_for(algorithm)
        sched, view = ctx["sched"], LegacyStateView(ctx["sched"])
        assert view.get_rr_ue_offset() == sched.rr_ue_offset == 0
        assert view.get_rr_rbg_offset() == sched.rr_rbg_offset == 0
        view.set_rr_ue_offset(2)
        view.set_rr_rbg_offset(3)
        assert (sched.rr_ue_offset, sched.rr_rbg_offset) == (2, 3), (
            f"{algorithm}: адаптер не пишет в поля планировщика")
        assert (view.get_rr_ue_offset(), view.get_rr_rbg_offset()) == (2, 3)


def test_state_view_absent_fields_read_as_zero():
    """У остальных шести алгоритмов полей нет — чтение даёт 0 (AS-IS, R2)."""
    for algorithm in bsc.ALL_ALGORITHMS:
        if algorithm in ("RoundRobin", "FD_FGS"):
            continue
        ctx = wb_ctx_for(algorithm)
        sched, view = ctx["sched"], LegacyStateView(ctx["sched"])
        assert not hasattr(sched, "rr_ue_offset"), f"{algorithm}: поле появилось"
        assert view.get_rr_ue_offset() == 0
        assert view.get_rr_rbg_offset() == 0


# ── 8. R1: потребители не переключены, legacy-строки не сдвинуты ────────────


def _module_lines():
    return inspect.getsource(SCHEDULER).splitlines()


def _section_start(lines):
    for idx, line in enumerate(lines, start=1):
        if line.strip() == SECTION_MARKER:
            return idx
    pytest.fail(f"раздел M1 не найден по маркеру {SECTION_MARKER!r}")


def test_no_consumer_switched_yet():
    """View-API используется только внутри нового раздела (правило R1).

    Если бы оркестратор или один из восьми алгоритмов уже вызывал адаптеры,
    совпадение нашлось бы до маркера раздела — это означало бы смену
    поведения в ветке M1, что запрещено (R2) и ломает baseline-снимки.
    """
    lines = _module_lines()
    section = _section_start(lines)
    offenders = [(n, text.strip())
                 for n, text in enumerate(lines, start=1)
                 if n < section and VIEW_API_RE.search(text)]
    assert not offenders, (
        "потребители переключены на views раньше срока (R1):\n"
        + "\n".join(f"  :{n}: {t}" for n, t in offenders[:20]))


def test_legacy_classes_do_not_reference_views():
    """Ни один legacy-класс не знает о протоколах и адаптерах (R1).

    Проверка по AST, а не по исходникам методов: у dataclass-классов часть
    методов (`SchedulingGrant.__eq__`) синтезирована и исходника не имеет,
    `inspect.getsource` на них падает с OSError.
    """
    import ast

    view_names = {cls.__name__ for _p, cls in VIEW_PAIRS} | {
        p.__name__ for p, _c in VIEW_PAIRS}
    tree = ast.parse(inspect.getsource(SCHEDULER))
    legacy_classes = [n for n in tree.body
                      if isinstance(n, ast.ClassDef) and n.name not in view_names]
    # 5 инфраструктурных классов + 8 алгоритмов; view-классы исключены
    assert len(legacy_classes) == 13, (
        f"состав legacy-классов изменился (ожидалось 13): "
        f"{[c.name for c in legacy_classes]}")
    for cls in legacy_classes:
        used = ({node.id for node in ast.walk(cls) if isinstance(node, ast.Name)}
                | {node.attr for node in ast.walk(cls)
                   if isinstance(node, ast.Attribute)})
        hits = view_names & used
        assert not hits, f"{cls.name} (:{cls.lineno}) ссылается на {sorted(hits)}"


def test_m1_section_appended_after_legacy_code():
    """Раздел добавлен в конец файла: номера строк legacy-кода сохранены (R3).

    Якоря `:NNN` в документации модуля и в отчётах остаются действительными,
    поэтому новая кода не вставляется в середину файла.
    """
    lines = _module_lines()
    section = _section_start(lines)
    legacy_end = max(cls.end_lineno for cls in _legacy_classes_ast())
    assert section > legacy_end, (
        f"раздел M1 на строке {section}, legacy-код кончается на {legacy_end}")


def _legacy_classes_ast():
    import ast
    tree = ast.parse(inspect.getsource(SCHEDULER))
    view_names = {cls.__name__ for _p, cls in VIEW_PAIRS} | {
        p.__name__ for p, _c in VIEW_PAIRS}
    return [n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name not in view_names]


def test_documentation_anchors_still_valid():
    """Якоря, на которые ссылается раздел M1, по-прежнему на своих строках.

    Сверяются только те строки, что процитированы в docstring адаптеров;
    на M2 (перенос кода) тест обновляется вместе с цитатами — молчаливого
    расхождения документации и кода быть не должно (DOC-SCH-003).
    """
    anchors = {
        574: "buffer_manager = self.lte_grid.bs.buffer_manager",
        1064: "def _get_wb_cqi",
        1084: "def _get_sb_cqi",
        1993: "self.rr_rbg_offset = 0",
        2133: "avg_throughput = user['ue'].average_throughput",
        2374: "self.rr_rbg_offset = 0",
        2790: "group = buffer_status.hol_delay // self.grouping_parameter",
        3029: "pdb = QCI(qci).get_delay_budget()",
        3088: "rbg_size  = self.lte_grid.GET_RBG_SIZE()",
        3089: "total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size",
        3126: "pdb = QCI(qci).get_delay_budget()",
    }
    lines = _module_lines()
    broken = {n: (lines[n - 1].strip(), expected)
              for n, expected in anchors.items() if expected not in lines[n - 1]}
    assert not broken, "якоря документации съехали:\n" + "\n".join(
        f"  :{n}: факт {fact!r}, ожидалось {exp!r}" for n, (fact, exp) in broken.items())


def test_module_header_version_bumped():
    """DOC-SCH-003: шапка модуля соответствует фактическому состоянию.

    До M1 шапка утверждала «Версия: 2.0.0 / Дата: 2025-10-26», хотя блок
    «Изменения v2.1.0» уже был в файле (:86) — расхождение закрыто в M1.
    """
    header = "\n".join(_module_lines()[:20])
    assert "Версия: 2.1.1" in header, f"шапка не обновлена:\n{header}"
    assert "Дата последнего изменения: 2026-11-01" in header
    assert "Изменения v2.1.0" in inspect.getsource(SCHEDULER), (
        "блок истории v2.1.0 потерян при правке шапки")


# ── 9. Реестр алгоритмов не затронут ────────────────────────────────────────


def test_create_accepts_all_eight_factory_keys():
    """Фабрика create() (:255-264) по-прежнему принимает все восемь имён (R2)."""
    ctx = wb_ctx_for("BestCQI")
    for algorithm in bsc.ALL_ALGORITHMS:
        sched = SCHEDULER.SchedulerInterface.create(
            algorithm, ctx["grid"], ctx["bs"])
        assert isinstance(sched, SCHEDULER.SchedulerInterface)
        assert sched.__class__.__name__ != "SchedulerInterface", algorithm


def test_available_algorithms_divergence_pinned_bug_sch_010(wb_ctx):
    """AS-IS: available_algorithms() (:284-292) расходится с реестром create().

    BUG-SCH-010 (латентный): `'FD_BestCQI'`/`'FD_FF'` рекламируются, но
    create() их отвергает (:268), а `'FD_BCQI'`/`'FD_PF'` работают, но не
    рекламируются. Вызовов available_algorithms() в коде dev нет. M1 не меняет
    наблюдаемый результат метода (R2) — синхронизация через единый
    SCHEDULER_REGISTRY запланирована на M8a. Тест фиксирует расхождение, чтобы
    оно не «поплыло» незаметно во время миграции.
    """
    registry = SCHEDULER.SchedulerInterface.available_algorithms()
    assert sorted(registry) == sorted([
        "BestCQI", "ProportionalFair", "RoundRobin", "FD_BestCQI", "FD_FGS",
        "FD_FF", "BasicQosAware", "QosAware"]), (
        f"список :284-292 изменился — сверьте с BUG-SCH-010 и планом M8a: {registry}")
    for advertised in ("FD_BestCQI", "FD_FF"):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            SCHEDULER.SchedulerInterface.create(
                advertised, wb_ctx["grid"], wb_ctx["bs"])
    for working in ("FD_BCQI", "FD_PF"):
        assert working not in registry, (
            f"{working} появился в available_algorithms() раньше M8a")


def test_views_are_not_registered_as_algorithms():
    """Адаптеры/протоколы не попали в фабрику create() и в реестр."""
    registry = SCHEDULER.SchedulerInterface.available_algorithms()
    for name in ("CqiView", "GridView", "AmcView", "BuffersView", "StatsView",
                 "StateView", "LegacyCqiView", "LegacyGridView"):
        assert name not in registry
    assert not any("View" in str(item) for item in registry), (
        f"в реестре появились view-имена: {registry}")


# ── 10. Побочные эффекты, выявленные инвентаризацией M1 (INVENTORY-M1.md §4) ─


def test_rr_allocator_mutates_shared_bs_buffer_size():
    """RR-аллокатор уменьшает общую оценку `user['bs_buffer_size']` (:2086).

    RoundRobin — единственный из восьми, кто пишет в словарь пользователя,
    подготовленный этапом 1 (:593); остальные работают с локальной копией
    `remaining_buffer`. Значение после schedule() читают `get_stats()` (:470)
    и оценка PDSCH (:739), то есть побочный эффект пересекает этапы —
    зафиксирован как AS-IS (R2), в baseline-снимки поле не входит
    (см. INVENTORY-M1.md §6, пробел покрытия).
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


@pytest.mark.parametrize(
    "algorithm", ["BestCQI", "ProportionalFair", "FD_BCQI", "FD_PF", "FD_FGS"])
def test_other_allocators_keep_bs_buffer_size_intact(algorithm):
    """Остальные пять non-QoS алгоритмов оценку этапа 1 не трогают."""
    ctx, pre, _result = run_tti(algorithm)
    for user in ctx["users"]:
        assert user["bs_buffer_size"] == pre[user["UE_ID"]], (
            f"{algorithm}: UE {user['UE_ID']} — оценка изменена аллокатором")


@pytest.mark.parametrize("algorithm", ["ProportionalFair", "FD_PF"])
def test_pf_writes_debug_field_instant_rate(algorithm):
    """PF и FD_PF добавляют отладочное поле `instant_rate` (:2143, :2614).

    Поле — часть наблюдаемого контракта `user`-словаря (его читает внешняя
    диагностика), поэтому при переносе метрики в UserRanker (M4b) оно должно
    остаться на месте (R2). В baseline-снимки поле не входит — пробел покрытия
    закрыт этим тестом (INVENTORY-M1.md §6).
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
