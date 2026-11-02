"""Общие построители контекстов для архитектурных тестов M1.

Вынесены из `conftest.py` по той же причине, что и `baseline_harness.py` в
M0-наборе: имя модуля `conftest` в дереве `PyScheduler/tests/` не уникально,
поэтому `from conftest import ...` при сборке нескольких каталогов одним
запуском разрешается в чужой файл. `conftest.py` собирает из этих хелперов
фикстуры, тесты импортируют хелперы напрямую.

Контексты строятся хелперами `baseline_scenarios` (`_new_ctx`, `_make_ue`,
`_mk_users`, `_fill_simple_buffer`, `_fill_layered_buffer`) — то есть на тех же
экземплярах grid/bs/sched и тех же профилях, на которых зафиксированы
baseline-снимки M0 (правило R5: не дублировать, а переиспользовать).
"""
import baseline_scenarios as bsc

FD_UE_PROFILE = [
    # (ue_id, x, wb_cqi, subband_cqi, buffer_bytes) — SB_FULL из baseline
    (1, 100.0, 12, bsc.SB_FULL[1], 2000),
    (2, 200.0, 10, bsc.SB_FULL[2], 2000),
    (3, 300.0, 14, bsc.SB_FULL[3], 2000),
]


def ctx_with_users(ctx, profile):
    """Заполнить буферы Simple-режима, собрать users, обновить cqi_map."""
    ues = [bsc._make_ue(ue_id, x, cqi, list(sb)) for ue_id, x, cqi, sb, _b in profile]
    for (ue_id, _x, _cqi, _sb, buf), _ue in zip(profile, ues, strict=True):
        bsc._fill_simple_buffer(ctx, ue_id, [(buf, None)])
    users = bsc._mk_users(ues)
    ctx["sched"]._refresh_cqi(5, users)
    ctx["ues"] = ues
    ctx["users"] = users
    return ctx


def wb_ctx_for(algorithm):
    """WB-контекст для произвольного алгоритма (профиль — как в baseline)."""
    return ctx_with_users(bsc._new_ctx(algorithm), bsc.WB_UE_PROFILE)


def layered_ctx_for(algorithm="QosAware"):
    """Контекст с LayeredBufferManager: per-QCI потоки + QciThroughputTracker.

    Нужен для `BuffersView.hol_delay` и `StatsView.qci_throughput`: в
    Simple-режиме этих данных нет (см. BUG-SCH-012). `bsc._new_ctx` жёстко
    задаёт `use_simple_buffer=True`, поэтому контекст собирается здесь —
    теми же конструкторами и тем же профилем QOS_MULTIFLOW, что в baseline.
    """
    from BS_MODULE import BaseStation
    from RES_GRID import RES_GRID_LTE
    from SCHEDULER import SchedulerInterface

    bs = BaseStation(bandwidth=10, use_simple_buffer=False)
    grid = RES_GRID_LTE(bandwidth=10, num_frames=2)
    grid.SET_BS(bs)
    ctx = {"algorithm": algorithm, "bs": bs, "grid": grid,
           "sched": SchedulerInterface.create(algorithm, grid, bs)}
    for ue_id, (bearers, _cqi) in bsc.QOS_MULTIFLOW.items():
        bsc._fill_layered_buffer(ctx, ue_id, bearers)
    ues = [bsc._make_ue(ue_id, 100.0 * ue_id, cqi, [])
           for ue_id, (_bearers, cqi) in bsc.QOS_MULTIFLOW.items()]
    ctx["ues"] = ues
    ctx["users"] = bsc._mk_users(ues)
    ctx["sched"]._refresh_cqi(5, ctx["users"])
    return ctx


def run_tti(algorithm, tti=5):
    """Прогнать один TTI на WB-профиле; вернуть контекст и буферы до вызова."""
    ctx = wb_ctx_for(algorithm)
    bm = ctx["bs"].buffer_manager
    pre = {u["UE_ID"]: bm.get_buffer_status(u["UE_ID"])[0].buffer_size
           for u in ctx["users"]}
    result = ctx["sched"].schedule(tti, ctx["users"])
    return ctx, pre, result
