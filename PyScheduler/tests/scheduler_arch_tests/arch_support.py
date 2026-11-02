"""Общие построители контекстов для архитектурных тестов M1.

Вынесены из `conftest.py` по той же причине, что и `baseline_harness.py` в
M0-наборе: имя модуля `conftest` в дереве `PyScheduler/tests/` не уникально,
поэтому `from conftest import ...` при сборке нескольких каталогов одним
запуском разрешается в чужой файл. `conftest.py` собирает из этих хелперов
фикстуры, тесты импортируют хелперы напрямую.

Контексты строятся хелперами `baseline_scenarios` (`new_ctx`,
`make_ue_with_cqi`, `build_users`, `fill_simple_buffer`,
`fill_layered_buffer`) — то есть на тех же экземплярах grid/bs/sched и тех же
профилях, на которых зафиксированы baseline-снимки M0 (правило R5: не
дублировать, а переиспользовать). Хелперы в `baseline_scenarios` публичные:
модуль не должен тянуть чужие приватные имена.
"""
from typing import Any, Dict, List, Tuple

import baseline_scenarios as bsc
from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE
from SCHEDULER import SchedulerInterface

# TTI 5 — тот же, что в золотом сценарии README §4 и в baseline-матрице
# (MULTI_TTI = [5, 6, 7]): сетка из 2 кадров покрывает TTI 0..19, поэтому 5
# попадает в первый же кадр и не требует пересборки сетки между прогонами.
TTI = 5

FD_UE_PROFILE = [
    # (ue_id, x, wb_cqi, subband_cqi, buffer_bytes) — SB_FULL из baseline
    (1, 100.0, 12, bsc.SB_FULL[1], 2000),
    (2, 200.0, 10, bsc.SB_FULL[2], 2000),
    (3, 300.0, 14, bsc.SB_FULL[3], 2000),
]


def ctx_with_users(ctx: Dict[str, Any],
                   profile: List[Tuple[int, float, int, List[int], int]]
                   ) -> Dict[str, Any]:
    """Заполнить буферы Simple-режима, собрать `users`, обновить `cqi_map`.

    Args:
        ctx: контекст из `baseline_scenarios.new_ctx` (ключи `bs`/`grid`/
            `sched`/`algorithm`).
        profile: строки `(ue_id, x, wb_cqi, sb_cqi, buffer_bytes)`.

    Returns:
        тот же `ctx`, дополненный ключами `ues` (объекты UserEquipment) и
        `users` (словари для `schedule()`).
    """
    ues = [bsc.make_ue_with_cqi(ue_id, x, cqi, list(sb)) for ue_id, x, cqi, sb, _b in profile]
    for (ue_id, _x, _cqi, _sb, buf), _ue in zip(profile, ues, strict=True):
        bsc.fill_simple_buffer(ctx, ue_id, [(buf, None)])
    users = bsc.build_users(ues)
    ctx["sched"]._refresh_cqi(TTI, users)
    ctx["ues"] = ues
    ctx["users"] = users
    return ctx


def wb_ctx_for(algorithm: str) -> Dict[str, Any]:
    """WB-контекст для произвольного алгоритма (профиль — как в baseline).

    Args:
        algorithm: ключ фабрики `SchedulerInterface.create()`.
    """
    return ctx_with_users(bsc.new_ctx(algorithm), bsc.WB_UE_PROFILE)


def layered_ctx_for(algorithm: str = "QosAware") -> Dict[str, Any]:
    """Контекст с LayeredBufferManager: per-QCI потоки + QciThroughputTracker.

    Нужен для `BuffersView.hol_delay` и `StatsView.qci_throughput`: в
    Simple-режиме этих данных нет (см. BUG-SCH-012). `bsc.new_ctx` жёстко
    задаёт `use_simple_buffer=True`, поэтому контекст собирается здесь —
    теми же конструкторами и тем же профилем QOS_MULTIFLOW, что в baseline.

    Args:
        algorithm: ключ фабрики; по умолчанию `QosAware` — единственный
            алгоритм, которому нужны несколько потоков на UE.
    """
    bs = BaseStation(bandwidth=10, use_simple_buffer=False)
    grid = RES_GRID_LTE(bandwidth=10, num_frames=2)
    grid.SET_BS(bs)
    ctx = {"algorithm": algorithm, "bs": bs, "grid": grid,
           "sched": SchedulerInterface.create(algorithm, grid, bs)}
    for ue_id, (bearers, _cqi) in bsc.QOS_MULTIFLOW.items():
        bsc.fill_layered_buffer(ctx, ue_id, bearers)
    ues = [bsc.make_ue_with_cqi(ue_id, 100.0 * ue_id, cqi, [])
           for ue_id, (_bearers, cqi) in bsc.QOS_MULTIFLOW.items()]
    ctx["ues"] = ues
    ctx["users"] = bsc.build_users(ues)
    ctx["sched"]._refresh_cqi(TTI, ctx["users"])
    return ctx


def run_tti(algorithm: str, tti: int = TTI
            ) -> Tuple[Dict[str, Any], Dict[int, int], Dict[str, Any]]:
    """Прогнать один TTI на WB-профиле; вернуть контекст и буферы до вызова.

    Returns:
        `(ctx, pre, result)`: контекст после `schedule()`, размеры буферов
        `{ue_id: bytes}` ДО вызова (чтобы тест мог сравнить их с состоянием
        после) и словарь, возвращённый `schedule()`.
    """
    ctx = wb_ctx_for(algorithm)
    bm = ctx["bs"].buffer_manager
    pre = {u["UE_ID"]: bm.get_buffer_status(u["UE_ID"])[0].buffer_size
           for u in ctx["users"]}
    result = ctx["sched"].schedule(tti, ctx["users"])
    return ctx, pre, result
