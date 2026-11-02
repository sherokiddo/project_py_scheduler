"""M0 (SPEC-FEAT-SCH-008, раздел 8/9): детерминированные baseline-сценарии.

Назначение
----------
Фиксация наблюдаемого поведения SCHEDULER.py на ревизии dev 6c0aae2 ДО
начала миграции (шаги M1+). Каждый сценарий полностью детерминирован:
фиксированные CQI/буферы/пакеты, без генераторов случайных чисел, поэтому
seed не требуется (в full-SIM контрольных точках seed задаётся отдельно).

Сценарии (имя файла baseline == ключ SCENARIOS):
- golden_tti5            — золотой сценарий README §4 (BestCQI, TTI 5);
- wb_<alg>               — WB-сценарий, 8 алгоритмов x 3 TTI (матрица §9);
- fd_sb_<full|missing|partial>_<alg> — SB / SB-missing->WB / частичный SB
                           для FD_BCQI, FD_PF, FD_FGS (таблица per-RBG выборов);
- rr_multitti            — ротация rr_ue_offset/rr_rbg_offset, TTI 5..8;
- qos_basic_single_bearer / qos_aware_multiflow — grants + per-QCI stats;
- conflict_rbg_contended / conflict_pdcch_limited — сеточный конфликт (§9.5).

Схема снимка TTI (capture_tti)
------------------------------
`tti`, `allocation`, `pdcch_stats`, `bitmap`, `statistics_keys`,
`last_ue_transmitted_bits`, `priority_list`, `grid_rb_owners`,
`rbg_choice_table`, `buffer_status`, `scheduler_extra`, `qci_tracker`,
`ues[]` — по каждому UE: `ue_id`, `average_throughput`,
`current_dl_throughput`, `cqi_map_wb`, `cqi_map_sb`, `user_dict_keys`,
`bs_buffer_size`, `instant_rate`.

Три последних поля добавлены 2026-11-01 (схема 2) после анализа пробела
покрытия: `user['bs_buffer_size']` (этап 1, :593; мутируется RR-аллокатором
:2086; читается :470 и :739) и `user['instant_rate']` (PF :2143, FD_PF :2614)
наблюдаемы после schedule(), но в схему 1 не входили. `user_dict_keys`
фиксирует весь контракт словаря `user`, чтобы добавление/удаление поля не
проходило молчаливо. Все значения схемы 1 при перегенерации совпали
побайтово (RUNBOOK §6.1) — расширение схемы поведения не меняло.

Запуск генерации снимков:  python tests/baseline/generate_baselines.py
Запуск тестов:             pytest tests/baseline -q
"""
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import GLOBALS
from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE
from SCHEDULER import SchedulerInterface
from TRAFFIC_MODEL import QCI, BearerInfo, Packet, UeBearersInfo
from UE_MODULE import UserEquipment

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

DEV_REVISION = "6c0aae2"

# Seed full-SIM контрольных точек (sim_checkpoint.py): одинаков для всех
# снимков sim*.json, поэтому тесты сверяют meta.seed с этой константой.
SIM_SEED = 42


def env_info() -> Dict[str, str]:
    """Информационная подпись окружения, в котором построен снимок.

    В сравнении не участвует (RUNBOOK §5): поле нужно, чтобы при расхождении
    снимков можно было отличить изменение поведения кода от смены окружения.
    Регенерация обновляет его автоматически (RUNBOOK §6) — его отличие
    в diff не является дефектом.
    """
    import numpy
    import pytest
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pytest": pytest.__version__,
    }


ALL_ALGORITHMS = [
    "BestCQI",
    "ProportionalFair",
    "RoundRobin",
    "FD_BCQI",
    "FD_FGS",
    "FD_PF",
    "BasicQosAware",
    "QosAware",
]
FD_ALGORITHMS = ["FD_BCQI", "FD_FGS", "FD_PF"]

# ── Фиксированные профили сценариев ──────────────────────────────────────

# Золотой сценарий (README §4): воспроизведён побитово на ae9db40 и 6c0aae2.
GOLDEN_UE_PROFILE = [
    # (ue_id, x, wb_cqi, sb_cqi, buffer_bytes)
    (1, 100.0, 13, [], 1000),
    (2, 200.0, 7, [], 5000),
    (3, 300.0, 15, [], 300),
]

# WB-матрица: у UE2 буфер заканчивается после TTI 5 (кейс пустого буфера),
# у UE4 низкий CQI (кейс низкой оценки AMC).
WB_UE_PROFILE = [
    (1, 100.0, 13, [], 2000),
    (2, 200.0, 9, [], 300),
    (3, 300.0, 15, [], 1500),
    (4, 400.0, 5, [], 4000),
]

# FD SB-профили: контрастные per-RBG CQI (13 RBG при bandwidth=10, RBG=2).
SB_FULL = {
    1: [15, 3, 15, 3, 15, 3, 15, 3, 15, 3, 15, 3, 15],
    2: [3, 15, 3, 15, 3, 15, 3, 15, 3, 15, 3, 15, 3],
    3: [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
}
# Режимы subband-CQI в FD-сценариях (единственный источник: из него строится
# и реестр SCENARIOS, и параметризация тестов).
SB_MODES = ("full", "missing", "partial")

SB_MISSING = {1: [], 2: [], 3: []}          # fallback SB -> WB
SB_PARTIAL = {                               # короткий список + нули внутри
    1: [15, 0, 15],
    2: [0, 12],
    3: [],
}
SB_WB_CQI = {1: 10, 2: 12, 3: 8}
SB_BUFFERS = {1: 2000, 2: 2000, 3: 2000}

# QoS, один bearer на UE (контракт BasicQosAware: один поток на абонента).
QOS_SINGLE_BEARER = {
    # ue_id: (qci, buffer_bytes, wb_cqi)
    1: (1, 2000, 12),   # GBR (VoIP)
    2: (4, 3000, 10),   # GBR
    3: (9, 1500, 14),   # non-GBR
}

# QoS, несколько bearer'ов на UE (QosAware: потоки (ue_id, qci)).
QOS_MULTIFLOW = {
    # ue_id: ([(bearer_id, qci, bytes)], wb_cqi)
    1: ([(1, 1, 1500), (2, 9, 2500)], 12),
    2: ([(1, 4, 2000)], 10),
    3: ([(1, 9, 1200), (2, 6, 1800)], 14),
}

CONFLICT_UE_PROFILE = [
    # CQI подобраны так, чтобы 95%-оценка PDSCH оставила в топ-2 UE1 и UE2
    # (одинаковые CQI и буферы -> спор за одни и те же RBG в FD-цикле).
    (1, 100.0, 15, [], 5000),
    (2, 200.0, 15, [], 5000),
    (3, 300.0, 11, [], 400),
]

MULTI_TTI = [5, 6, 7]        # WB-матрица и FD-сценарии
RR_TTI = [5, 6, 7, 8]        # RR-ротация: 3+ TTI (раздел 9.3)


# ── Построение сценариев ─────────────────────────────────────────────────

def make_ue_with_cqi(ue_id: int, x: float, cqi: int,
                     sb_cqi: List[int]) -> UserEquipment:
    """UE с фиксированным WB-CQI и subband-CQI (без модели канала).

    Имя отличается от `make_ue` в `tests/mobility_module_tests/`: там helper
    создаёт UE для моделей мобильности, здесь — для детерминированных
    сценариев планировщика. Совпадение имён в разных каталогах уже приводило
    к конфликту импортов (BUG-SCH-013), поэтому имя расширено.
    """
    ue = UserEquipment(UE_ID=ue_id, x=x, y=0.0)
    ue.cqi = cqi
    ue.cqi_subband = list(sb_cqi)
    return ue


def build_users(ues: List[UserEquipment]) -> List[Dict[str, Any]]:
    """Список словарей `users` в том виде, в котором его ожидает schedule().

    Этап 1 (`_filter_eligible_ues`, :559-600) дописывает в эти словари
    `bs_buffer_size` и (для PF/FD_PF) `instant_rate`; оба поля входят в
    снимок (схема 2, см. docstring модуля).
    """
    return [
        {
            "UE_ID": ue.UE_ID,
            "cqi": ue.cqi,
            "sbb_cqi": ue.cqi_subband,
            "ue": ue,
        }
        for ue in ues
    ]


def new_ctx(algorithm: str, bandwidth: int = 10, num_frames: int = 2,
            scheduler_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Свежие grid/bs/sched — состояние не переиспользуется между снимками.

    Args:
        algorithm: ключ фабрики `SchedulerInterface.create()`; неизвестное имя
            даёт `ValueError` с перечислением допустимых значений (:266-269).
        bandwidth: полоса БС, МГц (влияет на `rb_per_slot` и размер RBG).
        num_frames: число кадров сетки (должно покрывать максимальный TTI
            сценария, иначе `ALLOCATE_RBG` выйдет за пределы сетки).
        scheduler_kwargs: параметры конструктора планировщика
            (`grouping_parameter`, `metric_mode` и т.п.).

    Returns:
        dict с ключами `algorithm`, `bs`, `grid`, `sched`. Буферы пусты — их
        заполняют `fill_simple_buffer` / `fill_layered_buffer`.
    """
    scheduler_kwargs = dict(scheduler_kwargs or {})
    bs = BaseStation(bandwidth=bandwidth, use_simple_buffer=True)
    grid = RES_GRID_LTE(bandwidth=bandwidth, num_frames=num_frames)
    grid.SET_BS(bs)
    sched = SchedulerInterface.create(algorithm, grid, bs, **scheduler_kwargs)
    return {"algorithm": algorithm, "bs": bs, "grid": grid, "sched": sched}


def fill_simple_buffer(ctx: Dict[str, Any], ue_id: int,
                       packets: List[Tuple[int, Optional[int]]]) -> None:
    """Создать буфер UE в SimpleBufferManager и добавить пакеты.

    Args:
        ctx: контекст из `new_ctx`.
        ue_id: идентификатор UE.
        packets: пары `(размер в байтах, qci)`; `qci=None` — пакет без QCI
            (Simple-режим его не использует, `BufferStatus.qci` останется
            None — см. BUG-SCH-012).
    """
    bm = ctx["bs"].buffer_manager
    bm.create_ue_buffer(ue_id, ctx["bs"].per_ue_max)
    for size, qci in packets:
        pkt = Packet(size=size, ue_id=ue_id, creation_time=0)
        if qci is not None:
            pkt.qci = QCI(qci)
        bm.add_packet(ue_id, pkt)


def fill_layered_buffer(ctx: Dict[str, Any], ue_id: int,
                        bearers: List[Tuple[int, int, int]]) -> None:
    """Создать per-QCI буферы UE в LayeredBufferManager и добавить пакеты.

    Args:
        ctx: контекст из `new_ctx` (построенный с `use_simple_buffer=False`).
        ue_id: идентификатор UE.
        bearers: `[(bearer_id, qci, bytes)]` — по одному пакету на bearer.

    LayeredBufferManager маршрутизирует пакет по packet.bearer_id
    (BS_MODULE.py :1407), поэтому bearer_id обязателен.
    """
    bm = ctx["bs"].buffer_manager
    bearers_info = UeBearersInfo(
        ue_id=ue_id,
        num_bearers=len(bearers),
        active_bearers=len(bearers),
        bearers={
            bid: BearerInfo(bearer_id=bid, model="Poisson", qci=int(qci),
                            gbr=None, mbr=None, enabled=True)
            for bid, qci, _ in bearers
        },
    )
    bm.create_ue_buffer(ue_id, ctx["bs"].per_ue_max, bearers_info)
    for bid, qci, nbytes in bearers:
        bm.add_packet(
            ue_id,
            Packet(size=nbytes, ue_id=ue_id, creation_time=0,
                   qci=QCI(int(qci)), bearer_id=bid),
        )


# ── Сбор состояния (снимок) ──────────────────────────────────────────────

def grid_state(grid: RES_GRID_LTE, tti: int) -> Dict[int, Dict[str, int]]:
    """Владельцы RB в обоих слотах TTI: `{slot_id: {freq_idx: ue_id}}`.

    GET_TTI_STATUS на 6c0aae2 вызывает GET_RB(tti, freq_idx) при сигнатуре
    GET_RB(tti, slot_id, freq_idx) и неприменим; снимок строится напрямую
    из rb_map (внутренний формат зафиксирован на :334 _init_rb_map).
    """
    owners = {}
    for (t, slot_id, freq_idx), rb in grid.rb_map.items():
        if t == tti and rb.UE_ID is not None:
            owners.setdefault(slot_id, {})[str(freq_idx)] = rb.UE_ID
    return {slot: {k: m[k] for k in sorted(m, key=int)}
            for slot, m in sorted(owners.items())}


def _rbg_choice_table(grid: RES_GRID_LTE, tti: int,
                      rbg_size: int) -> Dict[str, int]:
    """Таблица per-RBG выборов (раздел 9.2): rbg_idx -> UE_ID (слот 0)."""
    slot_id = None
    for (t, s, _f) in grid.rb_map:
        if t == tti:
            slot_id = s
            break
    if slot_id is None:
        return {}
    table = {}
    for (t, s, freq_idx), rb in grid.rb_map.items():
        if t == tti and s == slot_id and rb.UE_ID is not None:
            table.setdefault(freq_idx // rbg_size, rb.UE_ID)
    return {str(k): table[k] for k in sorted(table)}


def _scheduler_extra_state(sched: Any) -> Dict[str, Any]:
    """Долгоживущее состояние планировщика, наблюдаемое между TTI.

    Собираются только те атрибуты, что реально есть у экземпляра: offsets
    ротации (RR/FD_FGS), параметры QoS-метрики, `allocation_flow`
    (накопленные биты потоков). Отсутствие атрибута — не ошибка: у восьми
    алгоритмов наборы полей разные (INVENTORY-M1.md §5).
    """
    extra = {}
    for attr in ("rr_ue_offset", "rr_rbg_offset", "grouping_parameter",
                 "metric_mode"):
        if hasattr(sched, attr):
            extra[attr] = getattr(sched, attr)
    allocation_flow = getattr(sched, "allocation_flow", None)
    if allocation_flow:
        extra["allocation_flow"] = {
            f"{int(flow[0])}:{int(flow[1])}": int(bits)
            for flow, bits in sorted(allocation_flow.items())
        }
    return extra


def _qci_tracker_state(ctx: Dict[str, Any],
                       ue_ids: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Снимок QciThroughputTracker: current/average throughput по (UE, QCI).

    В Simple-режиме трекера нет — возвращается пустой словарь. Доступ к
    защищённому `_stats` осознанный: публичного метода «отдать всё состояние»
    у трекера нет, а снимок обязан ловить изменение любого накопленного
    значения (иначе регрессия PF/QoS-метрики пройдёт незамеченной).
    """
    bm = ctx["bs"].buffer_manager
    tracker = getattr(bm, "qci_tput_tracker", None)
    stats = getattr(tracker, "_stats", {}) if tracker is not None else {}
    out = {}
    for ue_id in ue_ids:
        per_qci = {}
        for qci, st in sorted(stats.get(ue_id, {}).items()):
            per_qci[str(int(qci))] = {
                "current_throughput": st.current_throughput,
                "average_throughput": st.average_throughput,
            }
        if per_qci:
            out[str(ue_id)] = per_qci
    return out


def capture_tti(ctx: Dict[str, Any], tti: int, result: Dict[str, Any],
                users: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Один снимок TTI: всё наблюдаемое состояние после schedule().

    Снимок строится только из наблюдаемых результатов: возвращаемый
    `schedule()` словарь, состояние сетки, буферов, трекеров и словарей
    `users`. Внутренние локальные переменные алгоритмов не снимаются — их
    изменение без изменения наблюдаемого результата дефектом не считается.
    """
    sched, grid = ctx["sched"], ctx["grid"]
    rbg_size = grid.GET_RBG_SIZE()

    state = {
        "tti": tti,
        "allocation": {str(k): list(v) for k, v in sorted(result["allocation"].items())},
        "pdcch_stats": _normalize_pdcch(result.get("pdcch_stats") or {}),
        "bitmap": {str(k): list(v) for k, v in sorted(result.get("bitmap", {}).items())},
        "statistics_keys": sorted(result.get("statistics", {}).keys()),
        "last_ue_transmitted_bits": {
            str(k): int(v)
            for k, v in sorted(getattr(sched, "last_ue_transmitted_bits",
                                       {}).items())
        },
        "priority_list": [
            [int(u["UE_ID"]), u.get("priority")]
            for u in (getattr(sched, "_last_priority_list", None) or [])
        ],
        "ues": [
            {
                "ue_id": int(u["UE_ID"]),
                "average_throughput": u["ue"].average_throughput,
                "current_dl_throughput": u["ue"].current_dl_throughput,
                "cqi_map_wb": (sched.cqi_map[u["UE_ID"]].wb_cqi
                               if u["UE_ID"] in sched.cqi_map else None),
                "cqi_map_sb": (list(sched.cqi_map[u["UE_ID"]].sb_cqi)
                               if u["UE_ID"] in sched.cqi_map else None),
                # Контракт словаря `user` после schedule(): набор ключей +
                # два поля, которые алгоритмы пишут/мутируют поверх этапа 1.
                "user_dict_keys": sorted(u.keys()),
                "bs_buffer_size": u.get("bs_buffer_size"),
                "instant_rate": u.get("instant_rate"),
            }
            for u in sorted(users, key=lambda x: x["UE_ID"])
        ],
        "grid_rb_owners": grid_state(grid, tti),
        "rbg_choice_table": _rbg_choice_table(grid, tti, rbg_size),
        "buffer_status": {
            str(ue_id): [
                {
                    "buffer_size": int(bs_.buffer_size),
                    "lcid": bs_.lcid,
                    "qci": (int(bs_.qci) if bs_.qci is not None else None),
                    "hol_delay": bs_.hol_delay,
                }
                for bs_ in sorted(grid.bs.buffer_manager.get_buffer_status(ue_id),
                                  key=lambda b: (b.lcid if b.lcid is not None else -1,
                                                 b.qci if b.qci is not None else -1))
            ]
            for ue_id in sorted(u["UE_ID"] for u in users)
        },
        "scheduler_extra": _scheduler_extra_state(sched),
        "qci_tracker": _qci_tracker_state(ctx, sorted(u["UE_ID"] for u in users)),
    }
    return json.loads(json.dumps(state))  # нормализация типов в JSON-совместимые


def _normalize_pdcch(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Привести ключи статистики PDCCH к строкам (JSON не хранит int-ключи).

    Без нормализации `json.dumps` превращает `{1: ...}` в `{"1": ...}`, и
    снимок перестал бы отличаться от проверяемого значения типом ключа.
    """
    out = {}
    for k, v in stats.items():
        if k == "pdcch_ue_cce_allocations" and isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in sorted(v.items())}
        else:
            out[k] = v
    return out


# ── Сценарии ─────────────────────────────────────────────────────────────
# Каждый билдер возвращает {"meta": {...}, "ttis": [capture_tti, ...]}.

def _run_ttis(ctx: Dict[str, Any], users: List[Dict[str, Any]],
              ttis: List[int]) -> List[Dict[str, Any]]:
    """Прогнать TTI по порядку и снять состояние после каждого.

    `GLOBALS.CURRENT_TIME` выставляется явно перед каждым вызовом: буферный
    менеджер и метрики читают глобальное время (BS_MODULE.py :194, :443),
    поэтому без фиксации снимок зависел бы от порядка запуска тестов.
    """
    captures = []
    for tti in ttis:
        GLOBALS.CURRENT_TIME = tti
        result = ctx["sched"].schedule(tti, users)
        captures.append(capture_tti(ctx, tti, result, users))
    return captures


def _profile_meta(profile: List[Tuple[int, float, int, List[int], int]]
                  ) -> List[Dict[str, Any]]:
    """Профиль UE в meta снимка: по нему сценарий воспроизводится без кода."""
    return [
        {"ue_id": ue_id, "wb_cqi": cqi, "sb_cqi": list(sb), "buffer_bytes": buf}
        for ue_id, _x, cqi, sb, buf in profile
    ]


def build_simple_profile(algorithm: str, profile: List[Tuple], ttis: List[int],
                         scheduler_kwargs: Optional[Dict[str, Any]] = None,
                         scenario_name: Optional[str] = None,
                         extra_meta: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """Сценарий на SimpleBufferManager: профиль UE + список TTI.

    Returns:
        `{"meta": {...}, "ttis": [capture_tti, ...]}` — содержимое файла
        baseline-снимка.
    """
    ctx = new_ctx(algorithm, scheduler_kwargs=scheduler_kwargs)
    ues = [make_ue_with_cqi(ue_id, x, cqi, sb) for ue_id, x, cqi, sb, _ in profile]
    for (ue_id, _x, _cqi, _sb, buf), _ue in zip(profile, ues, strict=True):
        fill_simple_buffer(ctx, ue_id, [(buf, None)])
    users = build_users(ues)
    meta = {
        "scenario": scenario_name or algorithm,
        "algorithm": algorithm,
        "dev_revision": DEV_REVISION,
        "bandwidth_mhz": 10,
        "num_frames": 2,
        "scheduler_kwargs": scheduler_kwargs or {},
        "buffer_manager": "SimpleBufferManager (use_simple_buffer=True)",
        "ue_profile": _profile_meta(profile),
        "ttis": list(ttis),
        "seed": None,
        "rng_note": "детерминированный сценарий, ГСЧ не используется",
        "env": env_info(),
    }
    if extra_meta:
        meta.update(extra_meta)
    return {"meta": meta, "ttis": _run_ttis(ctx, users, ttis)}


def build_matrix(algorithm: str,
                 flow_profile: Optional[str] = None) -> Dict[str, Any]:
    """Ячейка WB-матрицы (раздел 9).

    Шесть «классических» алгоритмов — SimpleBufferManager, TTI 5..7.
    BasicQosAware/QosAware требуют LayeredBufferManager (per-QCI буферы и
    hol_delay; на SimpleBufferManager hol_delay=None и :2790/:3028 падают —
    ограничение AS-IS), поэтому их WB-ячейки строятся на layered-буфере.
    """
    if algorithm not in ("BasicQosAware", "QosAware"):
        return build_simple_profile(algorithm, WB_UE_PROFILE, MULTI_TTI,
                                    scenario_name=f"wb_{algorithm}")

    flow_profile = flow_profile or (
        "single_bearer" if algorithm == "BasicQosAware" else "multiflow")
    ctx = new_ctx(algorithm)
    bs = BaseStation(bandwidth=10, use_simple_buffer=False)
    grid = RES_GRID_LTE(bandwidth=10, num_frames=2)
    grid.SET_BS(bs)
    ctx["bs"], ctx["grid"] = bs, grid
    ctx["sched"] = SchedulerInterface.create(algorithm, grid, bs)

    if flow_profile == "single_bearer":
        flows = QOS_SINGLE_BEARER
        ues = []
        for ue_id in sorted(flows):
            qci, nbytes, cqi = flows[ue_id]
            ues.append(make_ue_with_cqi(ue_id, 100.0 * ue_id, cqi, []))
            fill_layered_buffer(ctx, ue_id, [(1, qci, nbytes)])
        flows_meta = {"flows": {str(k): list(v) for k, v in sorted(flows.items())}}
    else:
        flows = QOS_MULTIFLOW
        ues = []
        for ue_id in sorted(flows):
            bearers, cqi = flows[ue_id]
            ues.append(make_ue_with_cqi(ue_id, 100.0 * ue_id, cqi, []))
            fill_layered_buffer(ctx, ue_id, bearers)
        flows_meta = {"flows": {
            str(k): [list(b) for b in v[0]] + [v[1]] for k, v in sorted(flows.items())}}

    users = build_users(ues)
    meta = {
        "scenario": f"wb_{algorithm}",
        "algorithm": algorithm,
        "dev_revision": DEV_REVISION,
        "bandwidth_mhz": 10,
        "num_frames": 2,
        "scheduler_kwargs": {},
        "buffer_manager": "LayeredBufferManager (use_simple_buffer=False)",
        "flow_profile": flow_profile,
        "ttis": MULTI_TTI,
        "seed": None,
        "rng_note": "детерминированный сценарий, ГСЧ не используется",
        "note": ("QoS-алгоритмы на SimpleBufferManager неработоспособны "
                 "(hol_delay=None → TypeError :2790 / ValueError :3029) — "
                 "WB-ячейка матрицы на layered-буфере (BUG-SCH-012)"),
        "env": env_info(),
    }
    meta.update(flows_meta)
    return {"meta": meta, "ttis": _run_ttis(ctx, users, MULTI_TTI)}


def build_golden() -> Dict[str, Any]:
    """Золотой сценарий README §4: BestCQI, один TTI, три UE."""
    out = build_simple_profile("BestCQI", GOLDEN_UE_PROFILE, [5],
                               scenario_name="golden_tti5")
    # Эталонные значения README §4 (не из снимка): test_golden_tti5.py
    # сверяет с ними и снимок, и фактический прогон — независимая проверка.
    out["meta"]["expected_readme"] = {
        "allocation": {"1": list(range(6, 18)), "2": [], "3": list(range(0, 6))},
        "pdcch_ue_cce_allocations": {"3": 1, "1": 1},
        "last_ue_transmitted_bits": {"1": 8004, "2": 0, "3": 2404},
        "average_throughput": {"1": 16008.0, "2": 0.0, "3": 4808.0},
    }
    return out


def build_fd_sb(algorithm: str, sb_mode: str) -> Dict[str, Any]:
    """FD-сценарий с одним из трёх режимов subband-CQI (раздел 9.2).

    Args:
        algorithm: один из `FD_ALGORITHMS` (иначе subband-CQI не потребляется).
        sb_mode: ключ `SB_MODES` — `full` (SB есть у всех UE), `missing`
            (SB пустые, проверка fallback на WB), `partial` (короткий список
            SB с нулями внутри).
    """
    if sb_mode not in SB_MODES:
        raise ValueError(
            f"Недопустимый sb_mode '{sb_mode}'. "
            f"Допустимые значения: {', '.join(SB_MODES)}")
    sb_map = {"full": SB_FULL, "missing": SB_MISSING, "partial": SB_PARTIAL}[sb_mode]
    profile = [
        (ue_id, 100.0 * ue_id, SB_WB_CQI[ue_id], sb_map[ue_id], SB_BUFFERS[ue_id])
        for ue_id in sorted(sb_map)
    ]
    return build_simple_profile(
        algorithm, profile, MULTI_TTI,
        scenario_name=f"fd_sb_{sb_mode}_{algorithm}",
        extra_meta={"sb_mode": sb_mode})


def build_qos(algorithm):
    """QoS-сценарий раздела 9.4: grants + per-QCI stats, TTI 5..7.

    BasicQosAware — один bearer на UE (контракт класса), QosAware — несколько
    потоков (ue_id, qci). Обе версии на LayeredBufferManager.
    """
    flow_profile = "single_bearer" if algorithm == "BasicQosAware" else "multiflow"
    out = build_matrix(algorithm, flow_profile=flow_profile)
    out["meta"]["scenario"] = (
        "qos_basic_single_bearer" if algorithm == "BasicQosAware"
        else "qos_aware_multiflow")
    return out


def build_conflict(kind: str) -> Dict[str, Any]:
    """Сценарий конкуренции за ресурсы (раздел 9.5).

    Args:
        kind: `rbg_contended` — UE претендуют на одни и те же RBG;
            `pdcch_limited` — не хватает CCE (`max_dl_cce_allowance=1`),
            часть UE остаётся без гранта.
    """
    if kind not in ("rbg_contended", "pdcch_limited"):
        raise ValueError(
            f"Недопустимый kind '{kind}'. "
            f"Допустимые значения: rbg_contended, pdcch_limited")
    if kind == "rbg_contended":
        kwargs, name = {}, "conflict_rbg_contended"
    else:
        kwargs, name = {"max_dl_cce_allowance": 1}, "conflict_pdcch_limited"
    out = build_simple_profile("FD_BCQI", CONFLICT_UE_PROFILE, [5],
                               scheduler_kwargs=kwargs, scenario_name=name)
    out["meta"]["conflict_kind"] = kind
    out["meta"]["expected_invariant"] = (
        "каждый freq_idx в grid_rb_owners принадлежит не более чем одному UE "
        "(повторной выдачи нет, поведение AS-IS 6c0aae2; собственный rollback "
        "аллокатора отсутствует — BUG-SCH-009)"
    )
    return out


def build_rr_multitti() -> Dict[str, Any]:
    """Ротация RR на четырёх TTI: снимает offsets и порядок выдачи (раздел 9.3)."""
    return build_simple_profile("RoundRobin", WB_UE_PROFILE, RR_TTI,
                                scenario_name="rr_multitti")


# Реестр сценариев: имя файла снимка -> билдер. Единственный источник имён
# (generate_baselines.py и тесты берут ключи отсюда), поэтому расхождение
# «список сценариев vs список файлов» исключено.
#
# default-аргументы в лямбдах (`a=_alg`) обязательны: без них все лямбды
# замкнули бы одну и ту же переменную цикла и строили последний алгоритм.
SCENARIOS = {"golden_tti5": build_golden}
for _alg in ALL_ALGORITHMS:
    SCENARIOS[f"wb_{_alg}"] = (lambda a=_alg: build_matrix(a))
for _mode in SB_MODES:
    for _alg in FD_ALGORITHMS:
        SCENARIOS[f"fd_sb_{_mode}_{_alg}"] = (
            lambda m=_mode, a=_alg: build_fd_sb(a, m))
SCENARIOS["qos_basic_single_bearer"] = lambda: build_qos("BasicQosAware")
SCENARIOS["qos_aware_multiflow"] = lambda: build_qos("QosAware")
SCENARIOS["conflict_rbg_contended"] = lambda: build_conflict("rbg_contended")
SCENARIOS["conflict_pdcch_limited"] = lambda: build_conflict("pdcch_limited")
SCENARIOS["rr_multitti"] = build_rr_multitti


def build_all() -> Dict[str, Dict[str, Any]]:
    """Построить снимки всех сценариев реестра (используется generate_baselines)."""
    return {name: builder() for name, builder in SCENARIOS.items()}
