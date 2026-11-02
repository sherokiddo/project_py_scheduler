"""Матрица покрытия M0: 8 алгоритмов x режимы CSI (SPEC-FEAT-SCH-008 §9).

- wb_<alg> — WB-сценарий для всех восьми алгоритмов, TTI 5..7;
- fd_sb_<mode>_<alg> — SB / SB-missing->WB / частичный SB для FD_BCQI,
  FD_PF, FD_FGS (таблица per-RBG выборов фиксируется в снимке);
- rr_multitti — ротация offsets, TTI 5..8 (отдельный файл test_rr_multitti.py);
- qos_* — grants + per-QCI stats (test_qos.py);
- conflict_* — сеточный конфликт (test_conflict.py).

Списки алгоритмов и режимов SB не дублируются здесь: единственный источник —
`baseline_scenarios` (`ALL_ALGORITHMS`, `FD_ALGORITHMS`, `SB_MODES`), из него же
строится реестр `SCENARIOS`. Дублирование реестров в проекте уже приводило к
рассинхрону (BUG-SCH-010, `available_algorithms()` против `create()`).

CQI=0 через schedule() на 6c0aae2 недостижим (фильтр eligibility :592 +
continue :919) — кейс будет покрыт unit-тестами стратегий в M1+ (BUG-SCH-007).
"""
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import baseline_scenarios as bsc
import pytest
import SCHEDULER
from baseline_harness import assert_matches_baseline, load_snapshot
from SCHEDULER import SchedulerInterface

# Поля схемы снимка 2 (2026-11-01): контракт словаря `user` после schedule().
USER_CONTRACT_FIELDS = ("user_dict_keys", "bs_buffer_size", "instant_rate")


def _factory_registry() -> Optional[List[str]]:
    """Ключи реестра планировщиков внутри `SchedulerInterface.create()`.

    Извлечение по AST, а не вызовом `available_algorithms()`: последний
    расходится с фактическим реестром (BUG-SCH-010) и как источник имён
    непригоден. Поиск не привязан к имени локальной переменной — находится
    любой dict-литерал в `create()`, у которого все ключи строковые, а все
    значения — имена классов; иначе переименование `schedulers` в другое имя
    молчаливо лишило бы матрицу контроля.
    """
    tree = ast.parse(Path(SCHEDULER.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "create"):
            continue
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Dict)):
                continue
            keys, values = sub.value.keys, sub.value.values
            if keys and all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                            for k in keys) and all(isinstance(v, ast.Name)
                                                   for v in values):
                return [k.value for k in keys]
    return None


@pytest.mark.parametrize("alg", bsc.ALL_ALGORITHMS)
def test_wb_matrix_matches_baseline(alg):
    """WB-ячейка матрицы: прогон совпадает со снимком dev 6c0aae2."""
    assert_matches_baseline(f"wb_{alg}")


@pytest.mark.parametrize("mode", bsc.SB_MODES)
@pytest.mark.parametrize("alg", bsc.FD_ALGORITHMS)
def test_fd_sb_matrix_matches_baseline(alg, mode):
    """FD-ячейка матрицы (режим subband-CQI x алгоритм) совпадает со снимком."""
    assert_matches_baseline(f"fd_sb_{mode}_{alg}")


def test_all_baselines_pinned_to_dev_revision():
    """Каждый снимок несёт ревизию dev, seed/config — фиксируемы (раздел 9)."""
    for name in sorted(bsc.SCENARIOS):
        meta = load_snapshot(name)["meta"]
        assert meta["dev_revision"] == bsc.DEV_REVISION, name
        assert "seed" in meta and "ttis" in meta, name
    # full-SIM контрольные точки (RUNBOOK §7) — та же фиксация ревизии:
    # gate 8.1 требует meta.dev_revision у каждого снимка при регенерации.
    sim_names = sorted(p.stem for p in bsc.BASELINE_DIR.glob("sim*.json"))
    assert sim_names, "нет full-SIM снимков в baselines/"
    for name in sim_names:
        meta = load_snapshot(name)["meta"]
        assert meta["dev_revision"] == bsc.DEV_REVISION, name
        assert meta.get("seed") == bsc.SIM_SEED, name


def test_all_snapshots_carry_user_contract_fields():
    """Схема снимка 2: контракт словаря `user` зафиксирован в каждом снимке.

    `bs_buffer_size` (этап 1, :593; мутируется RR-аллокатором :2086; читается
    :470 и :739) и `instant_rate` (PF :2143, FD_PF :2614) наблюдаемы после
    schedule(), но в схему 1 не входили — пробел закрыт 2026-11-01 (RUNBOOK
    §6.1). `user_dict_keys` добавлен, чтобы появление/исчезновение ЛЮБОГО поля
    словаря меняло снимок, а не проходило молчаливо.
    """
    checked = 0
    for name in sorted(bsc.SCENARIOS):
        for tti in load_snapshot(name)["ttis"]:
            for ue in tti["ues"]:
                for field in USER_CONTRACT_FIELDS:
                    assert field in ue, f"{name}/tti{tti['tti']}: нет {field}"
                assert "UE_ID" in ue["user_dict_keys"], f"{name}: ключи user"
                assert "bs_buffer_size" in ue["user_dict_keys"], name
                assert isinstance(ue["bs_buffer_size"], int), (
                    f"{name}: bs_buffer_size = {ue['bs_buffer_size']!r}")
                checked += 1
    assert checked > 0, "в снимках нет ни одного UE"


@pytest.mark.parametrize("alg", bsc.FD_ALGORITHMS)
def test_fd_sb_full_choice_table_is_contrast(alg):
    """Sanity: в SB-сценарии выборы чередуются между UE1 и UE2 (контрастные CQI).

    Если таблица выборов вырождается — сценарий перестал различать SB-ветку,
    baseline нужно пересмотреть вместе с кодом.
    """
    snap = load_snapshot(f"fd_sb_full_{alg}")
    table = snap["ttis"][0]["rbg_choice_table"]
    assert table, "пустая таблица per-RBG выборов"
    chosen = set(table.values())
    assert {1, 2} <= chosen, f"контрастные UE не выбраны: {table}"


@pytest.mark.parametrize("alg", bsc.FD_ALGORITHMS)
def test_fd_sb_missing_falls_back_to_wb(alg):
    """Sanity: без SB CQI выборы определяются WB CQI (UE2=12 > UE1=10 > UE3=8)."""
    snap = load_snapshot(f"fd_sb_missing_{alg}")
    table = snap["ttis"][0]["rbg_choice_table"]
    assert table, "пустая таблица per-RBG выборов"
    assert table["0"] == 2, f"первый RBG должен уйти UE с лучшим WB CQI: {table}"


def test_wb_matrix_covers_all_factory_algorithms():
    """WB-ячейка существует для каждого ключа фабрики create().

    Эталон имён — реестр внутри `create()` (:255-264), а не
    `available_algorithms()` (:284-292), который с ним расходится
    (BUG-SCH-010). Тест ловит тихое расширение фабрики без обновления матрицы
    покрытия: новый алгоритм обязан получить хотя бы WB-снимок.
    """
    factory_keys = _factory_registry()
    assert factory_keys, "реестр планировщиков в create() не найден"
    assert sorted(factory_keys) == sorted(bsc.ALL_ALGORITHMS), (
        f"фабрика и ALL_ALGORITHMS разошлись: {sorted(factory_keys)}")
    for key in factory_keys:
        assert f"wb_{key}" in bsc.SCENARIOS, key
    # None для grid/bs безопасен: create() проверяет имя до конструирования
    # класса (:266-269), поэтому вызов падает на валидации, а не на аргументах.
    with pytest.raises(ValueError, match="Unknown algorithm"):
        SchedulerInterface.create("NoSuchAlgorithm", None, None)


def test_snapshot_meta_is_reproducible_from_code() -> None:
    """meta снимка описывает сценарий полностью: по ней он воспроизводится.

    Проверяется на одном снимке, что в meta есть все поля, нужные для
    пересборки без чтения кода (алгоритм, полоса, профиль UE, TTI, тип
    буферного менеджера, ревизия dev).
    """
    meta: Dict[str, Any] = load_snapshot("wb_RoundRobin")["meta"]
    required = ("scenario", "algorithm", "dev_revision", "bandwidth_mhz",
                "num_frames", "scheduler_kwargs", "buffer_manager",
                "ue_profile", "ttis", "seed", "rng_note")
    missing = [k for k in required if k not in meta]
    assert not missing, f"в meta не хватает полей: {missing}"
    assert meta["algorithm"] == "RoundRobin"
    assert [u["ue_id"] for u in meta["ue_profile"]] == [1, 2, 3, 4]
