"""Матрица покрытия M0: 8 алгоритмов x режимы CSI (SPEC-FEAT-SCH-008 §9).

- wb_<alg> — WB-сценарий для всех восьми алгоритмов, TTI 5..7;
- fd_sb_<mode>_<alg> — SB / SB-missing->WB / частичный SB для FD_BCQI,
  FD_PF, FD_FGS (таблица per-RBG выборов фиксируется в снимке);
- rr_multitti — ротация offsets, TTI 5..8 (отдельный файл test_rr_multitti.py);
- qos_* — grants + per-QCI stats (test_qos.py);
- conflict_* — сеточный конфликт (test_conflict.py).

CQI=0 через schedule() на 6c0aae2 недостижим (фильтр eligibility :592 +
continue :919) — кейс будет покрыт unit-тестами стратегий в M1+ (BUG-SCH-007).
"""
from pathlib import Path

import pytest
from conftest import assert_matches_baseline, load_snapshot
from SCHEDULER import SchedulerInterface

WB_ALGORITHMS = [
    "BestCQI", "ProportionalFair", "RoundRobin",
    "FD_BCQI", "FD_FGS", "FD_PF",
    "BasicQosAware", "QosAware",
]
FD_ALGORITHMS = ["FD_BCQI", "FD_FGS", "FD_PF"]
SB_MODES = ["full", "missing", "partial"]


@pytest.mark.parametrize("alg", WB_ALGORITHMS)
def test_wb_matrix_matches_baseline(alg):
    assert_matches_baseline(f"wb_{alg}")


@pytest.mark.parametrize("mode", SB_MODES)
@pytest.mark.parametrize("alg", FD_ALGORITHMS)
def test_fd_sb_matrix_matches_baseline(alg, mode):
    assert_matches_baseline(f"fd_sb_{mode}_{alg}")


def test_all_baselines_pinned_to_dev_revision():
    """Каждый снимок несёт ревизию dev, seed/config — фиксируемы (раздел 9)."""
    import baseline_scenarios as bsc
    for name in sorted(bsc.SCENARIOS):
        meta = load_snapshot(name)["meta"]
        assert meta["dev_revision"] == "6c0aae2", name
        assert "seed" in meta and "ttis" in meta, name
    # full-SIM контрольные точки (RUNBOOK §7) — та же фиксация ревизии:
    # gate 8.1 требует meta.dev_revision у каждого снимка при регенерации.
    sim_names = sorted(p.stem for p in bsc.BASELINE_DIR.glob("sim*.json"))
    assert sim_names, "нет full-SIM снимков в baselines/"
    for name in sim_names:
        meta = load_snapshot(name)["meta"]
        assert meta["dev_revision"] == "6c0aae2", name
        assert meta.get("seed") == 42, name


@pytest.mark.parametrize("alg", FD_ALGORITHMS)
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


@pytest.mark.parametrize("alg", FD_ALGORITHMS)
def test_fd_sb_missing_falls_back_to_wb(alg):
    """Sanity: без SB CQI выборы определяются WB CQI (UE2=12 > UE1=10 > UE3=8)."""
    snap = load_snapshot(f"fd_sb_missing_{alg}")
    table = snap["ttis"][0]["rbg_choice_table"]
    assert table, "пустая таблица per-RBG выборов"
    assert table["0"] == 2, f"первый RBG должен уйти UE с лучшим WB CQI: {table}"


def test_wb_matrix_covers_all_factory_algorithms():
    """WB-ячейка существует для каждого ключа фабрики create().

    Ключи извлекаются из AST SCHEDULER.py (dict `schedulers` в create(),
    :255-264). available_algorithms() (:284-292) с ними расходится —
    BUG-SCH-010, эталон именно create(). Тест ловит тихое расширение
    фабрики без обновления матрицы покрытия.
    """
    import ast

    import baseline_scenarios as bsc
    import SCHEDULER

    tree = ast.parse(Path(SCHEDULER.__file__).read_text(encoding="utf-8"))
    factory_keys = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "create":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign)
                        and any(getattr(t, "id", None) == "schedulers"
                                for t in sub.targets)
                        and isinstance(sub.value, ast.Dict)):
                    factory_keys = [k.value for k in sub.value.keys]
    assert factory_keys, "не найден реестр `schedulers` в create()"
    assert sorted(factory_keys) == sorted(bsc.ALL_ALGORITHMS), (
        f"фабрика и ALL_ALGORITHMS разошлись: {sorted(factory_keys)}")
    for key in factory_keys:
        assert f"wb_{key}" in bsc.SCENARIOS, key
    with pytest.raises(ValueError):
        SchedulerInterface.create("NoSuchAlgorithm", None, None)
