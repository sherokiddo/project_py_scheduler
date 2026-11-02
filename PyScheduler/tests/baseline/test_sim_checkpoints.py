"""Контрольные точки full-SIM прогона (SPEC-FEAT-SCH-008 §8.2, §9.6).

Конфигурация: sim_with_manager_qos (TEST_MODULES.py :234), seed=42,
verbose-лог и StatsManager отключены. Контрольные точки: 100/500/1000 TTI
(QosAware) и 1000 TTI (BestCQI). Сравнение: sha256 истории allocation —
побитово; агрегаты — по правилам раздела 8.2 (дискретные точно, float
rel tol 1e-9); wall-clock не сравнивается.

Тяжёлые прогонные тесты помечены `sim`; быстрый набор —
`pytest PyScheduler/tests/baseline -q -m "not sim"`.
"""
import importlib
import json

import baseline_compare
import pytest
from conftest import load_snapshot

sim_checkpoint = importlib.import_module("sim_checkpoint")

CHECKPOINTS = [
    (100, "QosAware", "sim100_qosaware"),
    (500, "QosAware", "sim500_qosaware"),
    (1000, "QosAware", "sim1000_qosaware"),
    (1000, "BestCQI", "sim1000_bestcqi"),
]


@pytest.mark.sim
@pytest.mark.parametrize("duration,algorithm,name", CHECKPOINTS)
def test_sim_checkpoint_matches_baseline(duration, algorithm, name):
    expected = load_snapshot(name)
    actual = sim_checkpoint.run(duration, seed=42, algorithm=algorithm)

    assert actual["allocation_history_sha256"] == \
        expected["allocation_history_sha256"], (
            f"[{name}] история allocation разошлась с baseline: "
            f"{actual['allocation_history_sha256']} != "
            f"{expected['allocation_history_sha256']}")

    diffs = []
    for key in ("tti_count_recorded", "total_rb_allocated", "buffer_stats",
                "ue_final"):
        diffs.extend(baseline_compare.compare(
            _json_norm(expected[key]), _json_norm(actual[key]), path=key))
    assert not diffs, (
        f"[{name}] агрегаты расходятся с baseline ({len(diffs)} отличий), "
        "первые 20:\n" + "\n".join(diffs[:20]))


def _json_norm(obj):
    """Нормализация к JSON-виду: ключи dict -> str (round-trip снимка)."""
    if isinstance(obj, dict):
        return {str(k): _json_norm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_norm(v) for v in obj]
    return obj


@pytest.mark.sim
def test_sim_determinism_same_seed():
    """Два прогона с одним seed в одном процессе дают идентичный снимок."""
    a = sim_checkpoint.run(30, seed=42, algorithm="QosAware")
    b = sim_checkpoint.run(30, seed=42, algorithm="QosAware")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
