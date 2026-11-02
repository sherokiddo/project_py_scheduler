"""Фикстуры baseline-тестов M0 (SPEC-FEAT-SCH-008, раздел 9).

Каждый тест пересобирает свой сценарий с нуля и сравнивает снимок состояния
с зафиксированным в baselines/*.json на ревизии dev 6c0aae2.
Правила сравнения — baseline_compare.compare (раздел 8.2):
дискретные значения точно, float с относительным допуском 1e-9.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PY_SCHEDULER = HERE.parents[1]
for path in (str(HERE), str(PY_SCHEDULER)):
    if path not in sys.path:
        sys.path.insert(0, path)

import baseline_scenarios as bsc  # noqa: E402
from baseline_compare import compare  # noqa: E402


def pytest_configure(config):
    """Регистрация маркера sim без корневого pytest.ini.

    Маркер нужен, чтобы отделять тяжёлые full-SIM контрольные точки
    (сотни-тысячи TTI на тест) от быстрого characterization-набора:
    `python -m pytest PyScheduler/tests/baseline -m "not sim"`.
    """
    config.addinivalue_line(
        "markers",
        "sim: тяжёлые full-SIM контрольные точки (сотни-тысячи TTI на тест)",
    )


ALL_SCENARIO_NAMES = sorted(bsc.SCENARIOS)


def load_snapshot(name):
    path = bsc.BASELINE_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"Нет baseline-снимка {path.name}. Сгенерируйте: "
            f"python PyScheduler/tests/baseline/generate_baselines.py {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_matches_baseline(name):
    """Пересобрать сценарий и побитово сравнить с baseline-снимком."""
    expected = load_snapshot(name)
    actual = bsc.SCENARIOS[name]()

    # meta.env — информационная подпись окружения, в сравнении не участвует
    # (RUNBOOK §5): иначе снимок нельзя было бы проверить на другой машине.
    exp_meta = {k: v for k, v in expected["meta"].items() if k != "env"}
    act_meta = {k: v for k, v in actual["meta"].items() if k != "env"}
    diffs = compare(exp_meta, act_meta, path="meta")
    assert not diffs, f"[{name}] meta расходится с baseline:\n" + "\n".join(diffs)

    assert len(expected["ttis"]) == len(actual["ttis"]), (
        f"[{name}] число TTI: baseline {len(expected['ttis'])}, "
        f"факт {len(actual['ttis'])}")

    diffs = []
    for exp_tti, act_tti in zip(expected["ttis"], actual["ttis"], strict=True):
        diffs.extend(compare(exp_tti, act_tti, path=f"tti{exp_tti['tti']}"))
    assert not diffs, (
        f"[{name}] поведение расходится с baseline dev 6c0aae2 "
        f"({len(diffs)} отличий), первые 40:\n" + "\n".join(diffs[:40]))


@pytest.fixture(params=ALL_SCENARIO_NAMES)
def scenario_name(request):
    return request.param
