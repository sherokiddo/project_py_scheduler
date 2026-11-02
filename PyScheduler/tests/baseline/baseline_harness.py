"""Хелперы сравнения baseline-снимков (M0, SPEC-FEAT-SCH-008 раздел 8.2).

Вынесены из `conftest.py` в отдельный модуль в шаге M1: имя модуля `conftest`
не уникально — в дереве `PyScheduler/tests/` четыре `conftest.py`
(`baseline/`, `scheduler_arch_tests/`, `layered_buffer_tests/`,
`scheduler_tests/`), и при сборке нескольких каталогов одним запуском
`from conftest import ...` разрешается в чужой файл:

    pytest tests/baseline tests/scheduler_arch_tests
    ImportError: cannot import name 'assert_matches_baseline' from 'conftest'
    (.../tests/scheduler_arch_tests/conftest.py)

Импорт из `baseline_harness` от порядка каталогов не зависит. `conftest.py`
эти имена НЕ переэкспортирует: тесты импортируют хелперы напрямую, а
переэкспорт «для совместимости» был бы мёртвым кодом.
"""
import json
from typing import Any, Dict

import baseline_scenarios as bsc
import pytest
from baseline_compare import compare


def load_snapshot(name: str) -> Dict[str, Any]:
    """Прочитать зафиксированный снимок `baselines/<name>.json`.

    Отсутствие файла — не ошибка импорта, а внятный провал с командой
    регенерации: снимок мог не попасть в checkout, и это надо отличать от
    расхождения поведения.
    """
    path = bsc.BASELINE_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"Нет baseline-снимка {path.name}. Сгенерируйте: "
            f"python PyScheduler/tests/baseline/generate_baselines.py {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_matches_baseline(name: str) -> None:
    """Пересобрать сценарий и побитово сравнить с baseline-снимком.

    Args:
        name: ключ реестра `baseline_scenarios.SCENARIOS` (он же имя файла).

    Raises:
        AssertionError: с перечнем первых отличий и ревизией dev, на которой
            снимок зафиксирован.
    """
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
