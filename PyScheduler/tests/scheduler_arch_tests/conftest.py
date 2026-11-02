"""Фикстуры архитектурных тестов M1 (SPEC-FEAT-SCH-008, шаг M1, раздел 5.7).

Каталог отделён от `tests/baseline/` намеренно: baseline — characterization
(«что делает legacy-код сейчас», снимки dev 6c0aae2), а этот набор — контрактный
(«какие интерфейсы введены и что они гарантируют»). Смешивать их нельзя
(правило R4: одна ветка — одна цель).

Построители контекстов — в `arch_support.py` (не здесь): имя модуля `conftest`
в дереве `PyScheduler/tests/` не уникально, и `from conftest import ...` при
сборке нескольких каталогов одним запуском разрешается в чужой файл.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PY_SCHEDULER = HERE.parents[1]
BASELINE_DIR = PY_SCHEDULER / "tests" / "baseline"
for path in (str(HERE), str(BASELINE_DIR), str(PY_SCHEDULER)):
    if path not in sys.path:
        sys.path.insert(0, path)

import baseline_scenarios as bsc  # noqa: E402
from arch_support import (  # noqa: E402
    FD_UE_PROFILE,
    ctx_with_users,
    layered_ctx_for,
    wb_ctx_for,
)


@pytest.fixture
def algorithms():
    """Все восемь реализаций планировщика (порядок — как в ALL_ALGORITHMS)."""
    return list(bsc.ALL_ALGORITHMS)


@pytest.fixture
def wb_ctx():
    """Контекст с WB-профилем и SimpleBufferManager (use_simple_buffer=True)."""
    return wb_ctx_for("BestCQI")


@pytest.fixture
def fd_ctx():
    """Контекст с subband CQI (FD) — проверяет `CqiView.get_sb`."""
    return ctx_with_users(bsc._new_ctx("FD_BCQI"), FD_UE_PROFILE)


@pytest.fixture
def layered_ctx():
    """Контекст с LayeredBufferManager: per-QCI потоки + QciThroughputTracker."""
    return layered_ctx_for("QosAware")
