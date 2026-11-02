"""Фикстуры baseline-тестов M0 (SPEC-FEAT-SCH-008, раздел 9).

Каждый тест пересобирает свой сценарий с нуля и сравнивает снимок состояния
с зафиксированным в baselines/*.json на ревизии dev 6c0aae2.
Правила сравнения — baseline_compare.compare (раздел 8.2):
дискретные значения точно, float с относительным допуском 1e-9.

Здесь только настройка `sys.path` и регистрация маркера `sim`: фикстур,
которые нужны тестам, нет — каждый тест пересобирает сценарий сам (функциями
`baseline_scenarios`), а общие хелперы живут в `baseline_harness.py`. Имя
модуля `conftest` в дереве `PyScheduler/tests/` не уникально (четыре каталога),
и `from conftest import ...` при сборке нескольких каталогов одним запуском
разрешается в чужой файл, поэтому тесты импортируют хелперы напрямую.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_SCHEDULER = HERE.parents[1]
for path in (str(HERE), str(PY_SCHEDULER)):
    if path not in sys.path:
        sys.path.insert(0, path)


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
