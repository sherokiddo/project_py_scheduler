"""
Пакет обучающих сред DRL для PyScheduler.

Здесь сосуществуют два класса сред:
- `pyscheduler_lte_env.py` — основная simulation-backed env на реальном runtime;
- `pyscheduler_lte_ranker_env.py` — simulation-backed env для TTI-level ranking;
- `lte_scheduler_env.py` и `lte_padded_env.py` — legacy standalone playground-path.

Импорт конкретных сред выполняется явно из соответствующих модулей, чтобы
не требовать `gymnasium` при простом импорте пакета.
"""
