# `drl`

Пакет для DRL-логики внутри `PyScheduler`.

Главная идея:
- `SimulationManager` остается оркестратором симуляции;
- baseline scheduler-ы продолжают жить в общем pipeline;
- вся DRL-специфика вынесена в отдельный слой `PyScheduler/drl`.

## Структура

Корень пакета:
- `simulation_bridge.py` — мост между runtime симулятора и DRL-слоем;
- `playground_adapter.py` — сборка observation/action-mask в формате `drl_playground`;
- `dqn_scheduler.py` — inference-only scheduler `DqnScheduler`;
- `dqn_model_runner.py` — рантайм-обертка для загрузки и вызова DQN модели.

Подпакеты:
- `agents/` — локальные реализации агентов, перенесенные из `drl_playground`;
- `envs/` — локальные LTE-среды для обучения и анализа;
- `scripts/` — скрипты обучения, оценки и визуализации.

## Как передавать параметры алгоритма

`SimulationManager` больше не хранит `dqn_*` поля в общем dataclass планировщика.
Теперь алгоритм-специфичные параметры передаются через `algorithm_kwargs`.

Пример:

```python
manager.set_scheduler(
    algorithm="DqnScheduler",
    max_dl_ue_tti=16,
    algorithm_kwargs={
        "dqn_model_path": "runs/lte_dqn/lte_dqn_shared_q.pt",
        "dqn_max_n_ue": 40,
        "dqn_wb_cqi_report_period_tti": 5,
        "dqn_deterministic": True,
    },
)
```

Это позволяет позже добавлять `PpoScheduler` и другие DRL-алгоритмы без раздувания
унифицированного интерфейса `SimulationManager`.

## Перенесенный playground-стек

Из `drl_playground` внутрь репозитория перенесены:
- LTE DQN агент;
- PPO агент;
- LTE scheduler env;
- padded LTE env;
- скрипт обучения LTE DQN;
- скрипт анимации scheduler-а.

Сейчас это локальный стек внутри репозитория, а не внешний импорт из `drl_playground`.

## Зависимости

Для inference-path внутри `DqnScheduler` требуется `torch`.

Для playground-сред и скриптов дополнительно требуется:
- `gymnasium`
- `matplotlib` для анимации

В текущем окружении эти зависимости могут отсутствовать. Поэтому тесты на перенесенные
`agents/envs` сделаны как optional smoke-тесты через `pytest.importorskip(...)`.
