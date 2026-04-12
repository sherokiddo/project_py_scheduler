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
- `dqn_scheduler.py` — inference-only scheduler `DqnScheduler` для запуска модели внутри реальной симуляции;
- `dqn_model_runner.py` — рантайм-обертка для загрузки и вызова DQN-модели.

Подпакеты:
- `agents/` — агенты DRL;
- `envs/` — обучающие среды;
- `scripts/` — сценарии обучения, оценки и визуализации.

## Что где живет

### `agents/`

В этой папке должны жить все агенты:
- `lte_dqn_agent.py` — основной LTE DQN агент;
- `ppo_agent.py` — PPO агент.

Файл `dqn_agent.py` в корне пакета был legacy-дубликатом и больше не нужен.

### `envs/`

Сейчас в `envs/` есть два разных класса сред.

Основная runtime-среда:
- `pyscheduler_lte_env.py` — simulation-backed env поверх реального runtime `PyScheduler`.

Legacy standalone-среды:
- `lte_scheduler_env.py` — standalone LTE env с собственной внутренней логикой;
- `lte_padded_env.py` — padded-обертка над standalone env для фиксированного observation/action space.

Standalone env не является runtime `PyScheduler` и не заменяет реальную симуляцию. Она остается как:
- быстрый smoke/debug training path;
- легкая исследовательская песочница;
- совместимый playground baseline.

## Почему старая env не заменяет новую

У нас уже есть:
- `simulation_bridge.py`, который умеет снимать состояние реальной симуляции;
- `playground_adapter.py`, который умеет строить observation/action-mask.

Но этого недостаточно для обучения на реальном симуляторе, потому что адаптер — это не Gym-среда.

Standalone env из `envs/`:
- умеет `reset()/step()`;
- но живет на своей внутренней логике, а не на `SimulationManager`.

Поэтому `PySchedulerLteEnv` нужна не вместо адаптера и не вместо старой env, а поверх уже существующего bridge/adapter-слоя.

## Скрипты обучения

В `scripts/` теперь есть два разных train-path:
- `train_lte_dqn_pyscheduler.py` — основной путь обучения на реальном runtime через `PySchedulerLteEnv`;
- `train_lte_dqn.py` — legacy playground-path на standalone env.

Идея простая:
- production-близкое обучение делаем через `train_lte_dqn_pyscheduler.py`;
- быстрый baseline и отладку — через `train_lte_dqn.py`.

Текущий runtime-baseline для обучения собран вокруг реального сценария из `TEST_MODULES.py`:
- `5 UE`;
- `10 MHz`;
- `wb_cqi_report_period_tti = 5`;
- `UMi`;
- `TDL = False`;
- `RandomWaypoint` с `pause_time = 0`;
- карта `[-500, 500] x [-500, 500]`;
- `mobility_update_interval_tti = 50`;
- `channel_update_interval_tti = 10`;
- якорный эпизод `2000 TTI`;
- saturated `Poisson` как якорный train/eval-case;
- дополнительный `OnOff` как сценарий на обобщение.

## Как передавать параметры алгоритма

`SimulationManager` больше не хранит `dqn_*` поля в общем dataclass планировщика.
Теперь алгоритм-специфичные параметры передаются через `algorithm_kwargs`.

Пример:

```python
manager.set_scheduler(
    algorithm="DqnScheduler",
    max_dl_ue_tti=16,
    algorithm_kwargs={
        "dqn_model_path": "PyScheduler/drl/runs/lte_dqn/lte_dqn_shared_q.pt",
        "dqn_max_n_ue": 40,
        "dqn_wb_cqi_report_period_tti": 5,
        "dqn_deterministic": True,
        "dqn_inference_device": "cpu",
    },
)
```

Это позволяет позже добавлять `PpoScheduler` и другие DRL-алгоритмы без раздувания интерфейса `SimulationManager`.

Параметр `dqn_inference_device` нужен именно для runtime-инференса внутри симулятора:
- по умолчанию `DqnScheduler` использует `cpu`;
- это сделано специально, потому что policy вызывается много раз за один TTI на очень маленьких observation;
- для такого `per-RBG` цикла GPU нередко медленнее CPU из-за накладных расходов на запуск kernels и перенос данных;
- если нужен эксперимент, можно явно передать `cuda` или `auto`.

## Зависимости

Для inference-path внутри `DqnScheduler` требуется `torch`.

Для training/env script-ов дополнительно требуются:
- `gymnasium`;
- `torch`;
- `numpy`.

Опциональные визуализации могут дополнительно требовать:
- `matplotlib`.

## Как запускать обучение

Из корня репозитория:

```bash
python PyScheduler/drl/scripts/train_lte_dqn_pyscheduler.py --run-dir PyScheduler/drl/runs/lte_dqn --total-env-steps 1000000 --learning-starts 25000 --target-update-freq 10000 --seed 42 --bootstrap-scenario anchor_5ue_10mhz_wb5_umi_fb
```

Скрипт печатает по эпизодам:
- train-сценарий;
- число UE, bandwidth и период CQI;
- длительность эпизода;
- channel model и traffic profile;
- mobility model и интервалы mobility/channel update;
- `mean_reward`, `mean_se_bps_hz`, `mean_jfi_active`, `mean_jfi_all`.

После обучения дополнительно печатается eval по runtime-сценариям:
- якорный `5 UE / 10 MHz / UMi / Poisson`;
- `5 UE / 10 MHz / UMi / OnOff`;
- robustness-check на `UMa`;
- сценарии по нагрузке, bandwidth и CQI-period.

Legacy playground-path:

```bash
python PyScheduler/drl/scripts/train_lte_dqn.py
```

Артефакты по умолчанию:
- runtime train-script пишет в `PyScheduler/drl/runs/lte_dqn/`;
- playground train-script пишет в `PyScheduler/drl/runs/lte_dqn_playground/`.

Это сделано специально, чтобы:
- не засорять корень репозитория служебными файлами обучения;
- держать все DRL-артефакты рядом с DRL-кодом;
- упростить дальнейшую стандартизацию путей для DQN/PPO и других алгоритмов.

Если этих зависимостей нет, optional smoke-тесты на `agents/envs` могут быть пропущены через `pytest.importorskip(...)`.

## End-to-End проверка

Минимальная end-to-end smoke-проверка покрывает цепочку:
- короткий runtime-train на `PySchedulerLteEnv`;
- сохранение checkpoint;
- запуск `DqnScheduler` с загрузкой этих весов внутри `SimulationManager`.

Запуск:

```bash
python -m pytest PyScheduler/tests/test_dqn_end_to_end_optional.py -q
```

Для ручного инференса через существующий сценарий:
- обучите или положите checkpoint в `PyScheduler/drl/runs/lte_dqn/lte_dqn_shared_q.pt`;
- запустите `python PyScheduler/TEST_MODULES.py`.

`TEST_MODULES.py` теперь по умолчанию смотрит именно в локальную папку `PyScheduler/drl/runs/lte_dqn/`.
Также в нем явно выставлен `dqn_inference_device="cpu"`, чтобы не уехать на GPU и не ухудшить scheduler latency.
Сам benchmark в `TEST_MODULES.py` теперь выровнен под runtime-train baseline: `UMi` и `enable_tdl=False`.
