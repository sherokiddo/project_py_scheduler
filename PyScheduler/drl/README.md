# `drl`

Пакет для DRL-логики внутри `PyScheduler`.

Главная идея:
- `SimulationManager` остается оркестратором симуляции;
- baseline scheduler-ы продолжают жить в общем pipeline;
- вся DRL-специфика вынесена в отдельный слой `PyScheduler/drl`.

Отдельно важно:
- `SCHEDULER.py` остается точкой фабричного входа;
- общий DRL runtime-pipeline не вшивается в `SCHEDULER.py`, а живет в `PyScheduler/drl/drl_scheduler.py`;
- это позволяет подключать новые DRL-алгоритмы без раздувания core-модуля планировщиков.

## Структура

Корень пакета:
- `simulation_bridge.py` — мост между runtime симулятора и DRL-слоем;
- `playground_adapter.py` — сборка observation/action-mask в формате `drl_playground`;
- `ranker_observation_adapter.py` — compact observation-contract для TTI-level ranker-моделей;
- `drl_scheduler.py` — общий базовый runtime-layer `DrlScheduler` для DRL-планировщиков;
- `dqn_scheduler.py` — DQN-специализация поверх `DrlScheduler`;
- `ppo_scheduler.py` — PPO-специализация поверх `DrlScheduler`;
- `ppo_ranker_scheduler.py` — TTI-level PPO ranker, где ML работает как priority/policy engine, а allocator отдельно завершает runtime-гибридный FD/PF allocation;
- `torchscript_ranker_export.py` — inference-only wrapper и export helper для TorchScript/LibTorch ranker path;
- `cpp_ranker_bridge.py` — Python `ctypes`-мост к C++/LibTorch ranker runtime;
- `cpp_ranker_runtime/` — C++ DLL для deterministic TorchScript ranker inference внутри Python runtime;
- `libtorch_ranker_smoke.py` — подготовка sample-pack для сверки Python ranker и LibTorch inference;
- `model_runners.py` — общий OOP-модуль с runtime-runner'ами для torch-агентов.

Подпакеты:
- `agents/` — агенты DRL;
- `envs/` — обучающие среды;
- `scripts/` — сценарии обучения, оценки и визуализации.
- `cpp_ranker_runtime/` — production-oriented DLL runtime для вызова TorchScript ranker через LibTorch.
- `cpp_libtorch_smoke/` — минимальный C++ smoke-test для проверки TorchScript ranker через LibTorch.

## Что где живет

### `agents/`

В этой папке должны жить все агенты:
- `lte_dqn_agent.py` — основной LTE DQN агент;
- `agents/lte_ppo_agent.py` — LTE-специфичный PPO агент с action-mask и shared UE encoder.
- `agents/lte_ppo_ranker_agent.py` — PPO-агент для TTI-level ranking, который выдает score по всем UE за один шаг.

Файл `dqn_agent.py` в корне пакета был legacy-дубликатом и больше не нужен.

### `envs/`

Сейчас в `envs/` есть несколько разных классов сред.

Основная runtime-среда:
- `pyscheduler_lte_env.py` — simulation-backed env поверх реального runtime `PyScheduler`.
- `pyscheduler_lte_ranker_env.py` — simulation-backed env для ranker-подхода, где одно действие задает score UE на весь TTI на priority-stage, а allocation затем завершается через hybrid FD/PF метрику.

Для ranker-env теперь используется отдельный compact observation-contract:
- per-UE: `reported_wb_cqi`, `wb_cqi_age_tti`, `buffer_bytes`, `average_throughput_bps`;
- global context: `active_ue_count`, `n_rbg`.

То есть на текущем этапе ranker-train path уже не использует признаки внутренней alloc-фазы внутри TTI:
- `alloc_rbg_frac_tti`;
- `current_rbg_index`;
- `allocated_fraction`;
- `current_tti / episode_len_tti`.

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

`PySchedulerLteRankerEnv` строится на той же базе и переиспользует:
- тот же runtime `SimulationManager`;
- тот же bridge/snapshot слой;
- ту же механику расчета reward;
- тот же scheduler-preparation pipeline до этапа allocation.

Отличие только в семантике действия:
- `PySchedulerLteEnv` делает одно решение per-RBG;
- `PySchedulerLteRankerEnv` делает одно score-решение per-TTI, после чего среда переводит score в bounded rank-weight и использует его как модификатор существующей FD/PF per-RBG метрики.

Важно по текущему состоянию проекта:
- compact observation уже включен в `PySchedulerLteRankerEnv` и используется для обучения ranker-модели;
- runtime `PpoRankerScheduler` теперь тоже переведен на тот же compact-contract, что и ranker-env;
- то есть train и runtime снова синхронизированы по входным признакам;
- старые PPO-ranker Python/C++ inference-цепочки и вспомогательные C++ примеры пока оставлены как временные reference-артефакты и позже должны быть удалены после окончательной стабилизации нового compact path.

## Скрипты обучения

В `scripts/` теперь есть два разных train-path:
- `train_lte_dqn_pyscheduler.py` — основной путь обучения на реальном runtime через `PySchedulerLteEnv`;
- `train_lte_ppo_scheduler.py` — PPO-обучение на том же runtime-env c последующей загрузкой весов в `PpoScheduler`;
- `train_lte_ppo_ranker.py` — PPO-обучение ranker-варианта на `PySchedulerLteRankerEnv`;
- `export_ppo_ranker_torchscript.py` — экспорт PPO ranker checkpoint в TorchScript для LibTorch/C++ inference;
- `prepare_ppo_ranker_libtorch_smoke.py` — подготовка sample-pack для C++ smoke-test;
- `plot_training_metrics.py` — постобработка сохраненных CSV в графики и summary;
- `train_lte_dqn.py` — legacy playground-path на standalone env.

Идея простая:
- production-близкое обучение делаем через `train_lte_dqn_pyscheduler.py`;
- PPO исследуем через `train_lte_ppo_scheduler.py` и можем прогонять его через тот же runtime pipeline;
- PPO ranker обучаем через `train_lte_ppo_ranker.py`, если хотим policy уровня "один ranking на весь TTI";
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

Аналогично для PPO:

```python
manager.set_scheduler(
    algorithm="PpoScheduler",
    max_dl_ue_tti=16,
    algorithm_kwargs={
        "ppo_model_path": "PyScheduler/drl/runs/lte_ppo/lte_ppo_policy.pt",
        "ppo_max_n_ue": 40,
        "ppo_wb_cqi_report_period_tti": 5,
        "ppo_deterministic": True,
        "ppo_inference_device": "cpu",
    },
)
```

Для PPO ranker:

```python
manager.set_scheduler(
    algorithm="PpoRankerScheduler",
    max_dl_ue_tti=16,
    algorithm_kwargs={
        "ppo_ranker_model_path": "PyScheduler/drl/runs/lte_ppo_ranker/lte_ppo_ranker_policy.pt",
        "ppo_ranker_max_n_ue": 40,
        "ppo_ranker_wb_cqi_report_period_tti": 5,
        "ppo_ranker_deterministic": True,
        "ppo_ranker_inference_device": "cpu",
        "ppo_ranker_rank_weight_beta": 0.3,
        "ppo_ranker_pf_epsilon_bps": 1e-6,
    },
)
```

Для PPO ranker через C++ backend:

```python
manager.set_scheduler(
    algorithm="PpoRankerScheduler",
    max_dl_ue_tti=16,
    algorithm_kwargs={
        "ppo_ranker_model_path": "PyScheduler/drl/runs/lte_ppo_ranker/lte_ppo_ranker_policy.ts",
        "ppo_ranker_backend": "cpp",
        "ppo_ranker_runtime_library_path": "PyScheduler/drl/cpp_ranker_runtime/build/Release/ppo_ranker_runtime.dll",
        "ppo_ranker_runtime_dll_search_paths": [
            "D:/libtorch_win_cpu/libtorch/lib",
        ],
        "ppo_ranker_max_n_ue": 40,
        "ppo_ranker_wb_cqi_report_period_tti": 5,
        "ppo_ranker_deterministic": True,
        "ppo_ranker_rank_weight_beta": 0.3,
        "ppo_ranker_pf_epsilon_bps": 1e-6,
    },
)
```

Важно:
- для backend `cpp` нужен именно TorchScript-файл `.ts`, а не training checkpoint `.pt`;
- `ppo_ranker_runtime_library_path` указывает на собранную DLL runtime;
- `ppo_ranker_runtime_dll_search_paths` должен включать каталог с LibTorch DLL, иначе Windows не найдет `torch_cpu.dll`, `c10.dll` и связанные зависимости.

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

PPO-path на той же env:

```bash
python PyScheduler/drl/scripts/train_lte_ppo_scheduler.py --run-dir PyScheduler/drl/runs/lte_ppo --total-env-steps 750000 --rollout-steps 4096 --seed 42 --bootstrap-scenario anchor_5ue_10mhz_wb5_umi_fb
```

PPO ranker-path на TTI-level env:

```bash
python PyScheduler/drl/scripts/train_lte_ppo_ranker.py --run-dir PyScheduler/drl/runs/lte_ppo_ranker_compact_v1 --total-env-steps 300000 --rollout-steps 2048 --seed 42 --bootstrap-scenario anchor_5ue_10mhz_wb5_umi_fb
```

С этого этапа `train_lte_ppo_ranker.py` дополнительно сохраняет:
- `run_config.json` — гиперпараметры запуска;
- `observation_contract.json` — точное описание compact observation-contract, на котором была обучена модель.

Это нужно, чтобы потом без гадания понимать:
- какие признаки реально подавались в policy;
- какой был `obs_dim`;
- можно ли этот checkpoint безопасно экспортировать и запускать в runtime/C++ path.

Для ranker-path по умолчанию дополнительно запускается сравнение:
- `deterministic eval` — то, как policy будет вести себя в боевом inference;
- `stochastic eval` — тот же checkpoint, но с sampling из policy.

Кроме финального eval, `train_lte_ppo_ranker.py` теперь умеет делать
периодический `probe-eval` прямо во время обучения:
- probe всегда идет в `deterministic` режиме;
- probe помогает сразу увидеть расхождение между train-rollout и реальным inference-path;
- это особенно полезно для ranker-policy, где стохастический sampling может временно давать красивую fairness-картину, но не переноситься в deterministic режим.

Это сделано специально, чтобы быстро увидеть важный эффект:
- если train-эпизоды выглядят fair, а deterministic eval резко хуже,
- policy могла опираться на шум при ранжировании UE.

Параметры:
- `--eval-compare-stochastic` / `--no-eval-compare-stochastic` — включить или выключить сравнительный eval;
- `--eval-stochastic-repeats N` — сколько stochastic прогонов усреднять.
- `--probe-every-episodes N` — как часто запускать детерминированный probe во время обучения;
- `--probe-scenario SCENARIO_KEY` — на каком сценарии гонять periodic probe.

Для smoke-запуска удобнее сразу ограничить eval и оставить его на CPU:

```bash
python PyScheduler/drl/scripts/train_lte_ppo_scheduler.py --run-dir PyScheduler/drl/runs/lte_ppo_smoke --total-env-steps 40000 --rollout-steps 2048 --eval-device cpu --eval-scenario-limit 3 --seed 42 --bootstrap-scenario anchor_5ue_10mhz_wb5_umi_fb
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

Построить графики по уже завершенному run:

```bash
python PyScheduler/drl/scripts/plot_training_metrics.py --run-dir PyScheduler/drl/runs/lte_ppo_ranker_compact_v1
```

Скрипт создает каталог `analysis/` внутри run-директории и сохраняет:
- `train_overview.png` — кривые reward, SE, JFI и episode-side optimizer signals;
- `ppo_updates.png` — loss, actor loss, critic loss и entropy по PPO updates;
- `eval_summary.png` — сводка deterministic eval по сценариям;
- `summary.json` — короткий агрегированный итог run.

Экспортировать PPO ranker в TorchScript для LibTorch:

```bash
python PyScheduler/drl/scripts/export_ppo_ranker_torchscript.py --checkpoint-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.pt --output-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts --device cpu
```

Скрипт сохраняет:
- `lte_ppo_ranker_policy.ts` — inference-only TorchScript artifact;
- `lte_ppo_ranker_policy.meta.json` — metadata с размерностями входа/выхода и зафиксированным inference-контрактом.

Текущий TorchScript runtime-path для PPO ranker теперь использует только actor-ветку:
- deterministic `score_vector` считается без вызова critic/value-head;
- это уменьшает стоимость runtime-inference и не влияет на train-path PPO.

TorchScript ranker экспортирует только deterministic inference-path:
- input: `obs [batch, obs_dim]`, `float32`;
- input: `action_mask [batch, max_n_ue]`, `bool`;
- output: `score_vector [batch, max_n_ue]`, `float32`.

Подготовить sample-pack для LibTorch smoke-test:

```bash
python PyScheduler/drl/scripts/prepare_ppo_ranker_libtorch_smoke.py --checkpoint-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.pt --torchscript-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts --output-dir PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/libtorch_smoke --scenario-key anchor_5ue_10mhz_wb5_umi_fb --seed 123
```

Скрипт сохраняет:
- `obs.txt`;
- `action_mask.txt`;
- `expected_scores.txt`;
- `smoke_manifest.json`.

Это нужно для smoke-проверки C++ path:
- Python снимает реальное `obs/action_mask` из runtime-based ranker env;
- Python вычисляет эталонный deterministic `score_vector`;
- C++/LibTorch загружает тот же `.ts` и сравнивает свой выход с `expected_scores.txt`.

Минимальный C++ smoke-test лежит в `PyScheduler/drl/cpp_libtorch_smoke/`.
Для сборки нужен `CMake` и Windows toolchain (`cl`) или совместимый C++ компилятор.
Для Windows практичнее использовать отдельный CPU-only LibTorch distribution, а не `Torch_DIR` из Python wheel.
В рабочем варианте `Torch_DIR` нужно направлять в каталог вида:
- `D:/libtorch_win_cpu/libtorch/share/cmake/Torch`

Перед запуском `.exe` Windows также должен видеть LibTorch DLL:
- `c10.dll`;
- `torch_cpu.dll`;
- и остальные `.dll` из `<libtorch>/lib`.

Самый простой вариант для smoke/benchmark-прогона:
- временно добавить `<libtorch>/lib` в `PATH` в том же терминале.

Пример последовательности:

```bash
cmake --fresh -S PyScheduler/drl/cpp_libtorch_smoke -B PyScheduler/drl/cpp_libtorch_smoke/build -DTorch_DIR=D:/libtorch_win_cpu/libtorch/share/cmake/Torch
cmake --build PyScheduler/drl/cpp_libtorch_smoke/build --config Release
set PATH=D:\libtorch_win_cpu\libtorch\lib;%PATH%
PyScheduler/drl/cpp_libtorch_smoke/build/Release/ppo_ranker_libtorch_smoke.exe PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/libtorch_smoke/obs.txt PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/libtorch_smoke/action_mask.txt PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/libtorch_smoke/expected_scores.txt
```

Ожидаемый результат smoke-теста:
- печать первых значений `expected_scores` и `actual_scores`;
- `max_abs_error` и `mean_abs_error`;
- финальная строка `Smoke-test PASSED`.

## C++ runtime для TEST_MODULES.py

После correctness smoke можно поднять тот же TorchScript ranker прямо в runtime-симуляции.

Сборка DLL:

```bash
cmake --fresh -S PyScheduler/drl/cpp_ranker_runtime -B PyScheduler/drl/cpp_ranker_runtime/build -DTorch_DIR=D:/libtorch_win_cpu/libtorch/share/cmake/Torch
cmake --build PyScheduler/drl/cpp_ranker_runtime/build --config Release
```

На Windows перед запуском Python-процесса LibTorch DLL должны быть видимы:
- либо через `ppo_ranker_runtime_dll_search_paths`;
- либо через `PATH` в том же терминале.

Пример ручного прогона через `TEST_MODULES.py`:

```python
sim_with_manager(
    scheduler_algorithm="PpoRankerScheduler",
    runtime_backend="cpp",
    runtime_model_path="PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts",
    runtime_library_path="PyScheduler/drl/cpp_ranker_runtime/build/Release/ppo_ranker_runtime.dll",
    runtime_dll_search_paths=[
        "D:/libtorch_win_cpu/libtorch/lib",
    ],
)
```

Этот path работает так:
- `TEST_MODULES.py` собирает обычный runtime-сценарий;
- `PpoRankerScheduler` формирует observation и action-mask как раньше;
- `CppPPORankerModelRunner` вызывает DLL через `ctypes`;
- C++ runtime делает deterministic forward TorchScript ranker на CPU;
- scheduler получает `score_vector` по UE и продолжает allocation в штатном pipeline PyScheduler.

То есть core-логика симулятора и allocator не переписываются:
- меняется только backend инференса policy;
- observation contract и scheduler contract остаются прежними.

Тот же бинарник умеет работать и как чистый micro-benchmark inference-path без симулятора:

```bash
PyScheduler/drl/cpp_libtorch_smoke/build/Release/ppo_ranker_libtorch_smoke.exe PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/libtorch_smoke/obs.txt PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/libtorch_smoke/action_mask.txt PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/libtorch_smoke/expected_scores.txt --benchmark 10000 --warmup 500
```

Benchmark-режим:
- сначала прогоняет обычную correctness smoke-проверку;
- затем делает `warmup`;
- затем меряет только `forward` по тем же `obs/action_mask`, без чтения файлов и без симулятора;
- печатает `mean_us`, `min_us`, `p50_us`, `p95_us`, `max_us`.

Артефакты по умолчанию:
- runtime train-script пишет в `PyScheduler/drl/runs/lte_dqn/`;
- runtime PPO train-script пишет в `PyScheduler/drl/runs/lte_ppo/`;
- runtime PPO ranker train-script пишет в `PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/`;
- playground train-script пишет в `PyScheduler/drl/runs/lte_dqn_playground/`.

Для `train_lte_ppo_scheduler.py` eval теперь:
- по умолчанию идет на `cpu`, даже если train шел на `cuda`;
- печатает явный прогресс по сценариям;
- может быть ограничен флагом `--eval-scenario-limit`;
- может быть полностью отключен через `--eval-scenario-limit 0`.

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

PPO runtime smoke:

```bash
python -m pytest PyScheduler/tests/test_ppo_end_to_end_optional.py -q
```

Для ручного инференса через существующий сценарий:
- обучите или положите checkpoint в `PyScheduler/drl/runs/lte_dqn/lte_dqn_shared_q.pt`;
- запустите `python PyScheduler/TEST_MODULES.py`.

`TEST_MODULES.py` теперь по умолчанию смотрит именно в локальную папку `PyScheduler/drl/runs/lte_dqn/`.
Также в нем явно выставлен `dqn_inference_device="cpu"`, чтобы не уехать на GPU и не ухудшить scheduler latency.
Сам benchmark в `TEST_MODULES.py` теперь выровнен под runtime-train baseline: `UMi` и `enable_tdl=False`.

Для PPO на том же сценарии:
- положите checkpoint в `PyScheduler/drl/runs/lte_ppo/lte_ppo_policy.pt`;
- вызовите `sim_with_manager(scheduler_algorithm="PpoScheduler")` в `PyScheduler/TEST_MODULES.py`.

Для PPO ranker на Python backend:
- положите checkpoint в `PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.pt`;
- вызовите `sim_with_manager(scheduler_algorithm="PpoRankerScheduler")`.

Для PPO ranker на C++/LibTorch backend:
- экспортируйте TorchScript в `PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts`;
- соберите DLL `PyScheduler/drl/cpp_ranker_runtime/build/Release/ppo_ranker_runtime.dll`;
- передайте `runtime_backend="cpp"` и `runtime_dll_search_paths=["D:/libtorch_win_cpu/libtorch/lib"]`;
- вызовите:

```python
sim_with_manager(
    scheduler_algorithm="PpoRankerScheduler",
    runtime_backend="cpp",
)
```

Если пути стандартные, `TEST_MODULES.py` подхватит:
- `PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts`;
- `PyScheduler/drl/cpp_ranker_runtime/build/Release/ppo_ranker_runtime.dll`.
