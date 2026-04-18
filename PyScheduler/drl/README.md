# `drl`

Папка `PyScheduler/drl` содержит весь DRL-слой проекта.

Базовый принцип:
- `SIMULATION_MANAGER.py` остается оркестратором симуляции;
- `SCHEDULER.py` остается фабричной точкой входа для планировщиков;
- внутри `drl` живут только DRL-алгоритмы, обучающие среды, runtime-инференс, export-цепочки и связанные тесты.

## Актуальные DRL-решения

Сейчас в проекте поддерживаются три актуальные ветки:
- `DQN per-RBG` — runtime-планировщик `dqn_scheduler.py`
- `PPO per-RBG` — runtime-планировщик `ppo_scheduler.py`
- `PPO ranker compact` — TTI-level ranker `ppo_ranker_scheduler.py`

Для `PPO ranker compact` поддерживаются три backend-а инференса:
- Python / PyTorch checkpoint `.pt`
- C++ / LibTorch через TorchScript `.ts`
- C++ / ONNX Runtime через `.onnx`

## Структура папки

```text
PyScheduler/drl/
├── agents/
│   ├── lte_dqn_agent.py
│   ├── lte_ppo_agent.py
│   └── lte_ppo_ranker_agent.py
├── envs/
│   ├── pyscheduler_lte_env.py
│   ├── pyscheduler_lte_ranker_env.py
│   ├── lte_scheduler_env.py
│   └── lte_padded_env.py
├── scripts/
│   ├── train_lte_dqn_pyscheduler.py
│   ├── train_lte_ppo_scheduler.py
│   ├── train_lte_ppo_ranker.py
│   ├── export_ppo_ranker_torchscript.py
│   ├── export_ppo_ranker_onnx.py
│   ├── plot_training_metrics.py
│   └── legacy/
│       ├── animate_playground_dqn.py
│       └── train_lte_dqn_playground.py
├── cpp_ranker_runtime/
│   ├── CMakeLists.txt
│   ├── ppo_ranker_runtime.cpp
│   ├── ppo_ranker_runtime.h
│   ├── ppo_ranker_c_api.cpp
│   └── ppo_ranker_c_api.h
├── cpp_onnx_runtime/
│   ├── CMakeLists.txt
│   ├── ppo_ranker_onnx_runtime.cpp
│   ├── ppo_ranker_onnx_runtime.h
│   ├── ppo_ranker_onnx_c_api.cpp
│   └── ppo_ranker_onnx_c_api.h
├── runs/
│   ├── lte_dqn/
│   ├── lte_ppo_tune_06_reference/
│   └── lte_ppo_ranker_compact_v1/
├── cpp_libtorch_ranker_bridge.py
├── cpp_onnx_ranker_bridge.py
├── dqn_scheduler.py
├── drl_scheduler.py
├── model_runners.py
├── onnx_ranker_export.py
├── paths.py
├── pdsch_allocation_session.py
├── playground_adapter.py
├── ppo_ranker_scheduler.py
├── ppo_scheduler.py
├── ranker_observation_adapter.py
├── runtime_scenario_factory.py
├── simulation_bridge.py
└── torchscript_ranker_export.py
```

## Что за что отвечает

### Scheduler-слой
- `dqn_scheduler.py` — DQN runtime scheduler per-RBG
- `ppo_scheduler.py` — PPO runtime scheduler per-RBG
- `ppo_ranker_scheduler.py` — compact PPO ranker с одним inference на TTI
- `drl_scheduler.py` — общий runtime-layer для per-RBG DRL scheduler-ов

### Runtime / inference
- `model_runners.py` — единый интерфейс запуска runtime-моделей
- `cpp_libtorch_ranker_bridge.py` — Python bridge к LibTorch DLL
- `cpp_onnx_ranker_bridge.py` — Python bridge к ONNX Runtime DLL
- `paths.py` — каноничные пути к весам и runtime-артефактам

### Env / adapters
- `simulation_bridge.py` — bridge между runtime PyScheduler и DRL-слоем
- `playground_adapter.py` — старый full observation-contract
- `ranker_observation_adapter.py` — compact observation-contract для ranker-модели
- `runtime_scenario_factory.py` — фабрика train/eval-сценариев

### Export
- `torchscript_ranker_export.py` — библиотечный export-helper для TorchScript
- `onnx_ranker_export.py` — библиотечный export-helper для ONNX

### CLI scripts
- `scripts/export_ppo_ranker_torchscript.py` — CLI-обертка над `torchscript_ranker_export.py`
- `scripts/export_ppo_ranker_onnx.py` — CLI-обертка над `onnx_ranker_export.py`
- `scripts/train_lte_dqn_pyscheduler.py` — канонический simulation-backed train для DQN
- `scripts/train_lte_ppo_scheduler.py` — канонический simulation-backed train для PPO per-RBG
- `scripts/train_lte_ppo_ranker.py` — канонический simulation-backed train для PPO ranker
- `scripts/plot_training_metrics.py` — построение графиков по train/eval CSV

### Legacy playground
- `scripts/legacy/train_lte_dqn_playground.py` — старый standalone train-path для playground env
- `scripts/legacy/animate_playground_dqn.py` — старая визуализация standalone DQN на playground env

Эти два legacy-скрипта не участвуют в текущем runtime-контура `TEST_MODULES.py`
и не являются каноническим путем для обучения моделей под PyScheduler runtime.

## Канонические `runs`

В `runs/` должны оставаться только актуальные артефакты:

- `runs/lte_dqn`
  - основной DQN checkpoint: `lte_dqn_shared_q.pt`
- `runs/lte_ppo_tune_06_reference`
  - основной PPO per-RBG checkpoint: `lte_ppo_policy.pt`
- `runs/lte_ppo_ranker_compact_v1`
  - основной compact PPO ranker checkpoint: `lte_ppo_ranker_policy.pt`
  - TorchScript: `lte_ppo_ranker_policy.ts`
  - ONNX: `lte_ppo_ranker_policy.onnx`
  - `run_config.json`
  - `observation_contract.json`
  - `train_metrics.csv`, `ppo_update_metrics.csv`, `eval_metrics.csv`
  - `analysis/` с итоговыми графиками

Временные `smoke_*.pt`, `tmp_*.pt` и прочие одноразовые чекпойнты не считаются частью канонического набора.

## Работа через `TEST_MODULES.py`

Функция `sim_with_manager()` в `PyScheduler/TEST_MODULES.py` должна уметь запускать:
- `DqnScheduler`
- `PpoScheduler`
- `PpoRankerScheduler` на backend `python`
- `PpoRankerScheduler` на backend `cpp` (LibTorch)
- `PpoRankerScheduler` на backend `onnx_cpp`

Стандартные пути к весам и runtime DLL берутся из `paths.py`.

## Сборка C++ runtime

### LibTorch runtime

```powershell
cmake --fresh -S PyScheduler\drl\cpp_ranker_runtime -B PyScheduler\drl\cpp_ranker_runtime\build -DTorch_DIR=D:/libtorch_cpu/libtorch/share/cmake/Torch
cmake --build PyScheduler\drl\cpp_ranker_runtime\build --config Release
```

### ONNX Runtime

```powershell
cmake --fresh -S PyScheduler\drl\cpp_onnx_runtime -B PyScheduler\drl\cpp_onnx_runtime\build -DONNXRUNTIME_ROOT=D:/onnxruntime_cpu/onnxruntime-win-x64-1.23.2
cmake --build PyScheduler\drl\cpp_onnx_runtime\build --config Release
```

## Основные команды обучения

### DQN per-RBG

```powershell
python PyScheduler/drl/scripts/train_lte_dqn_pyscheduler.py --run-dir PyScheduler/drl/runs/lte_dqn
```

### PPO per-RBG

```powershell
python PyScheduler/drl/scripts/train_lte_ppo_scheduler.py --run-dir PyScheduler/drl/runs/lte_ppo_tune_06_reference
```

### PPO ranker compact

```powershell
python PyScheduler/drl/scripts/train_lte_ppo_ranker.py --run-dir PyScheduler/drl/runs/lte_ppo_ranker_compact_v1
```

### Export ranker в TorchScript

```powershell
python PyScheduler/drl/scripts/export_ppo_ranker_torchscript.py --checkpoint-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.pt --output-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.ts --device cpu
```

### Export ranker в ONNX

```powershell
python PyScheduler/drl/scripts/export_ppo_ranker_onnx.py --checkpoint-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.pt --output-path PyScheduler/drl/runs/lte_ppo_ranker_compact_v1/lte_ppo_ranker_policy.onnx --device cpu
```

## Графики обучения

```powershell
python PyScheduler/drl/scripts/plot_training_metrics.py --run-dir PyScheduler/drl/runs/lte_ppo_ranker_compact_v1
```

## Тесты

DRL-тесты теперь собраны в отдельной папке:

```text
PyScheduler/tests/drl/
```

Старые scheduler/unit-тесты без DRL живут отдельно в:

```text
PyScheduler/tests/scheduler_tests/
```

Временные тестовые артефакты создаются в:

```text
.tmp_test_artifacts/
```

Это не часть репозитория и не часть runtime-цепочек.

## Базовый набор важных тестов

Минимальный набор тестов, который должен оставаться рабочим:
- `tests/drl/test_dqn_scheduler.py`
- `tests/drl/test_ppo_scheduler.py`
- `tests/drl/test_ppo_ranker_scheduler.py`
- `tests/drl/test_model_runners.py`
- `tests/drl/test_cpp_libtorch_ranker_bridge.py`
- `tests/drl/test_cpp_onnx_ranker_bridge.py`
- `tests/drl/test_ranker_observation_adapter.py`
- `tests/drl/test_runtime_scenario_factory.py`
- `tests/drl/test_simulation_runtime_init.py`
- `tests/drl/test_lte_ppo_agent.py`
- `tests/drl/test_lte_ppo_ranker_agent.py`
- `tests/drl/test_ppo_ranker_torchscript_optional.py`
- `tests/drl/test_ppo_ranker_onnx_optional.py`
- `tests/drl/test_plot_training_metrics_optional.py`

## Что сознательно убрано

Из `drl` убраны:
- старые benchmark/smoke-контуры, не являющиеся runtime-реализацией инференса;
- временные sample-pack и smoke-тесты для них;
- временные checkpoint-артефакты;
- `__pycache__`, `.pytest_cache`, локальные build-каталоги и прочий мусор.

Идея простая: в `drl` должна оставаться только логика, которая нужна для:
- обучения;
- экспорта;
- реального runtime-инференса;
- запуска через `TEST_MODULES.py`.
