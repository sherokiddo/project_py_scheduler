# SCHEDULER.py — планировщик радиоресурсов (LTE DL)

> Ответственный: Тимур (Scheduler)  
> Состояние: описан; шаги M0–M1 [FEAT-SCH-008](features.md#feat-sch-008) выполнены в ветке `feat/SCH-008-M1-protocols` (поставка в форк и патчами — §4.6, порядок интеграции — §4.8), следующий шаг — M1a; комплект документации модуля размещается в репозитории в `docs/scheduler/` (§8)  
> Проверено по коду: `6c0aae2` (ветка `dev`, AS-IS) и `9379b3d` (ветка `feat/SCH-008-M1-protocols`, шаги M0–M1 + фиксы по ревью PR #44) | Обновлено: `2026-11-02`

Источник: [`PyScheduler/SCHEDULER.py`](https://github.com/sherokiddo/project_py_scheduler/blob/6c0aae2/PyScheduler/SCHEDULER.py), версия модуля v2.1.0 + QoS-слияние PR #39, 3245 строк. Ссылки даны в формате `файл::символ (:строка)`; номера строк действительны для коммита `6c0aae2` (HEAD ветки `dev`). Все приведённые числа и поведение проверены фактическими запусками на этом коммите; контрольный сценарий §4 воспроизводится побитово и на `ae9db40`, и на `6c0aae2`. Авторская документация модуля: [`Documentation/doc_scheduler_v2.1.0.md`](https://github.com/sherokiddo/project_py_scheduler/blob/6c0aae2/PyScheduler/Documentation/doc_scheduler_v2.1.0.md) (не обновлялась после PR #26, см. [audit.md](audit.md) DOC-SCH-001). Якоря `:NNN` для строк ≤ 3245 действительны и в ветке FEAT-SCH-008: раздел шага M1 (:3246–3584) добавлен в конец файла, сдвига номеров нет (§4.7); строки > 3245 существуют только в ветке и сверены с коммитом `9379b3d`.

## 1. Назначение

Модуль решает задачу MAC-планировщика downlink: один раз в TTI (1 мс) определяет, какие пользователи (UE) получат радиоресурс и какие именно группы ресурсных блоков (RBG) им достанутся — с учётом качества канала (CQI), заполненности буферов базовой станции и бюджета управляющего канала PDCCH (CCE). Реализованы восемь алгоритмов планирования (BestCQI, ProportionalFair, RoundRobin, три частотно-селективных FD-варианта и два QoS-алгоритма, слитых в PR #39), выбираемых через фабрику. Кроме принятия решения модуль пока сам исполняет его: извлекает пакеты из буфера BS и обновляет статистику throughput UE; расслоение запланировано самим кодом (roadmap-комментарий :3227–3245, см. [FEAT-SCH-009](features.md#feat-sch-009)).

## 2. Границы ответственности

**Делает:**
- Полный pipeline одного TTI (этапы 0–8, `SCHEDULER.py::SchedulerInterface.schedule` :346): обновление карты CQI → отбор допущенных UE → скользящее окно → приоритеты → оценка потребности PDSCH → выделение CCE → распределение RBG → извлечение из буфера → статистика и результат.
- Восемь алгоритмов через фабрику `SchedulerInterface.create()` (:235): `'BestCQI'`, `'ProportionalFair'`, `'RoundRobin'`, `'FD_BCQI'`, `'FD_FGS'`, `'FD_PF'`, `'BasicQosAware'`, `'QosAware'` (ключи :262–263); список `available_algorithms()` (:277) расходится с ключами фабрики ([BUG-SCH-010](audit.md#bug-sch-010)).
- Учёт бюджета CCE (`PDCCHManager` :1262) и перевод CQI в оценку ёмкости RB (`AdaptiveModulationAndCoding` :1703).
- Хранение CQI между отчётами (`CQIMap` :191), формирование грантов извлечения (`SchedulingGrant` :117), мультиплексирование логических каналов (`_logical_channel_multiplexing` :811, режимы Simple/Layered).
- Накопление статистики TTI: `get_stats()` (:438), `amc.get_stats()` (:1755), `pdcch_manager.get_stats()` (:1660).

**Не делает (относится к соседним модулям):**
- Не измеряет канал: `cqi`/`sbb_cqi` приходят готовыми из `UE_MODULE.py::GET_USERS_FOR_SCHEDULER` (:901), где формируются `UE_MODULE.py::SINR_TO_CQI` (:693).
- Не генерирует трафик и не владеет буферами: их хранит `BS_MODULE` (`IBufferManager` :472, реализации `SimpleBufferManager` :580 и `LayeredBufferManager` :1356); планировщик читает статус и извлекает по грантам.
- Не владеет ресурсной сеткой: занятие RB выполняет `RES_GRID.py::ALLOCATE_RBG` (:430, атомарно с откатом `RELEASE_RBG` :440).
- Не реализует HARQ: `harq_manager = None` (:341), `_get_harq_retransmissions` (:1231) — заготовка; поля гранта `ndi`/`rv`/`harq_process_id` не используются.
- Не реализует QoS/LCP-приоритеты: в Layered-режиме TB делится поровну между активными логическими каналами (комментарий в коде называет стратегию несправедливой, :861–863); QoS-планировщики слиты в `dev` в PR #39 (`6c0aae2`), LCP-приоритизация извлечения остаётся открытой ([FEAT-SCH-009](features.md#feat-sch-009)).

## 3. Входы, выходы и зависимости

| Вход | Откуда | Формат/смысл |
|---|---|---|
| `tti: int` | `SIMULATION_MANAGER.py::run_tti` (:1657, вызов :1701) | номер текущего TTI (1 мс) |
| `users: List[Dict]` | `UE_MODULE.py::GET_USERS_FOR_SCHEDULER` (:901–920) | срез состояния на начало TTI, ровно 4 поля: `UE_ID`, `cqi` (WB 1–15), `sbb_cqi` (per-RBG список; **ключ с опечаткой — устоявшийся контракт обеих сторон**, [DOC-SCH-001](audit.md#doc-sch-001)), `ue` (живая ссылка на объект UE) |
| размеры буферов | `BS_MODULE.py::IBufferManager.ue_has_buffer/get_buffer_status` (:530/:492) | читаются напрямую на этапе 1, дописываются в `users[i]['bs_buffer_size']` |
| `lte_grid`, `bs` | конструктор (:294); конфигурация — `SIMULATION_MANAGER.py::SchedulerConfig` (:1030) | `GET_RBG_SIZE` (:421), `GET_RBG_INDICES` (:424), `ALLOCATE_RBG` (:430), `GENERATE_BITMAP` (:567); `bs.buffer_manager`, `bs.use_simple_buffer` (:1696) |

| Выход | Куда | Формат/смысл |
|---|---|---|
| возврат `schedule()` | `SIMULATION_MANAGER` | `{allocation: {UE_ID: [freq_idx]}, statistics: {}, bitmap: {UE_ID: [17 бит]}, pdcch_stats}`; **в главном цикле результат отбрасывается**: `_ = self.run_tti(...)` (SIMULATION_MANAGER.py:1451), [OPT-SCH-005](audit.md#opt-sch-005) |
| статистика | `StatsManager` (SIMULATION_MANAGER.py:109, `collect` :158) | фактический канал данных наружу: три `get_stats()` читают кэш `_last_*`, заполняемый на этапе 7.5 |
| извлечение пакетов | `BS_MODULE::get_packets(grants)` (:507, Layered :1441) | побочный эффект этапа 7: буфер BS физически уменьшается |
| per-QCI throughput | `BS_MODULE::QciThroughputTracker.update_all_qcis` (:223) | побочный эффект этапа 7 с `6c0aae2` (PR #39): `SCHEDULER.py` :901, :916–919, :941–944, :962–965, :967–975; на извлечение пакетов и гранты не влияет |
| throughput UE | `UE_MODULE.py::UPD_DL_THROUGHPUT_BPS` (:629) | побочный эффект этапа 7: `current_dl_throughput` и EWMA `average_throughput` (α=0.002 хардкод, :644; TODO «вывести в управление», issue #31) — знаменатель PF следующих TTI |
| занятые RB | `RES_GRID::ALLOCATE_RBG` (:430) | побочный эффект этапа 6: RBG заняты в обоих слотах подкадра до переиспользования (кадр = 10 подкадров) |

Зависимости: `GLOBALS` (`BITS_PER_BYTE` :140, `bytes_to_bits` :217), `time`, `dataclasses`, `typing`; runtime — RES_GRID, BS_MODULE, UE_MODULE, SIMULATION_MANAGER. Импорт `BaseStation` (:112) не используется ([OPT-SCH-006](audit.md#opt-sch-006)).

## 4. Как проверить работу

Разделы 4.1–4.2 — проверка AS-IS на `dev@6c0aae2`; разделы 4.3–4.7 — работа локальных веток FEAT-SCH-008 (шаги M0–M1: baseline-харнес, контрактный набор, фактические числа прогонов, поставка патчами, перепривязка якорей).

### 4.1 Минимальный сценарий AS-IS

Проверен на `6c0aae2`, воспроизводится побитово и на `ae9db40`; сохранить как `PyScheduler/_check.py` и запустить `python3 _check.py` из каталога `PyScheduler/`, нужен numpy):

```python
import GLOBALS
from RES_GRID import RES_GRID_LTE
from BS_MODULE import BaseStation
from UE_MODULE import UserEquipment
from TRAFFIC_MODEL import Packet
from SCHEDULER import SchedulerInterface

GLOBALS.CURRENT_TIME = 5
bs   = BaseStation(bandwidth=10, use_simple_buffer=True)
grid = RES_GRID_LTE(bandwidth=10, num_frames=2)
sched = SchedulerInterface.create('BestCQI', grid, bs)

ues = []
for ueid, cqi, nbytes in ((1, 13, 1000), (2, 7, 5000), (3, 15, 300)):
    ue = UserEquipment(UE_ID=ueid, x=100.0 * ueid, y=0.0)
    ue.cqi, ue.cqi_subband = cqi, []
    bs.buffer_manager.create_ue_buffer(ueid, bs.per_ue_max)
    bs.buffer_manager.add_packet(ueid, Packet(size=nbytes, ue_id=ueid, creation_time=0))
    ues.append(ue)

users = [{"UE_ID": u.UE_ID, "cqi": u.cqi, "sbb_cqi": u.cqi_subband, "ue": u} for u in ues]
res = sched.schedule(5, users)
print(res['allocation'])
```

Ожидаемый результат (фактический прогон): `allocation = {1: [6..17], 2: [], 3: [0..5]}` — 18 RB из 50 (PRB util 36%); `pdcch_ue_cce_allocations = {3: 1, 1: 1}` (2/25 CCE); UE2 отсечён оценкой PDSCH (порог 95%) и не тратит CCE; извлечено 8004/2404 бита у UE1/UE3; `average_throughput` = 16008/4808 бит/с (EWMA α=0.002).

### 4.2 Штатный тестовый набор репозитория

`PyScheduler/tests/scheduler_tests/` — 27 тестовых функций (`test_scheduler_common.py`, `test_fd_best_cqi.py`, `test_fd_proportional_fair.py`, `test_fd_round_robin.py`) и бенчмарк `FD_benchmark.py`. Запуск — из каталога `PyScheduler/`: `python3 -m pytest tests/scheduler_tests/ -q` (из корня репозитория импорт `SCHEDULER` не разрешается: тесты добавляют в `sys.path` только каталог `tests/`, а модули проекта разрешаются через cwd). **Внимание: на `dev` набор сейчас неработоспособен** — отсутствует импортируемый всеми файлами модуль `log_utils` (есть только в ветках `multithread`/`feature/AMC`). Проверено на свежих чистых клонах `ae9db40` и `6c0aae2` — картина одинаковая: (A) как есть — 4 ошибки сборки независимо от каталога запуска; (B) после возврата `log_utils.py` — 15 failed / 12 passed (14 — устаревшее имя `'FD_RR'`, 1 — содержательный инвариант FD_PF); (C) после замены `'FD_RR'`→`'FD_FGS'` — 26 passed / 1 failed. Подробности: [BUG-SCH-008](audit.md#bug-sch-008); восстановление набора — запрос к владельцу репозитория и автору тестового набора в [features.md](features.md) §2 (в объём работ по модулю не входит).

### 4.3 Baseline-харнес FEAT-SCH-008 (шаг M0): окружение, команды, снимки

Харнес существует в локальной ветке [FEAT-SCH-008](features.md#feat-sch-008) `feat/SCH-008-M1-protocols` (десять коммитов `9d50a94`…HEAD от `dev@6c0aae2`, топология — §4.6) и в `dev` не слит; поставка — веткой в форк и патчами (§4.6). Каталог `PyScheduler/tests/baseline/` автономен: `conftest.py` сам добавляет `PyScheduler/` в `sys.path` (приём `layered_buffer_tests/conftest.py`) и регистрирует маркер `sim` через `pytest_configure`; файлы корня репозитория и `PYTHONPATH` не требуются.

**Окружение.** Валидировано: Python 3.11.2, numpy 2.4.6, pytest 9.2.0. Снимки воспроизведены байт-в-байт на двух стендах с этими версиями, но побитовая воспроизводимость гарантируется только в рамках одной версии интерпретатора и numpy (другая BLAS способна сдвинуть float за пределы допуска 1e-9). Подпись окружения пишется в `meta.env` каждого снимка и в сравнении не участвует (§4.3, правила сравнения). Зависимости: `numpy`, `pytest`, `tqdm`, `matplotlib`, `psutil` (`statsmodels` в прогонах не используется).

**Команды** (все — из корня репозитория):

```bash
# Быстрый characterization-набор (44 теста, <1 c) — обязателен после каждого шага:
python -m pytest PyScheduler/tests/baseline -q -m "not sim"

# Полный baseline-набор вместе с full-SIM контрольными точками (49 тестов, ~9 c):
python -m pytest PyScheduler/tests/baseline -q

# Регенерация всех снимков (после слияния PR, gate SPEC §8.1):
python PyScheduler/tests/baseline/generate_baselines.py
# одного сценария:
python PyScheduler/tests/baseline/generate_baselines.py golden_tti5

# Full-SIM контрольная точка вручную:
python PyScheduler/tests/baseline/sim_checkpoint.py --duration 1000 \
    --out PyScheduler/tests/baseline/baselines/sim1000_qosaware.json

# Существующие наборы репозитория (своего conftest с sys.path нет — запуск из PyScheduler/):
cd PyScheduler && python -m pytest tests/scheduler_tests -q && cd -        # BUG-SCH-008: на dev не собирается
cd PyScheduler && python -m pytest tests/mobility_module_tests -q && cd -  # из корня — 5 ошибок коллекции
```

**Состав снимков (`baselines/`, 27 файлов: 23 сценарных + 4 full-SIM).**

| Снимок | Сценарий | Раздел SPEC |
|---|---|---|
| `golden_tti5.json` | золотой сценарий TTI 5 (BestCQI) — числа §4.1 | 9.1 |
| `wb_<alg>.json` (8) | WB-матрица: восемь алгоритмов, TTI 5–7 | 9, матрица покрытия |
| `fd_sb_full_<alg>.json` (3) | контрастные SB CQI, таблица per-RBG выборов | 9.2 |
| `fd_sb_missing_<alg>.json` (3) | пустой SB → fallback WB | 9.2 |
| `fd_sb_partial_<alg>.json` (3) | короткий список SB + нули внутри | 9.2 |
| `rr_multitti.json` | ротация `rr_*_offset`, TTI 5–8 | 9.3 |
| `qos_basic_single_bearer.json` | BasicQosAware: один bearer/UE, grants + per-QCI | 9.4 |
| `qos_aware_multiflow.json` | QosAware: потоки (ue, qci), `allocation_flow` | 9.4 |
| `conflict_rbg_contended.json` | спор двух UE за RBG (AS-IS, без повторной выдачи) | 9.5 |
| `conflict_pdcch_limited.json` | `max_dl_cce_allowance=1`, второй UE заблокирован | 9.5 |
| `sim100/500/1000_qosaware.json`, `sim1000_bestcqi.json` | full-SIM, seed 42, профиль `sim_with_manager_qos` (TEST_MODULES.py :234), verbose-лог и StatsManager отключены | 9.6, 8.2 |

Каждый снимок несёт `meta`: `dev_revision` (`6c0aae2`), `seed` (null для детерминированных сценариев — ГСЧ не используется), конфигурацию (bandwidth, num_frames, профили UE/потоков, kwargs планировщика), список TTI и `meta.env`.

**Что фиксирует снимок (наблюдаемое состояние после каждого TTI, схема 2):** `allocation`, `pdcch_stats` (включая `pdcch_ue_cce_allocations`), `bitmap`, `last_ue_transmitted_bits`, `priority_list` (порядок + метрики), владельцы RB в сетке по слотам (`grid_rb_owners`), таблица per-RBG выборов (`rbg_choice_table`), состояние UE (`average_throughput`, `current_dl_throughput`), `cqi_map` (WB+SB), контракт словаря `user` (`user_dict_keys` + два поля, которые алгоритмы пишут поверх этапа 1: `bs_buffer_size`, `instant_rate`), `buffer_status` (per-QCI: размер, lcid, qci, hol_delay), `qci_tracker` (current/average throughput per QCI), `scheduler_extra` (`rr_*_offset`, `allocation_flow`, параметры QoS). Зачем схема расширена и как доказано отсутствие изменения поведения — [SPEC §8.2](specs/SPEC-FEAT-SCH-008.md).

**Правила сравнения** ([SPEC §8.2](specs/SPEC-FEAT-SCH-008.md)): дискретные значения (списки PRB/RBG, UE, QCI/LCID, CCE, offsets, байты/биты) — **точно**; float — относительный допуск `1e-9` (`baseline_compare.FLOAT_REL_TOL`), широкий epsilon запрещён (R2); `sch_total_time_us` и прочие wall-clock показатели в снимки не входят; `meta.env` — информационное поле, исключено из сравнения (`baseline_harness.assert_matches_baseline` отфильтровывает ключ `env`), чтобы снимок проверялся на машине, отличной от той, на которой построен; full-SIM — sha256 полной истории allocation побитово, агрегаты по тем же правилам.

**Порядок обновления снимков (gate SPEC §8.1):** PR слит → обновить локальную `dev` → `generate_baselines.py` (все сценарии) → full-SIM точки (§4.3, команды) → в каждом снимке обновить `meta.dev_revision` → `pytest PyScheduler/tests/baseline -q` зелёный → коммит регенерации отдельным коммитом/PR с пометкой «baseline refresh после слияния <PR>». Если тест падает **до** слияния — миграцию остановить: дефект исправляется в текущей поставке или откатывается. Расширение схемы снимка — частный случай с дополнительным требованием доказать побайтовое совпадение прежних полей (порядок из пяти пунктов — [SPEC §8.2](specs/SPEC-FEAT-SCH-008.md)).

**Состав каталога.**

```
tests/baseline/
├── baseline_scenarios.py      — детерминированные сценарии + сбор снимков (capture_tti)
├── baseline_compare.py        — правила сравнения (SPEC §8.2)
├── generate_baselines.py      — генератор снимков (CLI)
├── sim_checkpoint.py          — full-SIM контрольные точки (CLI)
├── baseline_harness.py        — хелперы сравнения снимков (load/assert)
├── conftest.py                — sys.path + регистрация маркера `sim`
├── test_golden_tti5.py        — золотой сценарий (SPEC §9.1)
├── test_matrix.py             — WB/FD-матрица (SPEC §9, 9.2)
├── test_rr_multitti.py        — RR-ротация (SPEC §9.3)
├── test_qos.py                — QoS grants + per-QCI (SPEC §9.4)
├── test_conflict.py           — сеточный конфликт (SPEC §9.5)
├── test_sim_checkpoints.py    — full-SIM точки (SPEC §9.6), маркер `sim`
└── baselines/                 — 27 снимков (23 сценарных + 4 full-SIM)
```

**Ограничения харнеса и известные отклонения AS-IS** (зафиксированы, не исправляются — R2):

1. Глобальные кэши канала (`CHANNEL_MODEL.py` :158–160) не сбрасываются между прогонами в одном процессе — `sim_checkpoint.py` очищает их перед каждым прогоном; карточка [BUG-SCH-011](audit.md#bug-sch-011), запрос владельцу модели канала ([features.md §2](features.md)).
2. QoS-алгоритмы падают на `SimpleBufferManager` — WB-ячейки матрицы для BasicQosAware/QosAware строятся на `LayeredBufferManager`; карточка [BUG-SCH-012](audit.md#bug-sch-012) с готовым текстом запроса владельцу BS_MODULE.
3. `GET_TTI_STATUS` неприменим (RES_GRID.py :518 вызывает `GET_RB(tti, freq_idx)` при сигнатуре `GET_RB(tti, slot_id, freq_idx)`) — снимок сетки строится напрямую из `rb_map`.
4. `start_simulation()` выполняется только при `enable_verbose_log(to_file=True)` (SIMULATION_MANAGER.py :1331 — тело внутри `if self._to_file:`) — `sim_checkpoint.py` включает его и пишет `output.txt` во временный каталог.
5. `layered_buffer_tests`: 14 passed / 1 failed (`test_pdcp_discard_timer::test_packet_discard`) — существующее падение на `6c0aae2`, объект вне модуля SCHEDULER.
6. CQI=0 через `schedule()` недостижим (фильтр :592 + `continue` :919) — кейс покрывается юнит-тестами стратегий в M1+ ([BUG-SCH-007](audit.md#bug-sch-007)).
7. `scheduler_tests` на чистом `dev` не собираются (4 ошибки коллекции) — [BUG-SCH-008](audit.md#bug-sch-008); локальная верификация прогоняла набор, восстановленный по разделу «Как проверить» карточки: 26 passed + 1 xfailed. Восстановления в патчи не входят (правки чужого набора — по согласованию с владельцем репозитория).
8. `mobility_module_tests` недетерминирован **как набор**: `RandomWaypointModel` разыгрывает destination из глобального нефиксируемого `np.random` (MOBILITY_MODEL.py :305–306), поэтому от прогона к прогону падает 0–1 тест из разных мест набора (наблюдено на трёх полных прогонах одного стенда; изолированно каждый тест проходит). Практический вывод: стабильное падение полного прогона одно — `layered_buffer_tests::test_packet_discard`, а `mobility_module_tests` даёт плавающее второе падение, и это не дефект baseline. Зона MOBILITY_MODEL, пересекается с RNG-направлением (PR #38 открыт, PR #37 с тем же содержимым закрыт без слияния).
9. Связность харнеса full-SIM с публичным API оркестратора: `sim_checkpoint._Recorder` подменяет `SchedulerInterface.schedule` на уровне класса — планировщик создаётся внутри `start_simulation()` (SIMULATION_MANAGER.py :1357), и до экземпляра снаружи не добраться. Решение осознанное: перехватывается единственный публичный вход модуля, runtime не меняется. Следствие: если на шагах M1+ сигнатура `schedule()` или способ создания планировщика изменятся, харнес обновляется в той же поставке (R4); регенерация снимков при сохранении поведения не нужна.

### 4.4 Контрактный набор M1 (`tests/scheduler_arch_tests/`)

```bash
python -m pytest PyScheduler/tests/scheduler_arch_tests -q                 # 55 passed, <1 c
# оба каталога собираются вместе в любом порядке (из PyScheduler/):
cd PyScheduler && python -m pytest tests/baseline tests/scheduler_arch_tests -q      # 104 passed
cd PyScheduler && python -m pytest tests/scheduler_arch_tests tests/baseline -q      # 104 passed
```

Состав: `arch_support.py` (построители контекстов на хелперах `baseline_scenarios`), `conftest.py` (`sys.path` + фикстуры `wb_ctx`/`fd_ctx`/`layered_ctx`), `test_views_m1.py` (55 контрактных тестов: структурность протоколов, паритет сигнатур, делегирование адаптеров, сохранность якорей и шапки модуля, побочные эффекты инвентаризации). Что именно проверяется и как подтверждено мутациями — [SPEC §4.4](specs/SPEC-FEAT-SCH-008.md) и [SPEC §8.3](specs/SPEC-FEAT-SCH-008.md); инвентаризация входов восьми алгоритмов — [SPEC §4.2](specs/SPEC-FEAT-SCH-008.md).

Общие хелперы обоих каталогов лежат в модулях с уникальными именами (`baseline_harness.py`, `arch_support.py`), а не в `conftest.py`, и тесты не используют `from conftest import ...`: имя модуля `conftest` в дереве `PyScheduler/tests/` не уникально, и при сборке двух каталогов одним запуском импорт разрешается в чужой файл (`ImportError` → падение коллекции) — [BUG-SCH-013](audit.md#bug-sch-013), правка `41a163b`.

`ruff` запускается **из корня репозитория** (`ruff check --config .ruff.toml …`): в конфиге не задан `src`, поэтому классификация first-party зависит от каталога запуска, а вместе с ней — требование к порядку импортов (I001). Замеры и готовый текст запроса владельцу репозитория — [CFG-SCH-001](audit.md#cfg-sch-001).

### 4.5 Фактические числа прогонов (2026-11-01, повторно 2026-11-02)

Окружение: python 3.11.2, numpy 2.4.6, pytest 9.2.0, ruff 0.17.3. Проверено дважды: на ветке `feat/SCH-008-M1-protocols` (`f14885c`) и на свежем клоне `dev@6c0aae2` с применёнными пятью патчами (§4.6) — результаты совпадают, дерево побайтово идентично. Повторено 2026-11-02 на ветке с коммитом документации `docs/scheduler/`: числа те же (104 passed в двух каталогах; 264 passed / 1 failed в полном прогоне) — документация на тесты не влияет; с фикс-коммитами `c3c5164` и `9379b3d` числа также те же (тесты заменены/расширены без изменения счёта, снимки не тронуты, поведение не менялось).

```bash
python -m pytest PyScheduler/tests/baseline -q -m "not sim"                 # 44 passed, 5 deselected
python -m pytest PyScheduler/tests/baseline -q                              # 49 passed (~9 c)
python -m pytest PyScheduler/tests/scheduler_arch_tests -q                  # 55 passed (<1 c)
cd PyScheduler && python -m pytest tests/baseline tests/scheduler_arch_tests -q   # 104 passed
cd PyScheduler && python -m pytest tests -q --ignore=tests/scheduler_tests  # 264 passed, 1 failed, 473 subtests
ruff check --config .ruff.toml PyScheduler/tests/baseline PyScheduler/tests/scheduler_arch_tests  # All checks passed
# Замечания ruff считаются по строкам вывода, содержащим '.py:' — concise-вывод добавляет итоговые строки
# ('Found N errors.', '[*] N fixable…'), поэтому 'grep -c .' даёт N+2 и для сравнения не годится:
ruff check --config .ruff.toml --output-format concise PyScheduler/SCHEDULER.py | grep -c '\.py:'  # 60 (= dev, 0 новых)
ruff check --config .ruff.toml --output-format concise PyScheduler | grep -c '\.py:'               # 1095 (= dev)
ruff check --config .ruff.toml --output-format concise PyScheduler/SCHEDULER.py | awk -F: '$2>3245' | wc -l   # 0 (раздел M1 чист)
python PyScheduler/tests/baseline/generate_baselines.py                     # повторно: файлы не меняются
```

- Единственное падение полного прогона — `tests/layered_buffer_tests/test_pdcp_discard_timer.py::test_packet_discard`, существует на `dev@6c0aae2` (до патчей типичный результат 160 passed / 1 failed). Полный прогон на ветке повторён трижды подряд — 264 passed / 1 failed каждый раз. На `dev` изредка даёт 159 passed / 2 failed: вторым падает `mobility_module_tests/test_mobility_module_spcf.py::TestGaussMarkovSpecific::test_boundary_threshold_activates_near_y_max`, который в изолированном запуске проходит стабильно (5/5) — недетерминированность чужого набора, описана в §4.3; к модулю Scheduler отношения не имеет.
- `tests/scheduler_tests/` не собирается ни до, ни после патчей: `ModuleNotFoundError: log_utils` (4 ошибки коллекции) — [BUG-SCH-008](audit.md#bug-sch-008), зона владельца репозитория.
- В `SCHEDULER.py` 60 замечаний ruff — столько же, сколько на `dev` (60/60), во всём `PyScheduler/` — 1095 на `dev` и 1095 на ветке; в строках >3245 (раздел M1) — 0. Запуск — из корня репозитория; счёт — по строкам с `.py:` (см. команду выше).
- Детерминизм снимков: две регенерации подряд байт-в-байт идентичны, md5 набора из 27 снимков `313565fb71b291ea695daf1920a53317` совпадает на ветке и на чистом клоне с патчами.
- Мутационный контроль: 13/13 преднамеренных мутаций пойманы своими тестами (перечень — [SPEC §8.3](specs/SPEC-FEAT-SCH-008.md)).
- Докстринги и длина строк (AST- и построчный контроль): 0 узлов без докстринга из 128 в новых каталогах и из 56 в разделе M1 `SCHEDULER.py`; 0 строк длиннее 100 символов ([REV-SCH-001](audit.md#rev-sch-001)).

### 4.6 Поставка патчами: состав, порядок применения, md5

Основной способ поставки — ветка `feat/SCH-008-M1-protocols` в форке ответственного (`dokbrawn/project_py_scheduler`) и PR из неё в ветку `dev` апстрима (`sherokiddo/project_py_scheduler`); рабочая ветка проекта — `dev`, `main` отстаёт от неё и базой PR не является. Патчи — резервный способ передачи того же содержимого: получены `git diff` между коммитами ветки и применяются на чистое дерево `dev@6c0aae2`. Топология ветки: `dev@6c0aae2` → `9d50a94` (M0) → `41a163b` (инфраструктурная правка) → `2472cb8` (M1) → `ba479ee` (выравнивание по практикам кодревью апстрима + схема снимков 2) → `f14885c` (перенос рабочих файлов поставки в комплект документации модуля) → `bdd091f` (документация модуля в `docs/scheduler/`, §8) → `c3c5164` (фикс по замечанию внешнего ревью PR #44: view возвращает копию SB-списка) → коммит ревизии 13 комплекта → `9379b3d` (фикс второго замечания того же ревью: явная семантика отказа `GridView.allocate_rbg`, откат не owner-safe) → коммит ревизии 14 комплекта.

| # | Файл патча | Строк | Файлов | md5 | Коммит | Содержание |
|---|---|---|---|---|---|---|
| 1 | `SCH-008-M0.patch` | 2911 | 39 | `eb7a8b8156f5137c38523e31e1c4be45` | `9d50a94` | шаг M0: baseline-харнес — всё в `PyScheduler/tests/baseline/` (+2650 строк: characterization-тесты, 27 снимков с `meta.dev_revision=6c0aae2` и `meta.env`) |
| 2 | `SCH-008-fix-test-imports.patch` | 256 | 9 | `2baec380f56802fee320aea039758d37` | `41a163b` | инфраструктура: хелперы снимков вынесены из `conftest.py` в `baseline_harness.py` (+87/−47) — иначе два тестовых каталога не собираются одним запуском ([BUG-SCH-013](audit.md#bug-sch-013)) |
| 3 | `SCH-008-M1.patch` | 1407 | 5 | `6d5e2d1301f4573f8bc2cd0ea8cbf0b4` | `2472cb8` | шаг M1: шесть протоколов + шесть тонких адаптеров в `SCHEDULER.py` (+310/−4), каталог `PyScheduler/tests/scheduler_arch_tests/` |
| 4 | `SCH-008-M1-review-alignment.patch` | 2456 | 41 | `28c34ea154793c812fa10fa30104af59` | `ba479ee` | выравнивание M0/M1 по практикам кодревью апстрима ([REV-SCH-001](audit.md#rev-sch-001)) + схема снимков 2: 23 снимка перегенерированы, `SCHEDULER.py` +42/−9 (все вставки после строки 3269), докстринги, именованные константы, AST-проверки вместо архитектурозависимых тестов |
| 5 | `SCH-008-docs-consolidation.patch` | 782 | 8 | `9c3e5119ee89e16e2c925d3e7ee3b838` | `f14885c` | консолидация документации: удалены `RUNBOOK.md` и `INVENTORY-M1.md` из репозитория (содержимое растворено в §4.3–4.7 этого файла и в [SPEC §4.2–4.4, §8.1–8.2](specs/SPEC-FEAT-SCH-008.md)), 16 ссылок в тестовом коде перепривязаны на разделы документации — все замены построчные, `SCHEDULER.py` остался 3584 строки, 11 legacy-якорей на месте |
| 6 | `SCH-008-module-docs.patch` | — | 5 | — | коммит документации |
| 7 | `SCH-008-cqi-view-copy-fix.patch` | 62 | 2 | `306d5103fabaecd37b5c06969d79b5cf` | `c3c5164` | фикс по замечанию внешнего ревью PR #44 (Codex, P2): `LegacyCqiView.get_sb` возвращает копию SB-списка (read-only view, [SPEC §2.1](specs/SPEC-FEAT-SCH-008.md)); тест, фиксировавший живой список как AS-IS, заменён 1:1 — число тестов и снимки не изменились (+20/−9) |
| 8 | `SCH-008-docs-rev13.patch` | — | 5 | — | коммит ревизии 13 | ревизия 13 комплекта: фиксация фикс-коммита `c3c5164` в топологии, таблицах поставки и карточке [REV-SCH-001](audit.md#rev-sch-001). Md5 здесь нет намеренно: патч содержит этот самый файл |
| 9 | `SCH-008-gridview-failure-semantics.patch` | 63 | 2 | `da5337d4e004afc9e4a8c34b02f361e2` | `9379b3d` | фикс второго замечания внешнего ревью PR #44 (Codex, P2): docstring `GridView.allocate_rbg`/`LegacyGridView.allocate_rbg` вместо обещания отката формулируют owner-unsafe семантику (BUG-SCH-009); фаза 2 теста `test_grid_view_allocate_rbg_delegates_and_mutates_grid` фиксирует путь отказа (+25/−2) |
| 10 | `SCH-008-docs-rev14.patch` | — | 5 | — | коммит ревизии 14 | ревизия 14 комплекта: фиксация коммита `9379b3d` в карточке [BUG-SCH-009](audit.md#bug-sch-009), топологии и таблицах поставки. Md5 здесь нет намеренно: патч содержит этот самый файл | комплект документации модуля в `docs/scheduler/` (§8): этот файл, `audit.md`, `diagrams.md`, `features.md`, `specs/SPEC-FEAT-SCH-008.md`. Строк и md5 здесь нет намеренно: патч содержит этот самый файл, поэтому его хеш не может быть в нём зафиксирован — фактические значения лежат в манифесте поставки (REC-S5, `patches/manifest.txt`) |
| — | `SCH-008-all-steps-combined.patch` | — | 48 | — | `6c0aae2..` HEAD ветки | все десять коммитов одним патчем (для быстрой проверки, не для подачи); md5 — в манифесте поставки |

```bash
git clone <репозиторий> && cd project_py_scheduler
git checkout 6c0aae2                                   # dev
for p in SCH-008-M0 SCH-008-fix-test-imports SCH-008-M1 SCH-008-M1-review-alignment \
         SCH-008-docs-consolidation SCH-008-module-docs \
         SCH-008-cqi-view-copy-fix SCH-008-docs-rev13 \
         SCH-008-gridview-failure-semantics SCH-008-docs-rev14; do
    git apply --check /path/$p.patch && git apply /path/$p.patch
done
python -m pytest PyScheduler/tests/baseline PyScheduler/tests/scheduler_arch_tests -q   # 104 passed
```

Проверено 2026-11-02 на свежем клоне `dev@6c0aae2`: десять патчей применяются по порядку (`git apply --check` + `git apply`), дерево после применения побайтово совпадает с HEAD ветки `feat/SCH-008-M1-protocols` (`diff -r` — 0 различий), все прогоны §4.5 зелёные (44/49/55/104 passed, полный прогон 264 passed / 1 существующее падение), снимки после регенерации байт-в-байт идентичны (`313565fb71b291ea695daf1920a53317`), мутационный контроль 13/13. После применения патчей в `PyScheduler/tests/` не остаётся ни одного md-файла. Тела коммитов содержат разделы «Что сделано / Проверки / Что НЕ меняется / Находки / Вопросы ревьюеру» — то есть описание будущего PR уже готово в истории ветки. Оговорка: в теле коммита `ba479ee` упомянуты рабочие артефакты `REVIEW-PRACTICES.md` и `SELF-AUDIT-vs-PR.md` — с ревизии 11 они растворены в [SPEC §2.1](specs/SPEC-FEAT-SCH-008.md) (20 правил кодревью апстрима с цитатами) и в карточке [REV-SCH-001](audit.md#rev-sch-001) (сверка собственной поставки по каждому правилу); история коммитов не переписывалась намеренно — иначе поехали бы md5 всех патчей. Если подача PR всё же понадобится: каждый шаг — отдельный PR от своей ветки (база `dev`), squash-слияние, описание по структуре [SPEC §2.1.3](specs/SPEC-FEAT-SCH-008.md) («Что сделано» → «Проверки» → «Что НЕ меняется» → находки → вопросы ревьюеру); тела коммитов эти разделы уже содержат.

### 4.7 Перепривязка якорей `:NNN`

Документы ссылаются на конкретные строки `SCHEDULER.py`/`BS_MODULE.py`/`RES_GRID.py`, действительные для `dev@6c0aae2`. Порядок перепривязки при уходе `dev` или при правках чужого кода:

1. Найти фактические строки по цитате кода, а не по прежнему номеру: `grep -n "<цитата>" PyScheduler/SCHEDULER.py`.
2. Обновить номера одним коммитом с причиной («перепривязка якорей после <PR/коммита>»), не смешивая с содержательными правками.
3. Тесты на номера строк **не пишутся** намеренно: проверяется содержимое — `test_documentation_quotes_present_in_module` сверяет 9 уникальных цитат кода из документации M1 (прежний `test_documentation_anchors_still_valid` проверял 11 жёстких номеров строк, из которых две цитаты встречались дважды: :1993/:2374 и :3029/:3126). Тест, завязанный на номер строки чужого кода, падал бы на любой правке владельца модуля, то есть был бы архитектурозависимым — такие ревью апстрима отклоняет (PR #37, [SPEC §2.1.1](specs/SPEC-FEAT-SCH-008.md) A2).
4. Новый код добавляется в конец файла именно поэтому: якоря на чужой код не сдвигаются (контроль — `test_m1_section_appended_after_legacy_code`).

### 4.8 Порядок поставки и интеграции

Принятый в проекте порядок (договорённости команды, октябрь–ноябрь 2026): каждая крупная фича — отдельная ветка от актуальной `dev`, коммиты небольшие и логически завершённые, слияние — PR в `dev`, легаси сохраняется до проверки новой реализации.

1. **Ветка от актуальной `dev`.** Перед началом шага — `git fetch upstream && git rebase upstream/dev`; если `dev` ушёл с `6c0aae2`, перепривязать якоря (§4.7) и регенерировать снимки (§4.3).
2. **Один шаг — один PR в `dev`.** Gate шага — [SPEC §8.1](specs/SPEC-FEAT-SCH-008.md): зелёные прогоны (§4.5), 0 новых замечаний `ruff`, снимки не изменились либо изменение обосновано и согласовано (R2). Слияние — squash (практика апстрима).
3. **Легаси работает.** Маршрут фабрики `legacy` сохраняется до конца миграции (R5), удаление legacy-классов — отдельный PR после M8e (R8).
4. **Задачи владельцам других модулей — после слияния.** До слияния в `dev` чужие модули не правятся: пересечения оформляются запросами ([features.md](features.md) §2), а не изменениями в чужой зоне ответственности.
5. **Смежные модули подключаются в своей зоне.** Владелец соседнего модуля забирает актуальный `dev` в свою ветку, делает коннектор у себя, проверяет и льёт в `dev`. Пример для Scheduler: контракт коллектора статистики публикует владелец StatsManager, коллекторы в `SCHEDULER.py` добавляет ответственный за Scheduler ([features.md](features.md) §2).
6. **Проверка багов — пять шагов.** Обнаружение → тест, воспроизводящий проблему → подтверждение влияния → исправление → повторный прогон того же теста и фиксация результата; карточка в [audit.md](audit.md) хранит все пять пунктов (формат карточек — [audit.md §1](audit.md)).
7. **Профилирование до и после.** Для шагов, меняющих горячий путь (M4+), снимается профиль `cProfile` до и после на сценарии 1000 TTI; архитектурное разделение само по себе ускорением не считается (методика — [SPEC §8.2](specs/SPEC-FEAT-SCH-008.md)). Урок, зафиксированный практикой: правка шапки модуля на две строки сдвигает все якоря ниже, поэтому изменения шапки делаются заменой «строка в строку» с проверкой `wc -l` и `git diff --numstat` до и после.

## 5. Основные компоненты

| Файл / класс / функция | Назначение | Где используется |
|---|---|---|
| `SCHEDULER.py::SchedulingGrant` (:117) | dataclass гранта извлечения: `ue_id`, `num_bytes`, `lcid`, `ndi`, `harq_process_id`, `rv`; валидация `num_bytes ≥ 0`, `rv ∈ [0..3]` (:139); `to_dict()` (:160) | создаётся в `_logical_channel_multiplexing` (:811), потребляется `BS_MODULE::get_packets` (issue #27 — полная интеграция) |
| `SCHEDULER.py::CQIMap` (:191) | хранилище CQI на UE: `wb_cqi`, `sb_cqi` (per-RBG, длина = num_rbg), метки времени обновлений | пишет `_refresh_cqi` (:504), читают `_get_wb_cqi` (:1064) / `_get_sb_cqi` (:1084) |
| `SCHEDULER.py::SchedulerInterface` (:216) | оркестратор этапов 0–8 + фабрика: `create` (:235), `available_algorithms` (:277), `schedule` (:346), `get_stats` (:438), методы этапов (:504–1168), абстрактные `_calculate_priorities` (:1172) и `_allocate_pdsch` (:1197), HARQ-заготовка (:1231) | `SIMULATION_MANAGER` (создание :1357, вызов :1701) |
| `SCHEDULER.py::PDCCHManager` (:1262) | бюджет CCE: `_validate_parameters` (:1307), таблица `_calculate_total_cce` (:1347), `get_aggregation_level` (:1416), `check_cce_availability` (:1465), `allocate_cce` (:1496), `reset_tti` (:1545), `release_cce` — **заглушка** (:1568), `get_stats` (:1660) | этап 5 `_allocate_pdcch` (:769) |
| `SCHEDULER.py::AdaptiveModulationAndCoding` (:1703) | `CQI_TO_MCS` (:1709), `GET_BITS_PER_RB` (:1730) — единственная точка перевода CQI в биты (этапы 4.5, 6, 7, статистика); `get_stats` (:1755) читает кэш `_last_*` планировщика | весь модуль; TODO замены на TBS-таблицы :1749 (issue #30) |
| `BestCQIScheduler` (:1877) | priority = WB CQI (:1908); жадное TD-распределение (:1931) | фабрика `'BestCQI'` |
| `RoundRobinScheduler` (:1985) | ротация очереди от `rr_ue_offset` (:2015); по 1 RBG каждому по кругу (:2040); оценка PDSCH — pass-through (:1996) | фабрика `'RoundRobin'` |
| `ProportionalFairScheduler` (:2103) | priority = `instant_rate / (average_throughput/1000)` (:2118, fallback при нулевой истории :2135); жадное TD (:2160) | фабрика `'ProportionalFair'` |
| `FDxBestCQIScheduler` (:2215) | на каждом RBG побеждает max SB CQI с fallback на WB (:2269, выбор :2304) | фабрика `'FD_BCQI'` |
| `FDxFairGreedyScheduler` (:2339) | RR-ротация в TD (:2394) + per-RBG выбор по SB CQI с ограничением «один UE — один RBG за раунд» (`served_in_round` :2448–2505); до ревью PR #26 класс носил имя FD_RR | фабрика `'FD_FGS'` |
| `FDxProportionalFairScheduler` (:2524) | WB PF задаёт порядок выдачи CCE (:2588); на каждом RBG побеждает max PF-метрики по SB CQI (:2631, цикл :2645) | фабрика `'FD_PF'` |
| `BasicQosAwareScheduler` (:2704) | QoS (PR #39): ранжирование — HOL-группировка потоков + GBR-приоритет `1_000_000 + best_gbr_group` (:2764–2802); per-RBG выбор — произведение `hol_delay × gbr_weight × cqa_metric` (:2930; Coita или PF по `metric_mode`); docstring предупреждает: один bearer на UE | фабрика `'BasicQosAware'` (:262) |
| `QosAwareScheduler` (:2960) | QoS (PR #39): приоритет по потокам (UE, QCI): GBR — масштабное смещение + urgency от HOL delay и packet delay budget (`QCI.get_delay_budget`), per-RBG выбор потока по метрике `sigmoid(hol, pdb) × pf` (:3128–3133); переопределяет `_logical_channel_multiplexing` (:3175–3222) | фабрика `'QosAware'` (:263) |
| roadmap-комментарий (:3227–3245) | планы из самого кода: BufferManager, DCI + AllocationTracker, каноничный output, QoS, tie-break, исследование венгерского/жадного распределения RBG, оценка средней пропускной способности PF, защита голодающих UE | основа [features.md](features.md) |
| раздел M1 (:3247–3268), ветка `feat/SCH-008-M1-protocols` | шапка-комментарий «Интерфейсы стратегий и view-адаптеры»: что добавлено, какие правила миграции соблюдены (R1/R2/R5), почему раздел в конце файла, что намеренно отложено (целевая форма `CqiView` — CQI-трек C1, `abc.ABC`-иерархии аллокаторов — шаги M4+) | [SPEC-FEAT-SCH-008](specs/SPEC-FEAT-SCH-008.md) §4.2–4.4, §5.3, §5.7 |
| `CqiView` (:3278), `GridView` (:3296), `AmcView` (:3327), `BuffersView` (:3336), `StatsView` (:3355), `StateView` (:3376) | шесть `typing.Protocol` c `@runtime_checkable` — границы «что алгоритм вправе потреблять»: WB/SB CQI; операции и размеры сетки (`allocate_rbg`, `rbg_size`, `rbg_indices`, `total_rbg`, `prb_count`); bits/RB по CQI; наличие и статус буферов + HOL-задержка; throughput UE и per-QCI, GBR/PDB потока; чтение-запись RR-смещений | потребители переключаются не раньше шага M3 (правила R1/R2); в M1 только объявлены — контроль AST-проверкой `test_legacy_code_does_not_reference_views` |
| `LegacyCqiView` (:3403), `LegacyGridView` (:3424), `LegacyAmcView` (:3457), `LegacyBuffersView` (:3469), `LegacyStatsView` (:3516), `LegacyStateView` (:3557) | шесть тонких адаптеров поверх текущих полей оркестратора: делегируют существующим методам (`cqi_map`, `lte_grid`, `amc`, `bs.buffer_manager`, `ue.average_throughput`/`qci_tput_tracker`, `rr_*_offset`) и не дублируют логику (R5); `LegacyStatsView.users_by_id` типизирован `Dict[int, Any]`, так как `UserEquipment` модуль не импортирует | в M1 создаются только тестами (`tests/scheduler_arch_tests/`), в рантайм планировщика не внедрены |

Строки 3246–3584 существуют только в ветке `feat/SCH-008-M1-protocols` (коммиты `2472cb8`…`9379b3d`, поставка патчами — §4.6); в `dev@6c0aae2` файл заканчивается на строке 3245. Инвентаризация фактических входов и побочных эффектов всех восьми алгоритмов — [SPEC §4.2](specs/SPEC-FEAT-SCH-008.md), соответствие входов протоколам M1 — [SPEC §4.3](specs/SPEC-FEAT-SCH-008.md), гарантии шага и контролирующие их тесты — [SPEC §4.4](specs/SPEC-FEAT-SCH-008.md).

Матрица переопределений (различия алгоритмов локальны — базовый класс реализует весь цикл):

| Планировщик | `_calculate_priorities` | `_apply_pdsch_estimation` | `_allocate_pdsch` | доп. состояние |
|---|---|---|---|---|
| BestCQI (:1877) | WB CQI (:1908) | базовый, порог 95% (:712) | жадный TD (:1931) | — |
| RoundRobin (:1985) | ротация очереди (:2015) | pass-through (:1996) | 1 RBG по кругу (:2040) | `rr_rbg_offset`, `rr_ue_offset` |
| ProportionalFair (:2103) | r/R (:2118) | базовый (95%) | жадный TD (:2160) | — |
| FD_BCQI (:2215) | WB CQI (:2250) | pass-through (:2243) | per-RBG max SB CQI (:2269) | — |
| FD_FGS (:2339) | RR-ротация (:2394) | pass-through (:2377) | per-RBG SB CQI + `served_in_round` (:2423) | `rr_*_offset` |
| FD_PF (:2524) | WB PF (:2588) | pass-through (:2571) | per-RBG PF по SB CQI (:2631) | — |
| BasicQosAware (:2704) | HOL-группировка + GBR-приоритет (:2764–2802) | pass-through (:2753) | per-RBG произведение hol×gbr×cqa (:2930) | `metric_mode` (Coita/PF) |
| QosAware (:2960) | per-flow GBR/urgency (:2999–3038) | pass-through (:2988) | per-RBG выбор потока (:3044, метрика :3128–3133) | `allocation_flow`, override мультиплексирования (:3175–3222) |

Ключевые константы (дефолты `__init__` :294 и пороговые значения): `pcfich=2`; `max_dl_ue_tti=None` (без лимита); `window_size=100`, `enable_window=True`; `wb/sb_cqi_upd_interval=1` (каждый TTI); порог оценки PDSCH 95% `rb_per_slot` (:733); aggregation level: CQI 13–15→1, 10–12→2, 7–9→4, 1–6→8 CCE (:1416); бюджет CCE при 10 МГц: PCFICH 1/2/3 → 8/25/41 (:1347); RBG при 10 МГц: 3 RB → 17 RBG (`RES_GRID.py::RBG_SIZE_TABLE` :293); bits/RB (PCFICH=2, 1 порт): CQI 15→798, 13→755, 7→326 (:1730).

## 6. Один типичный сценарий

Вызов `schedule(tti=5, users)` из `SIMULATION_MANAGER.run_tti()` (BestCQI, 10 МГц, сценарий раздела 4 — все числа из фактического прогона на `ae9db40`):

1. **Этап 0** `_refresh_cqi` (:504) — WB CQI 13/7/15 записаны в `cqi_map` (SB-списки пусты — TD-сценарий).
2. **Этап 1** `_filter_eligible_ues` (:559) — все трое допущены (буферы существуют и непустые, CQI валидны); побочно: **всем** users сброшен `ue.current_dl_throughput = 0`, дописан `bs_buffer_size` = 1000/5000/300.
3. **Этапы 2–4** — окно: `_ue_queue` = [1,2,3] с ротацией (:610/:653); приоритеты = WB CQI; `priority_list` = [3, 1, 2] (:684).
4. **Этап 4.5** `_apply_pdsch_estimation` (:712) — потребность: UE3 = 2400//798 = 3 RB, UE1 = 8000//755 = 10 RB, UE2 = min(40000//326, 50) = 50 RB; с UE2 сумма 63 ≥ порога 47.5 → **UE2 отсечён** («13/50 RB = 26.0%, 1 UE excluded»), CCE не тратит.
5. **Этап 5** `_allocate_pdcch` (:769) — `reset_tti()`, CQI 15/13 ∈ [13–15] → по 1 CCE в порядке priority-списка: `{3: 1, 1: 1}`, занято 2 из 25 (8%).
6. **Этап 6** `_allocate_pdsch` (:1931) — жадно: RBG 0–1 → UE3 (остаток буфера исчерпан), RBG 2–5 → UE1; `allocation = {1: [6..17], 2: [], 3: [0..5]}`, 18 из 50 RB; каждый RBG занят атомарно через `ALLOCATE_RBG` (оба слота, откат при неудаче).
7. **Этап 7** `_process_buffers` (:880) — вместимость по WB CQI (UE1: 12·755 = 9060 бит → 1132 B; UE3: 6·798 = 4788 бит → 598 B) → гранты `SchedulingGrant` → `get_packets` извлекает 1000 B и 300 B → `UPD_DL_THROUGHPUT_BPS`: 8004 и 2404 бита (включая остаток `max_bits mod 8`, помеченный в коде «временный костыль» :934), EWMA-среднее 16008/4808 бит/с; с `6c0aae2` здесь же обновляется `qci_tput_tracker` (:967–975), а ветки «не запланирован»/«нет вместимости» завершаются `continue` (:919, :944) — один нулевой шаг EWMA вместо двух ([BUG-SCH-001](audit.md#bug-sch-001) закрыт).
8. **Этапы 7.5–8** — счётчики в `_last_*` (:1104/:1154); возврат `{allocation, statistics: {}, bitmap (17 бит на UE), pdcch_stats}` (:996); вызывающий код результат отбрасывает (:1451), статистика уходит наружу через `StatsManager` → `get_stats()`.

## 7. Ограничения и открытые вопросы

- Ёмкость RB считается формулой RE·Qm·R, а не таблицами TBS TS 36.213 §7.1.7.2 (TODO в коде :1749, отклонение ~8–10% по оценке автора TODO, issue #30); таблица `CQI_TO_MCS` (:1709) смещена относительно TS 36.213 Table 7.2.3-1 (+2 индекса для CQI 5–13) — ёмкость систематически завышена ([OPT-SCH-002](audit.md#opt-sch-002)); учёт CRS в первом слоте спорный (там же).
- Учёт CCE — счётчик без позиций и поисковых пространств: проверяется только необходимое условие размещения ([OPT-SCH-004](audit.md#opt-sch-004)); `release_cce` — заглушка (:1568), провал PDSCH после успешного PDCCH оставляет CCE занятыми до `reset_tti()` следующего TTI ([BUG-SCH-005](audit.md#bug-sch-005)).
- HARQ не реализован (заготовка :341/:1231, поля гранта не используются, issue #27); QoS-планировщики слиты в `dev` в PR #39 (`6c0aae2`: `BasicQosAwareScheduler` :2704, `QosAwareScheduler` :2960, ключи фабрики :262–263), LCP-приоритизация в извлечении пакетов по QCI остаётся открытой частью FEAT-SCH-009; скользящее окно — проприетарное решение с TODO-альтернативами в docstring (:610).
- Поведенческие нюансы: двойное нулевое обновление EWMA у незапланированных UE существовало до `6c0aae2` ([BUG-SCH-001](audit.md#bug-sch-001), закрыт побочно с qci-трекером PR #39 — проверено замером), «залипание» двух счётчиков статистики в пустом TTI ([BUG-SCH-004](audit.md#bug-sch-004)), сохранение устаревшего CQI при невалидном отчёте ([BUG-SCH-002](audit.md#bug-sch-002)), крах этапа 7 при CQI=0 у UE с живым буфером воспроизводился на `ae9db40`, на `6c0aae2` через `schedule()` недостижим ([BUG-SCH-007](audit.md#bug-sch-007)), откат `ALLOCATE_RBG` освобождает RB без проверки владельца ([BUG-SCH-009](audit.md#bug-sch-009), воспроизведено), `available_algorithms()` (:277) возвращает два имени, которые фабрика не принимает ([BUG-SCH-010](audit.md#bug-sch-010)) — детали и статусы в [audit.md](audit.md).
- Инфраструктура: тестовый набор на `dev` неработоспособен ([BUG-SCH-008](audit.md#bug-sch-008)); CI в репозитории отсутствует.
- Открытые вопросы: кто должен потреблять возврат `schedule()` и зачем `statistics` всегда пуст (:1030, [OPT-SCH-005](audit.md#opt-sch-005)); судьба незакрытых веток `feature/HARQ`, `feature/AMC`, `feature/drl-scheduler` относительно `dev` — требует отдельного уточнения.
- Целевая раскладка пакетов проекта (отдельные папки моделей, ядра, планировщиков, возможно MAC-уровня) на момент ревизии 12 не утверждена, поэтому шаг M8b (вынос стратегий в пакет) выполняется только после её фиксации; схема версионирования репозитория не определена; точные имена папок общей `docs/` уточняются; контракт коллектора статистики и единые правила именования метрик не опубликованы — сводка в [audit.md §3](audit.md#3-что-пока-не-удалось-понять), запросы — [features.md §2](features.md).
- Сводка запросов к владельцу репозитория и авторам смежных модулей — [audit.md §3](audit.md#3-что-пока-не-удалось-понять): канонический способ запуска `ruff` и отсутствие `src` в `.ruff.toml` ([CFG-SCH-001](audit.md#cfg-sch-001)); версия интерпретатора для прогонов (3.12.9 в окружении владельца против валидированной 3.11.2); планы по CI (в репозитории отсутствует); готовый текст запроса владельцу `BS_MODULE` по [BUG-SCH-012](audit.md#bug-sch-012); подтверждение двух решений карточки [REV-SCH-001](audit.md#rev-sch-001) (AST-проверки вместо архитектурозависимых тестов на номера строк; схема снимков 2 с побайтовой совместимостью со схемой 1).

## 8. Связанные материалы

Комплект документации модуля — четыре файла по ТЗ плюс спецификация архитектурной фичи. Других md-файлов ни в комплекте, ни в каталоге кода `PyScheduler/` не ведётся: прежние процессные документы (`docs/process/`: DoD, ROADMAP, CHANGELOG) и рабочие файлы поставки (`RUNBOOK.md`, `INVENTORY-M1.md` в каталогах тестов) с ревизии 11 растворены в этих пяти — куда именно, указано ниже.

**Размещение в репозитории (с ревизии 12).** Документация хранится не отдельно от кода, а в репозитории: общая папка `docs/`, внутри — папки модулей с четырьмя базовыми документами и спецификациями в отдельной подпапке. Для Scheduler это:

```
docs/scheduler/
├── README.md                    # этот файл: назначение, границы, как проверить
├── audit.md                     # ревизия модуля: карточки наблюдений
├── diagrams.md                  # диаграммы AS-IS и черновик TO-BE
├── features.md                  # план развития, карточки FEAT, межмодульные запросы
└── specs/
    └── SPEC-FEAT-SCH-008.md     # спецификация архитектурной фичи
```

Обновляет комплект ответственный за модуль вместе с кодом, а не передаёт его владельцу репозитория для ручного включения. Точные имена папок и файлов команда ещё уточняет: переход на другую раскладку — это `git mv` плюс правка относительных ссылок (внешних ресурсов в комплекте нет, только md и Mermaid, поэтому переносимость сохраняется). В каталогах тестов `PyScheduler/tests/` md-файлов нет: рабочие файлы поставки удалены в `f14885c`, их содержимое растворено в этом комплекте. Существующие `PyScheduler/Documentation/*.md` (авторская документация модуля `doc_scheduler_v2.1.0.md` и `TRAFFIC_MODEL.md`) не правятся и не дублируются: комплект описывает фактическое состояние кода на `6c0aae2`, а расхождения с авторским документом зафиксированы карточкой [DOC-SCH-001](audit.md#doc-sch-001). Отчёты по неделям и отчёт по разработке — материалы учебного процесса, в репозиторий кода не кладутся.

- [Диаграммы](diagrams.md) — контекст и классы (AS-IS), последовательность вызовов TTI, переходы состояний, черновик TO-BE; схема сравнения снимков шага M0 — [SPEC §8.2](specs/SPEC-FEAT-SCH-008.md)
- [Ревизия](audit.md) — 24 наблюдения (BUG-SCH-001…013, OPT-SCH-002…006, DOC-SCH-001…004, REV-SCH-001, CFG-SCH-001) с доказательствами, замерами и статусами; таблица сопоставления прежних ID и сводка запросов владельцам (§3) — там же
- [План развития](features.md) — карточки FEAT-SCH-001…009 (две из них — крупные архитектурные разработки: FEAT-SCH-008 с отдельной спецификацией и FEAT-SCH-009 в статусе черновика), дорожная карта модуля (§1.1), связи с issues/PR/ветками, межмодульные зависимости
- [Спецификация SPEC-FEAT-SCH-008](specs/SPEC-FEAT-SCH-008.md) — план реализации разделения Ranker/Allocator (v4): правила миграции R1–R8 и Definition of Done атомарного PR (§2), практика кодревью репозитория и операционный чек-лист поставки шага (§2.1), AS-IS-матрица и инвентаризация входов восьми алгоритмов (§4, §4.2–4.4), целевые контракты (§5), атомарные зелёные шаги M0–M8e и CQI-трек C0–C4 (§8), фактический статус шагов M0/M1 с подтверждениями (§8.3), регрессионная матрица (§9)
- [`Documentation/doc_scheduler_v2.1.0.md`](https://github.com/sherokiddo/project_py_scheduler/blob/6c0aae2/PyScheduler/Documentation/doc_scheduler_v2.1.0.md) — авторская документация модуля (PR #26, после не обновлялась)

**Где теперь живут прежние отдельные документы:**

| Прежний файл | Куда перенесено |
|---|---|
| `docs/process/DOD-FEAT-SCH-008.md` (критерии готовности атомарного PR, gate шага, статусы M0/M1) | [SPEC §2](specs/SPEC-FEAT-SCH-008.md) (правила R1–R8, DoD), [SPEC §2.1](specs/SPEC-FEAT-SCH-008.md) (чек-лист с практикой кодревью репозитория), [SPEC §8.1](specs/SPEC-FEAT-SCH-008.md) (gate), [SPEC §8.3](specs/SPEC-FEAT-SCH-008.md) (фактические статусы и подтверждения по шагам) |
| `docs/process/ROADMAP-SCH.md` (статусы пунктов развития: TODO из кода :3227–3245, шаги M0–M8e, CQI-трек, смежные issues) | [features.md §1.1](features.md) (дорожная карта модуля: TODO из кода, шаги фичи, issues репозитория) и карточки FEAT-SCH-001…009; шаги M0–M8e и трек C0–C4 — [SPEC §8](specs/SPEC-FEAT-SCH-008.md) |
| `docs/process/CHANGELOG-DOC-SCH-001.md` (ревизии комплекта, работы вне комплекта) | таблица ревизий ниже; работы FEAT-SCH-008 по шагам — §4.3–4.7 этого файла и [SPEC §8.3](specs/SPEC-FEAT-SCH-008.md) |
| `PyScheduler/tests/baseline/RUNBOOK.md` (окружение, команды, состав снимков, порядок обновления, известные ограничения AS-IS) | §4.3–4.5 и §4.7 этого файла; правила сравнения снимков — [SPEC §8.2](specs/SPEC-FEAT-SCH-008.md) |
| `PyScheduler/tests/scheduler_arch_tests/INVENTORY-M1.md` (инвентаризация входов и побочных эффектов восьми алгоритмов) | [SPEC §4.2](specs/SPEC-FEAT-SCH-008.md) (инвентаризация), §4.3 (соответствие входов протоколам M1), §4.4 (гарантии шага и контролирующие тесты) |
| `docs/specs/OPT-SCH-001.md` (черновик спецификации от 2026-10-17, статус «заменён») | содержимое давно заменено [SPEC-FEAT-SCH-008](specs/SPEC-FEAT-SCH-008.md) v4; прежние обозначения OPT-SCH-001/007 сохранены в таблице соответствия [audit.md §1](audit.md), соответствие старого плана T1–T7 шагам M0–M8e — [SPEC §8](specs/SPEC-FEAT-SCH-008.md); файл удалён в ревизии 11 |

Отчёты по неделям (недельные отчёты учебного процесса, хранятся в REC-S5) — сданные материалы: в комплект документации модуля не входят, не обновляются и правилом «четыре файла плюс спецификация» не затрагиваются. Отчёты за недели 4–5 отдельными файлами не составляются по решению ответственного от 2026-11-01; конкретные результаты, ссылки на код, обнаруженные проблемы и дальнейшие действия фиксируются в отчёте по разработке отчёт по разработке `dev_report_M1.md` (материал REC-S5, в репозиторий кода не кладётся) — это отчёт, а не документация модуля, поэтому правило «четыре файла плюс спецификация» на него не распространяется.

### История ревизий комплекта

| Рев. | Дата | База кода | Основное изменение |
|---|---|---|---|
| 1 | 2026-10-10 | `ae9db40` | Первый состав четырёх документов: README модуля, диаграммы, ревизия (audit), план развития (features) |
| 2 | после 2026-10-18 | `6c0aae2` | Добавлена спецификация [SPEC-FEAT-SCH-008](specs/SPEC-FEAT-SCH-008.md) (v1→v2 по замечаниям ревью: атомарность аллокации, правило «один писатель» для состояния, реестр пар стратегий, wb/sb как политика стратегий); база перепривязана `ae9db40` → `6c0aae2` (PR #39 QoS слит) — все якоря пересчитаны, QoS-классы вошли в AS-IS-схемы и матрицу алгоритмов; статусы BUG-SCH-001 и BUG-SCH-007 обновлены по контрольным замерам |
| 3 | октябрь 2026 | `6c0aae2` | Полная перепроверка: все прогоны повторены (золотой сценарий побитово, воспроизведения BUG-SCH-001/009, варианты A/B/C тестового набора, xfail-вариант); исправлены ~40 неточных якорей после сдвига строк PR #39; новая карточка [BUG-SCH-010](audit.md#bug-sch-010); «шесть алгоритмов» → восемь; три stateDiagram исправлены для парсера Mermaid; ссылки на карточки переведены на якоря |
| 4 | октябрь 2026 | `6c0aae2` | Две архитектурные записи перерегистрированы как фичи — [FEAT-SCH-008](features.md#feat-sch-008) и [FEAT-SCH-009](features.md#feat-sch-009) (новые сущности, контракты и точки расширения — фича, даже без изменения внешнего поведения); критерий «фича/оптимизация» зафиксирован во введении features §1; прежние обозначания — в таблице соответствия audit §1 |
| 5 | октябрь 2026 | `6c0aae2` | Атрибуция запросов в features §3 приведена к проверяемым источникам GitHub (issues, тела и комментарии PR, CODEOWNERS, git log): issues #27–#31 — автор владелец репозитория, мне не назначен ни один; две строки без подтверждённого источника удалены; устная договорённость по направлению srsRAN помечена «не проверено» |
| 6 | октябрь 2026 | `6c0aae2` | SPEC v3 по итогам архитектурного ревью v2: недельные задачи T1–T7 заменены атомарными зелёными шагами M0–M8 со строгим gate (§8); правила миграции R1–R8 и минимальный DoD атомарного PR (§2); политика CQI — независимая граница, трек C0–C4 (§5.7); внутренний `AllocationResult` для per-flow решений QoS; снято ложное утверждение об атомарности `ALLOCATE_RBG` — новая карточка [BUG-SCH-009](audit.md#bug-sch-009) с воспроизведением; наблюдений в audit — 17 |
| 7 | 2026-10-23 | `6c0aae2` | SPEC v4 по итогам ревью v3: четыре замечания P1 (экспериментальный путь — propose→validate→commit с точкой запуска после этапа PDCCH; контракт `MultiplexingPolicy` в M7b; противоречие порядка M1a/M3 снято самостоятельным подключением prefilter; неатомарные шаги разбиты M4c→M4c+M4d, M5→M5a+M5b, M8→M8a–M8e) и четыре уточнения границ; история замечаний — SPEC §11.1/§11.2 |
| 8 | 2026-10-23 | `6c0aae2` | Сверка комплекта с фактически выполненным шагом M0: новая карточка [BUG-SCH-011](audit.md#bug-sch-011) — классовые кэши `ChannelModel` (CHANNEL_MODEL.py :158–160) не сбрасываются между прогонами в одном процессе (воспроизведено, расхождение давал `O2I_INFO`); в SPEC — оговорка «M0 подготовлен до согласования v4» и очистка кэшей в методике full-SIM-снимков; в features §2 — запрос владельцу модели канала; наблюдений — 18 |
| 9 | 2026-10-24 | `6c0aae2` | Добавлены процессные документы `docs/process/` (DoD, ROADMAP, CHANGELOG — с ревизии 11 растворены в комплекте); разобрана практика изменений репозитория по истории PR #20/#21/#26/#32/#37/#38/#39 и issues — критерии внесены в DoD; объём M0-патча сужен до каталога тестов модуля (39 файлов вместо 41: корневые `conftest.py`/`pytest.ini` убраны, автономность каталога обеспечена его conftest по приёму `layered_buffer_tests`); код baseline вычищен до 0 ошибок `ruff check --config .ruff.toml` |
| 10 | 2026-11-01 | `6c0aae2` | Сверка комплекта с фактически выполненными шагами M0 (доработка) и M1. M0: патч перегенерирован — во все 27 снимков добавлено информационное поле `meta.env` (python/numpy/pytest, исключено из сравнения), md5 `841e21a5…` → `eb7a8b81…`, регенерация байт-в-байт стабильна, sha256 истории allocation full-SIM не изменился. M1: шесть `@runtime_checkable`-протоколов и шесть тонких legacy-адаптеров (раздел в конце `SCHEDULER.py`, номера строк legacy-кода сохранены), потребители не переключены (R1/R2), добавлены `tests/scheduler_arch_tests/`; шапка модуля приведена к факту — [DOC-SCH-003](audit.md#doc-sch-003) закрыт. Новые карточки: [BUG-SCH-012](audit.md#bug-sch-012) (QoS-планировщики падают на `SimpleBufferManager`, воспроизведено; сайты :2790/:2844/:3029/:3126), [BUG-SCH-013](audit.md#bug-sch-013) (не уникальное имя модуля `conftest` — закрыто `baseline_harness.py`/`arch_support.py`, коммит `41a163b`), [DOC-SCH-004](audit.md#doc-sch-004) (copy-paste docstring/verbose шести заглушек `_apply_pdsch_estimation`); наблюдений — 22. Решение ответственного от 2026-11-01: PR не подаются, разработка по плану локально, отчёты за недели 4–5 не составляются |
| 11 | 2026-11-01 | `6c0aae2` + ветка `f14885c` | Ревизия 11. (1) Комплект приведён к четырём файлам по ТЗ плюс спецификация: процессные документы и рабочие файлы поставки растворены (таблица выше), `docs/process/` и `docs/specs/OPT-SCH-001.md` удалены; история ревизий — эта таблица. (2) Из репозитория модуля удалены `RUNBOOK.md` и `INVENTORY-M1.md` (коммит `f14885c`): 16 ссылок в тестовом коде перепривязаны на разделы документации, все замены построчные — `SCHEDULER.py` остался 3584 строки, 11 legacy-якорей на месте; в `PyScheduler/tests/` md-файлов не осталось. (3) Пересмотрены **все 28 PR** апстрима (открытые и закрытые) — карточка [REV-SCH-001](audit.md#rev-sch-001): 20 правил кодревью, 9 нарушений в собственной поставке исправлено коммитом `ba479ee` (докстринги на всех 128+56 узлах, именованные константы вместо магических чисел, AST-проверки вместо архитектурозависимых тестов на номера строк, схема снимков 2), два решения вынесены на подтверждение владельца. (4) Пробел покрытия снимков, найденный инвентаризацией M1 (`bs_buffer_size`, `instant_rate` не фиксировались), закрыт расширением схемы снимков в той же ветке — побайтовая совместимость со схемой 1 проверена ([SPEC §8.2](specs/SPEC-FEAT-SCH-008.md)). (5) Новая карточка [CFG-SCH-001](audit.md#cfg-sch-001) — в `.ruff.toml` не задан `src`, результат линтера зависит от каталога запуска (замер: 1095 против 1102 замечаний, I001 32 → 39); запрос владельцу в audit §3. (6) В [BUG-SCH-012](audit.md#bug-sch-012) добавлен готовый текст запроса владельцу `BS_MODULE`. (7) Пятый патч `SCH-008-docs-consolidation.patch`, суммарный переименован в `SCH-008-all-steps-combined.patch`; цепочка из пяти патчей воспроизведена на свежем клоне `dev@6c0aae2` — дерево побайтово совпадает с веткой `f14885c`. Наблюдений в audit — 24 |
| 12 | 2026-11-02 | `6c0aae2` + ветка `feat/SCH-008-M1-protocols` | Эта ревизия. (1) Комплект размещается в репозитории по принятой структуре: общая `docs/` + папка модуля `docs/scheduler/` — четыре документа и `specs/SPEC-FEAT-SCH-008.md`, только md и Mermaid, внешних ресурсов нет (§8); дерево для репозитория генерируется из рабочей копии скриптом `tools/make_repo_docs.py` — он же проверяет относительные ссылки (прежнее зеркало комплекта и его скрипт синхронизации удалены). (2) SHA перепривязаны на фактическую топологию поставки: одна ветка, пять коммитов `9d50a94` → `41a163b` → `2472cb8` → `ba479ee` → `f14885c`; ссылки на прежние промежуточные коммиты и две промежуточные ветки заменены (149 замен), в [features.md](features.md) FEAT-SCH-008 исправлен объём коммита выравнивания (+899/−354 по `git show --shortstat`). (3) Добавлен §4.8 — порядок поставки и интеграции: ветка от актуальной `dev`, один шаг = один PR в `dev` (squash), легаси сохраняется до проверки, задачи смежным модулям выдаются после слияния, пять шагов проверки багов, профилирование до/после; §4.6 — основной способ поставки ветка в форк и PR в `dev`, патчи резервные, md5 патча документации не приводится из-за самоссылки. (4) SPEC v5: граница с AMC-направлением как единственный контракт `AmcView`/`GET_BITS_PER_RB` (§5.4), три новые зависимости и порядок слияния при параллельных правках одного файла (§7, §7.1), M8b привязан к целевой раскладке пакетов (§8), профилирование `cProfile` в методике сравнения (§8.2, §9), разбор изменений v4 → v5 (§11.3), чек-лист v5 (§13), non-goals (§14). (5) features: исследовательский трек R1–R5 после выделения аллокатора и ориентиры сроков (§1.1, слой 4), запросы AMC-направлению, StatsManager и владельцу репозитория по раскладке пакетов, версионированию и именам папок `docs/` (§2), запрос на согласование интерфейсов AMC (§3), уточнение к правилу шаблона про черновики — проект перешёл к реализации, результат этапа работающий и проверяемый код (блок «Правила:» сохранён дословно). (6) audit: формат карточек соотнесён с принятым порядком проверки предполагаемой ошибки (§1), пять новых открытых вопросов (§3). (7) diagrams: целевая раскладка пакетов проекта и что от неё не зависит (§3.1), черновик TO-BE приведён к v5. (8) Отчёт по разработке `dev_report_M1.md` дополнен результатами, ссылками на код и дальнейшими действиями. Наблюдений в audit — 24 |
| 13 | 2026-11-02 | `6c0aae2` + ветка `c3c5164` | Эта ревизия. (1) Закрыто P2-замечание внешнего ревью PR #44 (Codex, 2026-11-02) коммитом `c3c5164`: `LegacyCqiView.get_sb` возвращает копию SB-списка — контракт read-only view ([SPEC §2.1](specs/SPEC-FEAT-SCH-008.md), антипаттерн A3); тест, фиксировавший живой список как AS-IS, заменён 1:1: до фикса воспроизводит замечание (1 failed), после — зелёный; число тестов (49/55/104), полный прогон (264/1) и снимки не изменились; протокол из пяти шагов и факты — в карточке [REV-SCH-001](audit.md#rev-sch-001). (2) Поставка обновлена: седьмой патч `SCH-008-cqi-view-copy-fix.patch`, восьмой — эта ревизия, bundle и суммарный патч перегенерированы, топология ветки — восемь коммитов (§4.6). Наблюдений в audit — 24 |
| 14 | 2026-11-02 | `6c0aae2` + ветка `9379b3d` | Эта ревизия. (1) Закрыто второе P2-замечание внешнего ревью PR #44 (Codex, 2026-11-02) коммитом `9379b3d`: docstring `GridView.allocate_rbg`/`LegacyGridView.allocate_rbg` обещали «частичный отказ откатывает сетку» — ложная гарантия при owner-unsafe откате `RELEASE_RBG` ([BUG-SCH-009](audit.md#bug-sch-009)); контракт теперь формулирует семантику отказа явно, путь отказа зафиксирован фазой 2 теста `test_grid_view_allocate_rbg_delegates_and_mutates_grid` (поведение `RES_GRID` не менялось: R2, A1 — owner-safe откат за владельцем сетки). (2) Поставка обновлена: девятый патч `SCH-008-gridview-failure-semantics.patch`, десятый — эта ревизия, bundle и суммарный патч перегенерированы, топология ветки — десять коммитов (§4.6). Числа прогонов и состав снимков не изменились. Наблюдений в audit — 24 |

**Правила ведения комплекта.** Любое изменение — новая строка в таблице ревизий и повторный прогон проверок: внутренние ссылки и якоря (0 ошибок), соответствие разделов и полей карточек шаблонам ТЗ (audit — 8 полей, features — 11) и дословность блоков «Правила:» (0 отклонений), валидация ссылок `:NNN` против текущей базы кода (NO_TOKEN = 0), парсинг mermaid-блоков (headless), сканы на личные имена, упоминания встреч и внешних файлов как источников, маркеры генерации и гомоглифы; при размещении в репозитории дополнительно проверяются относительные ссылки в раскладке `docs/scheduler/` (генерация дерева `repo-docs/docs/scheduler/` и сверка — `tools/make_repo_docs.py --check`). При уходе `dev` с `6c0aae2` — перепривязка якорей по §4.7, обновление «Проверено по коду» в шапках, повторные прогоны затронутых измерений и регенерация снимков (§4.3). SPEC-FEAT-SCH-008 v5 содержательно не меняется до формального согласования; после завершения фичи (шаг M8e) его содержимое переносится в README/diagrams/features, а спецификация остаётся историей проектирования.
