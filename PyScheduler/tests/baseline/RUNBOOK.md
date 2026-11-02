# RUNBOOK — baseline M0 (SPEC-FEAT-SCH-008, шаги M0–M8)

Назначение: единая точка правды по командам запуска baseline-набора,
контрольным точкам full-SIM и правилам обновления снимков. Требование M0
(раздел 8 спеки): «команды запуска сохранены».

Ревизия dev, на которой зафиксированы снимки: **6c0aae2**.

## 1. Окружение

- Валидированное окружение: Python 3.11.2, numpy 2.4.6, pytest 9.2.0.
  Снимки воспроизведены байт-в-байт на двух разных стендах с этими версиями,
  но побитовая воспроизводимость гарантируется только в рамках одной версии
  интерпретатора и numpy (другая BLAS способна сдвинуть float за пределы
  допуска 1e-9). Подпись окружения пишется в `meta.env` каждого снимка и в
  сравнении не участвует (раздел 5).
- Зависимости: `numpy`, `pytest`, `tqdm`, `matplotlib`, `psutil`
  (`statsmodels` в прогонах не используется).
- Все команды — из корня репозитория. `PYTHONPATH` и файлы корня репозитория
  не требуются: каталог автономен — `tests/baseline/conftest.py` сам добавляет
  `PyScheduler/` в `sys.path` (тот же приём, что в `layered_buffer_tests/
  conftest.py`) и регистрирует маркер `sim` через `pytest_configure`.
  Патч M0 не создаёт и не меняет файлы вне `PyScheduler/tests/baseline/`.
- Хелперы сравнения снимков (`load_snapshot`, `assert_matches_baseline`,
  `ALL_SCENARIO_NAMES`) вынесены в `baseline_harness.py`, тесты импортируют их
  оттуда, а не из `conftest`. Причина: имя модуля `conftest` в дереве
  `PyScheduler/tests/` не уникально (свой `conftest.py` есть в `baseline/`,
  `layered_buffer_tests/`, `scheduler_tests/`, а с шага M1 — ещё и в
  `scheduler_arch_tests/`). Когда pytest собирает несколько таких каталогов
  одним запуском, `from conftest import ...` может разрешиться в чужой файл:
  `ImportError: cannot import name 'assert_matches_baseline' from 'conftest'`
  и падение коллекции. Воспроизведение: `cd PyScheduler && python -m pytest
  tests/baseline tests/<каталог со своим conftest.py> -q` (и обратный порядок).
  `conftest.py` переэкспортирует имена хелперов для совместимости.
  На снимки и поведение правка не влияет: регенерация не требуется.

## 2. Команды запуска

```bash
# Быстрый characterization-набор (42 теста, <1 c) — ОБЯЗАТЕЛЕН после каждого PR:
python -m pytest PyScheduler/tests/baseline -q -m "not sim"

# Полный baseline-набор вместе с full-SIM контрольными точками (~9 c):
python -m pytest PyScheduler/tests/baseline -q

# Существующий набор scheduler_tests (в M0 не изменяется). ВНИМАНИЕ: на чистом
# dev@6c0aae2 набор не собирается — ModuleNotFoundError: log_utils и устаревший
# ключ 'FD_RR' (BUG-SCH-008, audit). Команда применима после восстановления
# набора по разделу «Как проверить» карточки BUG-SCH-008 (восстановление —
# отдельный PR по согласованию с владельцем репозитория, в M0 не входит).
# ВАЖНО: у tests/scheduler_tests нет своего conftest с sys.path, поэтому из
# корня репозитория набор не собирается независимо от log_utils
# (ModuleNotFoundError: SCHEDULER) — запускать из каталога PyScheduler/:
cd PyScheduler && python -m pytest tests/scheduler_tests -q && cd -

# То же для mobility_module_tests (своего conftest нет; из корня — 5 ошибок
# коллекции ModuleNotFoundError: MOBILITY_MODEL):
cd PyScheduler && python -m pytest tests/mobility_module_tests -q && cd -

# Регенерация всех снимков (после слияния PR, gate 8.1):
python PyScheduler/tests/baseline/generate_baselines.py

# Регенерация одного сценария:
python PyScheduler/tests/baseline/generate_baselines.py golden_tti5

# Full-SIM контрольная точка вручную:
python PyScheduler/tests/baseline/sim_checkpoint.py --duration 1000 \
    --out PyScheduler/tests/baseline/baselines/sim1000_qosaware.json
```

## 3. Состав снимков (`baselines/`)

| Снимок | Сценарий | Раздел спеки |
|---|---|---|
| `golden_tti5.json` | золотой сценарий TTI 5 (BestCQI) | 9.1 |
| `wb_<alg>.json` (8) | WB-матрица: 8 алгоритмов, TTI 5–7 | 9, матрица покрытия |
| `fd_sb_full_<alg>.json` (3) | контрастные SB CQI, таблица per-RBG выборов | 9.2 |
| `fd_sb_missing_<alg>.json` (3) | пустой SB → fallback WB | 9.2 |
| `fd_sb_partial_<alg>.json` (3) | короткий список SB + нули внутри | 9.2 |
| `rr_multitti.json` | ротация `rr_*_offset`, TTI 5–8 | 9.3 |
| `qos_basic_single_bearer.json` | BasicQosAware: один bearer/UE, grants + per-QCI | 9.4 |
| `qos_aware_multiflow.json` | QosAware: потоки (ue,qci), `allocation_flow` | 9.4 |
| `conflict_rbg_contended.json` | спор двух UE за RBG (AS-IS, без повторной выдачи) | 9.5 |
| `conflict_pdcch_limited.json` | `max_dl_cce_allowance=1`, второй UE заблокирован | 9.5 |
| `sim100/500/1000_qosaware.json` | full-SIM, QosAware, seed 42 | 9.6, 8.2 |
| `sim1000_bestcqi.json` | full-SIM, BestCQI, seed 42 | 9.6 |

Каждый снимок несёт `meta`: `dev_revision`, `seed` (null для
детерминированных сценариев — ГСЧ не используется), конфигурацию
(bandwidth, num_frames, профили UE/потоков, kwargs планировщика) и список
TTI. Для full-SIM: seed=42, профиль `sim_with_manager_qos`
(TEST_MODULES.py :234), verbose-лог и StatsManager отключены.

## 4. Что фиксирует снимок (наблюдаемое состояние после каждого TTI)

`allocation`, `pdcch_stats` (включая `pdcch_ue_cce_allocations`), `bitmap`,
`last_ue_transmitted_bits`, `priority_list` (порядок + метрики), владельцы RB
в сетке по слотам (`grid_rb_owners`), таблица per-RBG выборов
(`rbg_choice_table`), состояние UE (`average_throughput`,
`current_dl_throughput`), `cqi_map` (WB+SB), `buffer_status`
(per-QCI: размер, lcid, qci, hol_delay), `qci_tracker` (current/average
throughput per QCI), `scheduler_extra` (`rr_*_offset`, `allocation_flow`,
параметры QoS).

## 5. Правила сравнения (раздел 8.2)

- дискретные значения (списки PRB/RBG, UE, QCI/LCID, CCE, offsets,
  байты/биты) — **точно**;
- float — относительный допуск `1e-9` (`baseline_compare.FLOAT_REL_TOL`);
  широкий epsilon запрещён (R2);
- `sch_total_time_us` и прочие wall-clock показатели в снимки **не входят**
  и не сравниваются;
- `meta.env` (версии python/numpy/pytest) — информационное поле, **исключено
  из сравнения** (`baseline_harness.assert_matches_baseline` отфильтровывает ключ
  `env`): снимок обязан проверяться на машине, отличной от той, на которой
  он построен;
- full-SIM: sha256 полной истории allocation — побитово; агрегаты — по тем
  же правилам.

## 6. Порядок обновления снимков (gate 8.1)

1. PR слит → обновить локальную dev.
2. `python PyScheduler/tests/baseline/generate_baselines.py` (все сценарии).
3. Full-SIM точки: команды из раздела 2 (100/500/1000 QosAware, 1000 BestCQI).
4. В каждом снимке обновить `meta.dev_revision` (для full-SIM — в
   `sim_checkpoint.py` поле meta формируется скриптом; ревизию указывать в
   коммите регенерации). `meta.env` генераторы обновляют сами: его отличие
   в diff — не дефект и не основание останавливать миграцию (раздел 5).
5. `python -m pytest PyScheduler/tests/baseline -q` — зелёный.
6. Коммит регенерации — отдельным коммитом в следующий PR или отдельным PR
   с пометкой «baseline refresh после слияния <PR>».

Если тест падает **до** слияния: миграцию остановить (8.1) — дефект
исправляется в текущем PR или откатывается.

## 7. Контрольные точки full-SIM (9.6)

Фиксируются точки **100 / 500 / 1000 TTI** (QosAware) и **1000 TTI**
(BestCQI). Сравнение метрик до/после миграции — ожидаемое расхождение 0
(sha256 истории + агрегаты). 1000-TTI прогон не заменяет короткий набор
после каждого PR (8.2).

Детерминизм: два запуска `sim_checkpoint.py` с одним seed обязаны давать
идентичный снимок (проверяется тестом `test_sim_determinism_same_seed`).

## 8. Известные ограничения AS-IS (зафиксированы, не исправляются в M0)

1. **Глобальные кэши канала.** `ChannelModel.SHADOW_FADING_INFO`,
   `CHANNEL_COND_INFO`, `O2I_INFO` — классовые dict (CHANNEL_MODEL.py
   :158–160). Последовательные симуляции **в одном процессе** без очистки
   кэшей дают другой результат: кэшированные o2i/SF-значения не
   разыгрываются, и весь поток ГСЧ рассинхронизируется (проверено: sha256
   истории allocation второго прогона отличается). `sim_checkpoint.py`
   очищает кэши перед каждым прогоном. Зафиксировано карточкой
   **BUG-SCH-011** ревизии модуля (комплект DOC-SCH-001, `docs/audit.md`) и
   запросом владельцу модели канала (`docs/features.md` §2); в scope M0 не
   исправляется — runtime не меняется.
2. **QoS-алгоритмы падают на SimpleBufferManager**:
   `SimpleBuffer.get_buffer_status()` (BS_MODULE.py :453–457) не заполняет
   `lcid`/`qci`/`hol_delay` (дефолты `None`, :47–50), а QoS-метрики используют
   эти поля без проверки → `BasicQosAware`: `TypeError: unsupported operand
   type(s) for //: 'NoneType' and 'int'` (SCHEDULER.py :2790); `QosAware`:
   `ValueError: None is not a valid QCI` (:3029, :3126). Воспроизведено на
   чистом `dev@6c0aae2`, зафиксировано карточкой **BUG-SCH-012** (комплект
   DOC-SCH-001, `docs/audit.md`) с запросом владельцу BS_MODULE и
   QoS-направлению. WB-ячейки матрицы для BasicQosAware/QosAware строятся на
   LayeredBufferManager; в M0 не исправляется (R2).
3. **`GET_TTI_STATUS` неприменим** (RES_GRID.py :518 вызывает
   `GET_RB(tti, freq_idx)` при сигнатуре `GET_RB(tti, slot_id, freq_idx)`);
   снимок сетки строится напрямую из `rb_map`.
4. **`start_simulation()` выполняется только при `enable_verbose_log(to_file=True)`**
   (SIMULATION_MANAGER.py :1331 — тело внутри `if self._to_file:`);
   `sim_checkpoint.py` включает его и пишет `output.txt` во временный каталог.
5. `layered_buffer_tests`: 14 passed / 1 failed
   (`test_pdcp_discard_timer::test_packet_discard`) — существующее падение на
   `6c0aae2`, не связано с M0; аудиторской карточкой не покрыто (объект —
   буферы/PDCP, вне модуля SCHEDULER).
6. CQI=0 через `schedule()` недостижим (фильтр :592 + continue :919) —
   кейс будет покрыт unit-тестами стратегий в M1+ (BUG-SCH-007).
7. `scheduler_tests` на чистом `dev` не собираются (4 ошибки коллекции:
   `ModuleNotFoundError: log_utils`; после его возврата — `ValueError: Unknown
   algorithm 'FD_RR'`) — BUG-SCH-008. Локальная верификация M0 прогоняла
   набор, восстановленный по разделу «Как проверить» карточки (log_utils из
   `feature/AMC`, замена `'FD_RR'`→`'FD_FGS'` в трёх файлах, `xfail` на
   `test_equal_ues_split_evenly`): 26 passed + 1 xfailed. Эти восстановления
   в патч M0 не входят (одна причина на PR; правки чужого набора — по
   согласованию с владельцем репозитория).
8. `mobility_module_tests` на `6c0aae2` **недетерминирован как набор**, а не
   одним тестом: `RandomWaypointModel` разыгрывает destination из глобального
   нефиксируемого `np.random` (MOBILITY_MODEL.py :305–306), поэтому от прогона
   к прогону падает 0–1 тест из разных мест набора. Наблюдено на трёх полных
   прогонах одного стенда: дважды падал
   `TestGaussMarkovSpecific::test_boundary_threshold_activates_near_y_max` и
   `TestRandomDirectionSpecific::test_subsequent_direction_in_half_circle`
   (по одному разу), ранее фиксировался `test_pause_time_zero_no_pause`
   (приход в новую точку ровно на последнем, 30-м шаге цикла — `is_paused` не
   успевает сняться); изолированно каждый тест проходит. Практический вывод
   для отчётов: стабильное падение в полном прогоне репозитория одно —
   `layered_buffer_tests::test_packet_discard`, а `mobility_module_tests`
   даёт плавающее второе падение, и это не дефект baseline. К baseline не
   относится (снимки full-SIM детерминированы: `sim_checkpoint` фиксирует
   глобальный seed и очищает кэши ChannelModel); зона модуля MOBILITY_MODEL,
   пересекается с RNG-направлением (открыт PR #38; PR #37 с тем же
   содержимым закрыт без слияния).

9. **Связность харнесса full-SIM с внутренним API оркестратора.**
   `sim_checkpoint._Recorder` подменяет `SchedulerInterface.schedule` на
   уровне класса: планировщик создаётся внутри `start_simulation()`
   (SIMULATION_MANAGER.py :1357), и до экземпляра снаружи не добраться.
   Это осознанное решение M0 — перехватывается единственный публичный вход
   модуля, runtime не меняется. Следствие: на шагах M1+, если сигнатура
   `schedule()` или способ создания планировщика изменятся, харнес
   обновляется в том же PR (R4: каждый PR запускается изолированно).
   Регенерация снимков при этом не нужна, если поведение сохранено.

## 9. Состав каталога

```
tests/baseline/
├── RUNBOOK.md                 — этот файл
├── baseline_scenarios.py      — детерминированные сценарии + сбор снимков
├── baseline_compare.py        — правила сравнения (8.2)
├── generate_baselines.py      — генератор снимков
├── sim_checkpoint.py          — full-SIM контрольные точки
├── baseline_harness.py        — хелперы сравнения снимков (load/assert)
├── conftest.py                — фикстуры, sys.path, маркер `sim`
├── test_golden_tti5.py        — золотой сценарий (9.1)
├── test_matrix.py             — WB/FD-матрица (9, 9.2)
├── test_rr_multitti.py        — RR-ротация (9.3)
├── test_qos.py                — QoS grants + per-QCI (9.4)
├── test_conflict.py           — сеточный конфликт (9.5)
├── test_sim_checkpoints.py    — full-SIM точки (9.6), маркер `sim`
└── baselines/                 — 27 снимков (23 сценарных + 4 full-SIM)
```
