# Исправления AMC по замечаниям куратора

Дата: 2026-11-01. Этот документ уточняет текущий объём работ и заменяет противоречащие ему решения прежнего плана AMC.

## Принятые решения и реализация

| Замечание | Реализация |
|---|---|
| Сохранить legacy-статистику | Тело amc.get_stats(), вызовы SimulationManager и позиционный аргумент scheduler сохранены. AMC_METRICS не создаётся; дальнейшая интеграция статистики ожидает требований Саши |
| Сохранить WB и предусмотреть SB | Добавлены McsSelectionPolicy, McsSelectionInput и LegacyWbMcsPolicy. Новый select_mcs(context) — явный интерфейс расширения; старые get_mcs_from_cqi и GET_BITS_PER_RB сохраняют поведение |
| Разделить выбор MCS и табличный расчёт | legacy_mcs_selection описан как политика проекта. Чистая get_tbs_bits(mcs, rb) в AMC.py выполняет только проверку входов и lookup |
| Не размещать TB в AMC | TransportBlock вынесен в TRANSPORT_BLOCK.py. SCHEDULER реэкспортирует класс; поля и имя метода сохранены. SchedulingGrant остаётся на месте |
| Исправить аномалию TBS | TBS_TABLE[6][0] = 88. Добавлен комментарий о 328 в исходной таблице. |
| Неизменяемая конфигурация | AmcConfig использует dataclass(frozen=True). Передаётся только новым keyword-аргументом config, старый конструктор не ломается |
| Не объявлять перенос полноценной регрессией | Выполнено отдельное сравнение симуляции до/после текущих исправлений.  |

## Совместимость и новые интерфейсы

```python
# Старые вызовы работают без изменений.
legacy = AdaptiveModulationAndCoding(scheduler)
legacy.get_stats()
legacy.GET_BITS_PER_RB(10)

# Необязательный новый путь для расчётов без Scheduler.
amc = AdaptiveModulationAndCoding(config=AmcConfig(control_symbols=2))
context = McsSelectionInput(wb_cqi=10)
mcs = amc.select_mcs(context)
tbs = get_tbs_bits(mcs, rb=10)
```

Если config указан, его control_symbols имеет приоритет. Без config сохраняется чтение актуального pcfich через Scheduler, либо значение 2 при отсутствии Scheduler. Неизменяемость конфигурации не ограничивает прежнюю динамическую настройку PDCCH.

McsSelectionInput содержит wb_cqi и tuple-поля sb_cqi, allocated_prbs, prb_to_subband (пары PRB → индекс поддиапазона). Политика получает значения, не объекты сети. select_mcs проверяет целочисленный результат 0–28, включая отклонение bool.

LegacyWbMcsPolicy использует только WB. Реальная allocated_sb-политика не реализована: правила агрегации, отсутствующих/устаревших отчётов и интеграцию со Scheduler нужно согласовать отдельно. Планировщики ещё не вызывают новый select_mcs; прежняя формула бюджетов не переключена на TBS. Существующий get_mcs_from_cqi остаётся совместимым legacy-методом независимо от выбранной новой политики.

## Проверки

Из корня проекта:

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider PyScheduler/tests/amc_tests
.venv/bin/python -B -m pytest -q -p no:cacheprovider PyScheduler/tests/amc_tests --amc-target=extracted
.venv/bin/python -B -m pytest -q -p no:cacheprovider PyScheduler/tests/amc_tests/contract_extracted.py
```

Результаты: 69 passed в legacy-режиме, те же 69 passed в extracted-режиме, 25 passed в отдельном контрактном наборе. Это 94 различных проверочных случая, а не 163 различных теста. Контрактный файл обновлён: больше не требует удаления legacy-статистики или размещения TB в AMC.

До исправлений было 68 passed / 1 failed: полный digest таблицы не совпадал из-за  I_TBS=6, N_PRB=0. Теперь тест требует 88 в корректируемой ячейке, 328 аномалия страндарта и проверяет остальные результаты против прежнего digest. Для сравнения digest он восстанавливает единственное разрешённое отличие во временном списке, не меняя рабочую таблицу.

AST-проверка подтвердила неизменность get_stats и остальных частей SCHEDULER, кроме переноса TransportBlock и его импорта.

Интеграция: sim_with_manager_qos(), 5 UE, ProportionalFair, 5000 TTI; отдельные процессы до/после, random.seed(42), numpy.random.seed(42), MPLBACKEND=Agg. Для «до» использованы копии AMC/SCHEDULER, снятые непосредственно перед текущими исправлениями. Формульный legacy-путь не использует изменяемую таблицу TBS.

- emp_stats.csv: 5000 строк совпали по всем полям, кроме исключённых sch_total_time_us, sch_priority_calc_time_us, sch_priority_sort_time_us.
- emp_stats_detailed.csv: 25000 строк совпали полностью.
- Выходы сохранены вне проекта в /tmp/amc_review_sim_before и /tmp/amc_review_sim_after; это временные артефакты проверки.

Совпадение подтверждено для данного сценария, не для всех алгоритмов и будущих SB/HARQ-путей. ACK/NACK и автоматические повторы в эту правку не входят.
