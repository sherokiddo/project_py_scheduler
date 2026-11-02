# INVENTORY-M1 — фактические входы и выходы восьми алгоритмов планировщика

**Фича:** FEAT-SCH-008 (SPEC-FEAT-SCH-008 v4), шаг **M1** — протоколы стратегий и
view-адаптеры.
**Раздел SPEC:** §5.4 (инвентаризация AS-IS), §5.7 (интерфейсы), §8 (план M0–M8).
**Ревизия кода:** `dev 6c0aae2` + ветка `feat/SCH-008-M1-protocols`
(секция протоколов добавлена в конец `SCHEDULER.py`, номера строк legacy-кода
не изменились — контроль: `test_views_m1.py::test_m1_section_appended_after_legacy_code`).
**Все номера строк** ниже сверены с фактическим файлом вручную на ревизии
`6e1b019` + правки кодревью 2026-11-01. Автоматически контролируется не номер,
а содержимое: тест `test_documentation_quotes_present_in_module` проверяет, что
каждая цитата кода из документации присутствует в модуле (проверка по номерам
строк была бы архитектурозависимой: её сдвигает любая правка владельца модуля
выше по файлу — порядок перепривязки номеров в RUNBOOK §8, п. 10).

> Инвентаризация выполнена чтением кода и AST-разбором (`ast.walk` по классам и
> методам), а не пересказом документации модуля. Каждый факт снабжён номером
> строки; спорные места отмечены как «наблюдение».

---

## 1. Как запустить проверки M1

```bash
# из корня репозитория
python -m pytest PyScheduler/tests/scheduler_arch_tests -q          # 55 passed
python -m pytest PyScheduler/tests/baseline -q                      # 49 passed
ruff check --config .ruff.toml PyScheduler/tests/scheduler_arch_tests   # All checks passed

# из PyScheduler/ — наборы должны собираться вместе и в любом порядке
cd PyScheduler
python -m pytest tests/baseline tests/scheduler_arch_tests -q       # 104 passed
python -m pytest tests/scheduler_arch_tests tests/baseline -q       # 104 passed
python -m pytest tests -q --ignore=tests/scheduler_tests
#   -> 264 passed, 1 failed (test_packet_discard — падение существовало на dev)
```

`ruff` — только из корня репозитория (канонический запуск). Причина в конфиге,
а не в коде: в `.ruff.toml` не задан `src`, поэтому классификация first-party
зависит от cwd, и результат различается:

| Запуск | Замечаний по `PyScheduler/` | Из них в новых каталогах M0/M1 |
|---|---|---|
| `ruff check --config .ruff.toml PyScheduler` из корня | 1095 | **0** |
| то же с `src = ["PyScheduler", "."]` | 1102 (I001: 32 → 39) | 4 (только I001) |
| из `PyScheduler/` с `--config ../.ruff.toml` | — | те же 4 (только I001) |

То есть собственных модулей проекта (`SCHEDULER`, `BS_MODULE`, `GLOBALS`, …)
ruff считает third-party, а при явном `src` — first-party, и требование к
порядку импортов меняется на противоположное (две конфигурации несовместимы:
проверено на probe-файле). Замечания E/F/W/N/B от `src` не зависят.
Карточка **CFG-SCH-001** (запрос владельцу репозитория) — в
`MR_doc_SCH-001/PyScheduler/scheduler/`; в M1 конфиг не правится (чужой файл,
нецелевое изменение).

### 1.1 Состав каталога

```
tests/scheduler_arch_tests/
├── INVENTORY-M1.md    — этот файл (инвентаризация AS-IS, SPEC §5.4)
├── arch_support.py    — построители контекстов (на хелперах baseline_scenarios)
├── conftest.py        — sys.path + фикстуры wb_ctx / fd_ctx / layered_ctx
└── test_views_m1.py   — 55 контрактных тестов (протоколы, адаптеры, R1/R2/R3)
```

Общие хелперы лежат в `arch_support.py`, а не в `conftest.py`, и тесты не
используют `from conftest import ...`: имя модуля `conftest` в дереве
`PyScheduler/tests/` не уникально, и при сборке двух каталогов одним запуском
импорт разрешается в чужой файл. Та же причина у `baseline_harness.py` в M0
(коммит `1556d48`, ветка `fix/SCH-008-test-import-collision` — M1 зависит от
него: без него `pytest tests/baseline tests/scheduler_arch_tests` падает на
коллекции с `ImportError: cannot import name 'assert_matches_baseline'`).

## 2. Карта этапов оркестратора (`SchedulerInterface.schedule`, :346–436)

| Этап | Вызов (строка) | Реализация | Переопределяют |
|---|---|---|---|
| 0. CQI refresh | `self._refresh_cqi(tti, users)` :366 (при `tti % wb_cqi_upd_interval == 0`, :365) | :504 | никто |
| 1. Eligibility | `self._filter_eligible_ues(tti, users)` :369 | :559 | никто |
| 2. Active window | `self._update_active_window(eligible_ues)` :374 | :610 | никто |
| 2.5. Window filter | `self.filter_by_window(eligible_ues)` :377 | :653 | никто |
| 3. Priorities | `self._calculate_priorities(windowed_ues, tti)` :384 | база :1172 (`NotImplementedError`-заглушка) | **все 8** |
| 4. PList formation | `self._form_priority_list(prioritized_ues, tti)` :392 | :684 (сортировка, tie-break отсутствует, top-N `max_dl_ue_tti`) | никто |
| 4.5. PDSCH estimation | `self._apply_pdsch_estimation(priority_list, tti)` :405 | база :712 (порог 95 % RB: `pdsch_threshold = total_rb * 0.95`, :733) | **6 из 8** (все — заглушки-пасстру) |
| 5. PDCCH allocation | `self._allocate_pdcch(priority_list_filtered)` :410 | :769 (`PDCCHManager`) | никто |
| 6. PDSCH allocation | `self._allocate_pdsch(tti, ues_with_pdcch, eligible_ues)` :416 | база :1197 | **все 8** |
| 7. Buffer processing | `self._process_buffers(tti, users, allocation)` :419 | :880 (гранты, `qci_tput_tracker`, `last_ue_transmitted_bits`) | никто |
| 7.5. Stats | `self._update_stats(...)` :424, `_save_timing_stats` :430 | :1104, :1154 | никто |
| 8. Result | `self._build_result(allocation, users, eligible_ues, tti)` :436 | :996 | никто |

Точки досрочного выхода `_empty_result()`: :371, :379, :402, :413.

**Вывод для M2:** из восьми алгоритмов варьируются только этапы **3**, **4.5**
и **6** (+ этап 7 у `QosAwareScheduler` через `_logical_channel_multiplexing`
:3175). Этапы 0/1/2/2.5/4/5/7/7.5/8 общие и уже сейчас живут в оркестраторе —
это и есть Template Method, который M2 раскладывает на `UserRanker` +
`ResourceAllocator` (+ необязательный `MultiplexingPolicy`) без изменения
порядка вызовов.

---

## 3. Матрица переопределений (AS-IS)

| Класс | Строки | `__init__` | `_apply_pdsch_estimation` | `_calculate_priorities` | `_allocate_pdsch` | `_logical_channel_multiplexing` |
|---|---|---|---|---|---|---|
| `BestCQIScheduler` | :1877–1983 | :1901 | — (база :712) | :1908 | :1931 | — |
| `RoundRobinScheduler` | :1985–2101 | :1991 | :1996 (пасстру) | :2015 | :2040 | — |
| `ProportionalFairScheduler` | :2103–2213 | :2115 | — (база :712) | :2118 | :2160 | — |
| `FDxBestCQIScheduler` | :2215–2337 | :2236 | :2243 (пасстру) | :2250 | :2269 | — |
| `FDxFairGreedyScheduler` | :2339–2522 | :2372 | :2377 (пасстру) | :2394 | :2423 | — |
| `FDxProportionalFairScheduler` | :2524–2702 | :2564 | :2571 (пасстру) | :2588 | :2631 | — |
| `BasicQosAwareScheduler` | :2704–2957 | :2727 | :2753 (пасстру) | :2764 | :2806 | — |
| `QosAwareScheduler` | :2960–3222 | :2965 | :2988 (пасстру) | :2999 | :3044 | :3175 |

Базовые реализации, которые никто не переопределяет: `_allocate_pdcch` :769,
`_process_buffers` :880, `_build_result` :996, `_form_priority_list` :684,
`_filter_eligible_ues` :559.

---

## 4. Входы: что фактически читает каждый алгоритм

Столбцы — источники данных; `×` — читается, в скобках строки первого чтения.

| Алгоритм | WB CQI (`_get_wb_cqi` :1064) | SB CQI (`_get_sb_cqi` :1084) | AMC (`GET_BITS_PER_RB` :1730) | Grid | Буферы БС | EWMA (`average_throughput`) | Состояние RR | QoS-профили | Per-QCI трекер |
|---|---|---|---|---|---|---|---|---|---|
| BestCQI | × :1919, :1970 | — | × :1971 | `ALLOCATE_RBG` :1965, `GET_RBG_INDICES` :1966 | `bs_buffer_size` из user-словаря (:593) → локальная копия `remaining_buffer` :1942–1943 | — | — | — | — |
| RoundRobin | × :2082 | — | × :2083 | :2071–2072 | читает `bs_buffer_size` :2070, **пишет обратно** :2086 | — | init :1993–1994; чтение :2028, :2036, :2063; запись :2090, :2093 | — | — |
| ProportionalFair | × :2129, :2202 | — | × :2130, :2203 | `rb_per_slot` :2131; :2198–2199 | `remaining_buffer` :2178 | × :2133 | — | — | — |
| FD_BestCQI | × :2261, :2307, :2309 | × :2304 | × :2326 | :2323; индексы RBG :2292 | `remaining_buffer` :2279–2280 | — | — | — | — |
| FD_FGS | × :2473, :2475 | × :2471 | × :2500 | :2497; индексы RBG :2455 | `remaining_buffer` :2443–2444 | — | init :2374–2375; чтение :2396, :2411, :2419, :2440; запись :2512, :2514; `served_in_round` — **локальная** :2448–2505 | — | — |
| FD_PF | × :2600, :2667 | × :2661 | × :2601, :2672 | :2689; индексы RBG :2649 | `remaining_buffer` :2640 | × :2604, :2675 | — | — | — |
| BasicQosAware | × :2906, :2915 | × :2891 | × :2920, :2947 | `GET_RBG_SIZE` :2852; :2941; индексы RBG :2859 | `buffer_manager` :2829; `get_buffer_status` :2837; `hol_delay`/`qci` :2790, :2843–2850 | — | — | `GLOBALS.is_gbr` :2792, :2847, :2883; `QCI_PROFILES['gbr']` :2884; `get_delay_budget` **не используется** (HOL-группы через `grouping_parameter`) | `qci_tput_tracker` :2885, :2922 |
| QosAware | × :3114, :3146 | × :3108 | × :3119, :3147 | `total_rbg` :3088–3089; :3140; индексы RBG :3093 | `buffer_manager` :3015, :3065; `get_buffer_status` :3019, :3072; `hol_delay` :3027, :3079 | — | — | `GLOBALS.is_gbr` :3033, :3125; `QCI.get_delay_budget` :3029, :3126 | `qci_tput_tracker` :3121 |

Общие входы всех восьми (через оркестратор, не через алгоритмы):
`cqi_map` (этап 0, :504), буферы БС на этапе 1 (`self.lte_grid.bs.buffer_manager`
:574, `ue_has_buffer` :580, `get_buffer_status` :584, запись `user['bs_buffer_size']`
:593), `pdcch_manager` (этап 5, :769), `lte_grid` (этапы 7/8).

**Частота чтения CQI (наблюдение, важно для CQI-трека C1):** `_get_wb_cqi`
вызывается в 20 местах файла; на один UE за один TTI приходится до **пяти**
чтений WB CQI — этап 1 (:559), этап 3 (метрика), этап 5 (:769), этап 6
(внутри цикла аллокации), этап 7 (:880). FD-алгоритмы дополнительно читают SB
один раз на RBG-цикл. Значения в пределах TTI не меняются (`cqi_map`
обновляется только на этапе 0), то есть повторные чтения — чистая нагрузка и
риск рассогласования при будущей политике TTL. Кэш/единая точка чтения —
задача `CqiView` целевой формы (`CqiEstimate` с `measurement_tti`), в M1
вводится только legacy-семантика (R6: отложенное не маскируется).

---

## 5. Выходы и побочные эффекты

| Куда | Что | Строки | Кто |
|---|---|---|---|
| `user['priority']` | метрика ранжирования | :1919, :2032, :2142, :2261, :2415, :2613, :2798/:2800/:2802, :3040 | все 8 |
| `user['instant_rate']` | отладочное поле PF-метрики | :2143, :2614 | ProportionalFair и FD_PF |
| `user['bs_buffer_size']` | оценка буфера этапа 1 (:593) **уменьшается аллокатором** | :2086 | только RoundRobin |
| `allocation` (возврат этапа 6) | `{UE_ID: [freq_idx]}` | вызовы `ALLOCATE_RBG`: :1965, :2071, :2198, :2323, :2497, :2689, :2941, :3140 | все 8 |
| `self.lte_grid` | владельцы RB/RBG (мутация сетки через `ALLOCATE_RBG`) | те же восемь строк | все 8 |
| `self.rr_ue_offset`, `self.rr_rbg_offset` | долгоживущее состояние ротации | инициализация :1993–1994, :2374–2375; запись :2090, :2093 (RR), :2512, :2514 (FD_FGS) | RR, FD_FGS |
| `self.allocation_flow` | per-flow бюджет битов — **побочный канал** из этапа 6 в этап 7 | инициализация :2986/:3087, запись :3149, чтение :3193–3207 | QosAware |
| `last_ue_transmitted_bits`, `qci_tput_tracker` | статистика/гранты | :880–995 | оркестратор (этап 7) |
| `_last_*` (14 полей) | диагностика, снимки для `get_stats()` | инициализация :316–331, запись :385–426 | оркестратор |

**Наблюдения, влияющие на декомпозицию (M2/M4/M7):**

1. **Состояние ротации живёт на экземпляре стратегии** (`rr_ue_offset`,
   `rr_rbg_offset`), писатель — аллокатор, читатель — ранкер: два разных
   владельца одного поля. Целевое правило «одно поле — один писатель»
   требует переноса в `SchedulerState` оркестратора (M2a/M2b); `StateView`
   и `LegacyStateView` введены именно как временный мост (критерий R8:
   адаптер удаляется вместе с переносом).
2. **`allocation_flow` — побочный канал**: результат этапа 6 доходит до
   мультиплексирования этапа 7 через поле экземпляра, а не через возвращаемое
   значение. В целевой форме — явное поле `AllocationResult.flow_budget_bits`
   (M7b).
3. **`served_in_round` — локальная переменная** FD_FGS (:2448, :2468, :2487–2488,
   :2505), не состояние: в `SchedulerState` переносить не нужно.
4. **RR мутирует общий user-словарь** (:2086), а остальные шесть аллокаторов
   строят локальную копию `remaining_buffer` (:1942, :2178, :2279, :2443,
   :2640, :2827); QosAware вместо копии ведёт per-flow бюджет. Расхождение
   зафиксировано тестами `test_rr_allocator_mutates_shared_bs_buffer_size` и
   `test_other_allocators_keep_bs_buffer_size_intact`.
5. **Доступ к буферам идёт через сетку**: `SchedulerInterface.__init__`
   (:294–304) принимает `bs`, но не сохраняет его — только
   `self.lte_grid.SET_BS(bs)` (:304). Все шесть точек доступа к буферам:
   `self.lte_grid.bs.buffer_manager` (:574, :900, :2776, :2829, :3015, :3065)
   и `self.lte_grid.bs.use_simple_buffer` (:834). Это и есть связь с
   `RES_GRID`, от которой модуль сам планирует избавиться (TODO :3232).
   `BuffersView` принимает менеджер буферов **явно**, чтобы на M2 зависимость
   передавалась стратегии напрямую.
6. **AMC держит обратную ссылку на планировщик**: `self.amc =
   AdaptiveModulationAndCoding(self)` (:305), конструктор
   `__init__(self, scheduler=None)` (:1727) — циклическая зависимость
   «оркестратор ↔ AMC». `AmcView.bits_per_rb(cqi)` разрывает её на уровне
   стратегий: ранкеру/аллокатору нужна только таблица MCS, не планировщик.
7. **Оценка PDSCH вырождена в 6 из 8**: базовая реализация (:712–768, порог
   95 % :731) используется только BestCQI и PF; RR (:1996), FD_BCQI (:2243),
   FD_FGS (:2377), FD_PF (:2571), BasicQos (:2753), QosAware (:2988) возвращают
   список без изменений (6 одинаковых verbose-строк «PDSCH estimation:
   SKIPPED» :2010, :2246, :2389, :2583, :2759, :2994). В целевой форме это
   `prefilter` стратегии аллокатора: у FD/QoS-аллокаторов предфильтра нет,
   у жадных — есть. Отдельный метод восьми классов для этого не нужен.
8. **Цикл аллокации продублирован восемь раз**: `ALLOCATE_RBG(tti, rbg_idx,
   ue_id)` → `GET_RBG_INDICES(rbg_idx)` → ёмкость `len(rb_indices) *
   GET_BITS_PER_RB(cqi)` → уменьшение оценки буфера. Вызовы: :1965/:1967,
   :2071/:2073, :2198/:2200, :2323/:2325, :2497/:2499, :2689, :2941, :3140;
   `GET_BITS_PER_RB` — 17 вызовов в файле. Это ядро будущего
   `ResourceAllocator` (M3+), `GridView` закрывает доступ к сетке уже сейчас.
9. **Формула числа RBG продублирована**: `total_rbg = (rb_per_slot +
   rbg_size - 1) // rbg_size` (:3088–3089) — в QosAware; остальные
   аллокаторы получают `total_rbg` тем же выражением в своих циклах.
   `LegacyGridView.total_rbg()` повторяет формулу дословно (тест
   `test_grid_view_total_rbg_matches_qos_formula`), чтобы общее FD-выделение
   не изменило число просматриваемых RBG (R2).

---

## 6. Пробел покрытия baseline-снимков — ЗАКРЫТ (2026-11-01)

`capture_tti` (tests/baseline/baseline_scenarios.py :348–412) фиксирует
`allocation`, `pdcch_stats`, `bitmap`, `priority_list`, `ues`
(`average_throughput`, `current_dl_throughput`, `cqi_map_wb/sb`,
`user_dict_keys`, `bs_buffer_size`, `instant_rate`), `grid_rb_owners`,
`rbg_choice_table`, `buffer_status`, `scheduler_extra`, `qci_tracker`,
`last_ue_transmitted_bits`.

Инвентаризация M1 нашла два наблюдаемых поля user-словаря (§5), которых в
схеме 1 не было:

- `user['bs_buffer_size']` (пишется на этапе 1 :593, уменьшается RR :2086,
  читается `get_stats()` :470 и оценкой PDSCH :739) — побочный эффект,
  пересекающий этапы;
- `user['instant_rate']` (отладочное поле PF :2143 и FD_PF :2614) — часть
  наблюдаемого контракта словаря `user`.

**Решение (принято 2026-11-01): расширить снимки, а не ограничиться тестами.**
Обоснование и фактическая проверка — в `MR_doc_SCH-001/PyScheduler/scheduler/`
(раздел «Пробел покрытия снимков»), коротко:

1. Назначение baseline-набора — ловить ЛЮБОЕ изменение наблюдаемого состояния
   при переносе кода на шагах M2+. Поля, которых нет в снимке, миграция может
   изменить незаметно; тесты ловят только то, что в них явно запрограммировано.
2. Расширение аддитивно: добавлены `ues[].bs_buffer_size`,
   `ues[].instant_rate` и `ues[].user_dict_keys` (весь набор ключей словаря
   `user`, чтобы появление/исчезновение любого поля меняло снимок).
3. Доказательство отсутствия изменения поведения: 23 сценария пересобраны и
   сверены со снимками схемы 1 — после удаления трёх новых ключей все поля
   схемы 1 совпадают **побайтово, 0 расхождений** (скрипт сверки и вывод —
   RUNBOOK §6.1).
4. Стоимость: регенерация 23 снимков занимает 0,5 с (`generate_baselines.py`),
   full-SIM контрольные точки (4 файла `sim*.json`) схемы не меняли и не
   перегенерировались.

Тесты, написанные в M1 для этого пробела, сохранены — они проверяют то, чего
снимок дать не может:

| Тест | Что добавляет поверх снимка |
|---|---|
| `test_rr_allocator_mutates_shared_bs_buffer_size` | направление изменения (оценка не может вырасти) на живом прогоне |
| `test_other_allocators_keep_bs_buffer_size_intact` | список алгоритмов выводится из `ALL_ALGORITHMS`: новый алгоритм попадёт под контроль автоматически |
| `test_pf_writes_debug_field_instant_rate` | независимый пересчёт `rb_per_slot * GET_BITS_PER_RB(wb_cqi)` и отсутствие поля у BestCQI |
| `test_all_snapshots_carry_user_contract_fields` | схема 2 присутствует в КАЖДОМ снимке (23 сценария), `bs_buffer_size` — int |

Мутационный контроль (2026-11-01): удаление `bs_buffer_size` из снимка,
подмена его типа на float, добавление несуществующей цитаты в
`DOCUMENTED_QUOTES`, отмена съёма `instant_rate` в `capture_tti` — каждая
мутация ловится соответствующим тестом (4/4 FAIL, после отката — PASS).

## 7. Соответствие входов и view-протоколов M1

| Вход (§4) | Протокол | Адаптер | Legacy-цель делегирования | Переключение потребителя |
|---|---|---|---|---|
| WB/SB CQI | `CqiView` (:3278) | `LegacyCqiView` (:3403) | `_get_wb_cqi` :1064, `_get_sb_cqi` :1084 | M3 (общий аллокатор) / C1 (целевая форма) |
| Сетка: RBG, индексы, PRB | `GridView` (:3296) | `LegacyGridView` (:3421) | `ALLOCATE_RBG`, `GET_RBG_SIZE`, `GET_RBG_INDICES`, `rb_per_slot` | M3 |
| Таблица MCS | `AmcView` (:3327) | `LegacyAmcView` (:3448) | `GET_BITS_PER_RB` :1730 | M3 |
| Буферы БС | `BuffersView` (:3336) | `LegacyBuffersView` (:3458) | `ue_has_buffer`, `get_buffer_status`, `hol_delay`/`qci` статусов | M2b (ранкеры), M6 (QoS) |
| EWMA, per-QCI, GBR/PDB | `StatsView` (:3355) | `LegacyStatsView` (:3502) | `user['ue'].average_throughput` :2133, `qci_tput_tracker`, `GLOBALS.is_gbr`/`QCI_PROFILES`, `QCI.get_delay_budget` | M4b (PF), M6 (QoS) |
| Состояние ротации | `StateView` (:3376) | `LegacyStateView` (:3538) | `rr_ue_offset`/`rr_rbg_offset` :1993–1994, :2374–2375 | M2a/M2b → `SchedulerState`, адаптер удаляется |

Номера строк протоколов/адаптеров — в текущей редакции файла; секция добавлена
в конец, поэтому при правках legacy-кода они сдвигаются последними.

---

## 8. Что в M1 НЕ менялось (правила R1/R2) и как это контролируется

| Гарантия | Контроль |
|---|---|
| Потребители не переключены на views | `test_legacy_code_does_not_reference_views` (AST: ни один legacy-узел модуля — класс ИЛИ функция — не упоминает 12 view-имён; полнота контроля подтверждается тем, что все восемь реализаций из фабрики попали в проверенный набор). Проверено мутацией 2026-11-01: метод `_mutation_probe_view_use` с `LegacyCqiView(self)` внутри `RoundRobinScheduler` роняет тест (FAIL), после отката — PASS. Скан строк по регулярному выражению (`test_no_consumer_switched_yet`) удалён как архитектурозависимый: шаблон `.hol_delay(` ложно срабатывал бы на легитимный legacy-код, а AST-проверка покрывает то же инвариантом |
| Поведение восьми алгоритмов не изменилось | 49 baseline-тестов проходят; снимки перегенерированы ТОЛЬКО из-за расширения схемы (§6): после удаления трёх новых ключей все поля схемы 1 совпадают со старыми снимками побайтово (23 сценария, 0 расхождений) |
| Номера строк legacy-кода сохранены | `test_m1_section_appended_after_legacy_code` (после маркера раздела нет ни одного legacy-узла), `test_documentation_quotes_present_in_module` (11 цитат кода из документации присутствуют в модуле). Проверено мутацией: узел, добавленный после раздела M1, роняет тест |
| Адаптеры делегируют, а не reimplement | `test_cqi_view_delegates_wb_cqi` (8 алгоритмов), `test_grid_view_allocate_rbg_delegates_and_mutates_grid` (сетки-близнецы), `test_amc_view_matches_get_bits_per_rb`, `test_buffers_view_matches_manager_simple/layered`, `test_stats_view_*`, `test_state_view_reads_and_writes_scheduler_fields`. Проверено мутациями: `+1` в `get_wb`, подмена формулы `total_rbg`, чтение не того поля состояния — каждая ловится |
| Реестр фабрики не затронут | `test_create_accepts_all_eight_factory_keys`, `test_views_are_not_registered_as_algorithms` |
| AS-IS дефекты не «подлечены» попутно | `test_buffers_view_hol_delay_none_on_simple_reproduces_bug_sch_012` (BUG-SCH-012), `test_available_algorithms_divergence_pinned_bug_sch_010` (BUG-SCH-010), `test_amc_view_propagates_invalid_cqi_error` (CQI=0 → ValueError :1735) |

---

## 9. Карточки и расхождения, подтверждённые инвентаризацией

| ID | Суть | Строки | Статус в M1 |
|---|---|---|---|
| **BUG-SCH-012** | QoS-метрики падают на `SimpleBufferManager`: `qci`/`hol_delay` = None (`BS_MODULE.py` :453–457) → `TypeError` на :2790 (BasicQos, этап 3) и :2844 (BasicQos, этап 6), `ValueError: None is not a valid QCI` на :3029 (QosAware, этап 3) и :3126 (QosAware, этап 6) | :2790, :2844, :3029, :3126 | воспроизводится тестом; **не исправляется** (R2). Уточнение M1: раньше в карточке не было второго `TypeError`-сайта :2844 |
| **BUG-SCH-010** | `available_algorithms()` (:284–292) возвращает `FD_BestCQI`/`FD_FF`, которых нет в реестре `create()` (:255–264), и не возвращает рабочих `FD_BCQI`/`FD_PF`; вызовов метода в коде нет | :255–264, :284–292 | расхождение зафиксировано тестом; синхронизация — M8a |
| **BUG-SCH-008** | `tests/scheduler_tests/` не собирается: `log_utils` отсутствует на `dev`; имя `FD_RR` устарело | — | вне M1, владелец — сопровождающий набора |
| **DOC-SCH-003** | Шапка модуля утверждала `Версия: 2.0.0 / Дата: 2025-10-26` при наличии блока «Изменения v2.1.0» (:86) | :10–11 | **закрыто в M1**: `Версия: 2.1.1`, `Дата последнего изменения: 2026-11-01`; тест `test_module_header_version_bumped` |
| DOC-кандидат (новый, требует карточки) | Copy-paste docstring/verbose у заглушек оценки PDSCH: в `FDxFairGreedy` (:2378–2389) и `FDxProportionalFair` (:2572–2583) текст начинается с «RR: Не применяем…», а `BasicQosAwareScheduler` (:2754–2759) печатает «FD-QoS per-RBG selection», не будучи FD-алгоритмом | :1996–2013, :2377–2389, :2571–2583, :2753–2759, :2988–2994 | найден в M1; правка — отдельный doc-шаг (не M1, R4) |
| Наблюдение (не дефект) | `Версия Python Kernel: 3.12.9` в шапке (:12) не совпадает с фактическим окружением прогонов (3.11, записано в `meta.env` снимков M0) | :12 | информация к сверке с владельцем репозитория |

---

## 10. Открытые вопросы к согласованию (не блокируют M1)

1. **Расширение baseline-снимков** полями `bs_buffer_size`/`instant_rate` —
   **РЕШЕНО 2026-11-01, вопрос снят**: снимки расширены (§6), регенерация
   23 сценариев выполнена в этой же ветке, все поля схемы 1 совпали побайтово.
   Исходная формулировка вопроса: «отдельная ветка с регенерацией 27 снимков —
   делать до M2 или после переноса аллокаторов?». Ответ: **до M2** — иначе M2
   менял бы одновременно и поведение, и схему снимков, и расхождение нельзя
   было бы атрибутировать одному шагу.
2. **Нормализация имён алгоритмов** (BUG-SCH-010): подтвердить, что
   единственное исправление — M8a через `SCHEDULER_REGISTRY`, и что имена
   `FD_BestCQI`/`FD_FF` не используются внешними конфигами (внутри репозитория
   вызовов нет).
3. **Doc-правка заглушек оценки PDSCH** (§9, DOC-кандидат): включать в M2 как
   сопутствующую или вести отдельным doc-PR (по опыту отклонённого PR #37 —
   отдельным).
