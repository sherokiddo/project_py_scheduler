# Диаграммы модуля SCHEDULER.py

> Ответственный: Тимур (Scheduler) | Проверено по коду: `6c0aae2` (ветка `dev`) | Обновлено: `2026-11-02`  
> Все схемы соответствуют коммиту `6c0aae2`; поведение подтверждено чтением кода и фактическими прогонами (сценарий TTI 5 из [README §4.1](README.md), pytest). Элементы, фактически добавленные шагом M1 [FEAT-SCH-008](features.md#feat-sch-008) в ветке `feat/SCH-008-M1-protocols` (коммит `f14885c`, строки :3246–3584), отмечены в §3 отдельной пометкой «реализовано в M1» — остальные элементы §3 остаются целевым черновиком. Baseline-снимки, которыми подтверждаются прогоны: состав и команды — [README §4.3](README.md), схема снимка и правила сравнения — [SPEC-FEAT-SCH-008 §8.2](specs/SPEC-FEAT-SCH-008.md). Наблюдения, упомянутые на схемах (BUG/OPT/DOC-SCH-xxx), — в [audit.md](audit.md); планируемые изменения (FEAT/OPT-SCH-xxx) — в [features.md](features.md). В метках переходов диаграмм состояний строки кода указаны как «стр. NNN»: второе двоеточие в метке перехода синтаксис Mermaid не допускает.

## 1. Текущая структура (AS-IS)

Кратко: показаны границы модуля, его внутренний состав и структура классов; каждое соединение подтверждено кодом (ссылки в §4) и прогоном сценария TTI 5.

### 1.1. Контекст модуля: границы и побочные эффекты

```mermaid
flowchart LR
    SM["SIMULATION_MANAGER<br/>run_tti :1657"] -->|"users: UE_ID, cqi, sbb_cqi, ue<br/>(срез на начало TTI, :1700)"| SCH["SCHEDULER.py<br/>schedule(tti, users) :346"]
    SCH -->|"возврат {allocation, statistics, bitmap,<br/>pdcch_stats} — отбрасывается (:1451)"| SM
    STATS["StatsManager :109<br/>collect :158"] -->|"scheduler.get_stats() :438,<br/>amc.get_stats() :1755,<br/>pdcch_manager.get_stats() :1660<br/>— читают кэш _last_*"| SCH

    UE["UE_MODULE<br/>GET_USERS_FOR_SCHEDULER :901"] -.->|"формирует список users"| SM
    SCH -->|"этап 1: сброс current_dl_throughput = 0 (:571)<br/>этап 7: UPD_DL_THROUGHPUT_BPS :629 (EWMA α=0.002)"| UEOBJ["объекты UE"]
    SCH -->|"этап 1: ue_has_buffer :530,<br/>get_buffer_status :492"| BS["BS_MODULE<br/>IBufferManager :472<br/>(Simple :580 / Layered :1356)"]
    SCH -->|"этап 7: get_packets(grants) :507 —<br/>физическое извлечение пакетов"| BS
    SCH -->|"этап 7 (с 6c0aae2, PR #39):<br/>update_all_qcis :223 — per-QCI throughput"| QCI["BS_MODULE<br/>QciThroughputTracker"]
    SCH -->|"этап 6: ALLOCATE_RBG :430 (атомарно,<br/>откат RELEASE_RBG :440); этап 8: GENERATE_BITMAP :567"| GRID["RES_GRID<br/>lte_grid"]
    SCH -->|"bytes_to_bits :217,<br/>BITS_PER_BYTE :140"| GL["GLOBALS"]
```

### 1.2. Внутренний состав модуля

```mermaid
flowchart TD
    SI["SchedulerInterface :216<br/>(оркестратор этапов 0–8 + фабрика create :235)"]
    PDM["PDCCHManager :1262<br/>бюджет CCE, aggregation level,<br/>release_cce — заглушка :1568"]
    AMC["AdaptiveModulationAndCoding :1703<br/>CQI_TO_MCS :1709, GET_BITS_PER_RB :1730"]
    CQM["CQIMap :191<br/>wb_cqi, sb_cqi (per-RBG), метки времени"]
    SG["SchedulingGrant :117<br/>ue_id, num_bytes, lcid,<br/>ndi, harq_process_id, rv"]
    SI -->|композиция| PDM
    SI -->|композиция| AMC
    SI -->|"агрегация: cqi_map"| CQM
    SI -.->|создаёт на этапе 7| SG
```

### 1.3. UML классов

```mermaid
classDiagram
    class SchedulerInterface {
        +lte_grid
        +amc
        +pdcch_manager
        +cqi_map
        +create(algorithm, lte_grid, bs)
        +schedule(tti, users) Dict
        +get_stats() Dict
        #_refresh_cqi()
        #_filter_eligible_ues()
        #_update_active_window()
        #filter_by_window()
        #_form_priority_list()
        #_apply_pdsch_estimation()
        #_allocate_pdcch()
        #_logical_channel_multiplexing()
        #_process_buffers()
        #_calculate_priorities()*
        #_allocate_pdsch()*
    }
    class BestCQIScheduler {
        #_calculate_priorities() priority = WB CQI
        #_allocate_pdsch() жадный TD
    }
    class RoundRobinScheduler {
        +rr_rbg_offset
        +rr_ue_offset
        #_calculate_priorities() ротация очереди
        #_apply_pdsch_estimation() pass-through
        #_allocate_pdsch() 1 RBG по кругу
    }
    class ProportionalFairScheduler {
        #_calculate_priorities() r/R
        #_allocate_pdsch() жадный TD
    }
    class FDxBestCQIScheduler {
        #_allocate_pdsch() per-RBG max SB CQI
    }
    class FDxFairGreedyScheduler {
        +rr_rbg_offset
        +rr_ue_offset
        #_allocate_pdsch() per-RBG + served_in_round
    }
    class FDxProportionalFairScheduler {
        #_allocate_pdsch() per-RBG PF по SB CQI
    }
    class BasicQosAwareScheduler {
        +metric_mode (Coita/PF)
        #_calculate_priorities() HOL-группировка + GBR-приоритет
        #_apply_pdsch_estimation() pass-through
        #_allocate_pdsch() per-RBG произведение hol×gbr×cqa
    }
    class QosAwareScheduler {
        +allocation_flow
        #_calculate_priorities() per-flow GBR/urgency
        #_apply_pdsch_estimation() pass-through
        #_allocate_pdsch() per-RBG выбор потока (sigmoid hol/PDB × pf)
        #_logical_channel_multiplexing() override QoS-мультиплексирования
    }
    class PDCCHManager {
        +total_cce
        +max_dl_cce
        +num_assigned_cce
        +get_aggregation_level(cqi)
        +allocate_cce(ue_id, n)
        +reset_tti()
        +release_cce(ue_id) заглушка
        +get_stats()
    }
    class AdaptiveModulationAndCoding {
        +CQI_TO_MCS
        +GET_BITS_PER_RB(cqi)
        +get_stats()
    }
    class CQIMap {
        +wb_cqi
        +sb_cqi
        +last_wb_update
        +last_sb_update
    }
    class SchedulingGrant {
        +ue_id
        +num_bytes
        +lcid
        +ndi
        +rv
        +to_dict()
    }
    SchedulerInterface <|-- BestCQIScheduler
    SchedulerInterface <|-- RoundRobinScheduler
    SchedulerInterface <|-- ProportionalFairScheduler
    SchedulerInterface <|-- FDxBestCQIScheduler
    SchedulerInterface <|-- FDxFairGreedyScheduler
    SchedulerInterface <|-- FDxProportionalFairScheduler
    SchedulerInterface <|-- BasicQosAwareScheduler
    SchedulerInterface <|-- QosAwareScheduler
    SchedulerInterface *-- PDCCHManager
    SchedulerInterface *-- AdaptiveModulationAndCoding
    SchedulerInterface o-- CQIMap : cqi_map
    SchedulerInterface ..> SchedulingGrant : создаёт на этапе 7
```

## 2. Типовой сценарий (AS-IS)

Сценарий: один TTI — вызов `schedule(tti, users)` из `SIMULATION_MANAGER.run_tti()`, включая все выходы за границы модуля. Порядок этапов подтверждён кодом :346–436 и прогоном ([README §4.1](README.md)); полный набор контрольных точек — 27 baseline-снимков ([README §4.3](README.md)).

### 2.1. Последовательность вызовов

```mermaid
sequenceDiagram
    participant SIM as SIMULATION_MANAGER
    participant SCH as schedule()
    participant CQI as cqi_map
    participant BS as BS_MODULE (буферы)
    participant PD as PDCCHManager
    participant GR as RES_GRID
    participant UE as UE_MODULE

    SIM->>SCH: schedule(tti, users) :1701
    SCH->>CQI: 0 _refresh_cqi :504 — записать WB/SB CQI
    SCH->>BS: 1 ue_has_buffer + get_buffer_status :559
    BS-->>SCH: размеры буферов → bs_buffer_size
    SCH->>UE: 1 current_dl_throughput = 0 (всем users) :571
    SCH->>SCH: 2–4 окно :610/:653 → приоритеты* → priority_list :684
    SCH->>SCH: 4.5 оценка PDSCH :712 (порог 95%)
    SCH->>PD: 5 reset_tti :1545 + allocate_cce :1496 (CQI → 1/2/4/8 CCE)
    PD-->>SCH: ues_with_pdcch
    loop каждый RBG (внутри _allocate_pdsch*)
        SCH->>GR: 6 ALLOCATE_RBG(tti, rbg_idx, ue_id) :430
        GR-->>SCH: True / False (при неудаче откат RELEASE_RBG :440)
    end
    SCH->>SCH: 7 вместимость RB → SchedulingGrant :811
    SCH->>BS: 7 get_packets(grants) :507 — извлечение пакетов
    BS-->>SCH: (packets, total_bytes)
    SCH->>UE: 7 UPD_DL_THROUGHPUT_BPS(bits, 1 ms) :629
    SCH->>BS: 7 update_all_qcis(ueid, bytes_per_qci) :967–975<br/>(с 6c0aae2, PR #39; в ветках отказа — :916–919/:941–944/:962–965)
    SCH->>GR: 8 GENERATE_BITMAP(tti, ue_id) :567
    SCH->>PD: 8 get_stats() :1660
    SCH-->>SIM: {allocation, statistics, bitmap, pdcch_stats}
```

Звёздочкой помечены виртуальные методы — точки расхождения восьми алгоритмов (`_calculate_priorities` :1172, `_allocate_pdsch` :1197); третья точка расхождения — переопределения `_apply_pdsch_estimation` (:712).

### 2.2. Досрочные выходы (dataflow с ветвлениями)

```mermaid
flowchart TD
    A["schedule(tti, users)"] --> B["0 _refresh_cqi :504"]
    B --> C["1 _filter_eligible_ues :559"]
    C -->|пусто| Z["_empty_result :1037<br/>(счётчики blocked/avg_priority<br/>не сбрасываются — BUG-SCH-004)"]
    C --> D["2/2.5 окно :610/:653"]
    D -->|пусто| Z
    D --> E["3 _calculate_priorities* :1172"]
    E --> F["4 _form_priority_list :684"]
    F -->|пусто| Z
    F --> G["4.5 _apply_pdsch_estimation :712"]
    G --> H["5 _allocate_pdcch :769"]
    H -->|пусто| Z
    H --> I["6 _allocate_pdsch* :1197"]
    I --> J["7 _process_buffers :880"]
    J --> K["7.5 _update_stats :1104"]
    K --> L["8 _build_result :996"]
    Z --> M["возврат в SIMULATION_MANAGER<br/>(в главном цикле отбрасывается :1451)"]
    L --> M
```

### 2.3. Контур обратной связи между TTI

Проверен замером на `ae9db40`: заранее выставленный `average_throughput` = 1 000 000 у незапланированного UE затухает до 996 004 за один TTI — двойной нулевой шаг EWMA ([BUG-SCH-001](audit.md#bug-sch-001)). На `6c0aae2` (PR #39) ветка получила `continue` (:919) — шаг один: 998 000; контур обратной связи ниже сохраняет смысл.

```mermaid
flowchart LR
    CH["Модель канала<br/>SINR_TO_CQI :693 → ue.cqi / cqi_subband"] --> CQ["cqi_map (_refresh_cqi :504)"]
    CQ --> PR["Приоритеты (этап 3) + AMC :1730"]
    TR["Трафик → буфер BS"] --> EL["Eligibility :559 + оценка :712"]
    PR --> AL["_allocate_pdsch :1197*"]
    EL --> AL
    AL --> BUF["get_packets :507 — буфер уменьшается"]
    BUF --> EW["UPD_DL_THROUGHPUT_BPS :629<br/>EWMA average_throughput (α=0.002 :644)"]
    EW -->|"знаменатель PF следующего TTI"| PR
    AL --> GRID["RES_GRID: RB заняты до переиспользования подкадра"]
    AL --> ST["_last_* :1104 → get_stats() :438 → StatsManager"]
```

### 2.4. Переходы состояний: жизненный цикл записи CQIMap (на UE)

Заявленный в docstring `CQIMap` (:197–201) жизненный цикл против фактического поведения кода:

```mermaid
stateDiagram-v2
    [*] --> Нет_записи : UE ещё не присылал валидный отчёт
    Нет_записи --> Created : первый валидный WB CQI 1–15<br/>_refresh_cqi стр. 529
    Created --> Updated : новый валидный отчёт<br/>wb_cqi, last_wb_update = tti стр. 539
    Updated --> Updated : невалидный CQI или пустой sbb_cqi —<br/>запись НЕ затирается (BUG-SCH-002) стр. 527/548
    Updated --> Deleted : дерегистрация UE<br/>НЕ РЕАЛИЗОВАНО — только в docstring стр. 200
    Created --> Deleted
    Deleted --> [*]
    note right of Updated
        _get_wb_cqi :1064 для UE без записи
        возвращает 0 — этап 1 отсеивает его (:592);
        на ae9db40 этап 7 при живом буфере падал
        ValueError (BUG-SCH-007), на 6c0aae2
        недостижим (continue :919)
    end note
```

### 2.5. Переходы состояний: бюджет CCE в пределах одного TTI (PDCCHManager)

```mermaid
stateDiagram-v2
    [*] --> Свободен : reset_tti() в начале этапа 5 стр. 1545
    Свободен --> Выделен : allocate_cce(ue_id, L) успех стр. 1496<br/>num_assigned_cce += L, cce_allocations[ue_id] = L
    Выделен --> Выделен : повторный запрос тому же UE → False стр. 1520
    Свободен --> Свободен : бюджет исчерпан → False стр. 1465
    Выделен --> Утёк : PDSCH для UE не разместились —<br/>release_cce() заглушка стр. 1568 (BUG-SCH-005)
    Утёк --> Свободен : только reset_tti() следующего TTI
```

### 2.6. Переходы состояний: UE в скользящем окне (`_ue_queue`)

```mermaid
stateDiagram-v2
    [*] --> В_очереди : новый eligible UE<br/>_update_active_window extend стр. 643
    В_очереди --> В_окне : позиция < window_size (100)<br/>filter_by_window стр. 668
    В_окне --> В_очереди : ротация обработанных в конец стр. 675
    В_очереди --> Исключён : UE пропал из eligible<br/>(буфер пуст / CQI невалиден) стр. 645–646
    Исключён --> В_очереди : UE снова eligible
    Исключён --> [*] : очередь переживает TTI
```

## 3. Целевая структура (TO-BE) — только если согласована

Не согласована: ниже — черновик к спецификации FEAT-SCH-008 v5 (структура TO-BE с v4 не менялась; в v5 добавлены граница с AMC-направлением, порядок слияния и целевой путь пакета — [SPEC §11.3](specs/SPEC-FEAT-SCH-008.md)) (статус и согласование — [features.md](features.md) §1, полный текст — [specs/SPEC-FEAT-SCH-008.md](specs/SPEC-FEAT-SCH-008.md)). Спецификация v4 учла замечания P1 архитектурного ревью v3 и уточнения границ смежных направлений: миграция — атомарными зелёными шагами (v4: M4c/M4d, M5a/M5b, M8a–M8e), в реестре до завершения переноса хранится legacy-маршрут (все старые имена работают на каждом PR); состояние планирования вынесено в типизированный `SchedulerState` во владении оркестратора (долговременные `rr_*_offset`; `served_in_round` остаётся локальной переменной FD_FGS), вторая сортировка и top-N остаются в оркестраторе; контракт аллокатора не содержит собственного rollback, и `ALLOCATE_RBG` **не объявляется гарантированно атомарной**: при частичном отказе `RELEASE_RBG` освобождает RB без проверки владельца ([BUG-SCH-009](audit.md#bug-sch-009)); результат аллокации — внутренний `AllocationResult` (per-flow бюджеты QoS доходят до мультиплексирования); переопределение мультиплексирования QoS становится явным контрактом `MultiplexingPolicy` — необязательным четвёртым элементом дескриптора пары, компонент вызывает оркестратор на этапе 7 (v4); экспериментальный путь работает по схеме propose→validate→commit с точкой запуска после этапа PDCCH (v4); политика CQI разделена на reporting/consumption с целевым контрактом `CqiEstimate`/`CqiView` и явной декларацией `channel_awareness` у алгоритмов (реализация — отдельный трек C1–C4). Направление «разделить ранжирование пользователей и аллокацию ресурсов» следует из roadmap модуля (:3227–3245) и дублирования механики аллокации в восьми `_allocate_pdsch` (шесть прежних + два QoS-класса из PR #39).

Чтение схемы: без пометки — существующий элемент; **(новый)** — добавляется; **(перенос)** — логика переезжает из указанного места; **(удалено)** — убирается из контракта; **(временный)** — существует только на период миграции (R5/R8).

```mermaid
flowchart TD
    subgraph NOW["AS-IS (6c0aae2)"]
        A1["SchedulerInterface :216<br/>этапы 0–8 + восемь _allocate_pdsch<br/>(:1931/:2040/:2160/:2269/:2423/:2631/:2806/:3044)"]
        A2["rr_ue_offset / rr_rbg_offset — поля классов RR (:1993–1994), FD_FGS (:2374–2375);<br/>served_in_round — локально в FD_FGS (:2448), остаётся локальным"]
        A3["переопределения _apply_pdsch_estimation<br/>(RR/FD/QoS — pass-through: :1996/:2243/:2377/:2571/:2753/:2988)"]
        A5["cqi_map (CQIMap :190–208) + _refresh_cqi :504–559 —<br/>копирование значений, не событие получения отчёта"]
        A6["allocation_flow — побочный канал QoS (:2986, :3087, :3149)<br/>→ _logical_channel_multiplexing (:3175–3222)"]
    end
    subgraph TOBE["TO-BE (FEAT-SCH-008 v5)"]
        B1["SchedulerInterface<br/>оркестратор этапов 0–8;<br/>вторая сортировка и top-N остаются здесь;<br/>общая точка проверки результата;<br/>commit экспериментального пути"]
        B2["SchedulerState (новый)<br/>dataclass: rr_ue_offset, rr_rbg_offset;<br/>владелец — оркестратор; один писатель на поле"]
        B3["UserRanker / scorer (новый, interface)<br/>rank(...) → prioritized_ues;<br/>целевая FD-форма — score_candidate(ue, rbg, flow?, tti)"]
        B4["ResourceAllocator (новый, interface)<br/>prefilter(...) — оценка PDSCH (перенос A3);<br/>allocate(...) → AllocationResult; без rollback"]
        B7["AllocationResult (новый, внутренний)<br/>ue_prbs + flow_budget_bits;<br/>LegacyAllocationAdapter — временный"]
        B5["SCHEDULER_REGISTRY (новый)<br/>имя → legacy | пара (+мультиплексирование) | custom;<br/>legacy-маршрут — временный (R5)"]
        B8["LegacySchedulerAdapter (временный)<br/>делегирование старым классам, без дублирования"]
        B9["CustomSchedulingPolicy (новый, контракт)<br/>propose → validate → commit после этапа PDCCH;<br/>mock proposal-only (M8e)"]
        B11["MultiplexingPolicy (новый)<br/>база :811 | QoS-компонент :3175–3222;<br/>вызов — оркестратор, этап 7"]
        B6["views-протоколы (новые)<br/>CqiView / GridView / AmcView / BuffersView / StatsView —<br/>точечная инъекция вместо ctx-контейнера"]
        B10["CQI: legacy-адаптер поверх cqi_map;<br/>целевые CqiEstimate/отчёты — трек C1–C4 (вне M-шагов)"]
    end
    A1 -->|"перенос: восемь _calculate_priorities"| B3
    A1 -->|"перенос: общая механика аллокации"| B4
    A1 -.->|"до переноса — legacy-маршрут по одному имени"| B8
    A2 -->|"перенос offsets — одному владельцу"| B2
    A3 -->|"перенос: стратегия оценки"| B4
    A5 -->|"чтение через адаптер (без изменения семантики)"| B10
    A6 -->|"перенос: явное поле результата"| B7
    A6 -->|"перенос override :3175-3222"| B11
    B1 --> B2
    B1 --> B5
    B5 -->|"pair"| B3
    B5 -->|"legacy"| B8
    B5 -->|"custom"| B9
    B3 --> B4
    B4 -->|"AllocationResult"| B1
    B4 -->|"flow_budget_bits"| B11
    B11 -->|"SchedulingGrant"| B1
    B9 -->|"AllocationProposal"| B1
    B3 --> B6
    B4 --> B6
    B9 --> B6
    B10 --> B6
    B4 -->|"пишет state (rr_*_offset);<br/>сетку меняет только через GridView.allocate_rbg"| B2
```

```mermaid
classDiagram
    class SchedulerInterface {
        +create(config) фабрика через SCHEDULER_REGISTRY (legacy | pair | custom)
        +schedule(tti, users) Dict — внешняя форма неизменна
        -state: SchedulerState
        -ranker: UserRanker
        -allocator: ResourceAllocator
        -multiplexing: MultiplexingPolicy (компонент QoS-пары, опционально)
        #_form_priority_list() сортировка+tie-break+top-N (остаётся)
        #validate_result() общая точка проверки результата
        #commit_plan() фаза commit экспериментального пути (v4)
    }
    class SchedulerState {
        +rr_ue_offset: int
        +rr_rbg_offset: int
    }
    class UserRanker {
        <<interface>>
        +rank(windowed_ues, views, state) prioritized_ues
        +score_candidate(ue, rbg, flow?, tti) целевая FD-форма
    }
    class ResourceAllocator {
        <<interface>>
        +prefilter(priority_list, views) List
        +allocate(tti, ues_with_pdcch, views, state) AllocationResult
    }
    class AllocationResult {
        <<value object>>
        +ue_prbs dict
        +flow_budget_bits dict or None
    }
    class CustomSchedulingPolicy {
        <<interface>>
        +propose(tti, eligible_ues, ues_with_pdcch, views, state) AllocationProposal
    }
    class MultiplexingPolicy {
        <<interface>>
        +multiplex(tti, ue, tb_bytes, buffer_statuses, flow_budgets, views) SchedulingGrant
    }
    class AllocationProposal {
        <<value object>>
        +ue_prbs dict
        +flow_budget_bits dict or None
    }
    class BestCQIRanker
    class ProportionalFairRanker
    class RoundRobinRanker
    class QosRanker
    class TDGreedyAllocator
    class RRCyclicAllocator
    class FDPerRBGAllocator
    class QosPerRBGAllocator
    UserRanker <|-- BestCQIRanker
    UserRanker <|-- ProportionalFairRanker
    UserRanker <|-- RoundRobinRanker
    UserRanker <|-- QosRanker
    ResourceAllocator <|-- TDGreedyAllocator
    ResourceAllocator <|-- RRCyclicAllocator
    ResourceAllocator <|-- FDPerRBGAllocator
    ResourceAllocator <|-- QosPerRBGAllocator
    ResourceAllocator ..> AllocationResult : возвращает
    CustomSchedulingPolicy ..> AllocationProposal : план до commit
    SchedulerInterface o-- SchedulerState : владеет
    SchedulerInterface o-- UserRanker : стратегия (путь pair)
    SchedulerInterface o-- ResourceAllocator : стратегия (путь pair)
    SchedulerInterface o-- MultiplexingPolicy : компонент QoS-пары, этап 7
    SchedulerInterface o-- CustomSchedulingPolicy : экспериментальный путь
```

**Реализовано в M1 (факт, не черновик):** узел `views-протоколы` — шесть `typing.Protocol` с `@runtime_checkable` (`CqiView` :3278, `GridView` :3296, `AmcView` :3327, `BuffersView` :3336, `StatsView` :3355, `StateView` :3376) и шесть тонких legacy-адаптеров (`LegacyCqiView` :3403, `LegacyGridView` :3424, `LegacyAmcView` :3457, `LegacyBuffersView` :3469, `LegacyStatsView` :3516, `LegacyStateView` :3557) введены в ветке `feat/SCH-008-M1-protocols` разделом в конце `SCHEDULER.py`; адаптеры делегируют существующим методам и не дублируют их (правило R5), потребители на views **не переключены** (правила R1/R2) — ни одна стрелка от оркестратора или алгоритмов к протоколам на схеме AS-IS (§1) не появилась. Целевая форма `CqiView` (`CqiEstimate`/`get_for_rbg`) и `abc.ABC`-иерархии `UserRanker`/`ResourceAllocator` в M1 не вводились: первое — CQI-трек C1 (не раньше M3), второе — шаги M4+ ([SPEC §5.3, §5.7, §8.3](specs/SPEC-FEAT-SCH-008.md)). Состав узлов и гарантии шага — [README §5](README.md) и [SPEC §4.3–4.4](specs/SPEC-FEAT-SCH-008.md).

Что изменится относительно AS-IS: общая механика аллокации (`remaining_buffer`, вызов `ALLOCATE_RBG`, сборка индексов RB, условия прерывания), продублированная сейчас в восьми `_allocate_pdsch`, уходит в иерархию `ResourceAllocator`; первоначальный порядок UE — в иерархию `UserRanker`, при этом ranker возвращает `prioritized_ues` как сейчас, а вторая сортировка, tie-break и top-N остаются в `_form_priority_list` (:684) — поведение отбора не меняется; долговременные RR-смещения (`rr_*_offset`) собираются в `SchedulerState` с правилом «один писатель на поле», `served_in_round` остаётся локальной переменной FD_FGS; переопределения оценки PDSCH становятся методом `prefilter` стратегии аллокатора (до M3 стратегии подключаются без реестра); per-flow решения QoS (`allocation_flow` :3087/:3149) переносятся в явное поле `AllocationResult.flow_budget_bits` и доходят до мультиплексирования без побочного канала, а само QoS-переопределение `_logical_channel_multiplexing` (:3175–3222) становится компонентом `MultiplexingPolicy` — необязательным четвёртым элементом дескриптора пары: оркестратор вызывает компонент из дескриптора на этапе 7 вместо `self._logical_channel_multiplexing`, внешняя форма грантов не меняется (M7b); дальнейшая эволюция QoS-мультиплексирования — карточки QoS-направления. Старые имена фабрики (`'BestCQI'`, `'FD_PF'`, `'QosAware'`, …) сохраняются через реестр с тремя видами маршрутов: `legacy` (делегирование старым классам — до переноса имени), `pair` (ranker+allocator плюс необязательный компонент мультиплексирования) и `custom` (целостная экспериментальная policy по схеме propose→validate→commit: точка запуска после этапа PDCCH, план на снимке свободных ресурсов, проверка до изменений, commit только оркестратором) — обратная совместимость с `SchedulerConfig` (SIMULATION_MANAGER.py:1044); совместимость фабрики с ядром подтверждается при интеграции, переписывания Simulation Manager не требуется. Перенос выполняется по одному алгоритму за PR (план M0–M8, v4: M4c/M4d, M5a/M5b, M8a–M8e, [specs/SPEC-FEAT-SCH-008.md](specs/SPEC-FEAT-SCH-008.md) §8) — на любом промежуточном шаге все восемь имён остаются рабочими; mock custom policy в M8e — proposal-only до реализации propose→validate→commit; синхронизация `available_algorithms()` (M8a) меняет наблюдаемый результат метода и сливается после отдельного согласования (R2). В ту же схему без изменения оркестратора встраиваются: DRL-ранжирование из ветки `feature/drl-scheduler` (PPO/DQN-ranker — как pair, либо как `CustomSchedulingPolicy`, если разложение на две стратегии искусственно) и венгерский бенчмарк ([FEAT-SCH-004](features.md#feat-sch-004) — офлайн эталонный Allocator; batch-путь `GridView.snapshot_free()/commit(plan)` — точка расширения, реализуемая после [BUG-SCH-009](audit.md#bug-sch-009)). Доступ к CQI стратегии получают через `CqiView`: на период миграции — legacy-адаптер поверх `cqi_map` с прежней семантикой WB/SB и fallback (поведение `6c0aae2` сохраняется), целевой контракт `CqiEstimate` с метками `measurement_tti`/`received_tti` и независимые политики reporting/consumption внедряются отдельным CQI-треком C1–C4; каждый алгоритм явно декларирует потребляемый набор CSI (`channel_awareness`), автоматического переключения режимов нет. Контракт без собственного rollback не отменяет [FEAT-SCH-001](features.md#feat-sch-001): откат CCE при провале PDSCH остаётся задачей оркестратора/менеджера контроля в точке отказа аллокатора.

### 3.1. Целевая раскладка пакетов проекта (обсуждается, не утверждена)

Проект переходит к модульной раскладке: отдельные папки моделей, ядра и планировщиков, при этом обсуждается вариант MAC-уровня, объединяющего планировщик и HARQ. Решение не принято, поэтому шаг M8b ([SPEC §8](specs/SPEC-FEAT-SCH-008.md)) — вынос стратегий, реестра и состояния в пакет — до фиксации раскладки не выполняется: путь пакета и маршруты импортов зависят от этого решения ([audit.md §3](audit.md), запрос в [features.md §2](features.md)).

```mermaid
flowchart LR
    subgraph FLAT["AS-IS: плоская раскладка (dev@6c0aae2)"]
        P["PyScheduler/<br/>SCHEDULER.py · SIMULATION_MANAGER.py<br/>CHANNEL_MODEL.py · MOBILITY_MODEL.py · TRAFFIC_MODEL.py<br/>BS_MODULE.py · UE_MODULE.py · RES_GRID.py<br/>GLOBALS.py · ENVIRONMENT.py · PLOT_METRICS.py · TEST_MODULES.py"]
        T["PyScheduler/tests/<br/>baseline/ · scheduler_arch_tests/ · scheduler_tests/ · …"]
    end
    subgraph V2["Целевая раскладка v2 (обсуждается)"]
        MM["models/<br/>мобильность, трафик, канал"]
        CC["core/<br/>ядро симулятора, сетка, статистика"]
        SS["scheduler/<br/>оркестратор, ранкеры, аллокаторы, реестр"]
        MC["mac/ — вариант<br/>scheduler + harq + pdcch"]
        DD["docs/<br/>документация по модулям"]
    end
    P -->|"M8b: вынос стратегий, реестра, состояния"| SS
    SS -.->|"если утвердят MAC-уровень"| MC
    DD --> DS["docs/scheduler/<br/>README · audit · diagrams · features · specs/"]
```

Что уже не зависит от этого решения:

- **Документация** размещается по принятому принципу «общая папка `docs/` + папка модуля»: `docs/scheduler/` с четырьмя документами и `specs/SPEC-FEAT-SCH-008.md` ([README §8](README.md)). Перенос в другую раскладку — `git mv` плюс правка относительных ссылок; внешних ресурсов в комплекте нет.
- **Код шагов M0–M7** живёт в `SCHEDULER.py` (добавления в конец файла, якоря legacy-кода не сдвигаются — [README §4.7](README.md)) и в каталогах `tests/`; на раскладку v2 это не завязано.
- **Контракты** (`AmcView`, `GridView`, `CqiView`, `BuffersView`, `StatsView`, `StateView` — :3278–3401, адаптеры :3403–3584) — точка независимости: при переезде в пакет меняются маршруты импортов, а не интерфейсы.
- **Версионирование** пакетов и релизов отложено; правило R8 (удаление legacy не ранее одного релизного цикла) до определения схемы версионирования опирается на решение владельца репозитория.

## 4. Ссылки на исходники

- `PyScheduler/SCHEDULER.py::SchedulerInterface.schedule` (:346–436) — порядок этапов 0–8, досрочные выходы через `_empty_result` (:1037)
- `PyScheduler/SCHEDULER.py::SchedulerInterface.create` (:235) — фабрика восьми алгоритмов (ключи QoS :262–263)
- `PyScheduler/SCHEDULER.py::BasicQosAwareScheduler/QosAwareScheduler` (:2704/:2960) — QoS-классы, слитые в PR #39 (`6c0aae2`)
- `PyScheduler/SCHEDULER.py::_process_buffers` (:880) — ветки отказа с `continue` (:919/:925/:944/:965) и обновление `qci_tput_tracker` (:916–919, :967–975) на `6c0aae2`
- `PyScheduler/SCHEDULER.py::_refresh_cqi` (:504, SB-часть :546) — поведение карты CQI, невалидные отчёты игнорируются
- `PyScheduler/SCHEDULER.py::PDCCHManager.allocate_cce/release_cce/reset_tti` (:1496/:1568/:1545) — учёт CCE и заглушка освобождения
- `PyScheduler/SCHEDULER.py::AdaptiveModulationAndCoding.GET_BITS_PER_RB` (:1730) — CQI → биты/RB, TODO TBS-таблиц :1749
- `PyScheduler/SIMULATION_MANAGER.py::run_tti` (:1657), создание планировщика (:1357), `SchedulerConfig` (:1030), отброшенный результат (:1451), `StatsManager` (:109, collect :158)
- `PyScheduler/UE_MODULE.py::GET_USERS_FOR_SCHEDULER` (:901–920) — контракт входа `users`, ключ `sbb_cqi` (:915)
- `PyScheduler/UE_MODULE.py::UPD_DL_THROUGHPUT_BPS` (:629, EWMA α=0.002 :644), `SINR_TO_CQI` (:693, клампит 1–15)
- `PyScheduler/BS_MODULE.py::IBufferManager` (:472; get_buffer_status :492, get_packets :507, ue_has_buffer :530), `LayeredBufferManager` (:1356), `BaseStation.use_simple_buffer` (:1696)
- `PyScheduler/RES_GRID.py::ALLOCATE_RBG/RELEASE_RBG/GET_RBG_INDICES/GET_RBG_SIZE/GENERATE_BITMAP` (:430/:440/:424/:421/:567), `BANDWIDTH_TO_RB` (:283), `RBG_SIZE_TABLE` (:293); `RELEASE_RB/RELEASE_RBG` (:399/:440) — освобождение без проверки владельца (BUG-SCH-009)
- `PyScheduler/BS_MODULE.py::QciThroughputTracker.update_all_qcis` (:223) — per-QCI throughput, точка встраивания этапа 7
- `PyScheduler/GLOBALS.py::bytes_to_bits` (:217), `BITS_PER_BYTE` (:140), `is_gbr` (:185)
- `PyScheduler/SCHEDULER.py` :3246–3584 — раздел M1 (протоколы и view-адаптеры), существует только в ветке `feat/SCH-008-M1-protocols` (коммит `f14885c`); в `dev@6c0aae2` файл заканчивается на строке 3245, поэтому якоря §1–§2 действительны в обоих случаях
- `PyScheduler/tests/baseline/` и `PyScheduler/tests/scheduler_arch_tests/` — characterization-снимки и контрактные тесты шагов M0/M1 (в `dev` отсутствуют, поставка патчами — [README §4.3–4.6](README.md))

Правила: схемы хранить **текстом Mermaid** в Markdown; AS-IS и TO-BE не смешивать; не рисовать каждую функцию; схема должна соответствовать конкретному коммиту.
