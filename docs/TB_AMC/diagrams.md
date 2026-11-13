## Для AMC:

> Актуализация 2026-11-01: [исправления по ревью AMC](AMC_REVIEW_FIXES.md) имеют приоритет над прежними предложениями ниже. Legacy-статистика и конструктор сохранены; TB вынесен в TRANSPORT_BLOCK.py, TBS исправлен на 88 в I_TBS=6/N_PRB=1, добавлен интерфейс политики MCS. AMC_METRICS и удаление get_stats сейчас не планируются.

```mermaid
flowchart LR
    SCH["Scheduler"] -->|"CQI"| AMC["AdaptiveModulationAndCoding"]
    CFG["Конфигурация PDCCH"] -->|"pcfich"| AMC
    AMC -->|"оценка bits_per_rb"| SCH

    CQI["CQI"] --> MCS["get_mcs_from_cqi"]
    MCS --> IDX["Индекс MCS"]
```
## Для HARQ:    
```mermaid
flowchart LR
    SCH["SchedulerInterface"] --> REF["harq_manager = None"]
    METHOD["_get_harq_retransmissions"] -->|"при отсутствии manager"| EMPTY["Пустой список"]
    TB["TB / SchedulingGrant"] --> FIELDS["Поля NDI, RV, process ID"]
```

## Вынос AMC из SCHEDULER.py

Статус: TO-BE обсуждается. Основание: [SPEC-FEAT-AMC-001](SPEC-FEAT-AMC-001.md).
Класс и таблицы уже перенесены в AMC.py, прежняя связь с Scheduler сохранена. `AdaptiveModulationAndConfig`
в обсуждениях соответствует существующему `AdaptiveModulationAndCoding`.

### Текущее размещение — AS-IS

```mermaid
classDiagram
    namespace SCHEDULER_py {
        class SchedulerInterface
        class PDCCHManager
    }
    namespace AMC_py {
        class AdaptiveModulationAndCoding
    }
    class StatsManager
    class SimulationManager
    SchedulerInterface --> AdaptiveModulationAndCoding : создаёт self.amc
    AdaptiveModulationAndCoding --> SchedulerInterface : читает состояние для статистики
    AdaptiveModulationAndCoding ..> PDCCHManager : читает pcfich через Scheduler
    StatsManager ..> AdaptiveModulationAndCoding : scheduler.amc.get_stats
    SimulationManager ..> AdaptiveModulationAndCoding : текущие показатели
```

### Самостоятельный AMC — TO-BE v2

Прежняя схема с чтением Scheduler внутри AMC отменена по замечанию куратора.
Ниже целевая граница; интерфейсы пока не реализованы и требуют согласования.

```mermaid
classDiagram
    class SchedulerInterface {
        +get_amc_stats()
    }
    class AmcConfig {
        +int control_symbols
    }
    class AdaptiveModulationAndCoding {
        +get_mcs_from_cqi(cqi)
        +get_tbs_bits(mcs, rb)
        +GET_BITS_PER_RB(cqi, n_ports)
    }
    class AmcViewAdapter {
        +bits_per_rb(cqi)
    }
    class SchedulingStrategies
    class TransportBlock
    class TbsCalculator
    class GlobalTables
    class AmcStatsSnapshot
    class AmcMetricsCollector
    class StatsManager
    class SimulationManager

    SchedulerInterface --> AdaptiveModulationAndCoding : числовые запросы
    AdaptiveModulationAndCoding --> AmcConfig : только значения
    AdaptiveModulationAndCoding ..> TbsCalculator : делегирует
    TransportBlock ..> TbsCalculator : делегирует
    TbsCalculator --> GlobalTables : lookup
    AmcViewAdapter --> AdaptiveModulationAndCoding : делегирует
    SchedulingStrategies --> AmcViewAdapter : AmcView
    SchedulerInterface ..> AmcStatsSnapshot : копирует значения
    AmcMetricsCollector ..> AmcStatsSnapshot : агрегирует
    SchedulerInterface ..> AmcMetricsCollector : передаёт снимок
    StatsManager --> SchedulerInterface : get_amc_stats
    SimulationManager --> SchedulerInterface : get_amc_stats
```

В целевой реализации AMC не содержит Scheduler, PDCCHManager, UE или callback на их состояние. Сейчас ссылка на Scheduler ещё существует.
Статистика формируется снаружи по снимку примитивных данных. TbsCalculator и AmcMetricsCollector обозначают функции, не обязательные классы реализации.
AmcViewAdapter принадлежит интеграционному слою, его TBS-расширение нужно обсуждать с Тимуром. Детали контрактов, первого этапа TBS и регрессии после каждого шага:
[SPEC-FEAT-AMC-001](SPEC-FEAT-AMC-001.md).

Основания AS-IS: SCHEDULER.py::SchedulerInterface.__init__,
AMC.py::AdaptiveModulationAndCoding.GET_BITS_PER_RB/get_stats;
SIMULATION_MANAGER.py::StatsManager.collect/SimulationManager.start_simulation.
