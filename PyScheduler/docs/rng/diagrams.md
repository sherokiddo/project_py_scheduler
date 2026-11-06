# Диаграммы модуля RNG

> Ответственный: Шаимов Богдан | Проверено по коду: `RNG.py` (v2.0.0)

## 1. Структура классов и контрактов модуля (`RNG.py` v2.0.0)

Фасад `RandomGenerator` реализует внешний контракт `RandomProvider` и создаёт подпотоки через внутренний контракт `UniformEngine` по выбранной стратегии из словаря `_ENGINES`.

```mermaid
classDiagram
    class UniformEngine {
        <<Protocol>>
        +rand_u01() float
    }

    class RandomProvider {
        <<Protocol>>
        +uniform(min_val, max_val, stream_offset) float
        +normal(mean, variance, bound, stream_offset) float
        +exponential(mean, bound, stream_offset) float
        +get_integer(min_val, max_val, stream_offset) int
    }

    class RngStream {
        +int seed
        +int stream_idx
        +int run_idx
        +list~float~ state
        +rand_u01() float
    }

    class PhiloxStream {
        +int seed
        +int stream_idx
        +int run_idx
        +list~int~ key
        +list~int~ counter
        +rand_u01() float
    }

    class RandomGenerator {
        +int seed
        +int base_stream_idx
        +int run_idx
        +str engine
        -dict _streams
        -dict _normal_cache
        -_get_stream(stream_offset) UniformEngine
        +uniform(min_val, max_val, stream_offset) float
        +normal(mean, variance, bound, stream_offset) float
        +exponential(mean, bound, stream_offset) float
        +get_integer(min_val, max_val, stream_offset) int
    }

    RngStream ..|> UniformEngine : реализует
    PhiloxStream ..|> UniformEngine : реализует
    RandomGenerator ..|> RandomProvider : реализует
    RandomGenerator "1" *-- "0..*" UniformEngine : создаёт в _streams
```

## 2. Сценарий вызова `RandomGenerator.normal()`

Показаны ленивая инициализация подпотока `stream_offset` и работа кэша пары чисел Бокса — Мюллера `_normal_cache`.

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Модель (Потребитель)
    participant RG as RandomGenerator
    participant Engine as RngStream / PhiloxStream

    Note over Caller,RG: Вызов №1 (подпоток stream_offset=1 ещё не создан)
    Caller->>RG: normal(mean=0.0, variance=1.0, bound=3.0, stream_offset=1)
    RG->>RG: Проверка variance >= 0, bound >= 0, расчёт std
    RG->>RG: _get_stream(stream_offset=1)
    alt Поток отсутствует в _streams
        RG->>Engine: __init__(seed, base_stream_idx + 1, run_idx)
        Engine-->>RG: экземпляр UniformEngine
    end
    loop Пока не выполнено 0 < w <= 1.0 и |x - mean| <= bound
        RG->>Engine: rand_u01()
        Engine-->>RG: u1
        RG->>Engine: rand_u01()
        Engine-->>RG: u2
        RG->>RG: Расчёт (x1, x2) по методу Бокса — Мюллера
    end
    RG->>RG: Сохранить (y, v2) в _normal_cache[1]
    RG-->>Caller: Возврат x1 (float)

    Note over Caller,RG: Вызов №2 (используется второе число из _normal_cache[1])
    Caller->>RG: normal(mean=0.0, variance=1.0, bound=3.0, stream_offset=1)
    RG->>RG: Извлечь (y, v2) из _normal_cache.pop(1) и вычислить x2
    alt |x2 - mean| <= bound
        RG-->>Caller: Возврат x2 (float)
    else |x2 - mean| > bound
        RG->>Engine: Генерация новой пары через rand_u01()
        RG-->>Caller: Возврат нового значения (float)
    end
```

## 3. Схема целевой интеграции с `SimulationManager` и моделями (TO-BE)

```mermaid
flowchart LR
    SM["SimulationManager\n(назначает base_stream_idx:\n1000, 2000, 3000)"] -->|"Создаёт"| RG["RandomGenerator\n(seed, base_stream_idx, run_idx, engine)"]
    RG -.->|"Передаётся как\nRandomProvider (DI)"| MOB["MobilityModel\n(stream_offset = UE_ID)"]
    RG -.->|"Передаётся как\nRandomProvider (DI)"| CH["ChannelModel\n(stream_offset = UE_ID)"]
    RG -.->|"Передаётся как\nRandomProvider (DI)"| TR["TrafficModel\n(stream_offset = UE_ID)"]
```

## 4. Ссылки на исходники

- `RNG.py::UniformEngine`, `RNG.py::RandomProvider` — протоколы модуля.
- `RNG.py::RngStream` — ядро `MRG32k3a` (`_build_cache`, `_advance_nth_by`, `rand_u01`).
- `RNG.py::PhiloxStream` — ядро `Philox4x64` (`_philox_round`, `_mulhilo`, `rand_u01`).
- `RNG.py::RandomGenerator` — фасад распределений (`_get_stream`, `uniform`, `normal`, `exponential`, `get_integer`).