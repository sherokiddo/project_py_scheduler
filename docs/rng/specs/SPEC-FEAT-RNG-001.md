# SPEC-FEAT-RNG-001 — Модуль генерации случайных последовательностей (RNG v2.0.0)

> Ответственный: Шаимов Богдан  
> Ветка: `feat/RNG-001-M1-protocols` | База: `dev` | Обновлено: `2026-11-06`  
> Статус: Реализовано в `RNG.py` (v2.0.0), проверено локальными модульными тестами.  
> Связанные документы: [README.md](../README.md) | [diagrams.md](../diagrams.md) | [audit.md](../audit.md) | [features.md](../features.md)

---

## 1. Назначение

В текущей версии симулятора модели используют локальные вызовы `np.random` и `random`, которые не позволяют централизованно управлять потоками случайных чисел.

Модуль `RNG.py` (v2.0.0) решает три задачи:
1. Предоставляет два детерминированных математических ядра с поддержкой независимых потоков (`stream_idx`) и прогонов (`run_idx`): `RngStream` (`MRG32k3a`, совместим с `ns-3`) и `PhiloxStream` (`Philox4x64`).
2. Задаёт статические контракты `typing.Protocol` (`UniformEngine` и `RandomProvider`) без жёсткой привязки моделей к конкретному классу генератора.
3. Предоставляет фасад распределений `RandomGenerator` (`uniform`, `normal`, `exponential`, `get_integer`) с выбором движка через реестр стратегий `_ENGINES`.

---

## 2. Границы текущего PR (Этап M1)

**Входит в текущий PR:**
- Автономный модуль `RNG.py` (v2.0.0) с контрактами `UniformEngine` и `RandomProvider`.
- Устранение дефектов `BUG-RNG-001..006` и оптимизация `OPT-RNG-001` (см. `docs/rng/audit.md`).
- Документация модуля в `docs/rng/`.

**Не входит (следующие этапы после слияния в `dev`):**
- Изменение кода `SIMULATION_MANAGER.py` и моделей (`MOBILITY_MODEL.py`, `CHANNEL_MODEL.py`, `TRAFFIC_MODEL.py`).
- Чтение предзаписанных трасс из файлов (вынесено в отдельную будущую задачу `FEAT-RNG-003`).

---

## 3. Архитектура и контракты (`RNG.py`)

| Компонент | Роль | Реализация |
|---|---|---|
| `UniformEngine` | Внутренний `Protocol` ядра ПСЧ | Метод `rand_u01() -> float` на полуинтервале `[0.0, 1.0)`. |
| `RandomProvider` | Внешний `Protocol` для моделей | Методы `uniform`, `normal`, `exponential`, `get_integer`. |
| `RngStream` | Стратегия `MRG32k3a` | Кэш матриц перехода, шаг между потоками $2^{127}$, между прогонами $2^{76}$. |
| `PhiloxStream` | Стратегия `Philox4x64` | 10 раундов, буфер на 4 числа, нормализация `(val >> 11) * (2.0 ** -53)`. |
| `RandomGenerator` | Фасад распределений | Реализует `RandomProvider`. Хранит словарь стратегий `_ENGINES`, потоки `_streams` и кэш Бокса — Мюллера `_normal_cache`. |

> **Примечание по докстрингам:** пометка `optional` в секциях `Args:` означает только наличие у параметра значения по умолчанию и не подразумевает передачу `None` (`typing.Optional`).

---

## 4. Контракт интеграции для смежных модулей (Этап M2)

### 4.1. Для `SimulationManager` (Александр С.)
`SimulationManager` создаёт экземпляры `RandomGenerator` и задаёт непересекающиеся `base_stream_idx` для каждого домена симуляции:

```python
from RNG import RandomGenerator, RandomProvider

STREAM_BASE_UE_INIT = 0
STREAM_BASE_MOBILITY = 1000
STREAM_BASE_CHANNEL = 2000
STREAM_BASE_TRAFFIC = 3000

self.mobility_rng: RandomProvider = RandomGenerator(
    seed=self.sim_config.rng_seed,
    base_stream_idx=STREAM_BASE_MOBILITY,
    run_idx=self.sim_config.rng_run_idx,
    engine=self.sim_config.rng_engine,
)
```

### 4.2. Для моделей `Mobility, Channel, Traffic` (Андрей К.) — Явная инъекция зависимости
Модели принимают `rng: RandomProvider` через конструктор как явную обязательную зависимость. Немой `fallback` вида `rng if rng is not None else RandomGenerator()` запрещён, так как при пропущенном аргументе разные модели молча создадут генераторы с одинаковыми (`seed=12345, base_stream_idx=0, run_idx=1`) и начнут выдавать искусственно скоррелированные последовательности:

```python
from RNG import RandomProvider

class MobilityInterface:
    def __init__(self, ue: UserEquipment, rng: RandomProvider, **kwargs):
        if rng is None:
            raise ValueError(
                "Parameter 'rng' (RandomProvider) must be explicitly provided via Dependency Injection."
            )
        self.ue = ue
        self.rng: RandomProvider = rng
```

---
### 4.3. Правила миграции вызовов `np.random` на RandomProvider (Этап M2)
При переводе `CHANNEL_MODEL.py, TRAFFIC_MODEL.py, MOBILITY_MODEL.py и UE_MODULE.py` с `np.random на RandomProvider` необходимо учитывать различия в семантике параметров:

# Таблица перевода вызовов `np.random` → `RandomProvider` (RNG.py)

| Вызов в существующем коде (`np.random`) | Семантика в `np.random` | Корректный вызов `RandomProvider` (`RNG.py`) | Важное правило перевода |
|---|---|---|---|
| `np.random.uniform(a, b)` | Полуинтервал [a, b) | `rng.uniform(a, b)` | Семантика границ совпадает: [a, b). |
| `np.random.normal(loc=mu, scale=sigma)` | `scale` — это СКО (σ, stddev) | `rng.normal(mean=mu, variance=sigma ** 2)` | Метод `normal()` принимает **дисперсию** (σ²), а не σ! Обязательно возводить `sigma` в квадрат. |
| `np.random.exponential(scale=beta)` | Математическое ожидание β > 0 | `rng.exponential(mean=beta)` | Семантика параметра совпадает (`mean = beta`). |
| `np.random.randint(low, high)` | Целое из [low, high) (верхняя граница exclusive) | `rng.get_integer(low, high - 1)` | В `get_integer(min_val, max_val)` **обе границы inclusive** ([min_val, max_val]). Верхнюю границу нужно уменьшать на `1`. |
| `rng.choice(seq)` (в `UE_MODULE`, `CHANNEL_MODEL`) | Выбор случайного элемента последовательности | `seq[rng.get_integer(0, len(seq) - 1)]` | Равновероятный выбор индекса через `get_integer(0, len(seq) - 1)`. |
| Векторные генерации `size=N` (в `CHANNEL_MODEL`, `TRAFFIC_MODEL`) | Массив из N случайных чисел | `np.array([rng.normal(...) for _ in range(N)])` | Последовательный вызов в генераторе списков сохраняет детерминированный порядок чтения из потока. |

## 5. Проверка (Definition of Done)
### 1. Добавление `RNG.py` не изменяет существующие файлы симулятора и не нарушает работу Legacy-кода.

### 2. Локальными модульными тестами подтверждены: воспроизводимость генерации `MRG32k3a` и `Philox4x64`, строгий полуинтервал `[0.0, 1.0)` у `PhiloxStream`, равномерность `get_integer` на отрицательных границах, защита от NaN/Inf и бесконечного цикла при `bound=0.0`, соблюдение верхних границ `seed <= 4294944442, stream_idx <= 2^63 - 1, run_idx <= 2^63 - 1,` а также работа реестра `StreamAllocator`.