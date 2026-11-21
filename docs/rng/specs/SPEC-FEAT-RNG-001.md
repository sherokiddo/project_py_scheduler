# SPEC-FEAT-RNG-001 — Модуль генерации случайных последовательностей (RNG v2.0.2)

> Ответственный: Шаимов Богдан  
> Ветка: `feat/RNG-001-M1-protocols` | База: `dev` | Обновлено: `2026-11-20`  
> Статус: Реализовано в `RNG.py` (v2.0.2), проверено локальными модульными тестами.  
> Связанные документы: [README.md](../README.md) | [diagrams.md](../diagrams.md) | [audit.md](../audit.md) | [features.md](../features.md)

---

## 1. Назначение

В текущей версии симулятора модели используют локальные вызовы `np.random` и `random`, которые не позволяют централизованно управлять потоками случайных чисел.

Модуль `RNG.py` (v2.0.2) решает четыре задачи:
1. Предоставляет два детерминированных математических ядра с поддержкой независимых потоков (`stream_idx`) и прогонов (`run_idx`): `RngStream` (`MRG32k3a`, совместим с `ns-3`) и `PhiloxStream` (`Philox4x64`).
2. Задаёт статические контракты `typing.Protocol` (`UniformEngine` и `RandomProvider`) без жёсткой привязки моделей к конкретному классу генератора.
3. Предоставляет фасад распределений `RandomGenerator` (`uniform`, `normal`, `exponential`, `get_integer`) с защитой граничных случаев `float64` и выбором движка через реестр стратегий `_ENGINES`.
4. Предоставляет централизованный реестр `StreamAllocator`, закрепляющий изолированные потоки за логическими ключами `(domain, purpose, entity_id)` независимо от порядка регистрации модулей.

---

## 2. Границы текущего PR (Этап M1)

**Входит в текущий PR:**
- Автономный модуль `RNG.py` (v2.0.2) с контрактами `UniformEngine` и `RandomProvider`, ядрами `RngStream` / `PhiloxStream`, фасадом `RandomGenerator` и реестром `StreamAllocator`.
- Устранение дефектов `BUG-RNG-001..006`, оптимизация `OPT-RNG-001` и закрытие граничных случаев по итогам ревью (см. `docs/rng/audit.md`).
- Документация модуля в `docs/rng/`, включая фиксацию происхождения портированных из `ns-3` математических блоков и план их последующей переписи по открытой математической спецификации P. L'Ecuyer (1999, 2002) с сохранением `ns-3` в качестве эталона для побитовых тестов.

**Не входит (следующие этапы после слияния в `dev`):**
- Изменение кода `SIMULATION_MANAGER.py` и моделей (`MOBILITY_MODEL.py`, `CHANNEL_MODEL.py`, `TRAFFIC_MODEL.py`).
- Чтение предзаписанных трасс из файлов (вынесено в отдельную будущую задачу `FEAT-RNG-003`).

---

## 3. Архитектура и контракты (`RNG.py`)

| Компонент | Роль | Реализация |
|---|---|---|
| `UniformEngine` | Внутренний `Protocol` ядра ПСЧ | Метод `rand_u01() -> float` на полуинтервале `[0.0, 1.0)`. |
| `RandomProvider` | Внешний `Protocol` для моделей | Методы `uniform`, `normal`, `exponential`, `get_integer` (без доступа к числовым смещениям потоков). |
| `RngStream` | Стратегия `MRG32k3a` | Потокобезопасный кэш матриц перехода (`threading.Lock` + `tuple`), шаг между потоками $2^{127}$, между прогонами $2^{76}$, предел `MAX_RUN_IDX = 2^51` ($2^{127} / 2^{76}$). |
| `PhiloxStream` | Стратегия `Philox4x64` | 10 раундов, буфер на 4 числа, нормализация `(val >> 11) * (2.0 ** -53)`. |
| `RandomGenerator` | Фасад распределений | Реализует `RandomProvider`. Хранит словарь стратегий `_ENGINES`, изолированный поток `_stream` и кэш Бокса — Мюллера `_normal_cache`. |
| `StreamAllocator` | Реестр и аллокатор потоков | Вычисляет индекс потока в `[0, 2^63 - 1]` через `blake2b` от `(domain, purpose, entity_id)` и контролирует единственного владельца каждого ключа. |

### 3.1. Обоснование пределов и граничных случаев

1. **Предел `MAX_RUN_IDX = 2**51` (`1 << 51`):** в `MRG32k3a` сдвиг на 1 `stream_idx` равен $2^{127}$ шагов, а сдвиг на 1 `run_idx` (при 1-based нумерации, число прыжков `run_idx - 1`) равен $2^{76}$ шагов. Внутри одного потока помещается $2^{127} / 2^{76} = 2^{51}$ непересекающихся прогонов:
   - при `stream_idx = 0, run_idx = 2**51` сдвиг равен $(2^{51} - 1) \cdot 2^{76} = 2^{127} - 2^{76}$ (остаётся внутри потока `0`);
   - при `stream_idx = 0, run_idx = 2**51 + 1` сдвиг достигает $2^{127}$ и совпадает с `(stream_idx = 1, run_idx = 1)`.
2. **Границы `uniform(min_val, max_val)`:** при переполнении разности `max_val - min_val` (например, `[-1e308, 1e308]`) используется масштабирование через половины интервала. Если при сложении на соседних представимых `float` округление дотягивает результат до `max_val`, значение сдвигается на один шаг назад через `math.nextafter(max_val, min_val)`, сохраняя строгий полуинтервал `[min_val, max_val)`.
3. **Завершение `exponential()` и `normal()` при малых `bound`:** при `bound > 0` в `exponential()` используется аналитическая формула усечённого распределения через `math.expm1` и `math.log1p` за $O(1)$ без цикла отбора. В `normal()` при `bound > 0` проверяется минимальное отношение `bound / std >= 1e-6` и ограничивается число попыток отбраковки с выбросом `ValueError`.
4. **Точность `get_integer(min_val, max_val)`:** отбор выполняется побитово в целых числах без приведения больших границ (`> 2^53`) к `float64`.

> **Примечание по докстрингам:** пометка `optional` в секциях `Args:` означает только наличие у параметра значения по умолчанию и не подразумевает передачу `None` (`typing.Optional`).

---

## 4. Контракт интеграции для смежных модулей (Этап M2)

### 4.1. Для `SimulationManager` (Александр С.) — Выделение потоков через `StreamAllocator`
Фиксированные числовые смещения (`1000`, `2000`, `3000`) и аргумент `stream_offset` не используются. `SimulationManager` создаёт один экземпляр `StreamAllocator` на прогон и запрашивает отдельный `RandomProvider` под каждый случайный процесс по ключу `(domain, purpose, entity_id)`:

```python
from RNG import RandomProvider, StreamAllocator

allocator = StreamAllocator(
    seed=self.sim_config.rng_seed,
    run_idx=self.sim_config.rng_run_idx,
    engine=self.sim_config.rng_engine,
)

# Выделение генератора для модели мобильности
mobility_rng: RandomProvider = allocator.get_provider(
    domain="MOBILITY",
    purpose="WAYPOINT",
    entity_id=ue_id,
)

# Пример модели с двумя независимыми случайными процессами (канал UE):
# каждый процесс получает собственный изолированный генератор и не может
# обратиться к чужому потоку по числовому смещению
channel_sf_rng: RandomProvider = allocator.get_provider(
    domain="CHANNEL",
    purpose="SHADOW_FADING",
    entity_id=ue_id,
)
channel_los_rng: RandomProvider = allocator.get_provider(
    domain="CHANNEL",
    purpose="LOS_STATE",
    entity_id=ue_id,
)
```

**Гарантии `StreamAllocator` для этапа M2:**
- **Независимость от порядка регистрации:** индекс потока вычисляется через хеш `blake2b` от нормализованного ключа `(domain, purpose, entity_id)`. Если поменять порядок инициализации моделей или добавить новый случайный процесс (например, логирование) перед `MOBILITY`, последовательности всех существующих процессов останутся прежними.
- **Один ключ — один владелец:** повторный вызов `allocator.get_provider(...)` с тем же ключом выбрасывает `ValueError`, чтобы два потребителя не делили один поток и не сбрасывали его состояние.

### 4.2. Для моделей `Mobility, Channel, Traffic` (Андрей К.) — Явная инъекция зависимости
Модели принимают `rng: RandomProvider` (или набор генераторов под каждый процесс) через конструктор как явную обязательную зависимость. Немой `fallback` вида `rng if rng is not None else RandomGenerator()` запрещён, так как при пропущенном аргументе разные модели молча создадут генераторы с одинаковыми (`seed=12345, base_stream_idx=0, run_idx=1`) и начнут выдавать искусственно скоррелированные последовательности:

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

### 4.3. Правила миграции вызовов `np.random` на `RandomProvider` (Этап M2)
При переводе `CHANNEL_MODEL.py`, `TRAFFIC_MODEL.py`, `MOBILITY_MODEL.py` и `UE_MODULE.py` с `np.random` на `RandomProvider` необходимо учитывать различия в семантике параметров:

#### Таблица перевода вызовов `np.random` → `RandomProvider` (`RNG.py`)

| Вызов в существующем коде (`np.random`) | Семантика в `np.random` | Корректный вызов `RandomProvider` (`RNG.py`) | Важное правило перевода |
|---|---|---|---|
| `np.random.uniform(a, b)` | Полуинтервал `[a, b)` | `rng.uniform(a, b)` | Семантика границ совпадает: `[a, b)`. |
| `np.random.normal(loc=mu, scale=sigma)` | `scale` — это СКО ($\sigma$, stddev) | `rng.normal(mean=mu, variance=sigma ** 2)` | Метод `normal()` принимает **дисперсию** ($\sigma^2$), а не $\sigma$. Обязательно возводить `sigma` в квадрат. |
| `np.random.exponential(scale=beta)` | Математическое ожидание $\beta > 0$ | `rng.exponential(mean=beta)` | Семантика параметра совпадает (`mean = beta`). |
| `np.random.randint(low, high)` | Целое из `[low, high)` (верхняя граница exclusive) | `rng.get_integer(low, high - 1)` | В `get_integer(min_val, max_val)` **обе границы inclusive** (`[min_val, max_val]`). Верхнюю границу нужно уменьшать на `1`. |
| `rng.choice(seq)` (равновероятный выбор, например `h_E` в `CHANNEL_MODEL`) | Выбор случайного элемента без весов | `seq[rng.get_integer(0, len(seq) - 1)]` | Равновероятный выбор индекса через `get_integer(0, len(seq) - 1)`. |
| `np.random.choice(states, p=weights)` (взвешенный выбор, например MMPP в `TRAFFIC_MODEL.py:512`) | Выбор с вероятностями / весами `weights` | `weighted_choice(rng, states, weights)` | Равномерный индекс не подходит. Используется выбор через `rng.uniform(0.0, sum(weights))` со строгим сравнением `u < cumulative` и пропуском нулевых весов (см. пример ниже). |
| Векторные генерации `size=N` (в `CHANNEL_MODEL`, `TRAFFIC_MODEL`) | Массив из `N` случайных чисел | `np.array([rng.normal(...) for _ in range(N)])` | Последовательный вызов в генераторе списков сохраняет детерминированный порядок чтения из потока. |

#### Взвешенный выбор состояния для модели трафика (`TRAFFIC_MODEL.py:512`)
Для переходов MMPP по вектору вероятностей/весов `weights` используется накопление суммы весов поверх `rng.uniform(0.0, total)`:

```python
import math
from typing import Sequence, TypeVar
from RNG import RandomProvider

T = TypeVar("T")

def weighted_choice(rng: RandomProvider, items: Sequence[T], weights: Sequence[float]) -> T:
    if len(items) == 0 or len(items) != len(weights):
        raise ValueError("items and weights must be non-empty and of equal length.")
    total = 0.0
    for w in weights:
        if math.isnan(w) or math.isinf(w) or w < 0.0:
            raise ValueError(f"Invalid weight value: {w}")
        total += w
    if total <= 0.0:
        raise ValueError("Total weight must be > 0.")

    u = rng.uniform(0.0, total)  # строго 0.0 <= u < total
    cumulative = 0.0
    last_positive_idx = -1
    for i, w in enumerate(weights):
        if w == 0.0:
            continue
        last_positive_idx = i
        cumulative += w
        if u < cumulative:  # строгое '<' гарантирует, что вес 0.0 никогда не сработает
            return items[i]
    return items[last_positive_idx]
```

**Пример с нулевым весом (`weights = [0.0, 1.0]`) и проверка на этапе M2:**
- При `items = [0, 1]` и `weights = [0.0, 1.0]` элемент с индексом `0` пропускается (`w == 0.0`), для индекса `1` выполняется `cumulative = 1.0`, а так как `rng.uniform(0.0, 1.0)` всегда возвращает число строго меньше `1.0`, всегда выбирается состояние `1`.
- В тестах этапа M2 для `TRAFFIC_MODEL` проверяются векторы весов `[0.0, 1.0]` и `[1.0, 0.0]`: как на граничных значениях (`u = 0.0` и `u = math.nextafter(1.0, 0.0)`), так и на серии из 10 000 выборок подтверждается, что переход в состояние с нулевым весом не происходит ни разу.

---

## 5. Проверка (Definition of Done)

1. Добавление `RNG.py` не изменяет существующие файлы симулятора и не нарушает работу Legacy-кода.
2. Локальными модульными тестами подтверждены: воспроизводимость генерации `MRG32k3a` и `Philox4x64`, строгий полуинтервал `[0.0, 1.0)` у `PhiloxStream`, отсутствие переполнения и попадания в правую границу у `uniform()`, завершение `exponential()` и `normal()` при малых `bound`, точность и равномерность `get_integer()` (включая отрицательные границы и значения $> 2^{53}$), защита от `NaN`/`Inf` и бесконечного цикла при `bound=0.0`, соблюдение верхних границ `seed <= 4294944442`, `stream_idx <= 2^63 - 1`, `run_idx <= 2^51`, потокобезопасность кэша матриц, а также независимость `StreamAllocator` от порядка регистрации и запрет повторного захвата ключа.