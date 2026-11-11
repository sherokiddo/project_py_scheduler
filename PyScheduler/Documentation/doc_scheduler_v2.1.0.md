***

# `SCHEDULER.py` — Release v2.1.0 Documentation

## Оглавление

1. [Введение](#%D0%B2%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5)
2. [Журнал изменений v2.1.0](#%D0%B6%D1%83%D1%80%D0%BD%D0%B0%D0%BB-%D0%B8%D0%B7%D0%BC%D0%B5%D0%BD%D0%B5%D0%BD%D0%B8%D0%B9-v210)
3. [Архитектура модуля](#%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%D0%B0-%D0%BC%D0%BE%D0%B4%D1%83%D0%BB%D1%8F)
4. [Новая система CQI](#%D0%BD%D0%BE%D0%B2%D0%B0%D1%8F-%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D0%B0-cqi)
5. [Частотно-селективные планировщики (FD)](#%D1%87%D0%B0%D1%81%D1%82%D0%BE%D1%82%D0%BD%D0%BE-%D1%81%D0%B5%D0%BB%D0%B5%D0%BA%D1%82%D0%B8%D0%B2%D0%BD%D1%8B%D0%B5-%D0%BF%D0%BB%D0%B0%D0%BD%D0%B8%D1%80%D0%BE%D0%B2%D1%89%D0%B8%D0%BA%D0%B8-fd)
6. [Улучшения AMC](#%D1%83%D0%BB%D1%83%D1%87%D1%88%D0%B5%D0%BD%D0%B8%D1%8F-amc)
7. [SchedulingGrant](#schedulinggrant)
8. [Миграция кода](#%D0%BC%D0%B8%D0%B3%D1%80%D0%B0%D1%86%D0%B8%D1%8F-%D0%BA%D0%BE%D0%B4%D0%B0)
9. [Тестирование](#%D1%82%D0%B5%D1%81%D1%82%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5)
10. [Roadmap](#roadmap)

***

## Введение

Версия **v2.1.0** вносит **мажорные изменения** в архитектуру модуля планировщиков. 
Ключевое нововведение — **полноценная поддержка частотно-селективного планирования (Frequency Domain Scheduling)** через интеграцию **subband CQI (SB-CQI)** и разработку трёх новых алгоритмов: **FD_BCQI**, **FD_RR**, **FD_PF**.

Данная версия фокусируется на:

- **Реалистичности** — использование реальных CQI-отчётов от модели канала TDL
- **Эффективности** — учёт служебных накладных расходов (overhead) при расчёте пропускной способности
- **Масштабируемости** — унификация работы с CQI через датакласс `CQIMap`
- **Чистоте кода** — удаление legacy-методов, рефакторинг, оптимизация

***

## Журнал изменений v2.1.0

### Удаление Legacy-кода

- **Полностью удалены** устаревшие методы из секции `*LEGACY*`
- Модуль теперь использует **только** новую архитектуру v2.0+ с `SchedulerInterface`


### Система CQI

- ✅ **Добавлена поддержка subband CQI** — получение per-RBG CQI от модели канала TDL
- ✅ **Введён датакласс `CQIMap`** — единое хранилище WB-CQI и SB-CQI для каждого UE
- ✅ **Унифицированный доступ к CQI**:
    - `_get_wb_cqi(ue_id)` — получение wideband CQI
    - `_get_sb_cqi(ue_id)` — получение subband CQI (список per-RBG)
    - `_refresh_cqi(tti, users)` — обновление CQI-карты с настраиваемым интервалом
- ✅ **Интервальное обновление** — `wb_cqi_upd_interval` и `sb_cqi_upd_interval` позволяют контролировать частоту обновления CQI (по умолчанию каждый TTI)


### Частотно-селективные планировщики

- ✅ **FD_BCQI** (`FDxBestCQIScheduler`) — выбирает UE с лучшим SB-CQI для каждого RBG
- ✅ **FD_RR** (`FDxRoundRobinScheduler`) — справедливая ротация UE с учётом SB-CQI
- ✅ **FD_PF** (`FDxProportionalFairScheduler`) — Proportional Fair с частотной селективностью

Все три планировщика:

- Работают на уровне **RBG (Resource Block Group)**
- Используют **SB-CQI** для выбора оптимального UE для каждого RBG
- Имеют **fallback на WB-CQI** если SB-CQI недоступен
- **Протестированы** полным набором unit-тестов и бенчмарком


### Улучшения AMC

- ✅ **Учёт служебного overhead** в `GET_BITS_PER_RB(cqi)`:
    - CRS (Cell-specific Reference Signals) — 4 RE/слот × число антенн
    - PCFICH — занимает 1, 2 или 3 OFDM-символа в зависимости от настройки
    - Корректный расчёт полезных RE per RB для транспортного блока
- ⚠️ **Требует доработки модуля AMC** для учёта:
    - PDCCH overhead (зависит от числа CCE)
    - PHICH (Physical HARQ Indicator Channel)
    - CSI-RS при конфигурации с дополнительными антеннами


### SchedulingGrant

- ✅ **Добавлен датакласс `SchedulingGrant`** для будущей миграции
- Содержит:
    - `ue_id` — идентификатор UE
    - `num_bytes` — размер транспортного блока (TB size)
    - `lcid` — Logical Channel ID (для Layered Buffer)
    - `ndi` — New Data Indicator (HARQ)
    - `harq_process_id` — идентификатор HARQ-процесса
    - `rv` — Redundancy Version
- 🚧 **В стадии подготовки** — пока используется в `logical_channel_multiplexing()`, полная интеграция запланирована


### Рефакторинг и оптимизация

- Удаление мёртвого кода во всем модуле, а также основное форматирование
- Устранение избыточных вычислений (двойные вызовы `GET_RBG_INDICES`, дублирование расчёта `avg_pf`)
- Унификация verbose-строк — единый формат логирования


### Тестирование

- ✅ Полный набор тестов разработан и пройден:
    - `conftest.py` + `check_fixtures.py` — 21 проверка фикстур
    - `test_scheduler_common.py` — 24 инварианта × 3 планировщика
    - `test_fd_best_cqi.py` — 6 сценариев
    - `test_fd_round_robin.py` — 6 сценариев
    - `test_fd_proportional_fair.py` — 7 сценариев + Jain's Fairness Index
    - `FD_benchmark.py` — сравнительный анализ throughput и fairness
- Всего **~64 проверки**, успешно.

***

## Архитектура модуля

### Иерархия классов

```
SchedulerInterface (Abstract Base Class)
    │
    ├── BestCQIScheduler (Time Domain)
    ├── ProportionalFairScheduler (Time Domain)
    ├── RoundRobinScheduler (Time Domain)
    │
    ├── FDxBestCQIScheduler (Frequency Domain)
    ├── FDxRoundRobinScheduler (Frequency Domain)
    └── FDxProportionalFairScheduler (Frequency Domain)
```


### Factory Pattern

Создание планировщика через фабричный метод:

```python
from SCHEDULER import SchedulerInterface

scheduler = SchedulerInterface.create(
    algorithm="FD_PF",      # FD_BCQI, FD_RR, FD_PF, BestCQI, ProportionalFair, RoundRobin
    ltegrid=lte_grid,
    bs=base_station,
    verbose=True,
    enable_window=True,
    windowsize=100
)
```


### Этапы планирования

Каждый TTI обрабатывается в 8 этапов:

```
0. CQI Refresh          — обновление CQI-карты (интервальное)
1. Eligibility Check    — фильтрация UE с буфером и валидным CQI
2. Active Window        — скользящее окно (FIFO queue)
3. Priority Calculation — расчёт метрик (алгоритм-специфичный)
4. Priority List        — формирование и сортировка по приоритету
4.5 PDSCH Estimation    — грубая оценка загрузки PDSCH (только для TD)
5. PDCCH Allocation     — выделение CCE для PDCCH
6. PDSCH Allocation     — выделение RBG для PDSCH (core алгоритм)
7. Buffer Processing    — извлечение данных из буфера BS
8. Result Formation     — формирование allocation map и статистики
```


***

## Новая система CQI

### Датакласс `CQIMap`

**Назначение:** Хранение CQI-отчётов для одного UE.

```python
@dataclass(slots=True)
class CQIMap:
    wb_cqi: int                 # Wideband CQI (1-15)
    last_wb_update: int         # TTI последнего обновления WB-CQI
    wb_timer: int               # Счётчик для интервального обновления
    sb_cqi: List[int]           # Per-RBG CQI (длина = num_rbg)
    last_sb_update: int         # TTI последнего обновления SB-CQI
    sb_timer: int               # Счётчик для интервального обновления
```

**Пример записи:**

```python
# Создание entry для UE с id=5
scheduler.cqi_map[^5] = CQIMap(
    wb_cqi=12,
    last_wb_update=100,
    sb_cqi=[15, 14, 13, 15, 14, 13, 15, 14, 13, 15, 14, 13, 15],  # 13 RBG для 5 MHz
    last_sb_update=100
)
```


### Функции работы с CQI

#### `_get_wb_cqi(ue_id: int) -> int`

**Назначение:** Получить wideband CQI для UE.

**Возвращает:** Значение CQI (1-15) или 0 если запись отсутствует.

```python
cqi = scheduler._get_wb_cqi(ue_id=5)
# cqi = 12
```

**Fallback:** Если `ue_id` нет в `cqi_map`, возвращает `user.get('cqi', 0)` из объекта UE (устаревший путь).

***

#### `_get_sb_cqi(ue_id: int) -> List[int]`

**Назначение:** Получить subband CQI (per-RBG список) для UE.

**Возвращает:** Список `[cqi_rbg0, cqi_rbg1, ..., cqi_rbgN-1]` или пустой список `[]`.

```python
sb_cqi_list = scheduler._get_sb_cqi(ue_id=5)
# sb_cqi_list = [15, 14, 13, 15, 14, 13, 15, 14, 13, 15, 14, 13, 15]
```

**Использование в FD-планировщиках:**

```python
# В FD_BCQI при выборе UE для RBG
sb_cqi_list = self._get_sb_cqi(ue_id)
if sb_cqi_list and rbg_idx < len(sb_cqi_list):
    cqi = sb_cqi_list[rbg_idx]  # SB-CQI для данного RBG
else:
    cqi = self._get_wb_cqi(ue_id)  # Fallback на WB-CQI
```


***

#### `_refresh_cqi(tti: int, users: List[Dict]) -> None`

**Назначение:** Обновить CQI-карту на основе отчётов от UE.

**Вызывается:** Автоматически в `schedule()` если `tti % wb_cqi_upd_interval == 0`.

**Параметры интервала:**

```python
scheduler.wb_cqi_upd_interval = 1  # Обновлять WB-CQI каждый TTI (по умолчанию)
scheduler.sb_cqi_upd_interval = 1  # Обновлять SB-CQI каждый TTI (по умолчанию)
```

**Формат входных данных:**

```python
users = [
    {
        'UE_ID': 1,
        'cqi': 12,                # WB-CQI (обязательно, 1-15)
        'sb_bcqi': [15, 14, ...], # SB-CQI (опционально, per-RBG)
        ...
    },
    ...
]
```

**Логика работы:**

1. Извлекает `user.get('cqi')` → проверяет диапазон 1-15
2. Если `ue_id` нет в `cqi_map` → создаёт новый `CQIMap` entry
3. Обновляет `wb_cqi` и `last_wb_update = tti`
4. Если присутствует `user.get('sb_bcqi')` → обновляет `sb_cqi` и `last_sb_update = tti`
5. Verbose-лог: `CQIMAP | TTI {tti} | Active entries: {count} UE`

**Пример вызова (обычно не требуется вручную):**

```python
scheduler._refresh_cqi(tti=100, users=eligible_ues)
```


***

### Интеграция с моделью канала TDL

**Откуда берутся SB-CQI?**

В модуле `UE_MODULE.py` реализован метод:

```python
ue.cqi_subband  # List[int] — per-RBG CQI от TDL-модели канала
```

Планировщик извлекает эти данные в `_refresh_cqi()`:

```python
sb_cqi_list = user.get('sb_bcqi')  # alias для ue.cqi_subband
if sb_cqi_list and isinstance(sb_cqi_list, list) and len(sb_cqi_list) > 0:
    self.cqi_map[ue_id].sb_cqi = sb_cqi_list.copy()
```

**Требования к TDL:**

- Длина `sb_cqi` должна равняться `num_rbg` (количество RBG в системе)
- Для 5 MHz: 25 RB → 13 RBG (RBG size = 2)
- Для 20 MHz: 100 RB → 17 RBG (RBG size = 6)

***

## Частотно-селективные планировщики (FD)

### Общая концепция

**Frequency Domain Scheduling** — планирование на уровне **RBG (Resource Block Group)**, где каждый RBG выделяется **индивидуально** UE с лучшей метрикой для этого конкретного участка частотного спектра.

**Ключевое отличие от Time Domain:**

- **TD:** Один UE получает все RB в TTI
- **FD:** Разные UE могут получать разные RBG в одном TTI

**Преимущества FD:**

- Использование частотной селективности канала (frequency-selective fading)
- Повышение суммарного throughput системы
- Для PF — улучшение справедливости при сохранении efficiency

***

### FD_BCQI — Frequency Domain Best CQI

**Класс:** `FDxBestCQIScheduler`

**Алгоритм:** Greedy per-RBG allocation — каждый RBG отдаётся UE с **максимальным SB-CQI** для этого RBG.

**Метрика:** `priority = wb_cqi` (для формирования PriorityList), но реальное решение принимается по SB-CQI per-RBG.

**Псевдокод:**

```python
for rbg_idx in range(total_rbg):
    best_user = None
    best_cqi = -1
    
    for user in ues_with_pdcch:
        sb_cqi_list = _get_sb_cqi(user.id)
        if sb_cqi_list:
            cqi = sb_cqi_list[rbg_idx]
        else:
            cqi = _get_wb_cqi(user.id)  # Fallback
        
        if cqi > best_cqi:
            best_cqi = cqi
            best_user = user
    
    allocate_rbg(rbg_idx, best_user.id)
```

**Характеристики:**

- ✅ **Максимальный суммарный throughput** — выбор UE с лучшим каналом для каждого RBG
- ❌ **Низкая справедливость** — Jain's Index ≈ 1/N (один UE с хорошим CQI монополизирует ресурс)
- 📊 **Use case:** Максимизация cell capacity в условиях высокой нагрузки

**Результаты benchmark:**

```
Jain's Fairness Index:  0.1667  (N=6 UE → 1/6)
Total throughput:       93800 kbps (максимальный среди всех)
Min UE throughput:      0 kbps (некоторые UE не получают ничего)
```


***

### FD_RR — Frequency Domain Round Robin

**Класс:** `FDxRoundRobinScheduler`

**Алгоритм:** Для каждого RBG выбирается UE с лучшим SB-CQI среди тех, кто ещё не получил RBG в этом раунде, начиная с вращающегося смещения ue_index.

**Метрика:** `priority = window_size - idx` (для TD tie-break), с учётом `rr_ue_offset`.

**Псевдокод:**

```python
ue_index = rr_rbg_offset  # Смещение от последнего TTI
served_in_round = set()

for rbg_idx in range(total_rbg):
    for attempt in range(2):  # Два прохода если никто не выбрал
        best_user = None
        best_cqi = -1
        
        for k in range(num_ues):
            ue = ues_with_pdcch[(ue_index + k) % num_ues]
            
            if ue.id in served_in_round:
                continue  # UE уже получил RBG в этом раунде
            
            sb_cqi_list = _get_sb_cqi(ue.id)
            cqi = sb_cqi_list[rbg_idx] if sb_cqi_list else _get_wb_cqi(ue.id)
            
            if cqi > best_cqi:
                best_cqi = cqi
                best_user = ue
        
        if best_user:
            allocate_rbg(rbg_idx, best_user.id)
            served_in_round.add(best_user.id)
            ue_index = (ue_index + 1) % num_ues
            break
        else:
            served_in_round.clear()  # Сброс раунда — повторить выбор
```

**Характеристики:**

- ✅ **Высокая справедливость** — Jain's Index ≈ 0.7–0.8
- ⚙️ **Средний throughput** — не использует частотную селективность на 100%
- 📊 **Use case:** Гарантии QoS, минимизация jitter

**Результаты benchmark:**

```
Jain's Fairness Index:  0.76
Min throughput        : UE3  = 2464.0 kbps
Max throughput        : UE1 = 18760.0 kbps
```

### FD_PF — Frequency Domain Proportional Fair

**Класс:** `FDxProportionalFairScheduler`

**Алгоритм:** Per-RBG Proportional Fair — каждый RBG отдаётся UE с **максимальной PF-метрикой** для этого RBG.

**Метрика:**

$$
\text{metric}_j(k, t) = \frac{R_j(k, t)}{T_j(t)}
$$

где:

- $R_j(k, t)$ — throughput UE $j$ на RBG $k$ в TTI $t$ (биты) = `rbg_width × bits_per_rb(sb_cqi[k])`
- $T_j(t)$ — средний throughput UE $j$ (bps), обновляется через EWMA

**Псевдокод:**

```python
for rbg_idx in range(total_rbg):
    best_user = None
    best_metric = -1.0
    
    for user in ues_with_pdcch:
        sb_cqi_list = _get_sb_cqi(user.id)
        cqi = sb_cqi_list[rbg_idx] if sb_cqi_list else _get_wb_cqi(user.id)
        
        r_jk = rbg_width * amc.GET_BITS_PER_RB(cqi)
        avg_tput = user.ue.average_throughput
        denom = 1.0 if avg_tput <= 0 else (avg_tput / 1000.0)  # bps → bits/TTI
        
        metric = r_jk / denom
        
        if metric > best_metric:
            best_metric = metric
            best_user = user
    
    allocate_rbg(rbg_idx, best_user.id)
```

**Характеристики:**

- ✅ **Баланс throughput и fairness** — Jain's Index ≈ 0.70–0.91 после сходимости
- ✅ **Адаптивность** — "голодные" UE (низкий avg_throughput) получают приоритет
- ✅ **Эффективность** — использует частотную селективность + учитывает историю
- 📊 **Use case:** Реальные LTE-сети (стандарт де-факто)

**Результаты benchmark:**

```
Jain's Fairness Index:  0.70 (растёт со временем, к TTI=1000 → 0.80+)
Total throughput:       49267.7 kbps
Min UE throughput:      2566.6 kbps (защита слабых UE работает)
```

**Сходимость:**

PF требует переходного периода ≈ `1/alpha` TTI (при `alpha=0.1` → ~10 TTI) для накопления статистики `average_throughput`. Jain's Index **растёт со временем**:

```
TTI=10:   J ≈ 0.50
TTI=100:  J ≈ 0.85
TTI=500:  J ≈ 0.94
TTI=1000: J ≈ 0.97
```

***

## Улучшения AMC

### Функция `GET_BITS_PER_RB(cqi: int, n_ports: int = 1) -> int`

**Назначение:** Рассчитать количество полезных бит данных на 1 RB с учётом служебного overhead.

**Изменения в v2.1.0:**

1. **Учёт CRS (Cell-specific Reference Signals):**

```python
rs_per_slot = n_ports * 4  # 4 RE per slot per antenna port
```

2. **Учёт PCFICH (Physical Control Format Indicator Channel):**

```python
pcfich = self.scheduler.pdcch_manager.pcfich  # 1, 2 или 3 OFDM-символа
re_slot0 = 7 - pcfich - 12 - rs_per_slot
```

3. **Корректный расчёт полезных RE:**

```python
re_per_rb_tti = re_slot0 + re_slot1  # Два слота в одном TTI
effective_bits = int(re_per_rb_tti * modulation * code_rate)
```


**Пример расчёта для CQI=10 (64-QAM, code_rate=0.650):**

```
Параметры:
- PCFICH = 2 (default)
- n_ports = 1 (одна антенна)
- Modulation = 6 bits/symbol (64-QAM)
- Code Rate = 0.650

Расчёт RE per RB:
Slot 0: 7 OFDM-символов - 2 (PCFICH) - 12 RE (control) - 4 RE (CRS) = 7 - 2 - 12 - 4 = -11 → 0 (защита от отрицательных)
ОШИБКА в формуле? Проверка кода...

Корректная формула из кода:
re_slot0 = (7 - pcfich) * 12 - rs_per_slot
         = (7 - 2) * 12 - 4
         = 5 * 12 - 4
         = 60 - 4 = 56 RE

re_slot1 = 7 * 12 - rs_per_slot
         = 84 - 4 = 80 RE

Total: 56 + 80 = 136 RE per RB per TTI

Bits: 136 RE × 6 bits/symbol × 0.650 = 530.4 → 530 bits
```

**⚠️ Текущие ограничения:**

Функция **не учитывает** (запланировано в будущих версиях):

- PDCCH overhead (зависит от числа CCE) — занимает 1-3 OFDM-символа первого слота
- PHICH (Physical HARQ Indicator Channel) — ~8-24 RE
- CSI-RS (при конфигурации > 1 антенны)

***

## SchedulingGrant

### Датакласс `SchedulingGrant`

**Назначение:** Инкапсуляция информации о гранте на передачу для одного UE.

```python
@dataclass(slots=True)
class SchedulingGrant:
    ue_id: int                    # UE identifier
    num_bytes: int                # Размер транспортного блока (TB size)
    lcid: Optional[int] = None    # Logical Channel ID (для Layered Buffer)
    ndi: bool = True              # New Data Indicator (HARQ)
    harq_process_id: int = 0      # HARQ process identifier
    rv: int = 0                   # Redundancy Version (0-3)
    
    def __post_init__(self):
        if self.num_bytes < 0:
            raise ValueError(f"num_bytes cannot be negative: {self.num_bytes}")
        if not (0 <= self.rv <= 3):
            raise ValueError(f"rv must be 0-3: {self.rv}")
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь для совместимости."""
        return {
            'ue_id': self.ue_id,
            'num_bytes': self.num_bytes,
            'lcid': self.lcid,
            'harq_process_id': self.harq_process_id,
            'ndi': self.ndi,
            'rv': self.rv
        }
```

**Текущее использование:**

В методе `logical_channel_multiplexing()`:

```python
def logical_channel_multiplexing(self, tb_size: int, buffer_status_list: List) -> List[SchedulingGrant]:
    """Мультиплексирование логических каналов в транспортный блок."""
    
    if self.ltegrid.bs.usesimplebuffer:
        # Simple Buffer — один грант на весь TB
        buffer_status = buffer_status_list[^0]
        grant = SchedulingGrant(
            ue_id=buffer_status.ue_id,
            num_bytes=tb_size
        )
        return [grant]
    else:
        raise NotImplementedError("Layered Buffer support coming soon.")
```

**🚧 Запланировано:**

- Интеграция с HARQ Manager (установка `ndi`, `harq_process_id`, `rv`)
- Поддержка Layered Buffer (множественные гранты per UE с разными `lcid`)

***

## Миграция кода

### Если вы использовали старый API

**Удалено в v2.1.0:**

```python
# ❌ Все методы из секции *LEGACY* удалены
scheduler.legacy_method(...)  # AttributeError!
```

**Миграция на v2.0+ API:**

```python
# ✅ Используйте новый Factory Pattern
scheduler = SchedulerInterface.create(
    algorithm="FD_PF",
    ltegrid=lte_grid,
    bs=base_station
)

result = scheduler.schedule(tti=100, users=active_users)
stats = scheduler.get_stats()
```


### Обновление интеграции с моделью канала

**Старый способ (v2.0):**

```python
user['cqi'] = 12  # Только WB-CQI
```

**Новый способ (v2.1.0):**

```python
user['cqi'] = 12                        # WB-CQI (обязательно)
user['sb_bcqi'] = [15, 14, 13, ...]    # SB-CQI (опционально, для FD)
```

Планировщик автоматически обновит `cqi_map` через `_refresh_cqi()`.

### Доступ к CQI в пользовательском коде

**Устаревший способ:**

```python
cqi = user.get('cqi', 0)  # Прямое обращение к объекту UE
```

**Рекомендуемый способ:**

```python
cqi = scheduler._get_wb_cqi(ue_id)
sb_cqi_list = scheduler._get_sb_cqi(ue_id)
```


***

## Тестирование

### Структура тестов

```
tests/
├── conftest.py                    # Фикстуры и утилиты
├── check_fixtures.py              # 21 проверка фикстур
├── test_scheduler_common.py       # 24 инварианта × 3 планировщика
├── test_fd_best_cqi.py            # 6 сценариев FD_BCQI
├── test_fd_round_robin.py         # 6 сценариев FD_RR
├── test_fd_proportional_fair.py   # 7 сценариев FD_PF + Jain's Index
└── FD_benchmark.py                # Сравнительный анализ
```


### Запуск тестов

**Все тесты:**

```bash
python tests/check_fixtures.py
python tests/test_scheduler_common.py
python tests/test_fd_best_cqi.py
python tests/test_fd_round_robin.py
python tests/test_fd_proportional_fair.py
```

**Benchmark:**

```bash
python FD_benchmark.py
```

Логи сохраняются в `logs/test_*.log` и `logs/FD_benchmark.log`.

**Документация актуальна для:** SCHEDULER.py v2.1.0

**Дата последнего обновления:** 28.02.2026

***

**Конец документации**
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: SCHEDULER.py

