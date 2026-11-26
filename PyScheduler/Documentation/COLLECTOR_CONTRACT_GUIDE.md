# Универсальный контракт сбора статистики (Stats Collector Contract)
## Руководство для разработчиков модулей симулятора LTE

> **Цель документа:** Служить техническим регламентом для владельцев модулей (`CHANNEL_MODEL`, `MOBILITY_MODEL`, `BS_MODULE`, `TRAFFIC_MODEL`, `HARQ` и др.) по интеграции телеметрии в `StatsManager`.
> 
> **Ключевой принцип:** **Open-Closed Principle (OCP)**. Ядро `StatsManager` закрыто для модификаций — чтобы собирать метрики нового модуля, **не требуется менять ни строчки** в `STATS_MANAGER.py`. Каждый владелец модуля пишет контракт коллектора внутри своей зоны ответственности и регистрирует его в менеджере.

---

## 1. Архитектура и жизненный цикл

### 1.1 Как работает сбор метрик
В цикле симуляции на каждом заданном интервале TTI (по умолчанию каждые `collect_interval` TTI):
1. `SimulationManager` вызывает `stats_manager.collect(current_tti)`.
2. `StatsManager` обходит реестр зарегистрированных коллекторов (`IStatsCollector`).
3. Для каждого коллектора запрашивается его сконфигурированный уровень детализации (`MetricLevel.NONE`, `BASIC`, `ADVANCED`, `FULL`).
4. Если уровень активен:
   - Вызывается `collect_aggregated(tti, level)` → данные объединяются в плоский снимок TTI для экспорта в общий системный CSV/JSON.
   - Если уровень `FULL`, вызывается `collect_detailed(tti, level)` → данные по каждому UE объединяются в поабонентный снимок TTI для детального CSV/JSON.
   - Если требуется специфичный многомерный формат (например, QoS per-UE-per-QCI), вызывается `collect_custom_records(tti, level)`.
5. При экспорте заголовки колонок CSV строятся **динамически**: новые поля из вашего коллектора автоматически станут столбцами выходных отчетов.

```mermaid
flowchart TD
    SM[SimulationManager / Loop] -->|collect(tti)| StatsMgr[StatsManager Registry]
    StatsMgr -->|collect_aggregated / detailed| C1[ChannelCollector]
    StatsMgr -->|collect_aggregated / detailed| C2[MobilityCollector]
    StatsMgr -->|collect_aggregated / detailed| C3[Scheduler / AMC / PDCCH]
    StatsMgr -->|collect_aggregated / detailed| C4[TrafficCollector]
    StatsMgr -->|collect_aggregated / detailed| C5[YourNewModuleCollector]
    StatsMgr -->|Автоматический экспорт| Out1[system_stats.csv / json]
    StatsMgr -->|Автоматический экспорт| Out2[detailed_ue_stats.csv / json]
```

---

## 2. Спецификация контракта: `IStatsCollector` и `BaseStatsCollector`

Контракты определены в модуле `PyScheduler.STATS_MANAGER`.

### 2.1 Интерфейс `IStatsCollector` (Protocol)

```python
from typing import Protocol, runtime_checkable, Dict, Any, Optional
from STATS_MANAGER import MetricLevel

@runtime_checkable
class IStatsCollector(Protocol):
    """Универсальный контракт коллектора телеметрии модуля."""

    def get_collector_name(self) -> str:
        """
        Уникальный строковый идентификатор модуля в нижнем регистре.
        Например: 'channel', 'mobility', 'traffic', 'harq', 'buffer'.
        """
        ...

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        """
        Собрать системные агрегированные метрики модуля за текущий TTI.
        Возвращает плоский словарь вида {'metric_name': value}.
        Если level == MetricLevel.NONE, возвращается {}.
        """
        ...

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        """
        (Опционально) Собрать поабонентные (per-UE) метрики модуля на текущем TTI.
        Возвращает словарь вида:
        {
            ue_id: {'metric_name': value, ...},
            ...
        }
        Если поабонентных данных нет, возвращает None или {}.
        """
        ...

    def collect_custom_records(self, tti: int, level: MetricLevel) -> Optional[Any]:
        """
        (Опционально) Собрать специфичные многомерные структуры (например, матрицу per-UE-per-QCI).
        По умолчанию возвращает None.
        """
        ...
```

### 2.2 Базовый класс `BaseStatsCollector` (Рекомендуется к наследованию)

Чтобы не реализовывать пустые методы-заглушки вручную, наследуйте свой коллектор от `BaseStatsCollector`:

```python
from STATS_MANAGER import BaseStatsCollector, MetricLevel

class MyModuleCollector(BaseStatsCollector):
    def __init__(self, my_module_instance):
        super().__init__(name="my_module")
        self.module = my_module_instance

    def collect_aggregated(self, tti: int, level: MetricLevel) -> dict:
        if level == MetricLevel.NONE:
            return {}
        # Ваша логика сбора агрегированных метрик
        return {
            "mymod_indicator_avg": self.module.get_avg_val(),
        }

    # collect_detailed и collect_custom_records уже имеют дефолтные реализации (возвращают None)
```

---

## 3. Градация уровней детализации (`MetricLevel`)

Конфигурация поддерживает уровни:
`MetricLevel.NONE < MetricLevel.BASIC < MetricLevel.ADVANCED < MetricLevel.FULL`

| Уровень | Что должен отдавать модуль | Нагрузка / Overhead | Назначение |
| :--- | :--- | :--- | :--- |
| **`NONE`** | Пустой словарь `{}`. Сбор выключен. | 0% (минимальная) | Модуль отключен или его метрики не требуются в эксперименте. |
| **`BASIC`** | Базовые ключевые показатели (1–3 скалярных числа: среднее, счетчик событий, сумма). | < 1% CPU | Длительные симуляции, продуктивные прогоны (50 000+ TTI). |
| **`ADVANCED`** | Расширенная агрегация: min, max, avg, ст. отклонение, процентили, гистограммы распределений. | ~ 2–5% CPU | Оптимизация алгоритмов, тюнинг радиоинтерфейса. |
| **`FULL`** | Все метрики `ADVANCED` + поабонентная детализация через `collect_detailed()` для каждого UE. | Требует памяти на per-UE структуры | Глубокий дебаг, анализ справедливости и поведения конкретных терминалов. |

---

## 4. Соглашение об именовании метрик и типы данных

1. **Уникальный префикс метрик (ОБЯЗАТЕЛЬНО):**
   Каждый модуль должен использовать свой префикс для ключей метрик, чтобы исключить конфликты имен в общем CSV:
   - `ch_` — Channel Model (`ch_sinr_avg`, `ch_pathloss_avg_db`)
   - `mob_` — Mobility Model (`mob_speed_avg_kmh`, `mob_distance_avg_m`)
   - `traf_` — Traffic Generator (`traf_bytes_generated_tti`, `traf_packets_count`)
   - `buf_` / `qos_` — Buffers & QoS (`buf_occupancy_total_bytes`, `buf_dropped_packets`)
   - `harq_` — HARQ Module (`harq_retrans_count`, `harq_bler_pct`)
   - `sch_` — Scheduler core
   - `amc_` — Adaptive Modulation and Coding
   - `pdcch_` — PDCCH Link

2. **Формат значений:**
   - В `collect_aggregated()` допускаются только скалярные типы: `int`, `float`, `str`, `bool`.
   - Не вкладывайте `dict` или `list` внутрь значений агрегата (если только это не специфичный JSON-снимок).
   - Если значение недоступно (например, деление на 0), возвращайте `0` или `0.0`, либо опускайте ключ. Не возвращайте `NaN` или `Infinity`.

3. **Формат per-UE метрик (`collect_detailed`):**
   - Ключом верхнего уровня должен быть идентификатор терминала (`int` или `str`).
   - Значением — плоский словарь метрик данного терминала.
   - Пример:
     ```python
     {
         1: {"ch_sinr": 18.2, "ch_cqi": 11, "mob_speed": 3.6},
         2: {"ch_sinr": 12.0, "ch_cqi": 7,  "mob_speed": 60.0},
     }
     ```

---

## 5. Пошаговое руководство: как создать коллектор для своего модуля

### Шаг 1: Создайте файл коллектора в своем модуле
Рекомендуется размещать класс коллектора прямо в файле вашего модуля или в отдельном файле рядом, например:
`PyScheduler/CHANNEL_STATS_COLLECTOR.py` или внутри `CHANNEL_MODEL.py`.

### Шаг 2: Реализуйте сбор метрик
Используйте внутренние структуры своего модуля. Защищайте вычисления от деления на ноль.

```python
from STATS_MANAGER import BaseStatsCollector, MetricLevel
from typing import Dict, Any, Optional

class ChannelStatsCollector(BaseStatsCollector):
    """Коллектор метрик радиоканала."""

    def __init__(self, channel_model):
        super().__init__(name="channel")
        self.channel_model = channel_model

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        if level == MetricLevel.NONE:
            return {}

        ue_sinr_dict = self.channel_model.get_all_ue_sinr()  # {ue_id: sinr_val}
        if not ue_sinr_dict:
            return {"ch_sinr_avg": 0.0, "ch_active_links": 0}

        avg_sinr = sum(ue_sinr_dict.values()) / len(ue_sinr_dict)
        metrics = {
            "ch_sinr_avg": round(avg_sinr, 2),
            "ch_active_links": len(ue_sinr_dict),
        }

        if level >= MetricLevel.ADVANCED:
            metrics["ch_sinr_min"] = round(min(ue_sinr_dict.values()), 2)
            metrics["ch_sinr_max"] = round(max(ue_sinr_dict.values()), 2)

        return metrics

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        if level < MetricLevel.FULL:
            return None

        detailed = {}
        for ue_id, sinr in self.channel_model.get_all_ue_sinr().items():
            detailed[ue_id] = {
                "ch_sinr": round(sinr, 2),
                "ch_cqi": self.channel_model.get_ue_cqi(ue_id),
            }
        return detailed
```

### Шаг 3: Как коллектор попадает в систему (Зоны ответственности)

> [!IMPORTANT]
> **Владельцы модулей НЕ должны редактировать `SIMULATION_MANAGER.py`!**
> Если каждый разработчик пойдёт прописывать свой уровень в общий файл `SIMULATION_MANAGER.py`, возникнут постоянные конфликты слияния в Git (merge conflicts).

**Разделение ответственности:**
1. **Владелец модуля (разработчик):**
   - Реализует класс коллектора в своём файле.
   - Задает рекомендуемый уровень по умолчанию прямо в классе коллектора через атрибут `default_level` (например, `MetricLevel.BASIC`):
     ```python
     class ChannelStatsCollector(BaseStatsCollector):
         default_level = MetricLevel.BASIC
         ...
     ```
   - Добавляет в свой основной класс модуля метод для получения коллектора:
     ```python
     def get_stats_collector(self) -> BaseStatsCollector:
         return ChannelStatsCollector(self)
     ```
   - Проверяет свой коллектор изолированным юнит-тестом. **На этом его задача выполнена!**

2. **Оркестратор симуляции / Интегратор эксперимента:**
   - Уровень детализации метрик — это параметр **конкретного эксперимента / запуска**, а не зашитая в код константа.
   - В точке сборки эксперимента (например, в сценарии запуска или `TEST_MODULES.py`) коллекторы регистрируются в `StatsManager`:
     ```python
     # 1. Создаем менеджер статистики
     stats_mgr = StatsManager(config=stats_config)

     # 2. Регистрируем коллекторы (уровень берется из default_level коллектора
     #    или переопределяется для конкретного эксперимента):
     stats_mgr.register_collector(channel_model.get_stats_collector())
     stats_mgr.register_collector(mobility_model.get_stats_collector(), level=MetricLevel.ADVANCED)
     ```
   - Если уровень не передан явно в `register_collector`, `StatsManager` автоматически применит `default_level`, указанный автором модуля.

**Все!** Никаких изменений внутри `STATS_MANAGER.py` и ручных правок `SIMULATION_MANAGER.py` со стороны разработчиков модулей не требуется.

---

## 6. Готовые эталонные шаблоны для модулей проекта

### 6.1 Модуль радиоканала (`CHANNEL_MODEL`)
```python
from STATS_MANAGER import BaseStatsCollector, MetricLevel

class ChannelStatsCollector(BaseStatsCollector):
    def __init__(self, channel_model):
        super().__init__(name="channel")
        self.channel = channel_model

    def collect_aggregated(self, tti: int, level: MetricLevel) -> dict:
        if level == MetricLevel.NONE:
            return {}
        sinrs = self.channel.get_current_sinr_all()
        if not sinrs:
            return {"ch_sinr_avg": 0.0}
        
        avg_sinr = sum(sinrs) / len(sinrs)
        res = {"ch_sinr_avg": round(avg_sinr, 2)}
        if level >= MetricLevel.ADVANCED:
            res["ch_sinr_min"] = round(min(sinrs), 2)
            res["ch_sinr_max"] = round(max(sinrs), 2)
            res["ch_fading_std_db"] = round(self.channel.get_fading_std(), 2)
        return res

    def collect_detailed(self, tti: int, level: MetricLevel):
        if level < MetricLevel.FULL:
            return None
        return {
            ue_id: {
                "ch_sinr": self.channel.get_ue_sinr(ue_id),
                "ch_pathloss_db": self.channel.get_ue_pathloss(ue_id),
            }
            for ue_id in self.channel.get_active_ue_ids()
        }
```

### 6.2 Модуль мобильности (`MOBILITY_MODEL`)
```python
from STATS_MANAGER import BaseStatsCollector, MetricLevel

class MobilityStatsCollector(BaseStatsCollector):
    def __init__(self, mobility_manager):
        super().__init__(name="mobility")
        self.mobility = mobility_manager

    def collect_aggregated(self, tti: int, level: MetricLevel) -> dict:
        if level == MetricLevel.NONE:
            return {}
        speeds = [ue.speed_kmh for ue in self.mobility.ues.values()]
        distances = [ue.distance_to_bs_m for ue in self.mobility.ues.values()]
        if not speeds:
            return {"mob_speed_avg_kmh": 0.0, "mob_distance_avg_m": 0.0}

        res = {
            "mob_speed_avg_kmh": round(sum(speeds) / len(speeds), 2),
            "mob_distance_avg_m": round(sum(distances) / len(distances), 1),
        }
        if level >= MetricLevel.ADVANCED:
            res["mob_cell_edge_ue_count"] = sum(1 for d in distances if d > 800)
            res["mob_handover_events_tti"] = self.mobility.get_handover_count(tti)
        return res

    def collect_detailed(self, tti: int, level: MetricLevel):
        if level < MetricLevel.FULL:
            return None
        return {
            ue_id: {
                "mob_x": ue.x,
                "mob_y": ue.y,
                "mob_speed_kmh": ue.speed_kmh,
                "mob_distance_m": ue.distance_to_bs_m,
            }
            for ue_id, ue in self.mobility.ues.items()
        }
```

### 6.3 Модуль генератора трафика (`TRAFFIC_MODEL`)
```python
from STATS_MANAGER import BaseStatsCollector, MetricLevel

class TrafficStatsCollector(BaseStatsCollector):
    def __init__(self, traffic_gen):
        super().__init__(name="traffic")
        self.traffic = traffic_gen

    def collect_aggregated(self, tti: int, level: MetricLevel) -> dict:
        if level == MetricLevel.NONE:
            return {}
        stats = self.traffic.get_tti_stats(tti)
        res = {
            "traf_bytes_generated_tti": stats.get("bytes_gen", 0),
            "traf_packets_generated_tti": stats.get("packets_gen", 0),
        }
        if level >= MetricLevel.ADVANCED:
            res["traf_active_sessions_count"] = stats.get("active_sessions", 0)
            res["traf_offered_load_kbps"] = round(stats.get("bytes_gen", 0) * 8 / 1000.0, 2)
        return res

    def collect_detailed(self, tti: int, level: MetricLevel):
        if level < MetricLevel.FULL:
            return None
        return {
            ue_id: {
                "traf_ue_bytes_gen": bytes_count,
            }
            for ue_id, bytes_count in self.traffic.get_per_ue_generated(tti).items()
        }
```

### 6.4 Модуль буферов и очередей (`BS_MODULE` / Buffer Manager)
```python
from STATS_MANAGER import BaseStatsCollector, MetricLevel

class BufferStatsCollector(BaseStatsCollector):
    def __init__(self, buffer_manager):
        super().__init__(name="buffer")
        self.buffer_mgr = buffer_manager

    def collect_aggregated(self, tti: int, level: MetricLevel) -> dict:
        if level == MetricLevel.NONE:
            return {}
        summary = self.buffer_mgr.get_summary()
        res = {
            "buf_occupancy_total_bytes": summary.get("total_bytes", 0),
            "buf_dropped_packets_tti": summary.get("dropped_packets", 0),
        }
        if level >= MetricLevel.ADVANCED:
            res["buf_avg_packet_delay_ms"] = round(summary.get("avg_delay_ms", 0.0), 2)
            res["buf_max_hol_delay_ms"] = round(summary.get("max_hol_delay_ms", 0.0), 2)
        return res

    def collect_detailed(self, tti: int, level: MetricLevel):
        if level < MetricLevel.FULL:
            return None
        return {
            ue_id: {
                "buf_ue_queue_bytes": ue_buf.get_size_bytes(),
                "buf_ue_hol_delay_ms": ue_buf.get_hol_delay_ms(),
            }
            for ue_id, ue_buf in self.buffer_mgr.get_all_ue_buffers().items()
        }
```

---

## 7. Руководство по тестированию для владельца модуля

Каждый разработчик может протестировать свой коллектор **изолированно** в юнит-тесте с помощью `pytest` без запуска всей симуляции на 5000 TTI.

### Пример юнит-теста (`tests/test_my_collector.py`):

```python
import pytest
from STATS_MANAGER import StatsManager, StatisticsConfig, MetricLevel
from my_module import ChannelModel, ChannelStatsCollector

def test_channel_collector_contract(tmp_path):
    # 1. Инициализация вашего модуля и коллектора
    model = ChannelModel()
    collector = ChannelStatsCollector(model)

    # 2. Инициализация StatsManager в режиме сбора
    config = StatisticsConfig(
        collect_interval=1,
        file_prefix=str(tmp_path / "test_channel"),
    )
    manager = StatsManager(config=config)

    # 3. Регистрация
    manager.register_collector(collector, level=MetricLevel.FULL)
    assert manager.has_collector("channel") is True

    # 4. Вызов сбора на шаге 0
    manager.collect(0)

    # 5. Проверка попадания метрик в историю
    history = manager.get_history()
    assert len(history) == 1
    assert "ch_sinr_avg" in history[0]

    # 6. Проверка детального экспорта
    detailed_csv = str(tmp_path / "test_channel_detailed.csv")
    manager.export_detailed_csv(detailed_csv)
    with open(detailed_csv, "r", encoding="utf-8") as f:
        header = f.readline()
        assert "ch_sinr" in header
```

Запуск проверки:
```bash
pytest tests/test_my_collector.py
```
