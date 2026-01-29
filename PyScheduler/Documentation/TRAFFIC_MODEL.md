# TRAFFIC_MODEL - Модели генерации сетевого трафика

**Версия:** 1.1.0

**Автор:** Норицин Иван, Дворников Андрей

**Последнее изменение:** 2026-01-23

**Python Kernel:** 3.12.9

---

## 1. Описание модуля
Модуль **TRAFFIC_MODEL** предоставляет инструменты для симуляции сетевого трафика в LTE/5G сетях. Он поддерживает как простую генерацию пакетов, так и продвинутую архитектуру QoS (Quality of Service) с поддержкой нескольких потоков (bearers) для одного пользователя.

### Основные возможности
1.  **Статистические модели:** Poisson, ON/OFF, MMPP.
2.  **Архитектура Multi-bearer:** Один UE может иметь несколько потоков трафика (например, VoIP + Web + Video) с разными приоритетами и QCI.
3.  **QoS Control:** Контроль битрейта (throttling), приоритезация на основе весов и QCI.
4.  **Advanced Statistics:** Сбор метрик по UE, по QCI и по каждому bearer.

---

## 2. Архитектура и паттерны
В модуле используются следующие паттерны проектирования:
*   **Strategy:** Каждая модель трафика (`Poisson`, `OnOff`, `MMPP`) реализует единый интерфейс `ITrafficModel`.
*   **Factory:** `TrafficModelFactory` централизует создание моделей.
*   **Composite:** `UeTrafficProfile` управляет множеством объектов `Bearer` для одного UE.
*   **Facade:** `PacketManager` предоставляет единый интерфейс для управления сложной логикой генерации, контроля скорости и статистики.
*   **Observer:** Система callback-ов (`packet_handler`) для передачи сгенерированных пакетов в буферы базовой станции.

---

## 3. Компоненты данных

### 3.1. TrafficType (Enum)
Классификация типов трафика и их параметров качества обслуживания (QoS).

| Тип (Enum) | QCI Value | Delay Budget | Описание |
| :--- | :--- | :--- | :--- |
| `VOIP` | 100 | 100 ms | Голосовые звонки |
| `CONV_VIDEO` | 150 | 150 ms | Видеозвонки (Conversational Video) |
| `REAL_TIME_GAMING` | 50 | 50 ms | Онлайн игры (Real-time Gaming) |
| `NON_CONV_VIDEO` | 300 | 300 ms | Потоковое видео (Non-Conversational) |
| `IMS` | 100 | - | IMS сервисы (Signaling) |
| `VIDEO_TCP` | 300 | - | Видео через TCP (Buffered Streaming) |
| `VOICE_VIDEO_GAMING`| 100 | - | Смешанный трафик реального времени |
| `WEB_SERVICES` | 300 | 300 ms | Веб-сервисы (WWW, Email) |
| `DEFAULT` | 300 | - | Стандартный трафик (Best Effort) |


### 3.2. Packet (Dataclass)
Представляет пакет данных. Содержит метаданные для маршрутизации и QoS.

**Поля:**
*   `size` (int): Размер в байтах.
*   `ue_id` (int): ID пользователя.
*   `creation_time` (float): Время создания (мс).
*   `qci` (int): Идентификатор класса QoS.
*   `priority` (int): Приоритет обработки (0 = наивысший).
*   `ttl_ms` (int): Время жизни пакета.
*   `deadline` (float): Абсолютное время дедлайна (вычисляется автоматически на основе Delay Budget).
*   `is_fragment` (bool): Флаг, указывающий, является ли пакет фрагментом.
*   `bearer_id` (Optional[int]): ID логического канала (bearer), к которому относится пакет.

---

## 4. Модели генерации трафика (Phase 1)

Все модели реализуют интерфейс `ITrafficModel` и обязаны иметь методы `generate_traffic` и `get_model_info`.

### 4.1. PoissonModel
Генерирует пакеты с экспоненциально распределенными интервалами.
*   **Тип:** Stateless (без состояния).
*   **Параметры:** `packet_rate` (пак/сек), `min_packet_size`, `max_packet_size`.

### 4.2. OnOffModel
Чередует периоды активности (ON) и тишины (OFF).
*   **Тип:** Stateful (хранит состояние для каждого UE).
*   **Параметры:** `duration_on`, `duration_off` (сек), `packet_rate` (в фазе ON).
*   **Управление памятью:** Требует вызова `clear_state(ue_id)` при удалении пользователя.

### 4.3. MMPPModel (Markov Modulated Poisson Process)
Система с несколькими состояниями интенсивности, переключающимися по марковской цепи.
*   **Тип:** Stateful.
*   **Параметры:** `packet_rates` (список интенсивностей), `transition_matrix` (матрица вероятностей переходов).

---

## 5. QoS Architecture

Внедрена для поддержки сценариев, где один пользователь потребляет несколько сервисов одновременно через разные логические каналы.

### 5.1. Bearer
Сущность, описывающая один поток трафика.
*   **Атрибуты:**
    *   `bearer_id`: Уникальный ID в рамках UE.
    *   `model`: Экземпляр `ITrafficModel` (источник данных).
    *   `qci`: Класс обслуживания для данного потока.
    *   `max_bitrate`: Ограничение скорости (Mbps) для данного потока.
    *   `weight`: Вес для взвешенной приоритизации.

### 5.2. UeTrafficProfile
Контейнер (Composite), хранящий все `Bearer` конкретного UE.
*   Позволяет добавлять/удалять потоки динамически.
*   Метод `generate_all_traffic` собирает пакеты со всех активных bearers и проставляет им корректные `bearer_id`.

### 5.3. BitrateController
Механизм ограничения пропускной способности (Traffic Shaping).
*   Использует скользящее окно (Sliding Window) для подсчета текущей скорости.
*   Если лимит превышен, новые пакеты отбрасываются (Drop).
*   **Методы:** `set_limit(ue_id, mbps)`, `check_and_throttle(packets)`.

---

## 6. Генераторы трафика и API

### 6.1. PacketManager (Рекомендуемый)
Основной фасад для работы с новой архитектурой.
*   **Функции:**
    *   Управление профилями пользователей (`UeTrafficProfile`).
    *   Интеграция с `BitrateController` для шейпинга трафика.
    *   Сбор детальной статистики через `TrafficStatistics`.
    *   Автоматическая отправка пакетов в буфер БС через callback (`packet_handler`).
*   **Ключевые методы:**
    *   `add_bearer(...)`: Добавить новый поток трафика пользователю.
    *   `set_bitrate_limit(...)`: Установить жесткий лимит скорости для UE.
    *   `generate_packets(...)`: Запуск генерации (вызывает throttling и callback).

### 6.2. SimpleGenerator (Legacy)
Упрощенная версия для обратной совместимости.
*   Поддерживает только одну модель на UE.
*   Не имеет контроля битрейта.
*   Используется для простых тестов, где QoS не требуется.

### 6.3. TrafficModelFactory
Фабрика для создания экземпляров моделей.
*   Метод `create_model(type, **kwargs)` валидирует параметры и возвращает нужный объект модели.

---

## 7. Примеры использования

### Настройка Multi-bearer трафика (через PacketManager)
```python
# Инициализация менеджера с callback-ом для отправки в БС
traffic_gen = PacketManager(packet_handler=bs.add_packets_callback)

# 1. Добавляем VoIP поток (QCI 100)
traffic_gen.add_bearer(
    ue_id=1,
    model_type="Poisson",
    qci=100,
    traffic_type=TrafficType.VOIP,
    packet_rate=50,
    max_bitrate=0.064  # 64 Kbps
)

# 2. Добавляем Video поток (QCI 150) тому же пользователю
traffic_gen.add_bearer(
    ue_id=1,
    model_type="OnOff",
    qci=150,
    traffic_type=TrafficType.CONV_VIDEO,
    duration_on=2,
    duration_off=1,
    packet_rate=200
)

# 3. Устанавливаем общий лимит скорости для пользователя
traffic_gen.set_bitrate_limit(ue_id=1, max_bitrate_mbps=5.0)

Сбор статистики

# Получить общую статистику по конкретному UE
stats = traffic_gen.get_statistics(ue_id=1)
print(f"Packets: {stats['packets']}, Bitrate: {stats['bitrate_mbps']} Mbps")

# Получить разбивку по QCI
qos_stats = traffic_gen.get_qos_statistics(ue_id=1)
