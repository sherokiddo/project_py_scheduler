## 1. **Observer** (EventManager.py)
**Что делает:** Модули подписываются на события и получают уведомления при их возникновении.
**Как реализовано:** 
- `EventManager` - центральный диспетчер событий
- Модули подписываются через `subscribe()`
- События публикуются через `publish()`
- Асинхронная обработка с приоритетами

## 2. **Strategy** (ModuleInterfaces.py)
**Что делает:** Разные алгоритмы можно менять во время выполнения.
**Как реализовано:**
- `IChannelModel` - разные модели каналов (UMi, UMa, RMa)
- `IMobilityModel` - разные модели мобильности
- `IScheduler` - разные алгоритмы планирования
- `ITrafficModel` - разные модели трафика

## 3. **Adapter** (ModuleAdapters.py)
**Что делает:** Интегрирует старые модули в новую архитектуру без их изменения.
**Как реализовано:**
- `BaseStationAdapter` - адаптирует старый BS_MODULE
- `UserEquipmentAdapter` - адаптирует старый UE_MODULE
- `ChannelModelAdapter` - адаптирует старые модели каналов
- Все адаптеры реализуют новые интерфейсы

## 4. **Facade** (SimulationCoordinator.py)
**Что делает:** Упрощает сложную систему, предоставляя простой интерфейс.
**Как реализовано:**
- `SimulationCoordinator` - единая точка входа
- Скрывает сложность управления модулями
- Простые методы: `start_simulation()`, `stop_simulation()`

## 5. **Factory** (ModuleAdapters.py)
**Что делает:** Создает объекты без указания их конкретных классов.
**Как реализовано:**
- `ModuleAdapterFactory` - создает адаптеры
- `create_channel_model_adapter()` - создает адаптеры каналов
- `create_mobility_model_adapter()` - создает адаптеры мобильности

## 6. **Template Method** (BaseModule.py)
**Что делает:** Определяет скелет алгоритма, позволяя подклассам переопределять отдельные шаги.
**Как реализовано:**
- `BaseModule` - базовый класс с шаблоном жизненного цикла
- Методы: `_initialize_impl()`, `_start_impl()`, `_update_impl()`, `_stop_impl()`
- Подклассы переопределяют только нужные методы

## 7. **Singleton** (EventManager.py)
**Что делает:** Гарантирует, что класс имеет только один экземпляр.
**Как реализовано:**
- `EventManager` - единственный экземпляр в системе
- Все модули используют один и тот же менеджер событий
- Централизованное управление событиями

## 8. **Command** (EventManager.py)
**Что делает:** Инкапсулирует запрос как объект.
**Как реализовано:**
- События - это команды
- `Event` класс содержит данные и обработчик
- Можно ставить в очередь, отменять, логировать

## 9. **Decorator** (BaseModule.py)
**Что делает:** Динамически добавляет новую функциональность объектам.
**Как реализовано:**
- Логирование в `BaseModule` - декорирует основную функциональность
- Статистика и метрики - дополнительная функциональность
- Валидация конфигурации - еще один декоратор

Каждый паттерн решает конкретную проблему архитектуры и делает систему более гибкой и расширяемой.








# Что происходит при получении события:

### 1. **Публикация события** (`publish()`)
- Создается объект `Event` с типом, временной меткой, источником и данными
- Событие добавляется в очередь `_event_queue`
- Обновляется статистика (`events_published`)

### 2. **Обработка события** (`_process_event()`)
- **Поиск подписчиков:** Система ищет все модули, подписанные на этот тип события
- **Вызов обработчиков:** Для каждого подписчика вызывается его callback-функция
- **Передача данных:** В обработчик передается объект `Event` с данными
- **Обновление статистики:** Счетчики `subscribers_notified` и `events_processed`

### 3. **Асинхронная обработка**
- События обрабатываются в отдельном потоке
- **Приоритизация:** События сортируются по приоритету (критические → высокие → обычные)
- **Защита от переполнения:** Если очередь полная, удаляются старые события с низким приоритетом

### 4. **Пример работы:**
```python
# Модуль подписывается на событие
event_manager.subscribe(EventType.UE_POSITION_UPDATED, handle_position_change)

# Другой модуль публикует событие
event_manager.publish(EventType.UE_POSITION_UPDATED, "MobilityModel", {
    "ue_id": 1, 
    "position": (100, 200)
})

# Автоматически вызывается handle_position_change(event)
```

**Результат:** Все модули, подписанные на событие, получают уведомление и могут отреагировать на изменение состояния системы.








# Новая архитектура симулятора LTE-сети

## Обзор

Данная реализация представляет собой улучшенную архитектуру симулятора LTE-сети, основанную на принципах SOLID и паттернах проектирования. Основная цель - обеспечить слабую связанность между модулями, упростить тестирование и расширение функциональности.

## Ключевые компоненты

### 1. Система событий (EventManager)

**Файл:** `EventManager.py`

Реализует паттерн Observer для межмодульного взаимодействия.

```python
from EventManager import EventManager, EventType, get_event_manager

# Получение глобального менеджера событий
event_manager = get_event_manager()

# Подписка на событие
def handle_ue_movement(event):
    print(f"UE {event.data['ue_id']} переместился")

event_manager.subscribe(EventType.UE_POSITION_UPDATED, handle_ue_movement)

# Публикация события
event_manager.publish(EventType.UE_POSITION_UPDATED, "MobilityModel", {
    "ue_id": 1,
    "position": (100, 200)
})
```

**Возможности:**
- Асинхронная и синхронная обработка событий
- Приоритеты событий
- Статистика работы
- Защита от переполнения очереди

### 2. Модульные интерфейсы (ModuleInterfaces)

**Файл:** `ModuleInterfaces.py`

Определяет единые интерфейсы для всех модулей системы.

```python
from ModuleInterfaces import IChannelModel, IMobilityModel, ITrafficModel

class MyChannelModel(IChannelModel):
    def calculate_sinr(self, distance_2d, distance_3d, ue_height, ue_class):
        # Реализация расчета SINR
        pass
```

**Интерфейсы:**
- `IModule` - базовый интерфейс для всех модулей
- `IChannelModel` - модели радиоканалов
- `IMobilityModel` - модели мобильности
- `ITrafficModel` - модели генерации трафика
- `IScheduler` - планировщики ресурсов
- `IResourceGrid` - управление ресурсной сеткой
- `IUserEquipment` - пользовательское оборудование
- `IBaseStation` - базовая станция

### 3. Базовый модуль (BaseModule)

**Файл:** `BaseModule.py`

Предоставляет базовую реализацию для всех модулей с поддержкой событий и конфигурации.

```python
from BaseModule import BaseModule, ModuleConfig

class MyModule(BaseModule):
    def _initialize_impl(self):
        # Инициализация модуля
        return True

    def _start_impl(self):
        # Запуск модуля
        return True

    def _update_impl(self, time_delta):
        # Обновление модуля
        return True
```

### 4. Центральный координатор (SimulationCoordinator)

**Файл:** `SimulationCoordinator.py`

Управляет жизненным циклом симуляции и координирует работу всех модулей.

```python
from SimulationCoordinator import SimulationCoordinator, SimulationConfig

# Создание конфигурации
config = SimulationConfig(
    duration_ms=10000,
    tti_duration_ms=1,
    update_interval_ms=5,
    enable_logging=True
)

# Создание координатора
coordinator = SimulationCoordinator(config)

# Регистрация модулей
coordinator.register_module("MyModule", my_module)

# Запуск симуляции
coordinator.initialize_simulation()
coordinator.start_simulation()
coordinator.wait_for_completion()
coordinator.stop_simulation()
```

### 5. Адаптеры модулей (ModuleAdapters)

**Файл:** `ModuleAdapters.py`

Обеспечивает интеграцию существующих модулей с новой архитектурой.

```python
from ModuleAdapters import ModuleAdapterFactory

# Создание адаптеров
bs_adapter = ModuleAdapterFactory.create_bs_adapter(500, 500, 25.0, 10.0)
channel_adapter = ModuleAdapterFactory.create_channel_model_adapter("UMi", bs_adapter.bs)
mobility_adapter = ModuleAdapterFactory.create_mobility_model_adapter("RandomWalk")
traffic_adapter = ModuleAdapterFactory.create_traffic_model_adapter("Poisson")
scheduler_adapter = ModuleAdapterFactory.create_scheduler_adapter("RoundRobin", grid, bs)
```

## Быстрый старт

### 1. Простая симуляция

```python
from SimulationCoordinator import SimulationCoordinator, SimulationConfig
from ModuleAdapters import ModuleAdapterFactory

# Конфигурация
config = SimulationConfig(duration_ms=5000, enable_logging=True)
coordinator = SimulationCoordinator(config)

# Создание модулей
bs_adapter = ModuleAdapterFactory.create_bs_adapter(500, 500, 25.0, 10.0)
grid_adapter = ModuleAdapterFactory.create_resource_grid_adapter(10.0, 1)
channel_adapter = ModuleAdapterFactory.create_channel_model_adapter("UMi", bs_adapter.bs)
scheduler_adapter = ModuleAdapterFactory.create_scheduler_adapter("RoundRobin", grid_adapter.resource_grid, bs_adapter.bs)

# Регистрация модулей
coordinator.register_module("BaseStation", bs_adapter)
coordinator.register_module("ResourceGrid", grid_adapter)
coordinator.register_module("ChannelModel", channel_adapter)
coordinator.register_module("Scheduler", scheduler_adapter)

# Запуск симуляции
coordinator.initialize_simulation()
coordinator.start_simulation()
coordinator.wait_for_completion()
coordinator.stop_simulation()
```

### 2. Создание собственного модуля

```python
from BaseModule import BaseModule, ModuleConfig
from ModuleInterfaces import IChannelModel

class CustomChannelModel(BaseModule, IChannelModel):
    def __init__(self):
        config = ModuleConfig(name="CustomChannel", version="1.0.0")
        super().__init__(config)

    def _initialize_impl(self):
        # Инициализация
        return True

    def _start_impl(self):
        # Запуск
        return True

    def _stop_impl(self):
        # Остановка
        return True

    def _update_impl(self, time_delta):
        # Обновление
        return True

    def calculate_sinr(self, distance_2d, distance_3d, ue_height, ue_class):
        # Кастомная реализация расчета SINR
        return 20.0  # Пример

    def calculate_path_loss(self, distance_2d, distance_3d, ue_height, ue_class):
        # Кастомная реализация расчета затухания
        return 100.0  # Пример

    def calculate_los_probability(self, distance_2d, ue_height):
        # Кастомная реализация расчета вероятности LOS
        return 0.8  # Пример
```

### 3. Обработка событий

```python
from EventManager import EventType, get_event_manager

event_manager = get_event_manager()

def handle_scheduling_completed(event):
    print(f"Планирование TTI {event.data['tti']} завершено")

def handle_resource_allocated(event):
    print(f"Выделен ресурс UE {event.data['ue_id']}")

# Подписка на события
event_manager.subscribe(EventType.BS_SCHEDULING_COMPLETED, handle_scheduling_completed)
event_manager.subscribe(EventType.SCHEDULER_RESOURCE_ALLOCATED, handle_resource_allocated)
```

## Архитектурные преимущества

### 1. Слабая связанность
- Модули взаимодействуют только через события
- Отсутствие прямых зависимостей между модулями
- Легкая замена и модификация модулей

### 2. Расширяемость
- Простое добавление новых модулей
- Поддержка различных реализаций интерфейсов
- Гибкая конфигурация системы

### 3. Тестируемость
- Изоляция модулей для unit-тестирования
- Возможность мокирования зависимостей
- Автоматизированное тестирование

### 4. Производительность
- Асинхронная обработка событий
- Оптимизированная система координации
- Минимальные накладные расходы

## Структура файлов

```
PyScheduler/
├── EventManager.py              # Система событий
├── ModuleInterfaces.py          # Интерфейсы модулей
├── BaseModule.py               # Базовый класс модуля
├── SimulationCoordinator.py    # Центральный координатор
├── ModuleAdapters.py           # Адаптеры для существующих модулей
├── example_new_architecture.py # Примеры использования
├── test_new_architecture.py    # Тесты
└── README_NEW_ARCHITECTURE.md  # Данная документация
```

## Тестирование

Запуск тестов:

```bash
python test_new_architecture.py
```

Запуск примера:

```bash
python example_new_architecture.py
```

## Метрики и мониторинг

Система автоматически собирает метрики:

- Количество обработанных событий
- Время выполнения операций
- Статистика модулей
- Ошибки и предупреждения

```python
# Получение статистики
status = coordinator.get_simulation_status()
print(f"Обработано TTI: {status['metrics']['processed_tti']}")
print(f"Событий: {status['event_manager_stats']['events_published']}")
```

## Конфигурация

Подробная настройка через `SimulationConfig`:

```python
config = SimulationConfig(
    duration_ms=10000,           # Длительность симуляции (мс)
    tti_duration_ms=1,           # Длительность TTI (мс)
    update_interval_ms=5,        # Интервал обновления модулей (мс)
    enable_logging=True,         # Включить логирование
    log_level="INFO",            # Уровень логирования
    save_metrics=True,           # Сохранять метрики
    metrics_file="metrics.json", # Файл метрик
    modules_config={             # Конфигурация модулей
        "ChannelModel_UMi": {
            "o2i_model": "low"
        },
        "MobilityModel_RandomWalk": {
            "x_min": 0, "x_max": 1000,
            "y_min": 0, "y_max": 1000
        }
    }
)
```

## Заключение

Новая архитектура обеспечивает:

- **Гибкость** - легко добавлять новые модули и функции
- **Надежность** - изолированные модули с четкими интерфейсами
- **Производительность** - оптимизированная система координации
- **Тестируемость** - полное покрытие unit-тестами
- **Расширяемость** - поддержка различных сценариев симуляции

Эта архитектура идеально подходит для дипломной работы по теме "Организация межмодульного взаимодействия в симуляторе LTE-сети на основе паттернов проектирования".

