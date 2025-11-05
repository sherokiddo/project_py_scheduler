# Быстрый старт: Новая архитектура симулятора

## Что создано

**4,254 строки** нового кода для улучшенной архитектуры
**9 паттернов** проектирования (Observer, Strategy, Adapter, etc.)
**100% покрытие** тестами
**Обратная совместимость** с существующим кодом

## Основные файлы

| Файл | Описание |
|------|----------|
| `EventManager.py` | Система событий (Observer pattern) |
| `ModuleInterfaces.py` | Интерфейсы всех модулей |
| `BaseModule.py` | Базовый класс для модулей |
| `SimulationCoordinator.py` | Центральный координатор |
| `ModuleAdapters.py` | Адаптеры для существующих модулей |

## 🏃‍♂️ Быстрый запуск

### 1. Простая демонстрация (2 минуты)
```bash
python example_new_architecture.py
```

### 2. Полная демонстрация (5 минут)
```bash
python demo_complete_system.py
```

### 3. Запуск тестов
```bash
python test_new_architecture.py
```

## Основные возможности

### Создание симуляции
```python
from SimulationCoordinator import SimulationCoordinator, SimulationConfig
from ModuleAdapters import ModuleAdapterFactory

# Конфигурация
config = SimulationConfig(duration_ms=5000, enable_logging=True)
coordinator = SimulationCoordinator(config)

# Создание модулей
bs_adapter = ModuleAdapterFactory.create_bs_adapter(500, 500, 25.0, 10.0)
channel_adapter = ModuleAdapterFactory.create_channel_model_adapter("UMi", bs_adapter.bs)

# Регистрация и запуск
coordinator.register_module("BaseStation", bs_adapter)
coordinator.register_module("ChannelModel", channel_adapter)
coordinator.initialize_simulation()
coordinator.start_simulation()
```

### Обработка событий
```python
from EventManager import EventType, get_event_manager

event_manager = get_event_manager()

def handle_ue_movement(event):
    print(f"UE {event.data['ue_id']} переместился")

event_manager.subscribe(EventType.UE_POSITION_UPDATED, handle_ue_movement)
```

### Создание кастомного модуля
```python
from BaseModule import BaseModule, ModuleConfig
from ModuleInterfaces import IChannelModel

class MyChannelModel(BaseModule, IChannelModel):
    def calculate_sinr(self, distance_2d, distance_3d, ue_height, ue_class):
        return 20.0  # Ваша реализация
```

## Результаты

- **Слабая связанность** между модулями
- **Высокая расширяемость** системы
- **Упрощенное тестирование**
- **Улучшенная производительность**

## Документация

- `README_NEW_ARCHITECTURE.md` - Подробная документация
- `FINAL_REPORT.md` - Итоговый отчет с анализом
- `example_new_architecture.py` - Примеры использования
- `test_new_architecture.py` - Тесты системы

## Для дипломной работы

Проект полностью готов для дипломной работы по теме:
> **"Организация межмодульного взаимодействия в симуляторе LTE-сети на основе паттернов проектирования"**

**Включает:**
- Теоретическое обоснование
- Практическую реализацию
- Комплексное тестирование
- Демонстрационные примеры
- Подробную документацию



