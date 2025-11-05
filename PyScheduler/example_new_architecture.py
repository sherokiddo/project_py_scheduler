"""
#------------------------------------------------------------------------------
# Модуль: example_new_architecture - Пример использования новой архитектуры
#------------------------------------------------------------------------------
# Описание:
#   Демонстрирует использование новой архитектуры симулятора с системой событий,
#   модульными интерфейсами и центральным координатором. Показывает как легко
#   создавать и настраивать симуляции с различными модулями.
#
# Версия: 1.0.0
# Дата создания: 2025-01-27
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
import time
import json
from typing import Dict, Any

from SimulationCoordinator import SimulationCoordinator, SimulationConfig
from ModuleAdapters import ModuleAdapterFactory
from EventManager import EventType, get_event_manager

def create_simulation_example():
    """
    Создание и запуск примера симуляции с новой архитектурой.
    Демонстрирует основные возможности системы.
    """
    print("=== Пример симуляции с новой архитектурой ===\n")

    # 1. Создание конфигурации симуляции
    config = SimulationConfig(
        duration_ms=5000,  # 5 секунд симуляции
        tti_duration_ms=1,  # 1 мс на TTI
        update_interval_ms=5,  # Обновление модулей каждые 5 мс
        enable_logging=True,
        log_level="INFO",
        save_metrics=True,
        metrics_file="example_simulation_metrics.json",
        modules_config={
            "ChannelModel_UMi": {
                "o2i_model": "low"
            },
            "MobilityModel_RandomWalk": {
                "x_min": 0,
                "x_max": 1000,
                "y_min": 0,
                "y_max": 1000
            },
            "TrafficModel_Poisson": {
                "packet_rate": 10.0,
                "min_packet_size": 150,
                "max_packet_size": 1500
            }
        }
    )

    # 2. Создание координатора симуляции
    coordinator = SimulationCoordinator(config)

    # 3. Создание и регистрация модулей
    print("Создание модулей...")

    # Базовая станция
    bs_adapter = ModuleAdapterFactory.create_bs_adapter(
        x=500, y=500, height=25.0, bandwidth=10.0
    )
    coordinator.register_module("BaseStation", bs_adapter)

    # Ресурсная сетка
    resource_grid_adapter = ModuleAdapterFactory.create_resource_grid_adapter(
        bandwidth=10.0, num_frames=1
    )
    coordinator.register_module("ResourceGrid", resource_grid_adapter)

    # Модель канала
    channel_adapter = ModuleAdapterFactory.create_channel_model_adapter(
        "UMi", bs_adapter.bs, config.modules_config["ChannelModel_UMi"]
    )
    coordinator.register_module("ChannelModel", channel_adapter)

    # Модель мобильности
    mobility_adapter = ModuleAdapterFactory.create_mobility_model_adapter(
        "RandomWalk", config.modules_config["MobilityModel_RandomWalk"]
    )
    coordinator.register_module("MobilityModel", mobility_adapter)

    # Модель трафика
    traffic_adapter = ModuleAdapterFactory.create_traffic_model_adapter(
        "Poisson", config.modules_config["TrafficModel_Poisson"]
    )
    coordinator.register_module("TrafficModel", traffic_adapter)

    # Планировщик
    scheduler_adapter = ModuleAdapterFactory.create_scheduler_adapter(
        "RoundRobin", resource_grid_adapter.resource_grid, bs_adapter.bs
    )
    coordinator.register_module("Scheduler", scheduler_adapter)

    # Пользовательские устройства
    ue_adapters = []
    for i in range(3):
        ue_adapter = ModuleAdapterFactory.create_ue_adapter(
            ue_id=i+1, x=100+i*200, y=100+i*200, ue_class="pedestrian"
        )

        # Настройка моделей для UE
        ue_adapter.set_channel_model(channel_adapter)
        ue_adapter.set_mobility_model(mobility_adapter)
        ue_adapter.set_traffic_model(traffic_adapter)

        # Регистрация в БС
        bs_adapter.register_user(ue_adapter)

        ue_adapters.append(ue_adapter)
        coordinator.register_module(f"UE_{i+1}", ue_adapter)

    print(f"Создано {len(ue_adapters)} пользовательских устройств")

    # 4. Настройка обработчиков событий для демонстрации
    event_manager = get_event_manager()

    def handle_ue_position_update(event):
        data = event.data
        print(f"UE {data['ue_id']} переместился в позицию {data['position']}")

    def handle_scheduling_completed(event):
        data = event.data
        print(f"Планирование TTI {data['tti']} завершено за {data['schedule_time']:.3f} мс")

    def handle_resource_allocated(event):
        data = event.data
        print(f"Выделен ресурс UE {data['ue_id']} на частоте {data['frequency_idx']}")

    # Подписка на события
    event_manager.subscribe(EventType.UE_POSITION_UPDATED, handle_ue_position_update)
    event_manager.subscribe(EventType.BS_SCHEDULING_COMPLETED, handle_scheduling_completed)
    event_manager.subscribe(EventType.SCHEDULER_RESOURCE_ALLOCATED, handle_resource_allocated)

    # 5. Инициализация и запуск симуляции
    print("\nИнициализация симуляции...")
    if not coordinator.initialize_simulation():
        print("Ошибка инициализации симуляции!")
        return

    print("Запуск симуляции...")
    if not coordinator.start_simulation():
        print("Ошибка запуска симуляции!")
        return

    # 6. Ожидание завершения симуляции
    print("Симуляция запущена. Ожидание завершения...")
    coordinator.wait_for_completion()

    # 7. Получение результатов
    print("\n=== Результаты симуляции ===")
    status = coordinator.get_simulation_status()

    print(f"Статус: {status['state']}")
    print(f"Обработано TTI: {status['metrics']['processed_tti']}")
    print(f"Опубликовано событий: {status['event_manager_stats']['events_published']}")
    print(f"Ошибок модулей: {status['metrics']['module_errors']}")

    # Статистика по модулям
    print("\nСтатистика модулей:")
    for module_name, module_info in status['modules'].items():
        print(f"  {module_name}: {module_info['status']}")

    # 8. Остановка симуляции
    print("\nОстановка симуляции...")
    coordinator.stop_simulation()

    print("Симуляция завершена!")
    print(f"Метрики сохранены в {config.metrics_file}")

def demonstrate_event_system():
    """
    Демонстрация работы системы событий.
    """
    print("\n=== Демонстрация системы событий ===")

    from EventManager import EventManager, EventType

    # Создание отдельного менеджера событий для демонстрации
    event_manager = EventManager(enable_async=False)  # Синхронная обработка для демонстрации

    # Счетчики событий
    event_counts = {}

    def create_event_handler(event_name):
        def handler(event):
            event_counts[event_name] = event_counts.get(event_name, 0) + 1
            print(f"Обработано событие {event_name}: {event.data}")
        return handler

    # Подписка на различные события
    event_manager.subscribe(EventType.SIMULATION_START, create_event_handler("SIMULATION_START"))
    event_manager.subscribe(EventType.TTI_START, create_event_handler("TTI_START"))
    event_manager.subscribe(EventType.UE_REGISTERED, create_event_handler("UE_REGISTERED"))
    event_manager.subscribe(EventType.ERROR_OCCURRED, create_event_handler("ERROR_OCCURRED"))

    # Публикация событий
    print("Публикация событий...")

    event_manager.publish(EventType.SIMULATION_START, "Demo", {"message": "Начало демонстрации"})
    event_manager.publish(EventType.UE_REGISTERED, "Demo", {"ue_id": 1, "position": (100, 100)})
    event_manager.publish(EventType.TTI_START, "Demo", {"tti": 0})
    event_manager.publish(EventType.TTI_START, "Demo", {"tti": 1})
    event_manager.publish(EventType.ERROR_OCCURRED, "Demo", {"error": "Тестовая ошибка"})

    # Статистика
    print(f"\nСтатистика событий: {event_counts}")
    print(f"Общая статистика: {event_manager.get_statistics()}")

def demonstrate_module_adapters():
    """
    Демонстрация работы адаптеров модулей.
    """
    print("\n=== Демонстрация адаптеров модулей ===")

    # Создание адаптеров
    print("Создание адаптеров...")

    # Базовая станция
    bs_adapter = ModuleAdapterFactory.create_bs_adapter(500, 500, 25.0, 10.0)
    print(f"Создана БС: {bs_adapter.get_info()}")

    # Модель канала
    channel_adapter = ModuleAdapterFactory.create_channel_model_adapter("UMi", bs_adapter.bs)
    print(f"Создана модель канала: {channel_adapter.get_info()}")

    # Модель мобильности
    mobility_adapter = ModuleAdapterFactory.create_mobility_model_adapter("RandomWalk")
    print(f"Создана модель мобильности: {mobility_adapter.get_info()}")

    # Модель трафика
    traffic_adapter = ModuleAdapterFactory.create_traffic_model_adapter("Poisson")
    print(f"Создана модель трафика: {traffic_adapter.get_info()}")

    # Инициализация и запуск модулей
    print("\nИнициализация модулей...")

    modules = [bs_adapter, channel_adapter, mobility_adapter, traffic_adapter]

    for module in modules:
        if module.initialize({}):
            print(f"✓ {module.config.name} инициализирован")
        else:
            print(f"✗ Ошибка инициализации {module.config.name}")

    # Запуск модулей
    print("\nЗапуск модулей...")
    for module in modules:
        if module.start():
            print(f"✓ {module.config.name} запущен")
        else:
            print(f"✗ Ошибка запуска {module.config.name}")

    # Демонстрация работы
    print("\nДемонстрация работы модулей...")

    # Тест модели канала
    sinr = channel_adapter.calculate_sinr(100, 120, 1.5, "pedestrian")
    print(f"SINR на расстоянии 100м: {sinr:.2f} дБ")

    # Тест модели мобильности
    new_pos, new_vel, new_dir = mobility_adapter.update_position(
        (200, 300), 2.0, 1.57, 1000
    )
    print(f"Новая позиция: {new_pos}, скорость: {new_vel:.2f} м/с")

    # Тест модели трафика
    packets = traffic_adapter.generate_traffic(1, 0, 1000)
    print(f"Сгенерировано пакетов: {len(packets)}")

    # Остановка модулей
    print("\nОстановка модулей...")
    for module in modules:
        if module.stop():
            print(f"✓ {module.config.name} остановлен")
        else:
            print(f"✗ Ошибка остановки {module.config.name}")

def main():
    """
    Главная функция для запуска всех демонстраций.
    """
    print("Демонстрация новой архитектуры симулятора LTE-сети")
    print("=" * 60)

    try:
        # Демонстрация системы событий
        demonstrate_event_system()

        # Демонстрация адаптеров модулей
        demonstrate_module_adapters()

        # Полная симуляция
        create_simulation_example()

        print("\n" + "=" * 60)
        print("Все демонстрации завершены успешно!")

    except Exception as e:
        print(f"\nОшибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

