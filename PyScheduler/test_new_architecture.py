"""
#------------------------------------------------------------------------------
# Модуль: test_new_architecture - Тесты для новой архитектуры симулятора
#------------------------------------------------------------------------------
# Описание:
#   Комплексные тесты для проверки работоспособности новой архитектуры
#   симулятора с системой событий, модульными интерфейсами и координатором.
#   Включает unit-тесты, интеграционные тесты и тесты производительности.
#
# Версия: 1.0.0
# Дата создания: 2025-01-27
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
import unittest
import time
import threading
from typing import Dict, Any
from unittest.mock import Mock, patch

from EventManager import EventManager, EventType, Event
from SimulationCoordinator import SimulationCoordinator, SimulationConfig
from ModuleAdapters import ModuleAdapterFactory
from BaseModule import BaseModule, ModuleConfig
from ModuleInterfaces import ModuleStatus

class TestEventManager(unittest.TestCase):
    """Тесты для системы событий"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.event_manager = EventManager(enable_async=False)
        self.received_events = []
    
    def test_event_publication_and_subscription(self):
        """Тест публикации и подписки на события"""
        def event_handler(event):
            self.received_events.append(event)
        
        # Подписка на событие
        self.assertTrue(
            self.event_manager.subscribe(EventType.SIMULATION_START, event_handler)
        )
        
        # Публикация события
        self.assertTrue(
            self.event_manager.publish(EventType.SIMULATION_START, "Test", {"test": "data"})
        )
        
        # Проверка получения события
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].event_type, EventType.SIMULATION_START)
        self.assertEqual(self.received_events[0].data["test"], "data")
    
    def test_event_priority(self):
        """Тест приоритетов событий"""
        high_priority_events = []
        low_priority_events = []
        
        def high_priority_handler(event):
            high_priority_events.append(event)
        
        def low_priority_handler(event):
            low_priority_events.append(event)
        
        # Подписка с разными приоритетами
        self.event_manager.subscribe(EventType.TTI_START, high_priority_handler, priority=1)
        self.event_manager.subscribe(EventType.TTI_START, low_priority_handler, priority=0)
        
        # Публикация события
        self.event_manager.publish(EventType.TTI_START, "Test", {"tti": 1})
        
        # Проверка обработки (высокий приоритет должен обработаться первым)
        self.assertEqual(len(high_priority_events), 1)
        self.assertEqual(len(low_priority_events), 1)
    
    def test_event_statistics(self):
        """Тест статистики событий"""
        def dummy_handler(event):
            pass
        
        # Подписка и публикация
        self.event_manager.subscribe(EventType.SIMULATION_START, dummy_handler)
        self.event_manager.publish(EventType.SIMULATION_START, "Test", {})
        self.event_manager.publish(EventType.SIMULATION_END, "Test", {})
        
        # Проверка статистики
        stats = self.event_manager.get_statistics()
        self.assertEqual(stats['events_published'], 2)
        self.assertEqual(stats['events_processed'], 2)
        self.assertEqual(stats['subscribers_notified'], 1)  # Только один подписчик

class TestBaseModule(unittest.TestCase):
    """Тесты для базового модуля"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.config = ModuleConfig(name="TestModule", version="1.0.0")
        self.module = TestModule(self.config)
    
    def test_module_initialization(self):
        """Тест инициализации модуля"""
        self.assertTrue(self.module.initialize({}))
        self.assertEqual(self.module.get_status(), ModuleStatus.INITIALIZED)
    
    def test_module_lifecycle(self):
        """Тест жизненного цикла модуля"""
        # Инициализация
        self.assertTrue(self.module.initialize({}))
        self.assertEqual(self.module.get_status(), ModuleStatus.INITIALIZED)
        
        # Запуск
        self.assertTrue(self.module.start())
        self.assertEqual(self.module.get_status(), ModuleStatus.RUNNING)
        
        # Приостановка
        self.assertTrue(self.module.pause())
        self.assertEqual(self.module.get_status(), ModuleStatus.PAUSED)
        
        # Возобновление
        self.assertTrue(self.module.resume())
        self.assertEqual(self.module.get_status(), ModuleStatus.RUNNING)
        
        # Остановка
        self.assertTrue(self.module.stop())
        self.assertEqual(self.module.get_status(), ModuleStatus.STOPPED)
    
    def test_module_info(self):
        """Тест получения информации о модуле"""
        self.module.initialize({})
        info = self.module.get_info()
        
        self.assertEqual(info.name, "TestModule")
        self.assertEqual(info.version, "1.0.0")
        self.assertEqual(info.status, ModuleStatus.INITIALIZED)
    
    def test_module_configuration(self):
        """Тест конфигурации модуля"""
        # Получение конфигурации
        config = self.module.get_configuration()
        self.assertEqual(config["name"], "TestModule")
        
        # Валидация конфигурации
        is_valid, errors = self.module.validate_configuration({"enabled": True})
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Установка конфигурации
        self.assertTrue(self.module.set_configuration({"enabled": False}))
        self.assertFalse(self.module.config.enabled)

class TestModuleAdapters(unittest.TestCase):
    """Тесты для адаптеров модулей"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.bs_adapter = ModuleAdapterFactory.create_bs_adapter(500, 500, 25.0, 10.0)
        self.channel_adapter = ModuleAdapterFactory.create_channel_model_adapter("UMi", self.bs_adapter.bs)
        self.mobility_adapter = ModuleAdapterFactory.create_mobility_model_adapter("RandomWalk")
        self.traffic_adapter = ModuleAdapterFactory.create_traffic_model_adapter("Poisson")
    
    def test_channel_model_adapter(self):
        """Тест адаптера модели канала"""
        self.channel_adapter.initialize({})
        self.channel_adapter.start()
        
        # Тест расчета SINR
        sinr = self.channel_adapter.calculate_sinr(100, 120, 1.5, "pedestrian")
        self.assertIsInstance(sinr, float)
        self.assertLess(sinr, 50)  # Разумное значение SINR
        
        # Тест расчета затухания
        path_loss = self.channel_adapter.calculate_path_loss(100, 120, 1.5, "pedestrian")
        self.assertIsInstance(path_loss, float)
        self.assertGreater(path_loss, 0)
    
    def test_mobility_model_adapter(self):
        """Тест адаптера модели мобильности"""
        self.mobility_adapter.initialize({})
        self.mobility_adapter.start()
        
        # Тест обновления позиции
        new_pos, new_vel, new_dir = self.mobility_adapter.update_position(
            (200, 300), 2.0, 1.57, 1000
        )
        
        self.assertIsInstance(new_pos, tuple)
        self.assertEqual(len(new_pos), 2)
        self.assertIsInstance(new_vel, float)
        self.assertIsInstance(new_dir, float)
        
        # Тест ограничений скорости
        vel_limits = self.mobility_adapter.get_velocity_limits()
        self.assertIsInstance(vel_limits, tuple)
        self.assertEqual(len(vel_limits), 2)
        self.assertLess(vel_limits[0], vel_limits[1])
    
    def test_traffic_model_adapter(self):
        """Тест адаптера модели трафика"""
        self.traffic_adapter.initialize({})
        self.traffic_adapter.start()
        
        # Тест генерации трафика
        packets = self.traffic_adapter.generate_traffic(1, 0, 1000)
        self.assertIsInstance(packets, list)
        
        # Тест статистики
        stats = self.traffic_adapter.get_traffic_statistics(1)
        self.assertIn('total_packets', stats)
        self.assertIn('total_bytes', stats)
    
    def test_ue_adapter(self):
        """Тест адаптера пользовательского оборудования"""
        ue_adapter = ModuleAdapterFactory.create_ue_adapter(1, 100, 200, "pedestrian")
        
        # Настройка моделей
        ue_adapter.set_channel_model(self.channel_adapter)
        ue_adapter.set_mobility_model(self.mobility_adapter)
        ue_adapter.set_traffic_model(self.traffic_adapter)
        
        # Инициализация и запуск
        self.assertTrue(ue_adapter.initialize({}))
        self.assertTrue(ue_adapter.start())
        
        # Тест обновления позиции
        ue_adapter.update_position(1000, (500, 500), 25.0)
        
        # Тест получения качества канала
        channel_quality = ue_adapter.get_channel_quality()
        self.assertIn('cqi', channel_quality)
        self.assertIn('sinr', channel_quality)

class TestSimulationCoordinator(unittest.TestCase):
    """Тесты для координатора симуляции"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.config = SimulationConfig(
            duration_ms=1000,
            tti_duration_ms=1,
            update_interval_ms=10,
            enable_logging=False
        )
        self.coordinator = SimulationCoordinator(self.config)
    
    def test_module_registration(self):
        """Тест регистрации модулей"""
        # Создание тестового модуля
        test_module = TestModule(ModuleConfig(name="TestModule"))
        
        # Регистрация модуля
        self.assertTrue(self.coordinator.register_module("TestModule", test_module))
        
        # Проверка регистрации
        self.assertIsNotNone(self.coordinator.get_module("TestModule"))
        
        # Отмена регистрации
        self.assertTrue(self.coordinator.unregister_module("TestModule"))
        self.assertIsNone(self.coordinator.get_module("TestModule"))
    
    def test_simulation_lifecycle(self):
        """Тест жизненного цикла симуляции"""
        # Создание и регистрация модулей
        bs_adapter = ModuleAdapterFactory.create_bs_adapter(500, 500, 25.0, 10.0)
        self.coordinator.register_module("BaseStation", bs_adapter)
        
        # Инициализация
        self.assertTrue(self.coordinator.initialize_simulation())
        
        # Запуск
        self.assertTrue(self.coordinator.start_simulation())
        
        # Ожидание завершения
        self.assertTrue(self.coordinator.wait_for_completion(timeout=2.0))
        
        # Остановка
        self.assertTrue(self.coordinator.stop_simulation())
    
    def test_simulation_status(self):
        """Тест получения статуса симуляции"""
        # Регистрация модуля
        test_module = TestModule(ModuleConfig(name="TestModule"))
        self.coordinator.register_module("TestModule", test_module)
        
        # Получение статуса
        status = self.coordinator.get_simulation_status()
        
        self.assertIn('state', status)
        self.assertIn('modules', status)
        self.assertIn('TestModule', status['modules'])

class TestPerformance(unittest.TestCase):
    """Тесты производительности"""
    
    def test_event_processing_performance(self):
        """Тест производительности обработки событий"""
        event_manager = EventManager(enable_async=False)
        
        def dummy_handler(event):
            pass
        
        # Подписка на событие
        event_manager.subscribe(EventType.TTI_START, dummy_handler)
        
        # Измерение времени обработки
        start_time = time.time()
        
        for i in range(1000):
            event_manager.publish(EventType.TTI_START, "Test", {"tti": i})
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Проверка что обработка достаточно быстрая
        self.assertLess(processing_time, 1.0)  # Менее 1 секунды для 1000 событий
        
        print(f"Обработано 1000 событий за {processing_time:.3f} секунд")
    
    def test_module_update_performance(self):
        """Тест производительности обновления модулей"""
        config = ModuleConfig(name="PerfTestModule")
        module = TestModule(config)
        module.initialize({})
        module.start()
        
        # Измерение времени обновления
        start_time = time.time()
        
        for i in range(1000):
            module.update(1.0)
        
        end_time = time.time()
        update_time = end_time - start_time
        
        # Проверка что обновления достаточно быстрые
        self.assertLess(update_time, 1.0)  # Менее 1 секунды для 1000 обновлений
        
        print(f"Выполнено 1000 обновлений модуля за {update_time:.3f} секунд")

class TestModule(BaseModule):
    """Тестовый модуль для unit-тестов"""
    
    def _initialize_impl(self) -> bool:
        return True
    
    def _start_impl(self) -> bool:
        return True
    
    def _stop_impl(self) -> bool:
        return True
    
    def _update_impl(self, time_delta: float) -> bool:
        return True

def run_performance_tests():
    """Запуск тестов производительности"""
    print("Запуск тестов производительности...")
    
    # Тест системы событий
    event_manager = EventManager(enable_async=True)
    
    def dummy_handler(event):
        pass
    
    event_manager.subscribe(EventType.TTI_START, dummy_handler)
    
    # Тест публикации событий
    start_time = time.time()
    for i in range(10000):
        event_manager.publish(EventType.TTI_START, "PerfTest", {"tti": i})
    
    # Ждем обработки всех событий
    while event_manager.get_statistics()['events_pending'] > 0:
        time.sleep(0.001)
    
    end_time = time.time()
    print(f"Обработано 10000 событий за {end_time - start_time:.3f} секунд")
    
    # Статистика
    stats = event_manager.get_statistics()
    print(f"Статистика: {stats}")
    
    event_manager.stop_async_processing()

def run_integration_test():
    """Запуск интеграционного теста"""
    print("\nЗапуск интеграционного теста...")
    
    # Создание простой симуляции
    config = SimulationConfig(
        duration_ms=2000,
        tti_duration_ms=1,
        update_interval_ms=10,
        enable_logging=False
    )
    
    coordinator = SimulationCoordinator(config)
    
    # Создание модулей
    bs_adapter = ModuleAdapterFactory.create_bs_adapter(500, 500, 25.0, 10.0)
    resource_grid_adapter = ModuleAdapterFactory.create_resource_grid_adapter(10.0, 1)
    channel_adapter = ModuleAdapterFactory.create_channel_model_adapter("UMi", bs_adapter.bs)
    mobility_adapter = ModuleAdapterFactory.create_mobility_model_adapter("RandomWalk")
    traffic_adapter = ModuleAdapterFactory.create_traffic_model_adapter("Poisson")
    scheduler_adapter = ModuleAdapterFactory.create_scheduler_adapter("RoundRobin", resource_grid_adapter.resource_grid, bs_adapter.bs)
    
    # Регистрация модулей
    coordinator.register_module("BaseStation", bs_adapter)
    coordinator.register_module("ResourceGrid", resource_grid_adapter)
    coordinator.register_module("ChannelModel", channel_adapter)
    coordinator.register_module("MobilityModel", mobility_adapter)
    coordinator.register_module("TrafficModel", traffic_adapter)
    coordinator.register_module("Scheduler", scheduler_adapter)
    
    # Создание пользователей
    for i in range(2):
        ue_adapter = ModuleAdapterFactory.create_ue_adapter(i+1, 100+i*200, 100+i*200)
        ue_adapter.set_channel_model(channel_adapter)
        ue_adapter.set_mobility_model(mobility_adapter)
        ue_adapter.set_traffic_model(traffic_adapter)
        bs_adapter.register_user(ue_adapter)
        coordinator.register_module(f"UE_{i+1}", ue_adapter)
    
    # Запуск симуляции
    print("Инициализация...")
    self.assertTrue(coordinator.initialize_simulation())
    
    print("Запуск...")
    self.assertTrue(coordinator.start_simulation())
    
    print("Ожидание завершения...")
    self.assertTrue(coordinator.wait_for_completion(timeout=5.0))
    
    print("Остановка...")
    self.assertTrue(coordinator.stop_simulation())
    
    # Проверка результатов
    status = coordinator.get_simulation_status()
    print(f"Статус: {status['state']}")
    print(f"Обработано TTI: {status['metrics']['processed_tti']}")
    print(f"Событий: {status['event_manager_stats']['events_published']}")

if __name__ == '__main__':
    # Запуск unit-тестов
    print("Запуск unit-тестов...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Запуск тестов производительности
    run_performance_tests()
    
    print("\nВсе тесты завершены!")

