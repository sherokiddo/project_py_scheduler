"""
#------------------------------------------------------------------------------
# Модуль: SimulationCoordinator - Центральный координатор симуляции
#------------------------------------------------------------------------------
# Описание:
#   Координирует работу всех модулей симулятора LTE-сети.
#   Управляет жизненным циклом симуляции, обработкой событий и взаимодействием
#   между модулями. Реализует паттерн Facade для упрощения работы с системой.
#
# Версия: 1.0.0
# Дата создания: 2025-10-11
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import json
import logging
from collections import defaultdict

from EventManager import EventManager, EventType, Event, get_event_manager
from ModuleInterfaces import IModule, ModuleStatus, ModuleInfo

class SimulationState(Enum):
    """Состояния симуляции"""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class SimulationConfig:
    """Конфигурация симуляции"""
    duration_ms: int = 10000
    tti_duration_ms: int = 1
    update_interval_ms: int = 5
    enable_logging: bool = True
    log_level: str = "INFO"
    save_metrics: bool = True
    metrics_file: str = "simulation_metrics.json"
    modules_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class SimulationMetrics:
    """Метрики симуляции"""
    start_time: float = 0.0
    end_time: float = 0.0
    total_tti: int = 0
    processed_tti: int = 0
    events_published: int = 0
    events_processed: int = 0
    module_errors: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class SimulationCoordinator:
    """
    Центральный координатор симуляции LTE-сети.
    Управляет всеми модулями и координирует их взаимодействие.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Инициализация координатора симуляции.

        Args:
            config: Конфигурация симуляции
        """
        self.config = config or SimulationConfig()
        self.state = SimulationState.STOPPED
        self.metrics = SimulationMetrics()

        # Менеджер событий
        self.event_manager = get_event_manager()

        # Регистрируем обработчики событий
        self._setup_event_handlers()

        # Модули системы
        self.modules: Dict[str, IModule] = {}
        self.module_dependencies: Dict[str, List[str]] = {}
        self.module_startup_order: List[str] = []

        # Логирование
        self._setup_logging()

        # Поток симуляции
        self._simulation_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Статистика производительности
        self._performance_stats = defaultdict(list)

    def _setup_logging(self) -> None:
        """Настройка системы логирования"""
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('simulation.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger('SimulationCoordinator')
        else:
            self.logger = logging.getLogger('SimulationCoordinator')
            self.logger.disabled = True

    def _setup_event_handlers(self) -> None:
        """Настройка обработчиков событий"""
        self.event_manager.subscribe(EventType.ERROR_OCCURRED, self._handle_error, priority=2)
        self.event_manager.subscribe(EventType.WARNING_ISSUED, self._handle_warning, priority=1)
        self.event_manager.subscribe(EventType.METRICS_UPDATED, self._handle_metrics_update, priority=0)

    def register_module(self, name: str, module: IModule,
                       dependencies: List[str] = None) -> bool:
        """
        Регистрация модуля в системе.

        Args:
            name: Имя модуля
            module: Объект модуля
            dependencies: Список зависимостей модуля

        Returns:
            bool: True если регистрация успешна
        """
        try:
            if name in self.modules:
                self.logger.warning(f"Модуль {name} уже зарегистрирован")
                return False

            self.modules[name] = module
            self.module_dependencies[name] = dependencies or []

            # Обновляем порядок запуска с учетом зависимостей
            self._update_startup_order()

            self.logger.info(f"Модуль {name} зарегистрирован")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка регистрации модуля {name}: {e}")
            return False

    def unregister_module(self, name: str) -> bool:
        """
        Отмена регистрации модуля.

        Args:
            name: Имя модуля

        Returns:
            bool: True если отмена регистрации успешна
        """
        try:
            if name not in self.modules:
                self.logger.warning(f"Модуль {name} не найден")
                return False

            # Останавливаем модуль если он запущен
            if self.modules[name].get_status() == ModuleStatus.RUNNING:
                self.modules[name].stop()

            del self.modules[name]
            del self.module_dependencies[name]

            # Обновляем порядок запуска
            self._update_startup_order()

            self.logger.info(f"Модуль {name} отменен")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка отмены регистрации модуля {name}: {e}")
            return False

    def _update_startup_order(self) -> None:
        """Обновление порядка запуска модулей с учетом зависимостей"""
        visited = set()
        temp_visited = set()
        order = []

        def visit(module_name):
            if module_name in temp_visited:
                raise ValueError(f"Циклическая зависимость обнаружена: {module_name}")
            if module_name in visited:
                return

            temp_visited.add(module_name)

            for dep in self.module_dependencies.get(module_name, []):
                if dep in self.modules:
                    visit(dep)

            temp_visited.remove(module_name)
            visited.add(module_name)
            order.append(module_name)

        for module_name in self.modules:
            if module_name not in visited:
                visit(module_name)

        self.module_startup_order = order

    def initialize_simulation(self) -> bool:
        """
        Инициализация симуляции.

        Returns:
            bool: True если инициализация успешна
        """
        try:
            self.state = SimulationState.INITIALIZING
            self.logger.info("Начало инициализации симуляции")

            # Инициализируем модули в правильном порядке
            for module_name in self.module_startup_order:
                module = self.modules[module_name]
                module_config = self.config.modules_config.get(module_name, {})

                if not module.initialize(module_config):
                    self.logger.error(f"Ошибка инициализации модуля {module_name}")
                    self.state = SimulationState.ERROR
                    return False

                self.logger.info(f"Модуль {module_name} инициализирован")

            # Публикуем событие начала инициализации
            self.event_manager.publish(
                EventType.SIMULATION_START,
                "SimulationCoordinator",
                {"config": self.config.__dict__}
            )

            self.state = SimulationState.STOPPED
            self.logger.info("Инициализация симуляции завершена")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка инициализации симуляции: {e}")
            self.state = SimulationState.ERROR
            return False

    def start_simulation(self) -> bool:
        """
        Запуск симуляции.

        Returns:
            bool: True если запуск успешен
        """
        try:
            if self.state != SimulationState.STOPPED:
                self.logger.warning("Симуляция уже запущена или в процессе инициализации")
                return False

            self.state = SimulationState.RUNNING
            self.metrics.start_time = time.time()
            self.metrics.total_tti = self.config.duration_ms // self.config.tti_duration_ms
            self.metrics.processed_tti = 0

            self.logger.info(f"Запуск симуляции на {self.config.duration_ms} мс")

            # Запускаем модули
            for module_name in self.module_startup_order:
                module = self.modules[module_name]
                if not module.start():
                    self.logger.error(f"Ошибка запуска модуля {module_name}")
                    self.state = SimulationState.ERROR
                    return False
                self.logger.info(f"Модуль {module_name} запущен")

            # Запускаем поток симуляции
            self._stop_event.clear()
            self._simulation_thread = threading.Thread(target=self._run_simulation)
            self._simulation_thread.start()

            return True

        except Exception as e:
            self.logger.error(f"Ошибка запуска симуляции: {e}")
            self.state = SimulationState.ERROR
            return False

    def pause_simulation(self) -> bool:
        """
        Приостановка симуляции.

        Returns:
            bool: True если приостановка успешна
        """
        try:
            if self.state != SimulationState.RUNNING:
                self.logger.warning("Симуляция не запущена")
                return False

            self.state = SimulationState.PAUSED

            # Приостанавливаем модули
            for module_name, module in self.modules.items():
                if module.get_status() == ModuleStatus.RUNNING:
                    module.pause()
                    self.logger.info(f"Модуль {module_name} приостановлен")

            self.logger.info("Симуляция приостановлена")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка приостановки симуляции: {e}")
            return False

    def resume_simulation(self) -> bool:
        """
        Возобновление симуляции.

        Returns:
            bool: True если возобновление успешно
        """
        try:
            if self.state != SimulationState.PAUSED:
                self.logger.warning("Симуляция не приостановлена")
                return False

            self.state = SimulationState.RUNNING

            # Возобновляем модули
            for module_name, module in self.modules.items():
                if module.get_status() == ModuleStatus.PAUSED:
                    module.resume()
                    self.logger.info(f"Модуль {module_name} возобновлен")

            self.logger.info("Симуляция возобновлена")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка возобновления симуляции: {e}")
            return False

    def stop_simulation(self) -> bool:
        """
        Остановка симуляции.

        Returns:
            bool: True если остановка успешна
        """
        try:
            if self.state not in [SimulationState.RUNNING, SimulationState.PAUSED]:
                self.logger.warning("Симуляция не запущена")
                return False

            self.state = SimulationState.STOPPING
            self.logger.info("Остановка симуляции")

            # Сигнализируем потоку симуляции о необходимости остановки
            self._stop_event.set()

            # Ждем завершения потока симуляции
            if self._simulation_thread and self._simulation_thread.is_alive():
                self._simulation_thread.join(timeout=5.0)

            # Останавливаем модули
            for module_name, module in self.modules.items():
                if module.get_status() in [ModuleStatus.RUNNING, ModuleStatus.PAUSED]:
                    module.stop()
                    self.logger.info(f"Модуль {module_name} остановлен")

            self.metrics.end_time = time.time()
            self.state = SimulationState.STOPPED

            # Публикуем событие окончания симуляции
            self.event_manager.publish(
                EventType.SIMULATION_END,
                "SimulationCoordinator",
                {"metrics": self.metrics.__dict__}
            )

            # Сохраняем метрики
            if self.config.save_metrics:
                self._save_metrics()

            self.logger.info("Симуляция остановлена")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка остановки симуляции: {e}")
            return False

    def _run_simulation(self) -> None:
        """Основной цикл симуляции"""
        try:
            current_time = 0
            last_update_time = 0

            while (current_time < self.config.duration_ms and
                   not self._stop_event.is_set() and
                   self.state == SimulationState.RUNNING):

                # Обработка TTI
                self._process_tti(current_time)

                # Обновление модулей
                if current_time - last_update_time >= self.config.update_interval_ms:
                    self._update_modules(current_time - last_update_time)
                    last_update_time = current_time

                # Пауза между TTI
                time.sleep(self.config.tti_duration_ms / 1000.0)
                current_time += self.config.tti_duration_ms

        except Exception as e:
            self.logger.error(f"Ошибка в цикле симуляции: {e}")
            self.state = SimulationState.ERROR

    def _process_tti(self, tti: int) -> None:
        """Обработка одного TTI"""
        try:
            # Публикуем событие начала TTI
            self.event_manager.publish(
                EventType.TTI_START,
                "SimulationCoordinator",
                {"tti": tti}
            )

            # Здесь можно добавить специфичную логику обработки TTI

            # Публикуем событие окончания TTI
            self.event_manager.publish(
                EventType.TTI_END,
                "SimulationCoordinator",
                {"tti": tti}
            )

            self.metrics.processed_tti += 1

        except Exception as e:
            self.logger.error(f"Ошибка обработки TTI {tti}: {e}")
            self.metrics.module_errors += 1

    def _update_modules(self, time_delta: float) -> None:
        """Обновление всех модулей"""
        for module_name, module in self.modules.items():
            try:
                if module.get_status() == ModuleStatus.RUNNING:
                    start_time = time.time()
                    module.update(time_delta)
                    update_time = time.time() - start_time

                    # Сохраняем метрики производительности
                    self._performance_stats[module_name].append(update_time)

            except Exception as e:
                self.logger.error(f"Ошибка обновления модуля {module_name}: {e}")
                self.metrics.module_errors += 1

    def _handle_error(self, event: Event) -> None:
        """Обработка событий ошибок"""
        self.logger.error(f"Ошибка в модуле {event.source}: {event.data}")
        self.metrics.module_errors += 1

    def _handle_warning(self, event: Event) -> None:
        """Обработка предупреждений"""
        self.logger.warning(f"Предупреждение от модуля {event.source}: {event.data}")

    def _handle_metrics_update(self, event: Event) -> None:
        """Обработка обновлений метрик"""
        self.metrics.performance_metrics.update(event.data)

    def _save_metrics(self) -> None:
        """Сохранение метрик в файл"""
        try:
            metrics_data = {
                "simulation_metrics": self.metrics.__dict__,
                "event_manager_stats": self.event_manager.get_statistics(),
                "module_performance": dict(self._performance_stats),
                "module_status": {
                    name: module.get_status().value
                    for name, module in self.modules.items()
                }
            }

            with open(self.config.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)

            self.logger.info(f"Метрики сохранены в {self.config.metrics_file}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения метрик: {e}")

    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Получение статуса симуляции.

        Returns:
            Dict: Статус симуляции и всех модулей
        """
        return {
            "state": self.state.value,
            "metrics": self.metrics.__dict__,
            "modules": {
                name: {
                    "status": module.get_status().value,
                    "info": module.get_info().__dict__
                }
                for name, module in self.modules.items()
            },
            "event_manager_stats": self.event_manager.get_statistics()
        }

    def get_module(self, name: str) -> Optional[IModule]:
        """
        Получение модуля по имени.

        Args:
            name: Имя модуля

        Returns:
            IModule или None если модуль не найден
        """
        return self.modules.get(name)

    def wait_for_completion(self, timeout: float = None) -> bool:
        """
        Ожидание завершения симуляции.

        Args:
            timeout: Максимальное время ожидания в секундах

        Returns:
            bool: True если симуляция завершилась успешно
        """
        if self._simulation_thread and self._simulation_thread.is_alive():
            self._simulation_thread.join(timeout=timeout)
            return not self._simulation_thread.is_alive()
        return True

