"""
#------------------------------------------------------------------------------
# Модуль: ModuleAdapters - Адаптеры для интеграции существующих модулей
#------------------------------------------------------------------------------
# Описание:
#   Создает адаптеры для существующих модулей симулятора, позволяя им работать
#   с новой архитектурой на основе событий и интерфейсов. Реализует паттерн Adapter
#   для обеспечения обратной совместимости.
#
# Версия: 1.0.0
# Дата создания: 2025-01-27
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np

from BaseModule import BaseModule, ModuleConfig
from ModuleInterfaces import IChannelModel, IMobilityModel, ITrafficModel, IScheduler, IResourceGrid, IUserEquipment, IBaseStation
from EventManager import EventType

# Импорты существующих модулей
from UE_MODULE import UserEquipment as OriginalUserEquipment, UECollection
from BS_MODULE import BaseStation as OriginalBaseStation
from RES_GRID import RES_GRID_LTE as OriginalResourceGrid
from SCHEDULER import RoundRobinScheduler, BestCQIScheduler, ProportionalFairScheduler
from CHANNEL_MODEL import RMaModel, UMaModel, UMiModel
from MOBILITY_MODEL import RandomWalkModel, RandomWaypointModel, RandomDirectionModel, GaussMarkovModel
from TRAFFIC_MODEL import PoissonModel, OnOffModel, MMPPModel

class ChannelModelAdapter(BaseModule, IChannelModel):
    """
    Адаптер для моделей радиоканалов.
    Интегрирует существующие модели каналов с новой архитектурой.
    """

    def __init__(self, config: ModuleConfig, channel_model: Any):
        super().__init__(config)
        self.channel_model = channel_model
        self.bs = None

    def _initialize_impl(self) -> bool:
        """Инициализация адаптера канала"""
        return True

    def _start_impl(self) -> bool:
        """Запуск адаптера канала"""
        return True

    def _stop_impl(self) -> bool:
        """Остановка адаптера канала"""
        return True

    def _update_impl(self, time_delta: float) -> bool:
        """Обновление адаптера канала"""
        return True

    def calculate_path_loss(self, distance_2d: float, distance_3d: float,
                           ue_height: float, ue_class: str) -> float:
        """Расчет затухания сигнала"""
        try:
            return self.channel_model.calculate_path_loss(
                distance_2d, 0, distance_3d, ue_height, ue_class
            )
        except Exception as e:
            self.logger.error(f"Ошибка расчета затухания: {e}")
            return 1000.0  # Большое затухание при ошибке

    def calculate_sinr(self, distance_2d: float, distance_3d: float,
                      ue_height: float, ue_class: str) -> float:
        """Расчет SINR"""
        try:
            return self.channel_model.calculate_SINR(
                distance_2d, 0, distance_3d, ue_height, ue_class
            )
        except Exception as e:
            self.logger.error(f"Ошибка расчета SINR: {e}")
            return -20.0  # Низкий SINR при ошибке

    def calculate_los_probability(self, distance_2d: float, ue_height: float) -> float:
        """Расчет вероятности LOS"""
        try:
            if hasattr(self.channel_model, 'calculate_los_probability'):
                return self.channel_model.calculate_los_probability(distance_2d, ue_height)
            else:
                # Простая эвристика для моделей без LOS
                return 1.0 if distance_2d < 100 else 0.5
        except Exception as e:
            self.logger.error(f"Ошибка расчета LOS: {e}")
            return 0.5

class MobilityModelAdapter(BaseModule, IMobilityModel):
    """
    Адаптер для моделей мобильности.
    Интегрирует существующие модели движения с новой архитектурой.
    """

    def __init__(self, config: ModuleConfig, mobility_model: Any):
        super().__init__(config)
        self.mobility_model = mobility_model
        self.boundaries = (0, 0, 1000, 1000)  # По умолчанию

    def _initialize_impl(self) -> bool:
        """Инициализация адаптера мобильности"""
        return True

    def _start_impl(self) -> bool:
        """Запуск адаптера мобильности"""
        return True

    def _stop_impl(self) -> bool:
        """Остановка адаптера мобильности"""
        return True

    def _update_impl(self, time_delta: float) -> bool:
        """Обновление адаптера мобильности"""
        return True

    def update_position(self, current_position: Tuple[float, float],
                       current_velocity: float, current_direction: float,
                       time_delta: float) -> Tuple[Tuple[float, float], float, float]:
        """Обновление позиции пользователя"""
        try:
            # Адаптируем вызов под разные модели
            if isinstance(self.mobility_model, RandomWalkModel):
                new_pos, new_vel, new_dir, _ = self.mobility_model.update(
                    current_position, current_velocity, 0.5, 5.0,
                    current_direction, False, int(time_delta)
                )
                return new_pos, new_vel, new_dir

            elif isinstance(self.mobility_model, RandomWaypointModel):
                new_pos, new_vel, new_dir, _, _, _ = self.mobility_model.update(
                    current_position, current_velocity, 0.5, 5.0,
                    current_direction, (0, 0), False, 0.0, int(time_delta)
                )
                return new_pos, new_vel, new_dir

            elif isinstance(self.mobility_model, GaussMarkovModel):
                new_pos, new_vel, new_dir, _ = self.mobility_model.update(
                    current_position, current_velocity, current_direction,
                    3.0, 0.0, int(time_delta)
                )
                return new_pos, new_vel, new_dir

            else:
                # Fallback для неизвестных моделей
                return current_position, current_velocity, current_direction

        except Exception as e:
            self.logger.error(f"Ошибка обновления позиции: {e}")
            return current_position, current_velocity, current_direction

    def get_velocity_limits(self) -> Tuple[float, float]:
        """Получение ограничений скорости"""
        if hasattr(self.mobility_model, 'x_min'):
            # Для моделей с границами
            return (0.5, 10.0)  # Примерные значения
        return (0.0, 5.0)

    def is_boundary_respected(self, position: Tuple[float, float]) -> bool:
        """Проверка соблюдения граничных условий"""
        x, y = position
        x_min, y_min, x_max, y_max = self.boundaries
        return x_min <= x <= x_max and y_min <= y <= y_max

class TrafficModelAdapter(BaseModule, ITrafficModel):
    """
    Адаптер для моделей генерации трафика.
    Интегрирует существующие модели трафика с новой архитектурой.
    """

    def __init__(self, config: ModuleConfig, traffic_model: Any):
        super().__init__(config)
        self.traffic_model = traffic_model
        self.traffic_stats = {}

    def _initialize_impl(self) -> bool:
        """Инициализация адаптера трафика"""
        return True

    def _start_impl(self) -> bool:
        """Запуск адаптера трафика"""
        return True

    def _stop_impl(self) -> bool:
        """Остановка адаптера трафика"""
        return True

    def _update_impl(self, time_delta: float) -> bool:
        """Обновление адаптера трафика"""
        return True

    def generate_traffic(self, ue_id: int, current_time: float,
                        time_interval: float) -> List[Dict[str, Any]]:
        """Генерация трафика для пользователя"""
        try:
            if isinstance(self.traffic_model, PoissonModel):
                packets = self.traffic_model.generate_traffic(
                    int(current_time), int(time_interval)
                )
            else:
                packets = self.traffic_model.generate_traffic(
                    ue_id, int(current_time), int(time_interval)
                )

            # Обновляем статистику
            if ue_id not in self.traffic_stats:
                self.traffic_stats[ue_id] = {
                    'total_packets': 0,
                    'total_bytes': 0,
                    'last_generation_time': current_time
                }

            self.traffic_stats[ue_id]['total_packets'] += len(packets)
            self.traffic_stats[ue_id]['total_bytes'] += sum(p['size'] for p in packets)
            self.traffic_stats[ue_id]['last_generation_time'] = current_time

            return packets

        except Exception as e:
            self.logger.error(f"Ошибка генерации трафика для UE {ue_id}: {e}")
            return []

    def get_traffic_statistics(self, ue_id: int) -> Dict[str, Any]:
        """Получение статистики генерации трафика"""
        return self.traffic_stats.get(ue_id, {
            'total_packets': 0,
            'total_bytes': 0,
            'last_generation_time': 0
        })

    def reset_statistics(self, ue_id: int) -> None:
        """Сброс статистики для пользователя"""
        if ue_id in self.traffic_stats:
            self.traffic_stats[ue_id] = {
                'total_packets': 0,
                'total_bytes': 0,
                'last_generation_time': 0
            }

class SchedulerAdapter(BaseModule, IScheduler):
    """
    Адаптер для планировщиков ресурсов.
    Интегрирует существующие планировщики с новой архитектурой.
    """

    def __init__(self, config: ModuleConfig, scheduler: Any):
        super().__init__(config)
        self.scheduler = scheduler
        self.scheduling_metrics = {
            'total_schedules': 0,
            'successful_schedules': 0,
            'failed_schedules': 0,
            'average_schedule_time': 0.0
        }

    def _initialize_impl(self) -> bool:
        """Инициализация адаптера планировщика"""
        return True

    def _start_impl(self) -> bool:
        """Запуск адаптера планировщика"""
        return True

    def _stop_impl(self) -> bool:
        """Остановка адаптера планировщика"""
        return True

    def _update_impl(self, time_delta: float) -> bool:
        """Обновление адаптера планировщика"""
        return True

    def schedule(self, tti: int, users: List[Dict[str, Any]],
                resources: Dict[str, Any]) -> Dict[str, Any]:
        """Планирование ресурсов для заданного TTI"""
        try:
            start_time = time.time()

            # Вызываем оригинальный планировщик
            result = self.scheduler.schedule(tti, users)

            # Обновляем метрики
            schedule_time = time.time() - start_time
            self.scheduling_metrics['total_schedules'] += 1
            self.scheduling_metrics['successful_schedules'] += 1

            # Обновляем среднее время планирования
            total = self.scheduling_metrics['total_schedules']
            current_avg = self.scheduling_metrics['average_schedule_time']
            self.scheduling_metrics['average_schedule_time'] = (
                (current_avg * (total - 1) + schedule_time) / total
            )

            # Публикуем событие о планировании
            self.publish_event("scheduling_completed", {
                "tti": tti,
                "users_count": len(users),
                "schedule_time": schedule_time,
                "result": result
            })

            return result

        except Exception as e:
            self.logger.error(f"Ошибка планирования для TTI {tti}: {e}")
            self.scheduling_metrics['failed_schedules'] += 1
            return {'allocation': {}, 'statistics': {}, 'bitmap': {}}

    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """Получение метрик планировщика"""
        return self.scheduling_metrics.copy()

    def reset_metrics(self) -> None:
        """Сброс метрик планировщика"""
        self.scheduling_metrics = {
            'total_schedules': 0,
            'successful_schedules': 0,
            'failed_schedules': 0,
            'average_schedule_time': 0.0
        }

class ResourceGridAdapter(BaseModule, IResourceGrid):
    """
    Адаптер для ресурсной сетки LTE.
    Интегрирует существующую ресурсную сетку с новой архитектурой.
    """

    def __init__(self, config: ModuleConfig, resource_grid: Any):
        super().__init__(config)
        self.resource_grid = resource_grid
        self.allocation_stats = {
            'total_allocations': 0,
            'total_releases': 0,
            'allocation_failures': 0
        }

    def _initialize_impl(self) -> bool:
        """Инициализация адаптера ресурсной сетки"""
        return True

    def _start_impl(self) -> bool:
        """Запуск адаптера ресурсной сетки"""
        return True

    def _stop_impl(self) -> bool:
        """Остановка адаптера ресурсной сетки"""
        return True

    def _update_impl(self, time_delta: float) -> bool:
        """Обновление адаптера ресурсной сетки"""
        return True

    def allocate_resource_block(self, tti: int, frequency_idx: int,
                              ue_id: int) -> bool:
        """Выделение ресурсного блока пользователю"""
        try:
            # Используем метод выделения пар RB (для обоих слотов)
            success = self.resource_grid.ALLOCATE_RB_PAIR(tti, frequency_idx, ue_id)

            if success:
                self.allocation_stats['total_allocations'] += 1
                self.publish_event("resource_allocated", {
                    "tti": tti,
                    "frequency_idx": frequency_idx,
                    "ue_id": ue_id
                })
            else:
                self.allocation_stats['allocation_failures'] += 1

            return success

        except Exception as e:
            self.logger.error(f"Ошибка выделения RB: {e}")
            self.allocation_stats['allocation_failures'] += 1
            return False

    def release_resource_block(self, tti: int, frequency_idx: int) -> bool:
        """Освобождение ресурсного блока"""
        try:
            # Освобождаем RB в обоих слотах
            subframe = tti % 10
            success = True

            for slot in [0, 1]:
                slot_id = f"sub_{subframe}_slot_{slot}"
                if not self.resource_grid.RELEASE_RB(tti, slot_id, frequency_idx):
                    success = False

            if success:
                self.allocation_stats['total_releases'] += 1
                self.publish_event("resource_released", {
                    "tti": tti,
                    "frequency_idx": frequency_idx
                })

            return success

        except Exception as e:
            self.logger.error(f"Ошибка освобождения RB: {e}")
            return False

    def get_available_resources(self, tti: int) -> List[int]:
        """Получение списка доступных ресурсов для TTI"""
        try:
            free_rbs = self.resource_grid.GET_FREE_RB_FOR_TTI(tti)
            return [rb.freq_idx for rb in free_rbs if rb.CHCK_RB()]
        except Exception as e:
            self.logger.error(f"Ошибка получения доступных ресурсов: {e}")
            return []

    def get_resource_utilization(self, tti: int) -> float:
        """Получение коэффициента использования ресурсов"""
        try:
            total_rbs = self.resource_grid.num_rb
            allocated_rbs = self.resource_grid.stats["allocation_by_tti"].get(tti, 0)
            return allocated_rbs / total_rbs if total_rbs > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Ошибка расчета утилизации ресурсов: {e}")
            return 0.0

class UserEquipmentAdapter(BaseModule, IUserEquipment):
    """
    Адаптер для пользовательского оборудования.
    Интегрирует существующий класс UE с новой архитектурой.
    """

    def __init__(self, config: ModuleConfig, ue: OriginalUserEquipment):
        super().__init__(config)
        self.ue = ue
        self.channel_model = None
        self.mobility_model = None
        self.traffic_model = None

    def _initialize_impl(self) -> bool:
        """Инициализация адаптера UE"""
        return True

    def _start_impl(self) -> bool:
        """Запуск адаптера UE"""
        return True

    def _stop_impl(self) -> bool:
        """Остановка адаптера UE"""
        return True

    def _update_impl(self, time_delta: float) -> bool:
        """Обновление адаптера UE"""
        return True

    def update_position(self, time_delta: float, bs_position: Tuple[float, float],
                       bs_height: float) -> None:
        """Обновление позиции пользователя"""
        try:
            if self.mobility_model:
                self.ue.UPD_POSITION(int(time_delta), bs_position, bs_height)
                self.publish_event("position_updated", {
                    "ue_id": self.ue.UE_ID,
                    "position": self.ue.position,
                    "velocity": self.ue.velocity
                })
        except Exception as e:
            self.logger.error(f"Ошибка обновления позиции UE {self.ue.UE_ID}: {e}")

    def update_channel_quality(self) -> None:
        """Обновление качества канала"""
        try:
            if self.channel_model:
                self.ue.UPD_CH_QUALITY()
                self.publish_event("channel_updated", {
                    "ue_id": self.ue.UE_ID,
                    "cqi": self.ue.cqi,
                    "sinr": self.ue.SINR
                })
        except Exception as e:
            self.logger.error(f"Ошибка обновления канала UE {self.ue.UE_ID}: {e}")

    def generate_traffic(self, time_delta: float) -> None:
        """Генерация трафика"""
        try:
            if self.traffic_model:
                self.ue.GEN_TRFFC(int(time_delta), int(time_delta))
                self.publish_event("traffic_generated", {
                    "ue_id": self.ue.UE_ID,
                    "buffer_size": self.ue.buffer.current_size
                })
        except Exception as e:
            self.logger.error(f"Ошибка генерации трафика UE {self.ue.UE_ID}: {e}")

    def get_channel_quality(self) -> Dict[str, Any]:
        """Получение текущего качества канала"""
        return {
            "cqi": self.ue.cqi,
            "sinr": self.ue.SINR,
            "distance": self.ue.dist_to_BS_2D
        }

    def get_buffer_status(self) -> Dict[str, Any]:
        """Получение статуса буфера"""
        return self.ue.buffer.GET_STATUS(int(time.time() * 1000))

    def set_channel_model(self, channel_model: IChannelModel) -> None:
        """Установка модели канала"""
        self.channel_model = channel_model
        if hasattr(channel_model, 'channel_model'):
            self.ue.SET_CH_MODEL(channel_model.channel_model)

    def set_mobility_model(self, mobility_model: IMobilityModel) -> None:
        """Установка модели мобильности"""
        self.mobility_model = mobility_model
        if hasattr(mobility_model, 'mobility_model'):
            self.ue.SET_MOBILITY_MODEL(mobility_model.mobility_model)

    def set_traffic_model(self, traffic_model: ITrafficModel) -> None:
        """Установка модели трафика"""
        self.traffic_model = traffic_model
        if hasattr(traffic_model, 'traffic_model'):
            self.ue.SET_TRAFFIC_MODEL(traffic_model.traffic_model)

class BaseStationAdapter(BaseModule, IBaseStation):
    """
    Адаптер для базовой станции.
    Интегрирует существующую БС с новой архитектурой.
    """

    def __init__(self, config: ModuleConfig, bs: OriginalBaseStation):
        super().__init__(config)
        self.bs = bs
        self.registered_users = {}

    def _initialize_impl(self) -> bool:
        """Инициализация адаптера БС"""
        return True

    def _start_impl(self) -> bool:
        """Запуск адаптера БС"""
        return True

    def _stop_impl(self) -> bool:
        """Остановка адаптера БС"""
        return True

    def _update_impl(self, time_delta: float) -> bool:
        """Обновление адаптера БС"""
        try:
            self.update_user_buffers(time_delta)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка обновления модуля BaseStation: {e}")
            return False

    def register_user(self, ue: IUserEquipment) -> bool:
        """Регистрация пользователя"""
        try:
            if hasattr(ue, 'ue'):
                self.bs.REG_UE(ue.ue)
                self.registered_users[ue.ue.UE_ID] = ue
                self.publish_event("user_registered", {
                    "ue_id": ue.ue.UE_ID,
                    "position": ue.ue.position
                })
                return True
            return False
        except Exception as e:
            self.logger.error(f"Ошибка регистрации пользователя: {e}")
            return False

    def deregister_user(self, ue_id: int) -> bool:
        """Отмена регистрации пользователя"""
        try:
            if ue_id in self.registered_users:
                del self.registered_users[ue_id]
                self.publish_event("user_deregistered", {"ue_id": ue_id})
                return True
            return False
        except Exception as e:
            self.logger.error(f"Ошибка отмены регистрации пользователя {ue_id}: {e}")
            return False

    def get_registered_users(self) -> List[IUserEquipment]:
        """Получение списка зарегистрированных пользователей"""
        return list(self.registered_users.values())

    def update_user_buffers(self, time_delta: float) -> None:
        """Обновление буферов пользователей"""
        try:
            # Преобразуем время в миллисекунды
            current_time_ms = int(time.time() * 1000)
            self.bs.UPD_GLOBAL_BUFFER(current_time_ms)
            self.publish_event("buffers_updated", {
                "total_size": self.bs.GET_GLOBAL_BUFFER_STATUS(current_time_ms)['total_size']
            })
        except Exception as e:
            self.logger.error(f"Ошибка обновления буферов: {e}")

    def get_buffer_status(self) -> Dict[str, Any]:
        """Получение статуса буферов базовой станции"""
        return self.bs.GET_GLOBAL_BUFFER_STATUS(int(time.time() * 1000))

# Фабрика для создания адаптеров
class ModuleAdapterFactory:
    """Фабрика для создания адаптеров модулей"""

    @staticmethod
    def create_channel_model_adapter(model_type: str, bs: Any, config: Dict[str, Any] = None) -> ChannelModelAdapter:
        """Создание адаптера модели канала"""
        if model_type == "RMa":
            model = RMaModel(bs)
        elif model_type == "UMa":
            model = UMaModel(bs)
        elif model_type == "UMi":
            model = UMiModel(bs)
        else:
            raise ValueError(f"Неизвестный тип модели канала: {model_type}")

        module_config = ModuleConfig(
            name=f"ChannelModel_{model_type}",
            custom_params=config or {}
        )

        return ChannelModelAdapter(module_config, model)

    @staticmethod
    def create_mobility_model_adapter(model_type: str, config: Dict[str, Any] = None) -> MobilityModelAdapter:
        """Создание адаптера модели мобильности"""
        params = config or {}
        x_min = params.get('x_min', 0)
        x_max = params.get('x_max', 1000)
        y_min = params.get('y_min', 0)
        y_max = params.get('y_max', 1000)

        if model_type == "RandomWalk":
            model = RandomWalkModel(x_min, x_max, y_min, y_max)
        elif model_type == "RandomWaypoint":
            pause_time = params.get('pause_time', 10.0)
            model = RandomWaypointModel(x_min, x_max, y_min, y_max, pause_time)
        elif model_type == "RandomDirection":
            pause_time = params.get('pause_time', 10.0)
            model = RandomDirectionModel(x_min, x_max, y_min, y_max, pause_time)
        elif model_type == "GaussMarkov":
            alpha = params.get('alpha', 0.75)
            boundary_threshold = params.get('boundary_threshold', 5.0)
            model = GaussMarkovModel(x_min, x_max, y_min, y_max, alpha, boundary_threshold)
        else:
            raise ValueError(f"Неизвестный тип модели мобильности: {model_type}")

        module_config = ModuleConfig(
            name=f"MobilityModel_{model_type}",
            custom_params=params
        )

        adapter = MobilityModelAdapter(module_config, model)
        adapter.boundaries = (x_min, y_min, x_max, y_max)
        return adapter

    @staticmethod
    def create_traffic_model_adapter(model_type: str, config: Dict[str, Any] = None) -> TrafficModelAdapter:
        """Создание адаптера модели трафика"""
        params = config or {}

        if model_type == "Poisson":
            packet_rate = params.get('packet_rate', 5.0)
            min_size = params.get('min_packet_size', 150)
            max_size = params.get('max_packet_size', 1500)
            model = PoissonModel(packet_rate, min_size, max_size)
        elif model_type == "OnOff":
            duration_on = params.get('duration_on', 2.0)
            duration_off = params.get('duration_off', 3.0)
            packet_rate = params.get('packet_rate', 25.0)
            min_size = params.get('min_packet_size', 150)
            max_size = params.get('max_packet_size', 1500)
            model = OnOffModel(duration_on, duration_off, packet_rate, min_size, max_size)
        elif model_type == "MMPP":
            packet_rates = params.get('packet_rates', [5, 20, 40])
            min_size = params.get('min_packet_size', 150)
            max_size = params.get('max_packet_size', 1500)
            model = MMPPModel(packet_rates, min_size, max_size)
        else:
            raise ValueError(f"Неизвестный тип модели трафика: {model_type}")

        module_config = ModuleConfig(
            name=f"TrafficModel_{model_type}",
            custom_params=params
        )

        return TrafficModelAdapter(module_config, model)

    @staticmethod
    def create_scheduler_adapter(scheduler_type: str, lte_grid: Any, bs: Any) -> SchedulerAdapter:
        """Создание адаптера планировщика"""
        if scheduler_type == "RoundRobin":
            scheduler = RoundRobinScheduler(lte_grid, bs)
        elif scheduler_type == "BestCQI":
            scheduler = BestCQIScheduler(lte_grid, bs)
        elif scheduler_type == "ProportionalFair":
            scheduler = ProportionalFairScheduler(lte_grid, bs)
        else:
            raise ValueError(f"Неизвестный тип планировщика: {scheduler_type}")

        module_config = ModuleConfig(name=f"Scheduler_{scheduler_type}")
        return SchedulerAdapter(module_config, scheduler)

    @staticmethod
    def create_resource_grid_adapter(bandwidth: float, num_frames: int) -> ResourceGridAdapter:
        """Создание адаптера ресурсной сетки"""
        grid = OriginalResourceGrid(bandwidth=bandwidth, num_frames=num_frames)
        module_config = ModuleConfig(name="ResourceGrid")
        return ResourceGridAdapter(module_config, grid)

    @staticmethod
    def create_ue_adapter(ue_id: int, x: float, y: float, ue_class: str = "pedestrian") -> UserEquipmentAdapter:
        """Создание адаптера пользовательского оборудования"""
        ue = OriginalUserEquipment(UE_ID=ue_id, x=x, y=y, ue_class=ue_class)
        module_config = ModuleConfig(name=f"UE_{ue_id}")
        return UserEquipmentAdapter(module_config, ue)

    @staticmethod
    def create_bs_adapter(x: float, y: float, height: float, bandwidth: float) -> BaseStationAdapter:
        """Создание адаптера базовой станции"""
        bs = OriginalBaseStation(x=x, y=y, height=height, bandwidth=bandwidth)
        module_config = ModuleConfig(name="BaseStation")
        return BaseStationAdapter(module_config, bs)

