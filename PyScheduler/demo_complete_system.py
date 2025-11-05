"""
#------------------------------------------------------------------------------
# Модуль: demo_complete_system - Полная демонстрация новой архитектуры
#------------------------------------------------------------------------------
# Описание:
#   Комплексная демонстрация всех возможностей новой архитектуры симулятора.
#   Показывает создание модулей, настройку событий, запуск симуляции и анализ
#   результатов. Идеально подходит для демонстрации в дипломной работе.
#
# Версия: 1.0.0
# Дата создания: 2025-01-27
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

from SimulationCoordinator import SimulationCoordinator, SimulationConfig
from ModuleAdapters import ModuleAdapterFactory
from EventManager import EventType, get_event_manager
from BaseModule import BaseModule, ModuleConfig
from ModuleInterfaces import IChannelModel

class CustomChannelModel(BaseModule, IChannelModel):
    """
    Кастомная модель канала для демонстрации расширяемости системы.
    Реализует упрощенную модель с учетом расстояния и препятствий.
    """

    def __init__(self, bs_position, bs_height):
        config = ModuleConfig(
            name="CustomChannelModel",
            version="1.0.0",
            custom_params={
                "bs_position": bs_position,
                "bs_height": bs_height,
                "path_loss_exponent": 2.5,
                "reference_distance": 1.0,
                "reference_loss": 32.4
            }
        )
        super().__init__(config)
        self.bs_position = bs_position
        self.bs_height = bs_height

    def _initialize_impl(self):
        """Инициализация кастомной модели канала"""
        self.logger.info("Инициализация кастомной модели канала")
        return True

    def _start_impl(self):
        """Запуск модели канала"""
        self.logger.info("Запуск кастомной модели канала")
        return True

    def _stop_impl(self):
        """Остановка модели канала"""
        self.logger.info("Остановка кастомной модели канала")
        return True

    def _update_impl(self, time_delta):
        """Обновление модели канала"""
        return True

    def calculate_path_loss(self, distance_2d, distance_3d, ue_height, ue_class):
        """Расчет затухания сигнала по упрощенной модели"""
        try:
            # Параметры модели
            n = self.config.custom_params["path_loss_exponent"]
            d0 = self.config.custom_params["reference_distance"]
            PL0 = self.config.custom_params["reference_loss"]
            fc = 1.8  # Частота в ГГц

            # Основное затухание
            path_loss = PL0 + 20 * np.log10(d0) + 10 * n * np.log10(distance_3d / d0) + 20 * np.log10(fc)

            # Дополнительные потери для разных типов UE
            if ue_class == "indoor":
                path_loss += 20  # Потери при проникновении в здание
            elif ue_class == "car":
                path_loss += 10  # Потери при проникновении в автомобиль

            # Случайные потери (затенение)
            shadow_fading = np.random.normal(0, 8)  # 8 дБ стандартное отклонение
            path_loss += shadow_fading

            return path_loss

        except Exception as e:
            self.logger.error(f"Ошибка расчета затухания: {e}")
            return 150.0  # Большое затухание при ошибке

    def calculate_sinr(self, distance_2d, distance_3d, ue_height, ue_class):
        """Расчет SINR"""
        try:
            # Расчет затухания
            path_loss = self.calculate_path_loss(distance_2d, distance_3d, ue_height, ue_class)

            # Параметры передатчика
            tx_power = 44  # дБм
            antenna_gain = 15  # дБи
            cable_loss = 2  # дБ

            # Расчет мощности сигнала
            signal_power = tx_power + antenna_gain - cable_loss - path_loss

            # Мощность шума и интерференции
            noise_power = -174 + 10 * np.log10(10e6)  # -104 дБм для 10 МГц
            interference_power = -95  # дБм

            # SINR
            total_noise = 10 * np.log10(10**(noise_power/10) + 10**(interference_power/10))
            sinr = signal_power - total_noise

            return sinr

        except Exception as e:
            self.logger.error(f"Ошибка расчета SINR: {e}")
            return -20.0  # Низкий SINR при ошибке

    def calculate_los_probability(self, distance_2d, ue_height):
        """Расчет вероятности прямой видимости"""
        try:
            # Простая модель: вероятность уменьшается с расстоянием
            if distance_2d < 50:
                return 1.0
            elif distance_2d < 200:
                return 0.8
            elif distance_2d < 500:
                return 0.5
            else:
                return 0.2

        except Exception as e:
            self.logger.error(f"Ошибка расчета LOS: {e}")
            return 0.5

class SimulationAnalyzer:
    """Анализатор результатов симуляции"""

    def __init__(self):
        self.metrics_history = []
        self.event_history = []

    def collect_metrics(self, coordinator):
        """Сбор метрик из координатора"""
        status = coordinator.get_simulation_status()
        self.metrics_history.append({
            'time': time.time(),
            'tti': status['metrics']['processed_tti'],
            'events': status['event_manager_stats']['events_published'],
            'errors': status['metrics']['module_errors']
        })

    def analyze_performance(self):
        """Анализ производительности симуляции"""
        if not self.metrics_history:
            return

        print("\n=== Анализ производительности ===")

        # Время симуляции
        total_time = self.metrics_history[-1]['time'] - self.metrics_history[0]['time']
        total_tti = self.metrics_history[-1]['tti']

        print(f"Общее время симуляции: {total_time:.2f} секунд")
        print(f"Обработано TTI: {total_tti}")
        print(f"Скорость обработки: {total_tti/total_time:.2f} TTI/сек")

        # События
        total_events = self.metrics_history[-1]['events']
        print(f"Всего событий: {total_events}")
        print(f"Скорость событий: {total_events/total_time:.2f} событий/сек")

        # Ошибки
        total_errors = self.metrics_history[-1]['errors']
        print(f"Всего ошибок: {total_errors}")
        if total_tti > 0:
            print(f"Частота ошибок: {total_errors/total_tti*100:.2f}%")

    def plot_metrics(self):
        """Построение графиков метрик"""
        if len(self.metrics_history) < 2:
            return

        times = [m['time'] - self.metrics_history[0]['time'] for m in self.metrics_history]
        tti_counts = [m['tti'] for m in self.metrics_history]
        event_counts = [m['events'] for m in self.metrics_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # График TTI
        ax1.plot(times, tti_counts, 'b-', linewidth=2)
        ax1.set_xlabel('Время (сек)')
        ax1.set_ylabel('Количество TTI')
        ax1.set_title('Прогресс обработки TTI')
        ax1.grid(True)

        # График событий
        ax2.plot(times, event_counts, 'r-', linewidth=2)
        ax2.set_xlabel('Время (сек)')
        ax2.set_ylabel('Количество событий')
        ax2.set_title('Накопление событий')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

def create_advanced_simulation():
    """Создание продвинутой симуляции с кастомными модулями"""
    print("=== Создание продвинутой симуляции ===")

    # Конфигурация симуляции
    config = SimulationConfig(
        duration_ms=10000,  # 10 секунд
        tti_duration_ms=1,
        update_interval_ms=10,
        enable_logging=True,
        log_level="INFO",
        save_metrics=True,
        metrics_file="advanced_simulation_metrics.json",
        modules_config={
            "CustomChannelModel": {
                "path_loss_exponent": 2.5,
                "reference_distance": 1.0,
                "reference_loss": 32.4
            },
            "MobilityModel_RandomWalk": {
                "x_min": 0, "x_max": 2000,
                "y_min": 0, "y_max": 2000
            },
            "TrafficModel_Poisson": {
                "packet_rate": 15.0,
                "min_packet_size": 100,
                "max_packet_size": 2000
            }
        }
    )

    # Создание координатора
    coordinator = SimulationCoordinator(config)
    analyzer = SimulationAnalyzer()

    # Создание базовой станции
    print("Создание базовой станции...")
    bs_adapter = ModuleAdapterFactory.create_bs_adapter(
        x=1000, y=1000, height=30.0, bandwidth=20.0
    )
    coordinator.register_module("BaseStation", bs_adapter)

    # Создание ресурсной сетки
    print("Создание ресурсной сетки...")
    resource_grid_adapter = ModuleAdapterFactory.create_resource_grid_adapter(
        bandwidth=20.0, num_frames=2
    )
    coordinator.register_module("ResourceGrid", resource_grid_adapter)

    # Создание кастомной модели канала
    print("Создание кастомной модели канала...")
    custom_channel = CustomChannelModel((1000, 1000), 30.0)
    coordinator.register_module("CustomChannelModel", custom_channel)

    # Создание стандартных моделей
    print("Создание стандартных моделей...")
    mobility_adapter = ModuleAdapterFactory.create_mobility_model_adapter(
        "RandomWalk", config.modules_config["MobilityModel_RandomWalk"]
    )
    traffic_adapter = ModuleAdapterFactory.create_traffic_model_adapter(
        "Poisson", config.modules_config["TrafficModel_Poisson"]
    )
    scheduler_adapter = ModuleAdapterFactory.create_scheduler_adapter(
        "ProportionalFair", resource_grid_adapter.resource_grid, bs_adapter.bs
    )

    coordinator.register_module("MobilityModel", mobility_adapter)
    coordinator.register_module("TrafficModel", traffic_adapter)
    coordinator.register_module("Scheduler", scheduler_adapter)

    # Создание пользователей с разными характеристиками
    print("Создание пользовательских устройств...")
    ue_configs = [
        {"id": 1, "x": 500, "y": 500, "class": "pedestrian"},
        {"id": 2, "x": 1500, "y": 500, "class": "car"},
        {"id": 3, "x": 500, "y": 1500, "class": "indoor"},
        {"id": 4, "x": 1500, "y": 1500, "class": "pedestrian"},
        {"id": 5, "x": 1000, "y": 1000, "class": "car"}
    ]

    ue_adapters = []
    for ue_config in ue_configs:
        ue_adapter = ModuleAdapterFactory.create_ue_adapter(
            ue_id=ue_config["id"],
            x=ue_config["x"],
            y=ue_config["y"],
            ue_class=ue_config["class"]
        )

        # Настройка моделей для UE
        ue_adapter.set_channel_model(custom_channel)
        ue_adapter.set_mobility_model(mobility_adapter)
        ue_adapter.set_traffic_model(traffic_adapter)

        # Регистрация в БС
        bs_adapter.register_user(ue_adapter)

        ue_adapters.append(ue_adapter)
        coordinator.register_module(f"UE_{ue_config['id']}", ue_adapter)

    print(f"Создано {len(ue_adapters)} пользовательских устройств")

    return coordinator, analyzer, ue_adapters

def setup_event_monitoring(analyzer):
    """Настройка мониторинга событий"""
    event_manager = get_event_manager()

    def handle_ue_movement(event):
        data = event.data
        print(f"📍 UE {data['ue_id']} переместился в {data['position']}")

    def handle_channel_update(event):
        data = event.data
        print(f"📡 UE {data['ue_id']}: CQI={data['cqi']}, SINR={data['sinr']:.1f} дБ")

    def handle_traffic_generation(event):
        data = event.data
        print(f"📦 UE {data['ue_id']}: буфер {data['buffer_size']} байт")

    def handle_scheduling(event):
        data = event.data
        print(f"Планирование TTI {data['tti']}: {data['users_count']} пользователей за {data['schedule_time']:.3f} мс")

    def handle_resource_allocation(event):
        data = event.data
        print(f"🔗 Выделен ресурс UE {data['ue_id']} на частоте {data['frequency_idx']}")

    # Подписка на события
    event_manager.subscribe(EventType.UE_POSITION_UPDATED, handle_ue_movement)
    event_manager.subscribe(EventType.UE_CHANNEL_UPDATED, handle_channel_update)
    event_manager.subscribe(EventType.UE_TRAFFIC_GENERATED, handle_traffic_generation)
    event_manager.subscribe(EventType.BS_SCHEDULING_COMPLETED, handle_scheduling)
    event_manager.subscribe(EventType.SCHEDULER_RESOURCE_ALLOCATED, handle_resource_allocation)

def run_comprehensive_simulation():
    """Запуск комплексной симуляции"""
    print("Запуск комплексной симуляции LTE-сети")
    print("=" * 60)

    # Создание симуляции
    coordinator, analyzer, ue_adapters = create_advanced_simulation()

    # Настройка мониторинга событий
    setup_event_monitoring(analyzer)

    # Инициализация
    print("\nИнициализация симуляции...")
    if not coordinator.initialize_simulation():
        print("Ошибка инициализации симуляции!")
        return

    print("Инициализация завершена")

    # Запуск симуляции
    print("\n▶️ Запуск симуляции...")
    if not coordinator.start_simulation():
        print("Ошибка запуска симуляции!")
        return

    print("Симуляция запущена")

    # Мониторинг в реальном времени
    print("\nМониторинг в реальном времени...")
    start_time = time.time()

    while coordinator.state.value == "running":
        time.sleep(1)  # Пауза для мониторинга
        analyzer.collect_metrics(coordinator)

        # Показываем прогресс каждые 2 секунды
        elapsed = time.time() - start_time
        if int(elapsed) % 2 == 0:
            status = coordinator.get_simulation_status()
            print(f"⏱️ Время: {elapsed:.1f}с, TTI: {status['metrics']['processed_tti']}, "
                  f"События: {status['event_manager_stats']['events_published']}")

    # Ожидание завершения
    print("\n⏳ Ожидание завершения...")
    coordinator.wait_for_completion()

    # Остановка
    print("\n⏹️ Остановка симуляции...")
    coordinator.stop_simulation()

    # Анализ результатов
    print("\nАнализ результатов...")
    analyzer.analyze_performance()

    # Финальная статистика
    final_status = coordinator.get_simulation_status()
    print(f"\nФинальная статистика:")
    print(f"   Статус: {final_status['state']}")
    print(f"   Обработано TTI: {final_status['metrics']['processed_tti']}")
    print(f"   Всего событий: {final_status['event_manager_stats']['events_published']}")
    print(f"   Ошибок модулей: {final_status['metrics']['module_errors']}")
    print(f"   Время симуляции: {final_status['metrics']['end_time'] - final_status['metrics']['start_time']:.2f}с")

    # Статистика по модулям
    print(f"\nСтатус модулей:")
    for module_name, module_info in final_status['modules'].items():
        status_emoji = "OK" if module_info['status'] == "stopped" else "WARN"
        print(f"   {status_emoji} {module_name}: {module_info['status']}")

    # Построение графиков
    print(f"\nПостроение графиков...")
    try:
        analyzer.plot_metrics()
    except Exception as e:
        print(f"⚠️ Не удалось построить графики: {e}")

    print(f"\n🎉 Симуляция завершена успешно!")
    print(f"📁 Метрики сохранены в {coordinator.config.metrics_file}")

def demonstrate_architecture_benefits():
    """Демонстрация преимуществ новой архитектуры"""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРЕИМУЩЕСТВ АРХИТЕКТУРЫ")
    print("=" * 60)

    print("\n1️⃣ СЛАБАЯ СВЯЗАННОСТЬ:")
    print("   • Модули взаимодействуют только через события")
    print("   • Отсутствие прямых зависимостей")
    print("   • Легкая замена компонентов")

    print("\n2️⃣ РАСШИРЯЕМОСТЬ:")
    print("   • Простое добавление новых модулей")
    print("   • Поддержка различных реализаций")
    print("   • Гибкая конфигурация")

    print("\n3️⃣ ТЕСТИРУЕМОСТЬ:")
    print("   • Изоляция модулей для unit-тестов")
    print("   • Возможность мокирования")
    print("   • Автоматизированное тестирование")

    print("\n4️⃣ ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print("   • Асинхронная обработка событий")
    print("   • Оптимизированная координация")
    print("   • Минимальные накладные расходы")

    print("\n5️⃣ НАБЛЮДАЕМОСТЬ:")
    print("   • Детальная статистика работы")
    print("   • Мониторинг в реальном времени")
    print("   • Автоматическое логирование")

def main():
    """Главная функция"""
    print("ДЕМОНСТРАЦИЯ НОВОЙ АРХИТЕКТУРЫ СИМУЛЯТОРА LTE-СЕТИ")
    print("   Тема дипломной работы: 'Организация межмодульного взаимодействия'")
    print("   'на основе паттернов проектирования'")

    try:
        # Демонстрация преимуществ
        demonstrate_architecture_benefits()

        # Запуск комплексной симуляции
        run_comprehensive_simulation()

        print("\n" + "=" * 60)
        print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("Готово для дипломной работы")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⏹️ Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n\nОшибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



