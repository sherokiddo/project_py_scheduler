import GLOBALS
import matplotlib.pyplot as plt
import numpy as np
from BS_MODULE import BaseStation
from MOBILITY_MODEL import MapBorders
from SIMULATION_MANAGER import SimulationManager
from TRAFFIC_MODEL import PoissonModel
from UE_MODULE import UECollection


def visualize_users_mobility(
    ue_collection: UECollection,
    bs: BaseStation,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    """
    Функция для построения карты передвижения пользователей.

    Args:
        ue_collection (UECollection): Объект коллекции пользователей.
        bs (BaseStation): Объект базовой станции.
        x_min (float): Минимальная граница отображения по оси X.
        x_max (float): Максимальная граница отображения по оси X.
        y_min (float): Минимальная граница отображения по оси Y.
        y_max (float): Максимальная граница отображения по оси Y.

    """
    x_bs, y_bs = bs.position

    plt.figure(figsize=(10, 6))
    plt.title("Карта передвижения пользователей")
    plt.xlabel("X координата (м)")
    plt.ylabel("Y координата (м)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot(x_bs, y_bs, marker="o", label="Base Station")

    for ue in ue_collection.GET_ALL_USERS():
        x_coords = [x for x, _ in ue.coordinates]
        y_coords = [y for _, y in ue.coordinates]
        plt.plot(x_coords, y_coords, marker=".", label=f"UE {ue.UE_ID}")

    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_users_sinr(ue_collection: UECollection, sim_duration: float, update_interval: float):
    """
    Функция для построения графика SINR для всех пользователей.

    Args:
        ue_collection (UECollection): Объект коллекции пользователей.
        sim_duration (float): Время симуляции (мс).
        update_interval (float): Интервал обновления симуляции (мс).
    """
    # Создаём диапазон по всем TTI (каждый миллисекунд)
    tti_range = np.arange(0, sim_duration)

    plt.figure(figsize=(12, 6))
    plt.title("График SINR во времени для всех пользователей")
    plt.xlabel("TTI (мс)")
    plt.ylabel("SINR (dB)")

    for ue in ue_collection.GET_ALL_USERS():
        sinr_values = ue.SINR_values

        # Выравнивание длин (на случай несовпадения)
        min_len = min(len(tti_range), len(sinr_values))

        plt.plot(tti_range[:min_len], sinr_values[:min_len], label=f"UE{ue.UE_ID}", alpha=0.7)

    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.ylim(-20, 65)
    plt.tight_layout()
    plt.show()


def print_users_stats(ue_collection: UECollection, tti: int, bs: BaseStation, sched_result: dict):
    """
    Функция для вывода статистики пользователей в определённом TTI.

    Args:
        ue_collection (UECollection): Объект коллекции пользователей.
        tti (int): Номер TTI.
        bs (BaseStation): Объект базовой станции.
        sched_result (dict): Результат работы планировщика.

    """
    print(f"\n[TTI {tti}]")
    print("=" * 40)

    allocation = sched_result["allocation"]

    for ue in ue_collection.GET_ALL_USERS():
        ue_id = ue.UE_ID
        num_rbs = len(allocation.get(ue_id, []))

        displacement = np.hypot(
            ue.position[0] - ue.coordinates[-2][0], ue.position[1] - ue.coordinates[-2][1]
        )

        print(f"UE {ue_id}:")
        print(f"\tRBs выделено        : {num_rbs}")
        print(f"\tТекущая скорость    : {ue.current_dl_throughput} bit/s")
        print(f"\tСредняя скорость    : {ue.average_throughput:.2f} bit/s")
        print("\t+" + "-" * 35)
        print(f"\tSINR                : {ue.SINR:.2f} dB")
        print(f"\tWideband CQI        : {ue.cqi}")
        print(f"\tSubband CQI         : {ue.cqi_subband}")
        print(f"\tРазмер буфера       : {bs.ue_buffers[ue.UE_ID].sizes[ue.UE_ID]} B")
        print("\t+" + "-" * 35)
        print(f"\tПред. позиция       : {ue.coordinates[-2]}")
        print(f"\tТекущая позиция     : {ue.position}")
        print(f"\tСмещение            : {displacement} m")
        print("-" * 40)


def sim_with_manager():
    """
    Пример запуска симуляции с использованием менеджера.

    """
    # =============================================================================
    #             НАСТРОЙКА БАЗОВОЙ СТАНЦИИ И КОЛЛЕКЦИИ ПОЛЬЗОВАТЕЛЕЙ
    # =============================================================================

    # Создание и настройка базовой станции
    bs = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa")

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Установка сида
    GLOBALS.SEED = 42

    # Генерация заданного числа UE в коллекцию
    ue_collection.ADD_RANDOM_USERS(num_ue=5)

    MapBorders(-1000, 1000, -1000, 1000)

    # Установка модели передвижения для всех пользователей коллекции
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint")

    # Регистрация всех пользователей коллекции в базовой станции
    ue_collection.REG_USERS_TO_BS(bs)

    sim = SimulationManager()

    # =============================================================================
    #                        НАСТРОЙКА МЕНЕДЖЕРА СИМУЛЯЦИИ
    # =============================================================================

    # Создание менеджера симуляции
    sim = SimulationManager()

    # Установка базовой станции
    sim.set_base_station(bs)

    # Установка коллекции пользователей
    sim.set_ue_collection(ue_collection)

    # Установка планировщика. Можно передвать параметры, которые
    # поддерживает SchedulerInterface.
    sim.set_scheduler(algorithm="RoundRobin")

    # Настраиваем модели для каждого UE
    for ue in ue_collection.GET_ALL_USERS():
        print(f"Setup UE: {ue.UE_ID} traffic (SimpleGenerator)")
        sim.setup_ue_traffic(
            ue_id=ue.UE_ID,
            model_type="Poisson",
            packet_rate=1000,
        )

    # Установка длительности симуляции
    sim.set_sim_duration(5000)

    # Включение verbose логирования. Для вывода всех логов в файл нужно
    # поставить флаг to_file=True.
    sim.enable_verbose_log()

    # Установка менеджера статистики
    sim.set_stats_manager(
        enabled=True,  # Включить сбор
        collect_interval=1,  # Собирать каждые 10 TTI
        history_max_len=5000,
        scheduler_level="full",  # Scheduler: только агрегированные метрики
        amc_level="full",  # AMC: total throughput + avg bits/RB
        pdcch_level="basic",  # PDCCH: отключен (можно включить "basic")
        file_prefix="emp_stats",  # Префикс файла: lte_stats.csv
    )

    # Запуск симуляции
    sim.start_simulation()


def chmdl_test():
    """
    Тестовый стенд для проверки работоспособности моделей канала

    """
    # =============================================================================
    #             НАСТРОЙКА БАЗОВОЙ СТАНЦИИ И КОЛЛЕКЦИИ ПОЛЬЗОВАТЕЛЕЙ
    # =============================================================================

    # Создание и настройка базовой станции
    bs1 = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa", enable_tdl=True)

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Установка сида
    GLOBALS.SEED = 24

    # Генерация заданного числа UE в коллекцию
    ue_collection.ADD_RANDOM_USERS(num_ue=8)

    MapBorders(-1000, 1000, -1000, 1000)

    # Установка модели передвижения для всех пользователей коллекции
    ue_collection.SET_MOBILITY_MODEL("DiagonalWalk", bs=bs1, pause_time=0)

    # Создание модели генерации трафика
    poisson = PoissonModel(packet_rate=5000)

    # Установка модели генерации трафика для всех пользователей коллекции
    ue_collection.SET_TRAFFIC_MODEL(poisson)

    # Регистрация всех пользователей коллекции в базовой станции
    ue_collection.REG_USERS_TO_BS(bs1)

    # =============================================================================
    #                        НАСТРОЙКА МЕНЕДЖЕРА СИМУЛЯЦИИ
    # =============================================================================

    # Создание менеджера симуляции
    sim = SimulationManager()

    # Установка базовой станции
    sim.set_base_station(bs1)

    # Установка коллекции пользователей
    sim.set_ue_collection(ue_collection)

    # Установка планировщика. Можно передвать параметры, которые
    # поддерживает SchedulerInterface.
    sim.set_scheduler(
        algorithm="ProportionalFair",
        max_dl_ue_tti=None,
        pcfich=2,
        enable_window=False,
        window_size=4,
        max_dl_cce_allowance=None,
    )

    # Установка длительности симуляции
    sim.set_sim_duration(5000)

    # Включение verbose логирования. Для вывода всех логов в файл нужно
    # поставить флаг to_file=True.
    sim.enable_verbose_log(to_file=True)

    # Установка менеджера статистики
    sim.set_stats_manager(
        enabled=True,  # Включить сбор
        collect_interval=1,  # Собирать каждые n TTI
        history_max_len=5000,
        scheduler_level="advanced",  # Scheduler: только агрегированные метрики
        amc_level="advanced",  # AMC: total throughput + avg bits/RB
        pdcch_level="advanced",  # PDCCH: отключен (можно включить "basic")
        file_prefix="emp_stats",  # Префикс файла: lte_stats.csv
    )

    # Запуск симуляции
    sim.start_simulation()

    # Визуализация передвижения пользователей
    visualize_users_mobility(
        ue_collection=ue_collection, bs=bs1, x_min=-1000, x_max=1000, y_min=-1000, y_max=1000
    )
    # Визуализация SINR пользователей во времени
    visualize_users_sinr(
        ue_collection=ue_collection, sim_duration=sim.sim_config.sim_duration, update_interval=100
    )


if __name__ == "__main__":
    # debug_simulation()
    # sim_with_ue_collection()
    sim_with_manager()
    # chmdl_test()
