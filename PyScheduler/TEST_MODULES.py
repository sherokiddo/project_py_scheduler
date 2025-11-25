import math

import GLOBALS
import matplotlib.pyplot as plt
import numpy as np
from BS_MODULE import BaseStation, Packet
from CHANNEL_MODEL import UMaModel
from MOBILITY_MODEL import MapBorders
from RES_GRID import RES_GRID_LTE
from SCHEDULER import ProportionalFairScheduler
from UE_MODULE import UECollection, UserEquipment


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
    tti_range = np.arange(0, sim_duration, update_interval)

    plt.figure(figsize=(10, 6))
    plt.title("График SINR во времени для всех пользователей")
    plt.xlabel("TTI")
    plt.ylabel("SINR (dB)")
    for ue in ue_collection.GET_ALL_USERS():
        sinr_values = ue.SINR_values
        plt.plot(tti_range, sinr_values, label=f"UE{ue.UE_ID}")
    plt.legend()
    plt.grid(True)
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
        print(f"\tCQI                 : {ue.cqi}")
        print(f"\tРазмер буфера       : {bs.ue_buffers[ue.UE_ID].sizes[ue.UE_ID]} B")
        print("\t+" + "-" * 35)
        print(f"\tПред. позиция       : {ue.coordinates[-2]}")
        print(f"\tТекущая позиция     : {ue.position}")
        print(f"\tСмещение            : {displacement} m")
        print("-" * 40)


def sim_with_ue_collection():
    """
    Пример сценария с использованием коллекций UE.

    """
    sim_duration = 3000  # Время симуляции (в мс)
    update_interval = 1  # Интервал обновления параметров пользователя (в мс)
    num_frames = int(np.ceil(sim_duration / 10))  # Кол-во кадров (для ресурсной сетки)
    bandwidth = 10  # Ширина полосы (в МГц)
    inf = math.inf

    # Создание и настройка базовой станции
    bs = BaseStation(x=0, y=0, bandwidth=bandwidth, global_max=inf, per_ue_max=inf)

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Установка сида. Нужен для того, чтобы ADD_RANDOM_USERS всегда генерировал
    # одинаковых пользователей в одинаковом месте. Если не задавать сид, пользователи
    # каждую симуляцию будут генерироваться абсолютно случайно
    GLOBALS.SEED = 42

    # Генерация заданного числа UE в коллекцию
    ue_collection.ADD_RANDOM_USERS(num_ue=3)

    # Создание модели передвижения пользователей
    MapBorders(-1000, 1000, -1000, 1000)

    # Установка модели передвижения для всех пользователей коллекции.
    # Есть возможность задавать для отдельных пользователей при помощи параметра ue_ids
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint")

    # @Andrey пишет: Для коллекций фабрика криво работает, надо пофикисть UPD_POSITION и
    # SET_MOBILITY_MODEL на основе тех, что были в UserEquipment

    # Создание модели радиоканала
    uma = UMaModel(bs=bs, cond_update_period=5)

    # Установка модели радиоканала для всех пользователей коллекции.
    # Есть возможность задавать для отдельных пользователей при помощи параметра ue_ids
    ue_collection.SET_CH_MODEL(uma)

    # Регистрация всех пользователей коллекции в базовой станции.
    ue_collection.REG_USERS_TO_BS(bs)

    # Имитация Full Buffer
    for ue in ue_collection.GET_ALL_USERS():
        bs.ue_buffers[ue.UE_ID].ADD_PACKET(
            Packet(size=inf, ue_id=ue.UE_ID, creation_time=0), current_time=0
        )

    # Создание ресурсной сетки
    lte_grid = RES_GRID_LTE(bandwidth=bandwidth, num_frames=num_frames)

    # Создание планировщика
    scheduler = ProportionalFairScheduler(lte_grid, bs)

    # Основной цикл симуляции
    for current_time in range(update_interval, sim_duration + 1, update_interval):
        # Обновление глобальной переменной текущего времени
        # (Временное решение)
        GLOBALS.CURRENT_TIME = current_time

        # Обновление состояния пользователей
        ue_collection.UPDATE_ALL_USERS(
            current_time=current_time,
            update_interval=update_interval,
            bs_position=bs.position,
            bs_height=bs.height,
        )

        # Цикл для планирования ресурсов (по TTI)
        for tti in range(current_time - update_interval, current_time):
            # Подготовка данных для планировщика
            users = ue_collection.GET_USERS_FOR_SCHEDULER()

            # Планирование ресурсов
            sched_result = scheduler.schedule(tti, users)

            # Вывод статистики для каждого пользователя
            print_users_stats(
                ue_collection=ue_collection, tti=tti, bs=bs, sched_result=sched_result
            )

    # Визуализация передвижения пользователей
    visualize_users_mobility(
        ue_collection=ue_collection, bs=bs, x_min=-1000, x_max=1000, y_min=-1000, y_max=1000
    )

    # Визуализация SINR пользователей во времени
    visualize_users_sinr(
        ue_collection=ue_collection, sim_duration=sim_duration, update_interval=update_interval
    )


def debug_simulation():
    """
    Симуляция для тестирования диагональной модели движения DiagonalWalkModel.
    Позволяет проверить работу новой модели в новой архитектуре с фабрикой.
    """
    sim_duration = 2500  # Время симуляции (в мс)
    update_interval = 1  # Интервал обновления параметров пользователя (в мс)
    num_frames = int(np.ceil(sim_duration / 10))  # Кол-во кадров (для ресурсной сетки)
    bandwidth = 10  # Ширина полосы (в МГц)
    inf = math.inf

    # Создание и настройка базовой станции
    bs = BaseStation(x=0, y=0, bandwidth=bandwidth, global_max=inf, per_ue_max=inf)

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Создание и настройка пользовательских устройств
    ue1 = UserEquipment(UE_ID=1, x=4, y=4, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=4, y=-2, ue_class="pedestrian")
    ue3 = UserEquipment(UE_ID=3, x=-2, y=-5, ue_class="pedestrian")

    # Через синглтон указываем границы карты
    MapBorders(-1000, 1000, -1000, 1000)

    ue1.SET_MOBILITY_MODEL("DiagonalWalk", bs=bs, pause_time=200)
    ue2.SET_MOBILITY_MODEL("RandomWaypoint", pause_time=0)
    ue3.SET_MOBILITY_MODEL("GaussMarkov", alpha=0.15, boundary_threshold=100)

    # Создание модели радиоканала
    uma = UMaModel(bs=bs, cond_update_period=5)

    # Назначение пользователям модели радиоканала
    ue1.SET_CH_MODEL(uma)
    ue2.SET_CH_MODEL(uma)
    ue3.SET_CH_MODEL(uma)

    # Регистрация пользователей в базовой станции
    bs.REG_UE(ue1)
    bs.REG_UE(ue2)
    bs.REG_UE(ue3)

    # Добавление пользовательских устройств в коллекцию
    ue_collection.ADD_USER(ue1)
    ue_collection.ADD_USER(ue2)
    ue_collection.ADD_USER(ue3)

    # Имитация Full Buffer
    bs.ue_buffers[1].ADD_PACKET(Packet(size=inf, ue_id=1, creation_time=0), current_time=0)
    bs.ue_buffers[2].ADD_PACKET(Packet(size=inf, ue_id=2, creation_time=0), current_time=0)
    bs.ue_buffers[3].ADD_PACKET(Packet(size=inf, ue_id=3, creation_time=0), current_time=0)

    # Создание ресурсной сетки
    lte_grid = RES_GRID_LTE(bandwidth=bandwidth, num_frames=num_frames)

    # Создание планировщика
    scheduler = ProportionalFairScheduler(lte_grid, bs)

    # Основной цикл симуляции
    for current_time in range(update_interval, sim_duration + 1, update_interval):
        # Обновление глобальной переменной текущего времени
        # (Временное решение)
        GLOBALS.CURRENT_TIME = current_time

        # Обновление состояния пользователей
        ue_collection.UPDATE_ALL_USERS(
            current_time=current_time,
            update_interval=update_interval,
            bs_position=bs.position,
            bs_height=bs.height,
        )

        # Цикл для планирования ресурсов (по TTI)
        for tti in range(current_time - update_interval, current_time):
            # Подготовка данных для планировщика
            users = ue_collection.GET_USERS_FOR_SCHEDULER()

            # Планирование ресурсов
            sched_result = scheduler.schedule(tti, users)

            # Вывод статистики для каждого пользователя
            print_users_stats(
                ue_collection=ue_collection, tti=tti, bs=bs, sched_result=sched_result
            )

    # Визуализация передвижения пользователей
    visualize_users_mobility(
        ue_collection=ue_collection, bs=bs, x_min=-100, x_max=100, y_min=-100, y_max=100
    )

    # Визуализация SINR пользователей во времени
    # visualize_users_sinr(ue_collection=ue_collection,
    #                         sim_duration=sim_duration,
    #                         update_interval=update_interval)


if __name__ == "__main__":
    debug_simulation()
    # sim_with_ue_collection()
