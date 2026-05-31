import csv
import random
import GLOBALS
import matplotlib.pyplot as plt
import numpy as np
from BS_MODULE import BaseStation
from MOBILITY_MODEL import MapBorders, MobilityInterface
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
    Тестовый стенд для проверки работоспособности моделей канала

    """
    # =============================================================================
    #             НАСТРОЙКА БАЗОВОЙ СТАНЦИИ И КОЛЛЕКЦИИ ПОЛЬЗОВАТЕЛЕЙ
    # =============================================================================

    # Сброс и создание карты
    MapBorders._instance = None
    MapBorders(-500, 500, -500, 500)
    x_min, x_max, y_min, y_max = MapBorders().get_borders()

    # Создание и настройка базовой станции
    bs1 = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa", enable_tdl=True)

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Установка сида
    GLOBALS.SEED = 42

    if GLOBALS.SEED is not None:
        np.random.seed(GLOBALS.SEED)   # покрывает numpy

    # Генерация заданного числа UE в коллекцию
    ue_collection.ADD_RANDOM_USERS(num_ue=5,
                                   x_min=x_min,
                                   x_max=x_max,
                                   y_min=y_min,
                                   y_max=y_max,
                                   ue_class="random")

    #TODO: Объединить MapBorders c рандомизацией UE

    # Установка модели передвижения для всех пользователей коллекции
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint", bs=bs1, pause_time = 0)

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
        algorithm="RoundRobin",
        max_dl_ue_tti=None,
        pcfich=2,
        enable_window=False,
        window_size=4,
        max_dl_cce_allowance=None,
    )

    # Настраиваем модели для каждого UE
    for ue in ue_collection.GET_ALL_USERS():
        #print(f"Setup UE: {ue.UE_ID} traffic (SimpleGenerator)")
        sim.setup_ue_traffic(
            ue_id=ue.UE_ID,
            model_type="Poisson",
            packet_rate=5000,
        )

    # Установка длительности симуляции
    sim.set_sim_duration(50000)
    sim.set_mobility_interval(500)
    sim.set_channel_interval(10)

    # Включение verbose логирования. Для вывода всех логов в файл нужно
    # поставить флаг to_file=True.
    sim.enable_verbose_log(to_file=True)

    # Установка менеджера статистики
    sim.set_stats_manager(
        enabled=True,  # Включить сбор
        collect_interval=1,  # Собирать каждые n TTI
        history_max_len=50000,
        scheduler_level="full",  # Scheduler: только агрегированные метрики
        amc_level="full",  # AMC: total throughput + avg bits/RB
        pdcch_level="full",  # PDCCH: отключен (можно включить "basic")
        file_prefix="emp_stats",  # Префикс файла: lte_stats.csv
    )

    # Запуск симуляции
    sim.start_simulation()

    # Визуализация передвижения пользователей
    visualize_users_mobility(
        ue_collection=ue_collection,
        bs=bs1,
        x_min=x_min,
        x_max=x_max,
        y_min=x_min,
        y_max=y_max)
    # Визуализация SINR пользователей во времени
    visualize_users_sinr(
        ue_collection=ue_collection, sim_duration=sim.sim_config.sim_duration, update_interval=10
    )


def sim_with_manager_qos():
    """
    Пример запуска симуляции с использованием менеджера и QoS трафиком.

    """
    # =============================================================================
    #             НАСТРОЙКА БАЗОВОЙ СТАНЦИИ И КОЛЛЕКЦИИ ПОЛЬЗОВАТЕЛЕЙ
    # =============================================================================
    
    # Сброс и создание карты
    MapBorders._instance = None
    MapBorders(-1000, 1000, -1000, 1000)
    x_min, x_max, y_min, y_max = MapBorders().get_borders()

    # Создание и настройка базовой станции
    bs = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa", use_simple_buffer=False)

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Установка сида
    GLOBALS.SEED = 42
    
    if GLOBALS.SEED is not None:
        np.random.seed(GLOBALS.SEED)

    # Генерация заданного числа UE в коллекцию
    ue_collection.ADD_RANDOM_USERS(num_ue=5,
                                   x_min=x_min,
                                   x_max=x_max,
                                   y_min=y_min,
                                   y_max=y_max,
                                   ue_class="random")

    # Установка модели передвижения для всех пользователей коллекции
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint")

    # Регистрация всех пользователей коллекции в базовой станции
    ue_collection.REG_USERS_TO_BS(bs)

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
    sim.set_scheduler(algorithm="ProportionalFair")

    # Настраиваем модели для каждого UE
    config = [
        {
        "ue_id": 1,
        "bearers": [
            {"model_type": "Poisson", "qci": 1, "packet_rate": 100},
            {"model_type": "Poisson", "qci": 5, "packet_rate": 200},
            {"model_type": "Poisson", "qci": 7, "packet_rate": 1500}]
        },
        {
        "ue_id": 2,
        "bearers": [
            {"model_type": "Poisson", "qci": 4, "packet_rate": 500},
            {"model_type": "Poisson", "qci": 8, "packet_rate": 1000}]
        },
        {
        "ue_id": 3,
        "bearers": [
            {"model_type": "Poisson", "packet_rate": 1000}]
        },
        {
        "ue_id": 4,
        "bearers": [
            {"model_type": "Poisson", "qci": 6, "packet_rate": 1000},
            {"model_type": "Poisson", "qci": 7, "packet_rate": 1500},
            {"model_type": "Poisson", "qci": 8, "packet_rate": 1200}]
        },
        {
        "ue_id": 5,
        "bearers": [
            {"model_type": "Poisson", "qci": 1, "packet_rate": 100}]
        },
    ]
    
    sim.setup_traffic_profiles(config)

    # Установка длительности симуляции
    sim.set_sim_duration(5000)

    # Включение verbose логирования. Для вывода всех логов в файл нужно
    # поставить флаг to_file=True.
    sim.enable_verbose_log(to_file=True)

    # Установка менеджера статистики
    sim.set_stats_manager(
        enabled=True,
        collect_interval=1,
        history_max_len=5000,
        scheduler_level="full",
        amc_level="full",
        pdcch_level="basic",
        file_prefix="emp_stats",
    )

    # Запуск симуляции
    sim.start_simulation()

def test_pure_mobility():
    """
    Тест моделей мобильности (без трафика и планировщика).
    Настроен для валидации с NS-3.
    """
    config = {
        'mobility_model': 'RandomDirection',  # 'RandomWalk', 'RandomWaypoint', 'RandomDirection', 'GaussMarkov'
        'map_min': 0.0, # Минимальная граница карты
        'map_max': 150.0, # Максимальная граница карты
        'sim_time': 150000,     # Общее время в мс
        'upd_interval': 1000,   # Шаг в мс
        'seed': 45,
        'run': 7,
        'num_ue': 5, # Количество абонентов
        'use_list_allocator': True, # True — расставить пользователей по координатам из списка positions_list, False — разбросать пользователей случайно в границах карты
        'positions_list': [
            (10.0, 10.0, 0.0),
            (20.0, 20.0, 0.0),
            (50.0, 50.0, 0.0),
            (100.0, 100.0, 0.0),
            (140.0, 140.0, 0.0)
        ],
        'ue_class': 'custom',     # 'pedestrian', 'cyclist', 'car', 'random', 'custom'
        'velocity_min': 5.0,      # Применяется только если ue_class == 'custom'
        'velocity_max': 10.0,     # Применяется только если ue_class == 'custom'
        'mean_velocity': 7.5,     # Применяется только если ue_class == 'custom'
        'pause_time': 3.0, # Время паузы
        'rw_mode': 'Time', # Режиме смены направления. Либо 'Time' либо 'Distance'
        'rw_mode_time': 5.0, # Каждые 5 секунд будет менять направление
        'rw_mode_distance': 15.0, # Каждые 15 метров будет менять направление
        'alpha': 0.98, # Параметр для Gauss-Markov
        'gm_time_step': 0.5, # Внутренний шаг времени в секундах для пересчета формул Гаусса-Маркова.
        'z_min': 0.0, # Минимальная граница карты по оси Z
        'z_max': 200.0, # Максимальная граница карты по оси Z
        'gm_pitch_min': 0.0, # Минимальный угол в радианах для движения 3D-моделей
        'gm_pitch_max': 1.57, # Максимальный угол в радианах для движения 3D-моделей
        'save_plot': 0  # 1 - сохранять PNG, 0 - только показать на экране
    }

    seed = config['seed']
    run = config['run']
    GLOBALS.SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    print(f"\n[*] Запуск теста мобильности | Seed: {seed} | Run: {run}")

    c_min = config['map_min']
    c_max = config['map_max']
    MapBorders(c_min, c_max, c_min, c_max)

    ue_collection = UECollection()
    num_ue = config['num_ue']

    ue_collection.ADD_RANDOM_USERS(
        num_ue=num_ue, x_min=0, x_max=0, y_min=0, y_max=0,
        ue_class=config['ue_class'],
        velocity_min=config['velocity_min'],
        velocity_max=config['velocity_max'],
        mean_velocity=config['mean_velocity']
    )
    all_ues = ue_collection.GET_ALL_USERS()

    use_list_allocator = config['use_list_allocator']
    if use_list_allocator:
        ue_collection.ALLOCATE_POSITIONS_FROM_LIST(config['positions_list'])
    else:
        ue_collection.ALLOCATE_POSITIONS_RANDOM(
            x_min=c_min, x_max=c_max, y_min=c_min, y_max=c_max,
            z_min=config['z_min'], z_max=config['z_max'],
            seed=seed, run=run
        )

    streams_per_model = {
        "RandomWalk": 2, "RandomWaypoint": 4,
        "RandomDirection": 3, "GaussMarkov": 6,
    }
    current_model = config['mobility_model']
    stride = streams_per_model.get(current_model, 2)

    for i, ue in enumerate(all_ues):
        ue.coordinates = [ue.position]

        stream_idx = (i * stride) if use_list_allocator else (2 + (i * stride))

        model_params = {
            'ue': ue, 'seed': seed, 'run': run, 'base_idx': stream_idx,
        }

        if current_model in ['RandomWaypoint', 'RandomDirection']:
            model_params['pause_time'] = config['pause_time']
        elif current_model == 'RandomWalk':
            model_params['mode'] = config['rw_mode']
            model_params['mode_time'] = config['rw_mode_time']
            model_params['mode_distance'] = config['rw_mode_distance']
        elif current_model == 'GaussMarkov':
            model_params['alpha'] = config['alpha']
            model_params['mode_time'] = config['gm_time_step']
            model_params['z_min'] = config['z_min']
            model_params['z_max'] = config['z_max']
            model_params['pitch_min'] = config['gm_pitch_min']
            model_params['pitch_max'] = config['gm_pitch_max']

        ue.mobility_model = MobilityInterface.create(current_model, **model_params)
        ue.mobility_model.update(0)

    sim_time = config['sim_time']
    update_interval = config['upd_interval']

    print(f"[*] Модель: {current_model} | Класс: {config['ue_class']} | Время: {sim_time} мс")

    for current_time in range(0, sim_time, update_interval):
        for ue in all_ues:
            new_pos, new_vel, new_dir = ue.mobility_model.update(time_ms=update_interval)
            ue.position = new_pos
            ue.velocity = new_vel
            ue.direction = new_dir
            ue.coordinates.append(new_pos)

    csv_filename = f"pure_mobility_{current_model}_seed{seed}_run{run}_ue{num_ue}_time{sim_time}.csv"
    import csv
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Time_s", "UE_ID", "X", "Y"])
        step_s = update_interval / 1000.0
        num_steps = len(all_ues[0].coordinates)
        for step_idx in range(num_steps):
            time_s = step_idx * step_s
            for i, ue in enumerate(all_ues):
                coords = ue.coordinates[step_idx]
                writer.writerow([time_s, i, f"{coords[0]:.6f}", f"{coords[1]:.6f}"])
    print(f"[*] Лог сохранён: {csv_filename}")

    if config['save_plot'] in [0, 1]:
        plt.figure(figsize=(8, 8))
        for i, ue in enumerate(all_ues):
            coords = np.array([c[:2] for c in ue.coordinates])
            plt.plot(coords[:, 0], coords[:, 1], label=f"UE {i} ({ue.ue_class})", linewidth=1.5)
            plt.scatter(coords[0, 0], coords[0, 1], color='green', s=50, label='Start' if i == 0 else "", zorder=5)
            plt.scatter(coords[-1, 0], coords[-1, 1], color='red', s=50, label='End' if i == 0 else "", zorder=5)

        plt.xlim(c_min, c_max)
        plt.ylim(c_min, c_max)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"Траектория: {current_model} (Seed {seed}, Run {run})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if config['save_plot'] == 1:
            plt.savefig(f"trajectory_{current_model}.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    # sim_with_manager()
    test_pure_mobility()
    #sim_with_manager_qos()
