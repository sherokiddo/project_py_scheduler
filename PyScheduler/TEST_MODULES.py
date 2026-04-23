import GLOBALS
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from collections import defaultdict
from BS_MODULE import BaseStation
from MOBILITY_MODEL import MapBorders
from SIMULATION_MANAGER import SimulationManager
from TRAFFIC_MODEL import PoissonModel
from UE_MODULE import UserEquipment, UECollection


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
    bs = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa", use_simple_buffer=False, enable_tdl=True, global_max=10e7)

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
    sim.set_scheduler(algorithm="FD_PF")

    # Настраиваем модели для каждого UE
    config = [
        {
        "ue_id": 1,
        "bearers": [
            {"model_type": "PereodicTraffic", "qci": 1, "packet_size": 320, "packet_interval_ms": 40}]
        },
        {
        "ue_id": 2,
        "bearers": [
            {"model_type": "PereodicTraffic", "qci": 4, "packet_size": 125, "packet_interval_ms": 1},
            {"model_type": "PereodicTraffic", "qci": 8, "packet_size": 1500, "packet_interval_ms": 1}]
        },
        {
        "ue_id": 3,
        "bearers": [
            {"model_type": "PereodicTraffic", "qci": 9, "packet_size": 1500, "packet_interval_ms": 1}]
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
            {"model_type": "PereodicTraffic", "qci": 2, "packet_size": 640, "packet_interval_ms": 20}]
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
    
    
def sim_with_manager_qos_2():
    """
    Пример запуска симуляции с использованием менеджера и QoS трафиком.

    """
    # =============================================================================
    #             НАСТРОЙКА БАЗОВОЙ СТАНЦИИ И КОЛЛЕКЦИИ ПОЛЬЗОВАТЕЛЕЙ
    # =============================================================================
    
    # Сброс и создание карты
    MapBorders._instance = None
    MapBorders(-2000, 2000, -2000, 2000)
    x_min, x_max, y_min, y_max = MapBorders().get_borders()

    # Создание и настройка базовой станции
    bs = BaseStation(x=0, y=0, bandwidth=5, ch_model_type="UMi", enable_tdl=True, use_simple_buffer=False, global_max=10e7)

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Установка сида
    GLOBALS.SEED = 42
    
    if GLOBALS.SEED is not None:
        np.random.seed(GLOBALS.SEED)

    # Генерация заданного числа UE в коллекцию
    ue1 = UserEquipment(UE_ID=1, x=0, y=2000, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=2000, y=0, ue_class="pedestrian")
    ue3 = UserEquipment(UE_ID=3, x=0, y=-300, ue_class="pedestrian")
    ue4 = UserEquipment(UE_ID=4, x=-300, y=0, ue_class="pedestrian")
    
# =============================================================================
#     ue5 = UserEquipment(UE_ID=5, x=0, y=1500, ue_class="pedestrian")
#     ue6 = UserEquipment(UE_ID=6, x=1500, y=0, ue_class="pedestrian")
#     ue7 = UserEquipment(UE_ID=7, x=0, y=-1500, ue_class="pedestrian")
#     ue8 = UserEquipment(UE_ID=8, x=-1500, y=0, ue_class="pedestrian")
# =============================================================================
    
    ue_collection.ADD_USER(ue1)
    ue_collection.ADD_USER(ue2)
    ue_collection.ADD_USER(ue3)
    ue_collection.ADD_USER(ue4)
    
# =============================================================================
#     ue_collection.ADD_USER(ue5)
#     ue_collection.ADD_USER(ue6)
#     ue_collection.ADD_USER(ue7)
#     ue_collection.ADD_USER(ue8)
# =============================================================================

    # Установка модели передвижения для всех пользователей коллекции
    ue_collection.SET_MOBILITY_MODEL(model="RandomWaypoint", pause_time=1000000)

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
    sim.set_scheduler(algorithm="FD_PF")

    # Настраиваем модели для каждого UE
    config = [
        {
        "ue_id": 1,
        "bearers": [
            {"model_type": "PereodicTraffic", "qci": 1, "packet_size": 160, "packet_interval_ms": 20}]
        },
        {
        "ue_id": 2,
        "bearers": [
            {"model_type": "PereodicTraffic", "qci": 2, "packet_size": 32, "packet_interval_ms": 1}]
        },
        {
        "ue_id": 3,
        "bearers": [
            {"model_type": "BitrateTraffic", "qci": 6, "packet_size": 1500, "bitrate_bps": 12000000}]
        },
        {
        "ue_id": 4,
        "bearers": [
            {"model_type": "BitrateTraffic", "qci": 9, "packet_size": 1500, "bitrate_bps": 12000000}]
        },
# =============================================================================
#         {
#         "ue_id": 5,
#         "bearers": [
#             {"model_type": "PereodicTraffic", "qci": 1, "packet_size": 160, "packet_interval_ms": 20}]
#         },
#         {
#         "ue_id": 6,
#         "bearers": [
#             {"model_type": "PereodicTraffic", "qci": 2, "packet_size": 640, "packet_interval_ms": 20}]
#         },
#         {
#         "ue_id": 7,
#         "bearers": [
#             {"model_type": "BitrateTraffic", "qci": 6, "packet_size": 1500, "bitrate_bps": 6000000}]
#         },
#         {
#         "ue_id": 8,
#         "bearers": [
#             {"model_type": "BitrateTraffic", "qci": 9, "packet_size": 1500, "bitrate_bps": 6000000}]
#         },
# =============================================================================
    ]
    
    sim.setup_traffic_profiles(config)

    # Установка длительности симуляции
    sim.set_sim_duration(30000)

    # Включение verbose логирования. Для вывода всех логов в файл нужно
    # поставить флаг to_file=True.
    sim.enable_verbose_log(to_file=True)

    # Установка менеджера статистики
    sim.set_stats_manager(
        enabled=True,
        collect_interval=1,
        history_max_len=30000,
        scheduler_level="full",
        amc_level="full",
        pdcch_level="basic",
        file_prefix="emp_stats",
    )

    # Запуск симуляции
    sim.start_simulation()
    
    visualize_users_mobility(
        ue_collection=ue_collection,
        bs=bs,
        x_min=x_min,
        x_max=x_max,
        y_min=x_min,
        y_max=y_max)


def _module_dir() -> str:
    """Абсолютный путь к директории с TEST_MODULES.py."""
    return os.path.dirname(os.path.abspath(__file__))


def _safe_float(value) -> float:
    """Преобразовать строку из CSV в float, понимая и ',' и '.' как разделитель."""
    if value is None:
        return 0.0

    text = str(value).strip()
    if text == "":
        return 0.0

    return float(text.replace(",", "."))


def _format_csv_number(value: float) -> str:
    """Компактное форматирование числа для итогового CSV."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))

    return f"{value:.6f}".rstrip("0").rstrip(".")


def _copy_stats_snapshot(run_dir: str, seed: int) -> None:
    """
    Сохранить результаты одного прогона в отдельную папку, не меняя
    итоговые emp_stats.csv/emp_stats_detailed.csv.
    """
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    base_dir = _module_dir()
    files_to_copy = [
        "emp_stats.csv",
        "emp_stats_detailed.csv",
        "output.txt",
    ]

    for filename in files_to_copy:
        src = os.path.join(base_dir, filename)
        dst = os.path.join(seed_dir, filename)
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def _aggregate_seed_csvs_to_emp_stats(run_dir: str, seeds) -> None:
    """
    Собрать усреднённые CSV-файлы в формате, совместимом с Tput_plots.py.

    Пишем только те поля, которые реально используются в Tput_plots.py:
      - emp_stats.csv:
            tti, dl_transmitted_bits_sum_tti
      - emp_stats_detailed.csv:
            tti, ue_id, qci,
            bits_transmitted_per_qci,
            packets_extracted_per_qci,
            packets_extracted_late_per_qci,
            packets_added_per_qci,
            packets_expired_per_qci
    """
    base_dir = _module_dir()
    seed_list = list(seeds)
    total_runs = len(seed_list)

    if total_runs == 0:
        raise ValueError("Seed list cannot be empty for aggregation.")

    cell_sums = defaultdict(float)
    detailed_sums = defaultdict(
        lambda: {
            "bits_transmitted_per_qci": 0.0,
            "packets_extracted_per_qci": 0.0,
            "packets_extracted_late_per_qci": 0.0,
            "packets_added_per_qci": 0.0,
            "packets_expired_per_qci": 0.0,
        }
    )

    for seed in seed_list:
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        cell_path = os.path.join(seed_dir, "emp_stats.csv")
        detailed_path = os.path.join(seed_dir, "emp_stats_detailed.csv")

        if not os.path.exists(cell_path):
            raise FileNotFoundError(f"Missing per-seed cell stats file: {cell_path}")
        if not os.path.exists(detailed_path):
            raise FileNotFoundError(f"Missing per-seed detailed stats file: {detailed_path}")

        with open(cell_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                tti = int(row["tti"])
                cell_sums[tti] += _safe_float(row.get("dl_transmitted_bits_sum_tti", 0))

        with open(detailed_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                qci = str(row.get("qci", "")).strip()
                if qci == "":
                    continue

                key = (
                    int(row["tti"]),
                    int(row["ue_id"]),
                    qci,
                )
                detailed_sums[key]["bits_transmitted_per_qci"] += _safe_float(
                    row.get("bits_transmitted_per_qci", 0)
                )
                detailed_sums[key]["packets_extracted_per_qci"] += _safe_float(
                    row.get("packets_extracted_per_qci", 0)
                )
                detailed_sums[key]["packets_extracted_late_per_qci"] += _safe_float(
                    row.get("packets_extracted_late_per_qci", 0)
                )
                detailed_sums[key]["packets_added_per_qci"] += _safe_float(
                    row.get("packets_added_per_qci", 0)
                )
                detailed_sums[key]["packets_expired_per_qci"] += _safe_float(
                    row.get("packets_expired_per_qci", 0)
                )

    avg_cell_path = os.path.join(base_dir, "emp_stats.csv")
    with open(avg_cell_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tti", "dl_transmitted_bits_sum_tti"],
            delimiter=";",
        )
        writer.writeheader()

        for tti in sorted(cell_sums):
            writer.writerow(
                {
                    "tti": tti,
                    "dl_transmitted_bits_sum_tti": _format_csv_number(
                        cell_sums[tti] / total_runs
                    ),
                }
            )

    avg_detailed_path = os.path.join(base_dir, "emp_stats_detailed.csv")
    with open(avg_detailed_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tti",
                "ue_id",
                "qci",
                "bits_transmitted_per_qci",
                "packets_extracted_per_qci",
                "packets_extracted_late_per_qci",
                "packets_added_per_qci",
                "packets_expired_per_qci",
            ],
            delimiter=";",
        )
        writer.writeheader()

        def _sort_key(item):
            tti, ue_id, qci = item[0]
            try:
                qci_sort = int(qci)
            except ValueError:
                qci_sort = qci
            return (tti, ue_id, qci_sort)

        for key, sums in sorted(detailed_sums.items(), key=_sort_key):
            tti, ue_id, qci = key
            writer.writerow(
                {
                    "tti": tti,
                    "ue_id": ue_id,
                    "qci": qci,
                    "bits_transmitted_per_qci": _format_csv_number(
                        sums["bits_transmitted_per_qci"] / total_runs
                    ),
                    "packets_extracted_per_qci": _format_csv_number(
                        sums["packets_extracted_per_qci"] / total_runs
                    ),
                    "packets_extracted_late_per_qci": _format_csv_number(
                        sums["packets_extracted_late_per_qci"] / total_runs
                    ),
                    "packets_added_per_qci": _format_csv_number(
                        sums["packets_added_per_qci"] / total_runs
                    ),
                    "packets_expired_per_qci": _format_csv_number(
                        sums["packets_expired_per_qci"] / total_runs
                    ),
                }
            )


def aggregate_existing_benchmark_runs(
    algorithm: str,
    seeds=None,
    scenario: str = None,
) -> None:
    """
    Пересобрать общие emp_stats.csv и emp_stats_detailed.csv из уже существующих
    per-seed результатов без повторного запуска симуляции.

    Args:
        algorithm: имя папки алгоритма внутри benchmark_runs, например:
                   "QosAware", "FD_PF", "FD_BCQI", "RoundRobin"
        seeds:     список seed для включения в агрегацию. Если None, будут взяты
                   все найденные папки вида seed_<N>.
        scenario:  опциональная папка сценария внутри benchmark_runs. Например,
                   для broken_lcm_multi_bearer путь будет:
                   benchmark_runs/broken_lcm_multi_bearer/<algorithm>/seed_*
    """
    if scenario is None:
        run_root = os.path.join(_module_dir(), "benchmark_runs", algorithm)
    else:
        run_root = os.path.join(_module_dir(), "benchmark_runs", scenario, algorithm)

    if not os.path.isdir(run_root):
        raise FileNotFoundError(
            f"Benchmark directory does not exist: {run_root}"
        )

    if seeds is None:
        seed_list = []
        for entry in sorted(os.listdir(run_root)):
            entry_path = os.path.join(run_root, entry)
            if not os.path.isdir(entry_path):
                continue
            if not entry.startswith("seed_"):
                continue
            seed_text = entry.removeprefix("seed_")
            if seed_text.isdigit():
                seed_list.append(int(seed_text))
    else:
        seed_list = list(seeds)

    if len(seed_list) == 0:
        raise ValueError(
            f"No seed directories were found for aggregation in: {run_root}"
        )

    if scenario is None:
        print(f"[AGGREGATE] Rebuilding emp_stats for algorithm: {algorithm}")
    else:
        print(f"[AGGREGATE] Rebuilding emp_stats for scenario: {scenario}, algorithm: {algorithm}")
    print(f"[AGGREGATE] Seeds: {seed_list}")
    _aggregate_seed_csvs_to_emp_stats(run_root, seed_list)
    print("[AGGREGATE] Aggregated emp_stats.csv and emp_stats_detailed.csv have been generated.")


def benchmark_qosaware_vs_baselines(
    algorithm: str = "QosAware",
    sim_duration: int = 30000,
    seed: int = 42,
    seeds=None,
    show_plots: bool = False,
):
    """
    Воспроизводимый benchmark-сценарий для сравнения QoS-aware планировщика
    с baseline-алгоритмами.

    Важно: за один запуск выполняется только один планировщик.
    Для сравнения нужно перезапускать сценарий с разными значениями algorithm.

    Если передан параметр seeds, сценарий будет выполнен несколько раз
    с разными сидaми, после чего в корне PyScheduler будут собраны
    усреднённые emp_stats.csv и emp_stats_detailed.csv, совместимые
    с Tput_plots.py.

    Recommended algorithms:
        - "QosAware"
        - "FD_PF"
        - "FD_BCQI"
        - "RoundRobin"

    Сценарий специально создаёт перегрузку соты и конфликт интересов:
        - 4 GBR bearer'а (QCI 1-4) с разными delay/bitrate требованиями
        - 4 non-GBR bearer'а (QCI 6/9), которые конкурируют за остаточную ёмкость
        - смесь edge/mid/near UE по расстоянию до BS
    """
    allowed_algorithms = {"QosAware", "FD_PF", "FD_BCQI", "RoundRobin"}
    if algorithm not in allowed_algorithms:
        valid = ", ".join(sorted(allowed_algorithms))
        raise ValueError(
            f"Unsupported benchmark algorithm '{algorithm}'. "
            f"Use one of: {valid}"
        )

    seed_list = list(seeds) if seeds is not None else [seed]
    if len(seed_list) == 0:
        raise ValueError("Parameter 'seeds' cannot be empty.")

    run_root = os.path.join(_module_dir(), "benchmark_runs", algorithm)
    if os.path.isdir(run_root):
        shutil.rmtree(run_root)
    os.makedirs(run_root, exist_ok=True)

    print(f"[BENCHMARK] Scheduler: {algorithm}")
    print(f"[BENCHMARK] Seeds: {seed_list}")
    print(f"[BENCHMARK] Duration: {sim_duration} TTI")
    print("[BENCHMARK] Traffic mix: GBR(QCI1-4) + BE(QCI6/QCI9) under controlled overload")

    last_context = None

    for run_idx, current_seed in enumerate(seed_list, start=1):
        MapBorders._instance = None
        MapBorders(-2000, 2000, -2000, 2000)
        x_min, x_max, y_min, y_max = MapBorders().get_borders()

        GLOBALS.SEED = current_seed
        if GLOBALS.SEED is not None:
            np.random.seed(GLOBALS.SEED)

        bs = BaseStation(
            x=0,
            y=0,
            bandwidth=5,
            ch_model_type="UMi",
            enable_tdl=True,
            use_simple_buffer=False,
            global_max=10e7,
        )

        ue_collection = UECollection()

        # Две edge GBR-UE, две middle GBR-UE, две near BE-UE и две middle BE-UE.
        # Такой расклад заставляет алгоритмы балансировать между delay/QoS и raw CQI.
        ue_positions = [
            (1, 0, 1600),
            (2, 1600, 0),
            (3, 0, -800),
            (4, -800, 0),
            (5, 220, 180),
            (6, -220, -180),
            (7, 900, 250),
            (8, -900, -250),
        ]

        for ue_id, x_pos, y_pos in ue_positions:
            ue_collection.ADD_USER(
                UserEquipment(
                    UE_ID=ue_id,
                    x=x_pos,
                    y=y_pos,
                    ue_class="pedestrian",
                )
            )

        # Практически статический сценарий: канал обновляется, но траектории
        # не "гуляют", поэтому межалгоритмное сравнение чище.
        ue_collection.SET_MOBILITY_MODEL(model="RandomWaypoint", pause_time=1000000)
        ue_collection.REG_USERS_TO_BS(bs)

        sim = SimulationManager()
        sim.set_base_station(bs)
        sim.set_ue_collection(ue_collection)
        sim.set_scheduler(algorithm=algorithm)

        config = [
            {
                "ue_id": 1,
                "bearers": [
                    {"model_type": "PereodicTraffic", "qci": 1, "packet_size": 160, "packet_interval_ms": 20}
                ],
            },
            {
                "ue_id": 2,
                "bearers": [
                    {"model_type": "PereodicTraffic", "qci": 2, "packet_size": 32, "packet_interval_ms": 1}
                ],
            },
            {
                "ue_id": 3,
                "bearers": [
                    {"model_type": "PereodicTraffic", "qci": 3, "packet_size": 64, "packet_interval_ms": 1}
                ],
            },
            {
                "ue_id": 4,
                "bearers": [
                    {"model_type": "PereodicTraffic", "qci": 4, "packet_size": 125, "packet_interval_ms": 1}
                ],
            },
            {
                "ue_id": 5,
                "bearers": [
                    {"model_type": "BitrateTraffic", "qci": 6, "packet_size": 1500, "bitrate_bps": 3000000}
                ],
            },
            {
                "ue_id": 6,
                "bearers": [
                    {"model_type": "BitrateTraffic", "qci": 6, "packet_size": 1500, "bitrate_bps": 3000000}
                ],
            },
            {
                "ue_id": 7,
                "bearers": [
                    {"model_type": "BitrateTraffic", "qci": 9, "packet_size": 1500, "bitrate_bps": 4000000}
                ],
            },
            {
                "ue_id": 8,
                "bearers": [
                    {"model_type": "BitrateTraffic", "qci": 9, "packet_size": 1500, "bitrate_bps": 4000000}
                ],
            },
        ]

        sim.setup_traffic_profiles(config)
        sim.set_sim_duration(sim_duration)
        sim.set_mobility_interval(500)
        sim.set_channel_interval(10)
        # sim.enable_verbose_log(to_file=True)

        sim.set_stats_manager(
            enabled=True,
            collect_interval=1,
            history_max_len=sim_duration,
            scheduler_level="full",
            amc_level="full",
            pdcch_level="basic",
            file_prefix="emp_stats",
        )

        print(
            f"[BENCHMARK] Run {run_idx}/{len(seed_list)} "
            f"for {algorithm}, seed={current_seed}"
        )
        sim.start_simulation()
        _copy_stats_snapshot(run_root, current_seed)

        last_context = {
            "ue_collection": ue_collection,
            "bs": bs,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "sim_duration": sim.sim_config.sim_duration,
        }

    _aggregate_seed_csvs_to_emp_stats(run_root, seed_list)
    print("[BENCHMARK] Averaged emp_stats.csv and emp_stats_detailed.csv have been generated.")
    print(f"[BENCHMARK] Per-seed raw CSV files are stored in: {run_root}")

    if show_plots and len(seed_list) == 1 and last_context is not None:
        visualize_users_mobility(
            ue_collection=last_context["ue_collection"],
            bs=last_context["bs"],
            x_min=last_context["x_min"],
            x_max=last_context["x_max"],
            y_min=last_context["y_min"],
            y_max=last_context["y_max"],
        )
        visualize_users_sinr(
            ue_collection=last_context["ue_collection"],
            sim_duration=last_context["sim_duration"],
            update_interval=10,
        )
    elif show_plots and len(seed_list) > 1:
        print("[BENCHMARK] show_plots=True is ignored for multi-seed mode.")


def broken_lcm_multi_bearer(
    algorithm: str = "FD_BCQI",
    sim_duration: int = 10000,
    seed: int = 42,
    seeds=None,
    show_plots: bool = False,
) -> None:
    """
    Diagnostic scenario for the current logical_channel_multiplexing behavior.

    One UE has several logical channels:
      - three small GBR-like flows
      - one large QCI9 flow

    If the TB is split almost evenly between channels, the large flow should
    lose a disproportionate amount of capacity even though the small flows
    need only a small fraction of the transport block.
    """
    allowed_algorithms = {"QosAware", "FD_PF", "FD_BCQI", "RoundRobin"}
    if algorithm not in allowed_algorithms:
        valid = ", ".join(sorted(allowed_algorithms))
        raise ValueError(
            f"Unsupported LCM diagnostic algorithm '{algorithm}'. "
            f"Use one of: {valid}"
        )

    seed_list = list(seeds) if seeds is not None else [seed]
    if len(seed_list) == 0:
        raise ValueError("Parameter 'seeds' cannot be empty.")

    scenario_name = "broken_lcm_multi_bearer"
    run_root = os.path.join(_module_dir(), "benchmark_runs", scenario_name, algorithm)
    if os.path.isdir(run_root):
        shutil.rmtree(run_root)
    os.makedirs(run_root, exist_ok=True)

    print(f"[LCM-BENCHMARK] Scenario: {scenario_name}")
    print(f"[LCM-BENCHMARK] Scheduler: {algorithm}")
    print(f"[LCM-BENCHMARK] Seeds: {seed_list}")
    print(f"[LCM-BENCHMARK] Duration: {sim_duration} TTI")
    print("[LCM-BENCHMARK] UE1 has 3 small flows + 1 heavy QCI9 flow")

    last_context = None

    for run_idx, current_seed in enumerate(seed_list, start=1):
        MapBorders._instance = None
        MapBorders(-500, 500, -500, 500)
        x_min, x_max, y_min, y_max = MapBorders().get_borders()

        GLOBALS.SEED = current_seed
        if GLOBALS.SEED is not None:
            np.random.seed(GLOBALS.SEED)

        bs = BaseStation(
            x=0,
            y=0,
            bandwidth=5,
            ch_model_type="UMi",
            enable_tdl=True,
            use_simple_buffer=False,
            global_max=10e7,
        )

        ue_collection = UECollection()
        ue_collection.ADD_USER(
            UserEquipment(
                UE_ID=1,
                x=80,
                y=0,
                ue_class="pedestrian",
            )
        )

        # Keep the UE near the BS and almost static, so the experiment isolates
        # intra-UE multiplexing rather than inter-UE scheduling or mobility.
        ue_collection.SET_MOBILITY_MODEL(model="RandomWaypoint", pause_time=1000000)
        ue_collection.REG_USERS_TO_BS(bs)

        sim = SimulationManager()
        sim.set_base_station(bs)
        sim.set_ue_collection(ue_collection)
        sim.set_scheduler(
            algorithm=algorithm,
            enable_window=False,
            max_dl_ue_tti=None,
        )

        config = [
            {
                "ue_id": 1,
                "bearers": [
                    {
                        "model_type": "PereodicTraffic",
                        "qci": 1,
                        "packet_size": 160,
                        "packet_interval_ms": 20,
                    },
                    {
                        "model_type": "PereodicTraffic",
                        "qci": 2,
                        "packet_size": 32,
                        "packet_interval_ms": 1,
                    },
                    {
                        "model_type": "PereodicTraffic",
                        "qci": 3,
                        "packet_size": 64,
                        "packet_interval_ms": 1,
                    },
                    {
                        "model_type": "BitrateTraffic",
                        "qci": 9,
                        "packet_size": 1500,
                        "bitrate_bps": 12000000,
                    },
                ],
            }
        ]

        sim.setup_traffic_profiles(config)
        sim.set_sim_duration(sim_duration)
        sim.set_mobility_interval(500)
        sim.set_channel_interval(10)

        sim.set_stats_manager(
            enabled=True,
            collect_interval=1,
            history_max_len=sim_duration,
            scheduler_level="full",
            amc_level="full",
            pdcch_level="basic",
            file_prefix="emp_stats",
        )

        print(
            f"[LCM-BENCHMARK] Run {run_idx}/{len(seed_list)} "
            f"for {algorithm}, seed={current_seed}"
        )
        sim.start_simulation()
        _copy_stats_snapshot(run_root, current_seed)

        last_context = {
            "ue_collection": ue_collection,
            "bs": bs,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "sim_duration": sim.sim_config.sim_duration,
        }

    _aggregate_seed_csvs_to_emp_stats(run_root, seed_list)
    print("[LCM-BENCHMARK] Averaged emp_stats.csv and emp_stats_detailed.csv have been generated.")
    print(f"[LCM-BENCHMARK] Per-seed raw CSV files are stored in: {run_root}")

    if show_plots and len(seed_list) == 1 and last_context is not None:
        visualize_users_mobility(
            ue_collection=last_context["ue_collection"],
            bs=last_context["bs"],
            x_min=last_context["x_min"],
            x_max=last_context["x_max"],
            y_min=last_context["y_min"],
            y_max=last_context["y_max"],
        )
        visualize_users_sinr(
            ue_collection=last_context["ue_collection"],
            sim_duration=last_context["sim_duration"],
            update_interval=10,
        )
    elif show_plots and len(seed_list) > 1:
        print("[LCM-BENCHMARK] show_plots=True is ignored for multi-seed mode.")


if __name__ == "__main__":
    # sim_with_manager()
    # sim_with_manager_qos()
    # sim_with_manager_qos_2()
# =============================================================================
#     selected_algorithm = "FD_BCQI"
#     selected_seeds = list(range(0, 51))
#     # selected_seeds = [42]
#     benchmark_qosaware_vs_baselines(
#         algorithm=selected_algorithm,
#         seeds=selected_seeds,
#     )
# =============================================================================
    # aggregate_existing_benchmark_runs("FD_BCQI")
    broken_lcm_multi_bearer(
    algorithm="QosAware",
    seeds=[42],
    sim_duration=10000,
)

