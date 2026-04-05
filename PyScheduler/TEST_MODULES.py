import math

import numpy as np
 
import GLOBALS
from BS_MODULE import BaseStation
from CHANNEL_MODEL import UMaModel
from MOBILITY_MODEL import MapBorders
from RES_GRID import RES_GRID_LTE
from SCHEDULER import ProportionalFairScheduler
from SIMULATION_MANAGER import SimulationManager
from UE_MODULE import Packet, UECollection, UserEquipment


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
    """
    import matplotlib.pyplot as plt

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
    """
    import matplotlib.pyplot as plt

    tti_range = np.arange(0, sim_duration)

    plt.figure(figsize=(12, 6))
    plt.title("График SINR во времени для всех пользователей")
    plt.xlabel("TTI (мс)")
    plt.ylabel("SINR (dB)")

    for ue in ue_collection.GET_ALL_USERS():
        sinr_values = ue.SINR_values
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
        print(f"\tРазмер буфера       : {bs.buffer_manager.buffers[ue_id].current_size} B")
        print("\t+" + "-" * 35)
        print(f"\tПред. позиция       : {ue.coordinates[-2]}")
        print(f"\tТекущая позиция     : {ue.position}")
        print(f"\tСмещение            : {displacement} m")
        print("-" * 40)


def sim_with_ue_collection():
    """
    Пример сценария с использованием коллекций UE.
    """
    sim_duration = 3000
    update_interval = 1
    num_frames = int(np.ceil(sim_duration / 10))
    bandwidth = 10
    inf = math.inf

    bs = BaseStation(x=0, y=0, bandwidth=bandwidth, global_max=inf, per_ue_max=inf)
    ue_collection = UECollection()

    ue1 = UserEquipment(UE_ID=1, x=10, y=10, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=4, y=-2, ue_class="cyclist")
    ue3 = UserEquipment(UE_ID=3, x=-5, y=-5, ue_class="car")

    MapBorders._instance = None
    MapBorders(-1000, 1000, -1000, 1000)

    ue1.SET_MOBILITY_MODEL("DiagonalWalk", bs=bs, pause_time=0)
    ue2.SET_MOBILITY_MODEL("DiagonalWalk", bs=bs, pause_time=0)
    ue3.SET_MOBILITY_MODEL("DiagonalWalk", bs=bs, pause_time=0)

    uma = UMaModel(bs=bs, cond_update_period=5)
    ue1.SET_CH_MODEL(uma)
    ue2.SET_CH_MODEL(uma)
    ue3.SET_CH_MODEL(uma)

    bs.REG_UE(ue1)
    bs.REG_UE(ue2)
    bs.REG_UE(ue3)

    ue_collection.ADD_USER(ue1)
    ue_collection.ADD_USER(ue2)
    ue_collection.ADD_USER(ue3)

    bs.buffer_manager.add_packet(1, Packet(size=1000000, creation_time=0))
    bs.buffer_manager.add_packet(2, Packet(size=1000000, creation_time=0))
    bs.buffer_manager.add_packet(3, Packet(size=1000000, creation_time=0))

    lte_grid = RES_GRID_LTE(bandwidth=bandwidth, num_frames=num_frames)
    scheduler = ProportionalFairScheduler(lte_grid, bs)

    for current_time in range(update_interval, sim_duration + 1, update_interval):
        GLOBALS.CURRENT_TIME = current_time
        ue_collection.UPDATE_ALL_USERS(
            current_time=current_time,
            update_interval=update_interval,
        )

        for tti in range(current_time - update_interval, current_time):
            users = ue_collection.GET_USERS_FOR_SCHEDULER()
            sched_result = scheduler.schedule(tti, users)
            print_users_stats(
                ue_collection=ue_collection,
                tti=tti,
                bs=bs,
                sched_result=sched_result,
            )

    visualize_users_mobility(
        ue_collection=ue_collection,
        bs=bs,
        x_min=-11,
        x_max=11,
        y_min=-11,
        y_max=11,
    )
    visualize_users_sinr(
        ue_collection=ue_collection,
        sim_duration=sim_duration,
        update_interval=update_interval,
    )


def plot_harq_comparison(sim_off, sim_on, x_max=4999, window=100):
    import matplotlib.pyplot as plt

    data_off = sim_off.get_harq_plot_data(x_max=x_max, window=window)
    data_on = sim_on.get_harq_plot_data(x_max=x_max, window=window)

    if data_off is None or data_on is None:
        print("No HARQ data to compare")
        return

    y_max_tx = max(max(data_off["avg_tx_smooth"]), max(data_on["avg_tx_smooth"]), 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(data_off["tti"], data_off["avg_tx_smooth"], linewidth=2, label="HARQ выключен")
    axes[0].plot(data_on["tti"], data_on["avg_tx_smooth"], linewidth=2, label="HARQ включен")
    axes[0].set_title("Сравнение HARQ OFF и HARQ ON по передаче транспортных блоков")
    axes[0].set_xlabel("TTI")
    axes[0].set_ylabel("Среднее число\nпередач одного TB")
    axes[0].set_xlim(0, x_max)
    axes[0].set_ylim(0, y_max_tx * 1.05)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(data_off["tti"], data_off["ack_rate_smooth"], linewidth=2, label="HARQ выключен")
    axes[1].plot(data_on["tti"], data_on["ack_rate_smooth"], linewidth=2, label="HARQ включен")
    axes[1].set_xlabel("TTI")
    axes[1].set_ylabel("Доля успешно\nпринятых TB")
    axes[1].set_xlim(0, x_max)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def build_sim(harq_enabled: bool):
    MapBorders._instance = None
    MapBorders(-1000, 1000, -1000, 1000)

    bs = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa")
    ue_collection = UECollection()

    GLOBALS.SEED = 42
    np.random.seed(GLOBALS.SEED)

    ue_collection.ADD_RANDOM_USERS(
        num_ue=3,
        x_min=-1000,
        x_max=1000,
        y_min=-1000,
        y_max=1000,
        ue_class="random",
    )
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint", bs=bs, pause_time=0)
    ue_collection.REG_USERS_TO_BS(bs)

    sim = SimulationManager()
    sim.sched_config.harq_enabled = harq_enabled
    sim.set_base_station(bs)
    sim.set_ue_collection(ue_collection)
    sim.set_scheduler(algorithm="RoundRobin")

    for ue in ue_collection.GET_ALL_USERS():
        sim.setup_ue_traffic(ue_id=ue.UE_ID, model_type="Poisson", packet_rate=1000)

    sim.set_sim_duration(5000)
    return sim


def sim_compare_harq():
    sim_off = build_sim(False)
    sim_off.start_simulation()

    sim_on = build_sim(True)
    sim_on.start_simulation()

    plot_harq_comparison(sim_off, sim_on, x_max=4999, window=100)


def sim_with_manager():
    """
    Тестовый стенд для проверки работоспособности моделей канала.
    """
    MapBorders._instance = None
    MapBorders(-500, 500, -500, 500)
    x_min, x_max, y_min, y_max = MapBorders().get_borders()

    bs = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa", enable_tdl=True)
    ue_collection = UECollection()

    GLOBALS.SEED = 42
    np.random.seed(GLOBALS.SEED)

    ue_collection.ADD_RANDOM_USERS(
        num_ue=5,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        ue_class="random",
    )
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint", bs=bs, pause_time=0)
    ue_collection.REG_USERS_TO_BS(bs)

    sim = SimulationManager()
    sim.set_base_station(bs)
    sim.set_ue_collection(ue_collection)
    sim.set_scheduler(
        algorithm="RoundRobin",
        max_dl_ue_tti=None,
        pcfich=2,
        enable_window=False,
        window_size=4,
        max_dl_cce_allowance=None,
    )

    for ue in ue_collection.GET_ALL_USERS():
        sim.setup_ue_traffic(
            ue_id=ue.UE_ID,
            model_type="Poisson",
            packet_rate=5000,
        )

    sim.set_sim_duration(50000)
    sim.set_mobility_interval(500)
    sim.set_channel_interval(10)
    sim.enable_verbose_log(to_file=True)
    sim.set_stats_manager(
        enabled=True,
        collect_interval=1,
        history_max_len=50000,
        scheduler_level="full",
        amc_level="full",
        pdcch_level="full",
        file_prefix="emp_stats",
    )
    sim.start_simulation()

    visualize_users_mobility(
        ue_collection=ue_collection,
        bs=bs,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    visualize_users_sinr(
        ue_collection=ue_collection,
        sim_duration=sim.sim_config.sim_duration,
        update_interval=10,
    )


if __name__ == "__main__":
    # sim_with_ue_collection()
    # sim_compare_harq()
    sim_with_manager()
