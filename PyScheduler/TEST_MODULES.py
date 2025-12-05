import math

import matplotlib.pyplot as plt
import numpy as np

import GLOBALS
from BS_MODULE import BaseStation, Packet
from CHANNEL_MODEL import UMaModel
from MOBILITY_MODEL import RandomWaypointModel, DiagonalWalkModel
from RES_GRID import RES_GRID_LTE
from SCHEDULER import ProportionalFairScheduler, HARQManager, AdaptiveModulationAndCoding
from SIMULATION_MANAGER import SimulationManager
from TRAFFIC_MODEL import PoissonModel
from UE_MODULE import UECollection, UserEquipment

def visualize_users_mobility(ue_collection: UECollection, bs: BaseStation,
                            x_min: float, x_max: float, y_min: float, y_max: float):
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
    plt.plot(x_bs, y_bs, marker = 'o', label='Base Station')

    for ue in ue_collection.GET_ALL_USERS():
        x_coords = [x for x, _ in ue.coordinates]
        y_coords = [y for _, y in ue.coordinates]
        plt.plot(x_coords, y_coords, marker = '.', label=f"UE {ue.UE_ID}")

    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_users_sinr(ue_collection: UECollection, sim_duration: float,
                            update_interval: float):
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
        plt.plot(tti_range, sinr_values, label=f'UE{ue.UE_ID}')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_users_stats(ue_collection: UECollection, tti: int, bs: BaseStation,
                      sched_result: dict):
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
            ue.position[0] - ue.coordinates[-2][0],
            ue.position[1] - ue.coordinates[-2][1]
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

def debug_simulation():
    """
    Симуляция, предназначенная для отладки. 
    Позволяет проверить корректность взаимодействия компонентов системы при 
    помощи вывода различных графиков и статистики для каждого пользователя.
    
    """
    sim_duration = 3000 # Время симуляции (в мс)
    update_interval = 1 # Интервал обновления параметров пользователя (в мс)
    num_frames = int(np.ceil(sim_duration / 10)) # Кол-во кадров (для ресурсной сетки)
    bandwidth = 10 # Ширина полосы (в МГц)
    inf = math.inf

    # Создание и настройка базовой станции
    bs = BaseStation(x=0, y=0, bandwidth=bandwidth, global_max=inf, per_ue_max=inf)

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Создание и настройка пользовательских устройств
    ue1 = UserEquipment(UE_ID=1, x=10, y=10, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=4, y=-2, ue_class="cyclist")
    ue3 = UserEquipment(UE_ID=3, x=-5, y=-5, ue_class="car")

    # Создание модели передвижения пользователей
    diagonalwalk = DiagonalWalkModel(bs_x=0,
                                    bs_y=0,
                                    pause_time = 0)

    # Назначение пользователям модели передвижения
    ue1.SET_MOBILITY_MODEL(diagonalwalk)
    ue2.SET_MOBILITY_MODEL(diagonalwalk)
    ue3.SET_MOBILITY_MODEL(diagonalwalk)

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
        ue_collection.UPDATE_ALL_USERS(current_time=current_time,
                                       update_interval=update_interval)

        # Цикл для планирования ресурсов (по TTI)
        for tti in range(current_time - update_interval, current_time):

            # Подготовка данных для планировщика
            users = ue_collection.GET_USERS_FOR_SCHEDULER()

            # Планирование ресурсов
            sched_result = scheduler.schedule(tti, users)

            # Вывод статистики для каждого пользователя
            print_users_stats(ue_collection=ue_collection,
                              tti=tti,
                              bs=bs,
                              sched_result=sched_result)


    # Визуализация передвижения пользователей
    visualize_users_mobility(ue_collection=ue_collection,
                            bs=bs,
                            x_min=-11,
                            x_max=11,
                            y_min=-11,
                            y_max=11)

    # Визуализация SINR пользователей во времени
    visualize_users_sinr(ue_collection=ue_collection,
                            sim_duration=sim_duration,
                            update_interval=update_interval)

def sim_with_ue_collection():
    """
    Пример сценария с использованием коллекций UE.

    """
    sim_duration = 20 # Время симуляции (в мс)
    update_interval = 1 # Интервал обновления параметров пользователя (в мс)
    num_frames = int(np.ceil(sim_duration / 10)) # Кол-во кадров (для ресурсной сетки)
    bandwidth = 10 # Ширина полосы (в МГц)
    inf = math.inf

    # Создание и настройка базовой станции
    bs = BaseStation(
        x=0, y=0, bandwidth=bandwidth, global_max=inf, 
        per_ue_max=inf, ch_model_type="UMa", ch_model_params={"cond_update_period": 5}
    )

    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()

    # Установка сида. Нужен для того, чтобы ADD_RANDOM_USERS всегда генерировал
    # одинаковых пользователей в одинаковом месте. Если не задавать сид, пользователи
    # каждую симуляцию будут генерироваться абсолютно случайно
    GLOBALS.SEED = 42

    # Генерация заданного числа UE в коллекцию
    ue_collection.ADD_RANDOM_USERS(num_ue=3)

    # Создание модели передвижения пользователей
    diagonalwalk = DiagonalWalkModel(bs.position[0], 
                                     bs.position[1], 
                                     pause_time=0)

    # Установка модели передвижения для всех пользователей коллекции.
    # Есть возможность задавать для отдельных пользователей при помощи параметра ue_ids
    ue_collection.SET_MOBILITY_MODEL(diagonalwalk)

    # Регистрация всех пользователей коллекции в базовой станции.
    ue_collection.REG_USERS_TO_BS(bs)

    # Имитация Full Buffer
    for ue in ue_collection.GET_ALL_USERS():
        bs.ue_buffers[ue.UE_ID].ADD_PACKET(Packet(size=inf,
                                                  ue_id=ue.UE_ID,
                                                  creation_time=0), current_time=0)

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
        ue_collection.UPDATE_ALL_USERS(current_time=current_time,
                                       update_interval=update_interval)

        # Цикл для планирования ресурсов (по TTI)
        for tti in range(current_time - update_interval, current_time):

            # Подготовка данных для планировщика
            users = ue_collection.GET_USERS_FOR_SCHEDULER()

            # Планирование ресурсов
            sched_result = scheduler.schedule(tti, users)

            # Вывод статистики для каждого пользователя
            print_users_stats(ue_collection=ue_collection,
                              tti=tti,
                              bs=bs,
                              sched_result=sched_result)


    # Визуализация передвижения пользователей
    visualize_users_mobility(ue_collection=ue_collection,
                            bs=bs,
                            x_min=-1000,
                            x_max=1000,
                            y_min=-1000,
                            y_max=1000)

    # Визуализация SINR пользователей во времени
    visualize_users_sinr(ue_collection=ue_collection,
                            sim_duration=sim_duration,
                            update_interval=update_interval)
    
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
    ue_collection.ADD_RANDOM_USERS(num_ue=3)    
    
    # Создание модели передвижения пользователей
    random_waypoint = RandomWaypointModel(x_min=-1000, 
                                          x_max=1000, 
                                          y_min=-1000, 
                                          y_max=1000, 
                                          pause_time=0)
    
    # Установка модели передвижения для всех пользователей коллекции
    ue_collection.SET_MOBILITY_MODEL(random_waypoint)    
    
    # Создание модели генерации трафика
    poisson = PoissonModel(packet_rate=1000)
    
    # Установка модели генерации трафика для всех пользователей коллекции
    ue_collection.SET_TRAFFIC_MODEL(poisson)
    
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
    sim.set_scheduler(algorithm="RoundRobin")
    
    # Установка длительности симуляции
    sim.set_sim_duration(5000)
    
    # Включение verbose логирования. Для вывода всех логов в файл нужно
    # поставить флаг to_file=True.
    sim.enable_verbose_log()
    
    # Включение логирования статистики в CSV-файл
    sim.enable_stats_log()
    
    # Запуск симуляции
    sim.start_simulation()
    
# =============================================================================
#                    ТЕСТИРОВАНИЕ BLER_TABLE ИЗ GLOBALS                
# =============================================================================

def test_bler_table():
    print("BLER_TABLE contents:")
    for cqi in range(1, 16):  # CQI от 1 до 15 
        entries = GLOBALS.BLER_TABLE[cqi]
        print(f"CQI {cqi}:")
        for sinr, bler in entries:
            print(f"\tSINR: {sinr:.2f}, BLER: {bler:.3f}")
   
def print_bler_range_by_cqi():
    print("Диапазон BLER для каждого CQI:")
    print("CQI | min(BLER) | max(BLER) | mean(BLER)")
    for cqi in range(1, 16):
        entries = GLOBALS.BLER_TABLE[cqi]
        bler_list = [b for s, b in entries]
        min_bler = min(bler_list)
        max_bler = max(bler_list)
        mean_bler = sum(bler_list)/len(bler_list)
        print(f"{cqi:>3} | {min_bler:9.3f} | {max_bler:9.3f} | {mean_bler:9.3f}")

def plot_bler_vs_sinr():
    plt.figure(figsize=(10, 7))
    for cqi in range(1, 16):
        entries = GLOBALS.BLER_TABLE[cqi]
        sinr = []
        bler = []
        for s, b in entries:
            sinr.append(s)
            bler.append(b)
        plt.plot(sinr, bler, marker='o', label=f'CQI {cqi}')
    plt.xlabel("SINR (дБ)")
    plt.ylabel("BLER (Block Error Rate)")
    plt.yscale('log')
    plt.title("BLER vs SINR для разных CQI")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
#                  ТЕСТИРОВАНИЕ RV/IR/RTT/Soft Combining             
# =============================================================================

def test_harq_ir_rtt():
    manager = HARQManager(num_processes=2)
    manager.init_ue(1)
    proc = manager.get_idle_process(1)
    tb_data = b"hello"
    cqi = 15
    rbs = [0, 1]
    manager.current_tti = 0
    # Начальная передача
    proc.start_transmission(tb_data, cqi, rbs, soft_bits_received=[1.1, -0.9, 0.3])
    # эмитируем серию NACK с новыми soft bits для IR/soft combining
    print("HARQ IR cycle:")
    for attempt in range(1, 5):
        success = proc.handle_nack(soft_bits_received=[0.5*attempt, -0.2*attempt, 0.1*attempt])
        rv = proc.get_current_rv()
        print(f"Attempt {attempt}: RV={rv}, Success={success}, Combined soft bits={proc.get_combined_soft_bits()}")
    # Проверить правильность RV последовательности, накопление soft bits

# =============================================================================
#             ТЕСТИРОВАНИЕ HARQ RTT (ОТЛОЖЕННАЯ ОБРАТНАЯ СВЯЗЬ)           
# =============================================================================

def test_harq_rtt():
    manager = HARQManager(num_processes=2)
    manager.init_ue(1)
    manager.current_tti = 5
    ack = False
    process_id = 0
    soft_bits = [0.7, -0.5, 0.12]
    manager.handle_feedback(1, process_id, ack, soft_bits_received=soft_bits)
    
    # Очередь сразу после постановки feedback 
    print("Feedback Queue before RTT advance:", manager.feedback_queue)

    # Сдвигаем TTI, чтобы элемент должен был быть обработан (RTT=4)
    manager.set_current_tti(9)
    
    # Очередь после обработки 
    print("Feedback Queue after RTT:", manager.feedback_queue)

# =============================================================================
#             ТЕСТИРОВАНИЕ ПОЛУЧЕНИЯ ДАННЫХ ИЗ ТАБЛИЦ        
# =============================================================================

from CHANNEL_MODEL import ChannelModel
from BS_MODULE import BaseStation
def test_channel_model_tables():
    bs = BaseStation()
    ch = ChannelModel(bs)
    for cqi in range(1, 16):
        bler = ch.lookup_bler(0, cqi)
        mcs = ch.get_mcs_from_cqi(cqi)
        itbs, qam = ch.get_itbs_from_mcs(mcs)
        tbs = ch.get_tbs(itbs, 10)
        print(f"CQI={cqi}, BLER@0dB={bler}, MCS={mcs}, ITBS={itbs}, QBAM={qam}, TBS@10RB={tbs}")

# =============================================================================
#                        ТЕСТИРОВАНИЕ CQI to MCS, ITBS, TBS           
# =============================================================================

def test_cqi_to_mcs_table():
    amc = AdaptiveModulationAndCoding()
    CQI_vals = list(range(1, 16))

    for cqi in CQI_vals:
        mcs = amc.cqi_to_mcs(cqi)
        qm, itbs = amc.mcs_to_qm_itbs(mcs)
        tbs = GLOBALS.TB_SIZE_TABLE[itbs][10]
        print(f"CQI={cqi:2d} -> MCS={mcs:2d}, Qm={qm}, ITBS={itbs}, TBS(11PRB)={tbs}")

if __name__ == "__main__":
    #debug_simulation()
    #sim_with_ue_collection()
    #sim_with_manager()
    #test_bler_table() #1
    #print_bler_range_by_cqi() #1
    #plot_bler_vs_sinr() #1
    #test_tables() #2
    #test_harq_ir_rtt() #3
    #test_harq_rtt() #4
    #test_channel_model_tables() #5
    test_cqi_to_mcs_table() #6