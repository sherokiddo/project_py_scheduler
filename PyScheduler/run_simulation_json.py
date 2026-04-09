"""
Скрипт для запуска симуляции с экспортом статистики в JSON формат.
"""
import GLOBALS
from BS_MODULE import BaseStation
from MOBILITY_MODEL import MapBorders
from SIMULATION_MANAGER import SimulationManager
from TRAFFIC_MODEL import PoissonModel
from UE_MODULE import UECollection


def run_simulation_with_json_export():
    """
    Запуск симуляции с экспортом статистики в JSON.
    
    Генерирует файлы:
    - sim_stats.json (основная статистика)
    - sim_stats_detailed.json (детальная per-UE статистика)
    """
    print("=" * 60)
    print("ЗАПУСК СИМУЛЯЦИИ С ЭКСПОРТОМ В JSON")
    print("=" * 60)

    # =====================================================================
    # 1. НАСТРОЙКА БАЗОВОЙ СТАНЦИИ
    # =====================================================================
    bs = BaseStation(
        x=0, 
        y=0, 
        bandwidth=10,  # 10 MHz
        ch_model_type="UMa"  # Urban Macro
    )
    print(f"[OK] Базовая станция создана (bandwidth={bs.bandwidth} MHz)")

    # =====================================================================
    # 2. СОЗДАНИЕ КОЛЛЕКЦИИ ПОЛЬЗОВАТЕЛЕЙ
    # =====================================================================
    ue_collection = UECollection()
    GLOBALS.SEED = 42  # Для воспроизводимости

    # Генерация 5 UE
    ue_collection.ADD_RANDOM_USERS(num_ue=5)
    print(f"[OK] Создано {len(ue_collection.GET_ALL_USERS())} пользователей")

    # Границы карты
    MapBorders(-1000, 1000, -1000, 1000)

    # Модель передвижения
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint")
    print("[OK] Модель передвижения: RandomWaypoint")

    # Модель трафика
    poisson = PoissonModel(packet_rate=1000)
    ue_collection.SET_TRAFFIC_MODEL(poisson)
    print("[OK] Модель трафика: Poisson (rate=1000 pkt/s)")

    # Регистрация в BS
    ue_collection.REG_USERS_TO_BS(bs)
    print("[OK] Пользователи зарегистрированы в BS")

    # =====================================================================
    # 3. НАСТРОЙКА МЕНЕДЖЕРА СИМУЛЯЦИИ
    # =====================================================================
    sim = SimulationManager()
    sim.set_base_station(bs)
    sim.set_ue_collection(ue_collection)
    
    # Планировщик (можно поменять на 'BestCQI' или 'ProportionalFair')
    sim.set_scheduler(
        algorithm="RoundRobin",
        max_dl_ue_tti=None,
        enable_window=True,
        window_size=10
    )
    print("[OK] Планировщик: RoundRobin")

    # Длительность симуляции
    sim.set_sim_duration(5000)  # 5000 мс = 5 секунд
    print("[OK] Длительность симуляции: 5000 мс (5 сек)")

    # =====================================================================
    # 4. НАСТРОЙКА ЭКСПОРТА В JSON
    # =====================================================================
    sim.set_stats_manager(
        enabled             =True,
        collect_interval    =1,          # Собирать каждый TTI
        history_max_len     =5000,
        scheduler_level     ="advanced",  # Расширенная статистика
        amc_level           ="advanced",        # AMC метрики
        pdcch_level         ="basic",         # Базовая PDCCH статистика
        export_format       ="json",        # <<< ЭКСПОРТ В JSON
        export_detailed_format="json",  # <<< ДЕТАЛЬНЫЙ ЭКСПОРТ В JSON
        file_prefix         ="sim_stats"      # Префикс файлов: sim_stats.json
    )
    print("[OK] StatsManager настроен на экспорт в JSON")

    # =====================================================================
    # 5. ЗАПУСК СИМУЛЯЦИИ
    # =====================================================================
    print("\n" + "=" * 60)
    print("НАЧАЛО СИМУЛЯЦИИ...")
    print("=" * 60 + "\n")

    sim.start_simulation()

    print("\n" + "=" * 60)
    print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print("\nСгенерированные файлы:")
    print("  - sim_stats.json (основная статистика)")
    print("  - sim_stats_detailed.json (детальная per-UE статистика)")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation_with_json_export()