"""
#------------------------------------------------------------------------------
# Модуль: ENVIRONMENT - Среда симуляции для системы LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет среду для симуляции работы системы LTE, включая взаимодействие
# пользовательских устройств, планировщиков ресурсов и визуализацию результатов.
#------------------------------------------------------------------------------
# Версия: 1.0.0
# Дата последнего изменения: 2025-03-18
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл

# Зависимости:
# - UE_MODULE.py (модели пользовательского оборудования)
# - RES_GRID (модель ресурсной сетки LTE)
# - SCHEDULER.py (модель планировщика)
#------------------------------------------------------------------------------
"""

"""
Модуль: ENVIRONMENT - Интеграционная среда для симуляции LTE-сети
"""

import time
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')  # Или 'Qt5Agg' для GUI бэкенда
import matplotlib.pyplot as plt
%matplotlib
from matplotlib.lines import Line2D
from UE_MODULE import UECollection, UserEquipment
from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE
from SCHEDULER import RoundRobinScheduler, BestCQIScheduler
from MOBILITY_MODEL import RandomWalkModel, RandomWaypointModel
from TRAFFIC_MODEL import PoissonModel
from CHANNEL_MODEL import UMiModel

class LTEGridVisualizer:
    def __init__(self, lte_grid):
        self.lte_grid = lte_grid
    
    def visualize_timeline_grid(self, tti_start=0, tti_end=None, scheduler_name: str = "Round Robin", ue_colors=None):
        """Визуализирует ресурсную сетку LTE с правильным расположением слотов по временной оси"""
        
        title = f"{scheduler_name} планировщик (TTI {tti_start}-{tti_end-1})\n"
        
        if tti_end is None:
            tti_end = min(tti_start + 5, self.lte_grid.total_tti)
        
        # Если цвета не заданы, создаем их по умолчанию
        if ue_colors is None:
            ue_colors = {}
                
        # Количество слотов = количество TTI × 2
        num_slots = (tti_end - tti_start) * 2
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Настраиваем оси
        rb_range = self.lte_grid.rb_per_slot
        ax.set_xlim(-0.5, num_slots - 0.5)
        ax.set_ylim(-0.5, rb_range - 0.5)
        ax.set_title("LTE Resource Grid (Timeline View)")
        ax.set_xlabel('Time (slot)')
        ax.set_ylabel('Frequency (RB index)')
        
        # Добавляем сетку
        ax.set_xticks(range(num_slots))
        # Создаем метки для слотов (TTI.slot)
        slot_labels = []
        for tti in range(tti_start, tti_end):
            slot_labels.extend([f"{tti}.0", f"{tti}.1"])
        ax.set_xticklabels(slot_labels, rotation=45)
        
        ax.set_yticks(range(0, rb_range, 5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        
        # Рисуем вертикальные линии, разделяющие TTI
        for i in range(0, num_slots, 2):
            if i > 0:  # Не рисуем линию в начале графика
                ax.axvline(x=i - 0.5, color='red', linestyle='-', linewidth=1)
        
        # Создаем цветовую карту для UE_ID
        default_colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # Сохраняем UE_ID для легенды
        unique_ue_ids = set()
        
        # Отображаем занятые ресурсные блоки
        for tti in range(tti_start, tti_end):
            # Вычисляем индексы кадра и подкадра
            frame_idx = tti // 10
            subframe_idx = tti % 10
            
            # Вычисляем начальный индекс слота для данного TTI
            slot_start_idx = (tti - tti_start) * 2
            
            # Обрабатываем оба слота
            for slot_num in [0, 1]:
                slot_id = f"sub_{subframe_idx}_slot_{slot_num}"  # Используем subframe_idx вместо tti
                slot_idx = slot_start_idx + slot_num
                
                for freq_idx in range(rb_range):
                    rb = self.lte_grid.GET_RB(tti, slot_id, freq_idx)
                    if rb and not rb.CHCK_RB():
                        # Отрисовка блока
                        ue_id = rb.UE_ID
                        unique_ue_ids.add(ue_id)
                        
                        # Используем цвет из словаря или берем стандартный
                        if ue_id in ue_colors:
                            color = ue_colors[ue_id]
                        else:
                            # Автоматически назначаем цвет и сохраняем для будущего использования
                            color = default_colors[ue_id % 10]
                            ue_colors[ue_id] = color
                        
                        # Рисуем прямоугольник
                        rect = plt.Rectangle(
                            (slot_idx - 0.4, freq_idx - 0.4),
                            0.8, 0.8,
                            facecolor=color,
                            alpha=0.8,
                            edgecolor='black'
                        )
                        ax.add_patch(rect)
                        
                        # Добавляем метку UE_ID
                        ax.text(
                            slot_idx, freq_idx,
                            str(ue_id),
                            ha='center', va='center',
                            fontsize=9, fontweight='bold',
                            color='white'
                        )
        
        # Добавляем поясняющие метки для TTI
        for tti in range(tti_start, tti_end):
            slot_start_idx = (tti - tti_start) * 2
            ax.text(slot_start_idx + 0.5, -3, f"TTI {tti}", 
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.plot([slot_start_idx - 0.5, slot_start_idx + 1.5], [-2, -2], 'k-', linewidth=2)
        
        plt.tight_layout()
        plt.show()
        
        ax.set_title(title)
        return fig, ax, ue_colors

    
    def _plot_slot_data(self, ax, data, tti_start, tti_end, title):
        """Отображение данных для конкретного слота"""
        # Создаем пустую сетку
        rb_range = self.lte_grid.rb_per_slot
        tti_range = tti_end - tti_start
        
        # Настраиваем оси и заголовок
        ax.set_xlim(-0.5, tti_range - 0.5)
        ax.set_ylim(-0.5, rb_range - 0.5)
        ax.set_title(title)
        ax.set_xlabel('TTI')
        ax.set_ylabel('Frequency (RB index)')
        
        # Добавляем линии сетки
        ax.set_xticks(range(tti_range))
        ax.set_xticklabels(range(tti_start, tti_end))
        ax.set_yticks(range(0, rb_range, 5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        
        # Создаем цветовую карту для UE_ID
        colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 различных цветов
        
        # Отображаем каждый занятый ресурсный блок
        for item in data:
            tti_rel = item['tti'] - tti_start
            freq_idx = item['freq_idx']
            ue_id = item['UE_ID']
            
            # Определяем цвет блока (по UE_ID)
            color_idx = ue_id % 10
            
            # Рисуем прямоугольник для блока
            rect = plt.Rectangle(
                (tti_rel - 0.4, freq_idx - 0.4),
                0.8, 0.8,
                facecolor=colors[color_idx],
                alpha=0.8,
                edgecolor='black'
            )
            ax.add_patch(rect)
            
            # Добавляем текстовую метку с UE_ID
            ax.text(
                tti_rel, freq_idx,
                str(ue_id),
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white'
            )

def test_visualize_lte_timeline():
    """Тестовая функция для проверки корректной визуализации слотов по временной оси"""
    print("Создание ресурсной сетки LTE...")
    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=2)
    
    # Выделение различных ресурсных блоков
    # TTI 0
    lte_grid.ALLOCATE_RB(0, "sub_0_slot_0", 10, 1)
    lte_grid.ALLOCATE_RB(0, "sub_0_slot_1", 20, 2)
    lte_grid.ALLOCATE_RB(0, "sub_0_slot_1", 21, 2)
    
    # TTI 1
    lte_grid.ALLOCATE_RB_GROUP(1, 30, 3)  # Выделяет RB с индексом 30 в обоих слотах TTI 1
    
    # TTI 2 - демонстрация разных пользователей в одном частотном индексе в разных слотах
    lte_grid.ALLOCATE_RB(2, "sub_2_slot_0", 15, 4)
    lte_grid.ALLOCATE_RB(2, "sub_2_slot_1", 15, 5)
    
    # TTI 3 - выделение нескольких последовательных RB одному пользователю
    for rb_idx in range(40, 45):
        lte_grid.ALLOCATE_RB(3, "sub_3_slot_0", rb_idx, 6)
    
    # Визуализация с корректным отображением слотов по временной оси
    print("\nЗапуск визуализации по временной оси...")
    visualizer = LTEGridVisualizer(lte_grid)
    visualizer.visualize_timeline_grid(tti_start=0, tti_end=4)
    
    print("Визуализация завершена.")
    return lte_grid

def test_scheduler_with_buffer():
    """
    Расширенный тест работы планировщика Round Robin с буфером пользователей.
    Теперь выводит требования RB и фактические выделения без фиксированных проверок.
    """
    print("\n=== Расширенный тест планировщика с буфером ===")

    #------------------------------------------------------------------
    # Шаг 1: Инициализация ресурсной сетки (10 МГц, 1 кадр = 10 TTI)
    #------------------------------------------------------------------
    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=1)
    print("[OK] Ресурсная сеть инициализирована")

    #------------------------------------------------------------------
    # Шаг 2: Создание планировщика
    #------------------------------------------------------------------
    scheduler = BestCQIScheduler(lte_grid)
    print(f"[OK] Планировщик {scheduler.__class__.__name__} инициализирован")

    #------------------------------------------------------------------
    # Шаг 3: Создание пользователей и базовой станции
    #------------------------------------------------------------------
    bs = BaseStation(x=0, y=0, height=25.0, bandwidth=10)
    ue1 = UserEquipment(UE_ID=1, x=500, y=500, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=1000, y=1000, ue_class="car")
    
    # Установка моделей канала
    ue1.SET_CH_MODEL(UMiModel(bs))
    ue2.SET_CH_MODEL(UMiModel(bs))
    print("[OK] Пользователи созданы")

    #------------------------------------------------------------------
    # Шаг 4: Добавление пакетов в буфер
    #------------------------------------------------------------------
    current_time = 0
    ue1.buffer.ADD_PACKET(1500, creation_time=current_time, current_time=current_time)  # 1500 Б
    ue2.buffer.ADD_PACKET(3000, creation_time=current_time, current_time=current_time)  # 3000 Б
    print("[OK] Пакеты добавлены в буферы")

    #------------------------------------------------------------------
    # Шаг 5: Обновление позиций и качества канала
    #------------------------------------------------------------------
    ue1.UPD_POSITION(current_time, bs.position, bs.height)
    ue2.UPD_POSITION(current_time, bs.position, bs.height)
    ue1.UPD_CH_QUALITY()
    ue2.UPD_CH_QUALITY()
    print(f"[OK] CQI рассчитан: UE1={ue1.cqi}, UE2={ue2.cqi}")

    #------------------------------------------------------------------
    # Шаг 6: Подготовка данных для планировщика
    #------------------------------------------------------------------
    users = [
        {'UE_ID': 1, 'buffer_size': 1500, 'cqi': ue1.cqi, 'ue': ue1},
        {'UE_ID': 2, 'buffer_size': 3000, 'cqi': ue2.cqi, 'ue': ue2}
    ]

    #------------------------------------------------------------------
    # Шаг 7: Запуск планировщика для TTI=0
    #------------------------------------------------------------------
    result = scheduler.schedule(0, users)
    allocation = result['allocation']
    stats = result['statistics']

    #------------------------------------------------------------------
    # Шаг 8: Анализ распределения RB по слотам
    #------------------------------------------------------------------
    print("\nДетализация распределения RB:")
    
    # Получение подкадра (TTI=0)
    subframe = lte_grid.GET_SUBFRAME(0)
    
    # Анализ слота 0
    slot0 = subframe.slots[0]
    slot0_rbs = {rb.freq_idx: rb.UE_ID for rb in slot0.GET_ALL_RES_BLCK()}
    
    # Анализ слота 1
    slot1 = subframe.slots[1]
    slot1_rbs = {rb.freq_idx: rb.UE_ID for rb in slot1.GET_ALL_RES_BLCK()}

    print("\nСлот 0:")
    print("RB Индекс | Пользователь")
    print("-----------------------")
    for freq_idx in sorted(slot0_rbs.keys()):
        print(f"{freq_idx:8} | {slot0_rbs[freq_idx] or 'Свободен'}")

    print("\nСлот 1:")
    print("RB Индекс | Пользователь")
    print("-----------------------")
    for freq_idx in sorted(slot1_rbs.keys()):
        print(f"{freq_idx:8} | {slot1_rbs[freq_idx] or 'Свободен'}")

    #------------------------------------------------------------------
    # Шаг 9: Вывод параметров MCS для каждого пользователя
    #------------------------------------------------------------------
    print("\nПараметры MCS:")
    for user in users:
        cqi = user['cqi']
        mcs_params = scheduler.amc.CQI_TO_MCS.get(cqi, (0, 0))
        bits_per_rb = scheduler.amc.GET_BITS_PER_RB(cqi)
        
        print(f"\nUE {user['UE_ID']} (CQI={cqi}):")
        print(f"- Modulation: {'QPSK' if mcs_params[0] == 2 else '16QAM' if mcs_params[0] == 4 else '64QAM'}")
        print(f"- Code Rate: {mcs_params[1]:.3f}")
        print(f"- Биты/RB: {bits_per_rb}")

    #------------------------------------------------------------------
    # Шаг 10: Общая статистика
    #------------------------------------------------------------------
    print("\nИтоговая статистика:")
    total_throughput = 0
    
    for user in users:
        ue_id = user['UE_ID']
        rb_count = len(allocation.get(ue_id, []))
        bits_per_rb = scheduler.amc.GET_BITS_PER_RB(user['cqi'])
        throughput = rb_count * bits_per_rb  # Бит/мс = кбит/с
        
        print(f"\nUE {ue_id}:")
        print(f"  RB выделено: {rb_count}")
        print(f"  Пропускная способность: {throughput * 1000:.2f} бит/с")
        total_throughput += throughput

    # Средняя пропускная способность на TTI
    num_users = len([u for u in users if len(allocation.get(u['UE_ID'], [])) > 0])
    avg_throughput = (total_throughput / num_users) * 1000 if num_users > 0 else 0
    
    print(f"\nСредняя пропускная способность в TTI: {avg_throughput:.2f} бит/с")
    print(f"Всего выделено RB: {stats['total_allocated_rbs']}/100")
    print("[OK] Тест пройден успешно!")
    
def test_scheduler_grid():
    """Тест работы Round Robin планировщика с визуализацией полного фрейма"""
    print("\n=== Тест Round Robin (полный фрейм) ===")
    
    # Шаг 1: Инициализация компонентов
    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=2)  # 1 фрейм = 10 TTI
    visualizer = LTEGridVisualizer(lte_grid)
    scheduler = BestCQIScheduler(lte_grid)

    # Шаг 2: Создание пользователей
    bs = BaseStation(x=500, y=500, height=25.0, bandwidth=10)
    ue1 = UserEquipment(UE_ID=1, x=300, y=300, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=700, y=700, ue_class="car")
    ue3 = UserEquipment(UE_ID=3, x=100, y=200, ue_class="car")
    ue1.buffer.ADD_PACKET(10000, creation_time=0, current_time=0,ttl_ms=10000)
    ue2.buffer.ADD_PACKET(10000, creation_time=0, current_time=0,ttl_ms=10000)
    ue1.buffer.ADD_PACKET(2000, creation_time=0, current_time=0,ttl_ms=10000)
    ue2.buffer.ADD_PACKET(3000, creation_time=0, current_time=0,ttl_ms=10000)
    ue3.buffer.ADD_PACKET(30000, creation_time=0, current_time=0,ttl_ms=10000)
    
    # Настройка моделей
    ue1.SET_MOBILITY_MODEL(RandomWaypointModel(x_min=0, x_max=1000, y_min=0, y_max=1000, pause_time=10))
    ue1.SET_TRAFFIC_MODEL(PoissonModel(packet_rate=1000))
    ue1.SET_CH_MODEL(UMiModel(bs))
    
    ue2.SET_MOBILITY_MODEL(RandomWalkModel(x_min=0, x_max=1000, y_min=0, y_max=1000))
    ue2.SET_TRAFFIC_MODEL(PoissonModel(packet_rate=100))
    ue2.SET_CH_MODEL(UMiModel(bs))
    
    ue3.SET_MOBILITY_MODEL(RandomWalkModel(x_min=0, x_max=1000, y_min=0, y_max=1000))
    ue3.SET_TRAFFIC_MODEL(PoissonModel(packet_rate=100))
    ue3.SET_CH_MODEL(UMiModel(bs))
    
    sim_duration = 20
    prev_time = 0

    cqi_history = {
        1: [],
        2: [],
        3: []
    }    

    # Шаг 3: Основной цикл симуляции
    for tti in range(20):  # 0-9 TTI (полный фрейм)
        current_time = tti
        interval = current_time - prev_time
        interval = max(0,interval)
        
        # Генерация трафика
        ue1.GEN_TRFFC(current_time, interval)
        ue2.GEN_TRFFC(current_time, interval)
        ue3.GEN_TRFFC(current_time, interval)
        
        # Обновление состояния
        ue1.UPD_POSITION(interval, bs.position, bs.height)
        ue2.UPD_POSITION(interval, bs.position, bs.height)
        ue3.UPD_POSITION(interval, bs.position, bs.height)
        ue1.UPD_CH_QUALITY()
        ue2.UPD_CH_QUALITY()
        ue3.UPD_CH_QUALITY()
        
        prev_time = current_time
        
        # Логирование CQI
        cqi_history[1].append(ue1.cqi)
        cqi_history[2].append(ue2.cqi)
        cqi_history[3].append(ue3.cqi)
        
        print(f"[TTI {tti}] CQI: UE1={ue1.cqi}, UE2={ue2.cqi}, UE3={ue3.cqi}")
        print(f"TTI {tti}: UE1={ue1.buffer.current_size}B, UE2={ue2.buffer.current_size}B, UE3={ue3.buffer.current_size}B")
        
        # Подготовка данных для планировщика
        users = [
            {
                'UE_ID': 1,
                'buffer_size': ue1.buffer.current_size,
                'cqi': ue1.cqi,
                'ue': ue1
            },
            {
                'UE_ID': 2,
                'buffer_size': ue2.buffer.current_size,
                'cqi': ue2.cqi,
                'ue': ue2
            },
            {
                'UE_ID': 3,
                'buffer_size': ue3.buffer.current_size,
                'cqi': ue3.cqi,
                'ue': ue3
            }
        ]
        
        # Запуск планировщика
        scheduler.schedule(tti, users)

    # Шаг 4: Визуализация всего фрейма
    print("\nВизуализация распределения ресурсов:")
    fig, ax, ue_colors = visualizer.visualize_timeline_grid(
        tti_start=0, 
        tti_end=sim_duration,
        scheduler_name= {scheduler.__class__.__name__} # <-- Добавлен параметр
    )
    
    # Дополнительные аннотации
    ax.set_title(
        f"{scheduler.__class__.__name__} планировщик (полный фрейм)\n"
        f"UE1: Средний CQI={np.mean(ue1.CQI_values):.1f}\n"
        f"UE2: Средний CQI={np.mean(ue2.CQI_values):.1f}"
        f"UE3: Средний CQI={np.mean(ue3.CQI_values):.1f}"
    )
    
    # Легенда
    legend_elements = [
        Line2D([0], [0], color=ue_colors[ue_id], lw=4, label=f'UE{ue_id}')
        for ue_id in sorted(ue_colors.keys())
    ]
    
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()
    print("[OK] Тест завершен с визуализацией")


if __name__ == "__main__":
    #test_scheduler_with_buffer()
    #test_visualize_lte_timeline()
    test_scheduler_grid()
    print("Все тесты успешно пройдены!")

