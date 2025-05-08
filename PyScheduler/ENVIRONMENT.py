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
import math
import matplotlib
import json
import os
#matplotlib.use('TkAgg')  # Или 'Qt5Agg' для GUI бэкенда
import matplotlib.pyplot as plt
#%matplotlib
from matplotlib.lines import Line2D
from UE_MODULE import UECollection, UserEquipment
from BS_MODULE import BaseStation, Buffer, Packet
from RES_GRID import RES_GRID_LTE
from SCHEDULER import RoundRobinScheduler, BestCQIScheduler, ProportionalFairScheduler, ProportionalFairScheduler_v2
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
        rb_range = self.lte_grid.rb_per_slot
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Настраиваем оси
        rb_range = self.lte_grid.rb_per_slot
        ax.set_xlim(-0.5, num_slots - 0.5)
        ax.set_ylim(-0.5, rb_range - 0.5)
        ax.set_title("LTE Resource Grid (Timeline View)")
        ax.set_xlabel('Time (slot)')
        ax.set_ylabel('Frequency (RB index)')
        
        # Метки только для начала каждого TTI (каждые два слота)
        ms_labels = []
        for tti in range(tti_start, tti_end):
            ms_labels.extend([f"{tti} мс", ""])  # Метка для первого слота, пусто для второго

        ax.set_xticks(range(num_slots))
        ax.set_xticklabels(ms_labels, rotation=45)
        
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
        
        # Отрисовка занятых RB
        for tti in range(tti_start, tti_end):
            frame_idx = tti // 10
            subframe_idx = tti % 10
            slot_start_idx = (tti - tti_start) * 2
            
            # Получаем подфрейм напрямую
            subframe = self.lte_grid.GET_SUBFRAME(tti)
            for slot_num in [0, 1]:
                # Получаем слот напрямую из подфрейма
                slot = subframe.slots[slot_num]
                slot_idx = slot_start_idx + slot_num
                
                # Получаем все ресурсные блоки в слоте
                all_rbs = slot.GET_ALL_RES_BLCK()
                for rb in all_rbs:
                    if not rb.CHCK_RB():  # Если блок занят
                        freq_idx = rb.freq_idx
                        ue_id = rb.UE_ID
                        unique_ue_ids.add(ue_id)
                        
                        color = ue_colors.get(ue_id, default_colors[ue_id % 10])
                        ue_colors[ue_id] = color
                        
                        rect = plt.Rectangle((slot_idx - 0.4, freq_idx - 0.4), 0.8, 0.8,
                                            facecolor=color, alpha=0.8, edgecolor='black')
                        ax.add_patch(rect)
                        ax.text(slot_idx, freq_idx, str(ue_id), ha='center', va='center',
                               fontsize=9, fontweight='bold', color='white')
    
        plt.tight_layout()
        plt.show()
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
    lte_grid.ALLOCATE_RB_PAIR(1, 30, 3)  # Выделяет RB с индексом 30 в обоих слотах TTI 1
    
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
    bandwidth = 10 #Mhz
    lte_grid = RES_GRID_LTE(bandwidth=bandwidth, num_frames=1)
    bs = BaseStation(x=0, y=0, height=25.0, bandwidth=bandwidth)
    print("[OK] Ресурсная сеть инициализирована")

    #------------------------------------------------------------------
    # Шаг 2: Создание планировщика
    #------------------------------------------------------------------
    scheduler = BestCQIScheduler(lte_grid, bs)
    print(f"[OK] Планировщик {scheduler.__class__.__name__} инициализирован")

    #------------------------------------------------------------------
    # Шаг 3: Создание пользователей + регистрация
    #------------------------------------------------------------------
    ue1 = UserEquipment(UE_ID=1, x=500, y=500, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=1000, y=1000, ue_class="car")
    bs.REG_UE(ue1)
    bs.REG_UE(ue2)
    
    # Установка моделей канала
    ue1.SET_CH_MODEL(UMiModel(bs))
    ue2.SET_CH_MODEL(UMiModel(bs))
    
    print(f"UE1 состояние буфера, Б: {bs.ue_buffers[1].sizes[1]}")
    print(f"UE2 состояние буфера, Б: {bs.ue_buffers[2].sizes[2]}")
    
    print("[OK] Пользователи зарегестрированы")

    #------------------------------------------------------------------
    # Шаг 4: Добавление пакетов в буфер
    #------------------------------------------------------------------

    current_time = 0
    bs.ue_buffers[1].ADD_PACKET(Packet(size=5, ue_id=1, creation_time=current_time), current_time=current_time)
    bs.ue_buffers[2].ADD_PACKET(Packet(size=100, ue_id=2, creation_time=current_time), current_time=current_time)
    print(f"UE1 состояние буфера, Б: {bs.ue_buffers[1].sizes[1]}")
    print(f"UE2 состояние буфера, Б: {bs.ue_buffers[2].sizes[2]}")
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
        {
            'UE_ID': 1,
            'buffer_size': bs.ue_buffers[1].sizes[1],
            'cqi': ue1.cqi,
            'ue': ue1
        },
        {
            'UE_ID': 2,
            'buffer_size': bs.ue_buffers[2].sizes[2],
            'cqi': ue2.cqi,
            'ue': ue2
        }
    ]

    #------------------------------------------------------------------
    # Шаг 7: Запуск планировщика для TTI=0
    #------------------------------------------------------------------
    result = scheduler.schedule(0, users)
    allocation = result['allocation']
    stats = result['statistics']
    #dl_thpt = result['dl_throughput']

    print(f"UE1 состояние буфера, Б: {bs.ue_buffers[1].sizes[1]}")
    print(f"UE2 состояние буфера, Б: {bs.ue_buffers[2].sizes[2]}")

    #------------------------------------------------------------------
    # Шаг 8: Анализ распределения RB по слотам
    #------------------------------------------------------------------
    print("\nДетализация распределения RB:")
    subframe = lte_grid.GET_SUBFRAME(0)
    slot0 = subframe.slots[0]
    slot1 = subframe.slots[1]
    slot0_rbs = {rb.freq_idx: rb.UE_ID for rb in slot0.GET_ALL_RES_BLCK()}
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
    print("\nИтоговая DL статистика базовой станции:")
    total_effective = 0
    total_max = 0
    total_throughput = 0
    total_users = len(users)
    #current_time += 1
    
    for user in users:
        ue_id = user['UE_ID']
        rb_count = len(allocation.get(ue_id,[])) * 2
        
        # Использование статистики из планировщика
        throughput = stats['user_throughput'].get(ue_id, 0)
        #effective_bits_d = dl_thpt.get(ue_id, 0)   
        effective_bits = stats['user_effective_throughput'].get(ue_id, 0)
        max_throughput = stats['user_max_throughput'].get(ue_id, 0)
        utilization = (effective_bits / max_throughput * 100) if max_throughput > 0 else 0
        
        print(f"\nUE {ue_id}:")
        print(f"  RB выделено: {rb_count}")
        print(f"  Макс. пропускная способность: {max_throughput:.2f} бит/мс")
        #print(f"  Фактическая: {effective_bits:.2f} бит/мс")
        print(f"  Фактическая: {user['ue'].current_dl_throughput / 1000:.2f} бит/мс")
        #print(f"  Фактическая_дебаг: {effective_bits_d:.2f} бит/мс")
        
        # 3. Расчет утилизации
        utilization = (effective_bits / max_throughput * 100) if max_throughput > 0 else 0
        print(f" Утилизация RBG: {utilization:.1f}%")
        
        # 4. Обновление агрегированных метрик
        total_effective += effective_bits
        total_max += max_throughput
        total_throughput += throughput
    
    # 5. Системные метрики
    system_utilization = (total_effective / total_max * 100) if total_max > 0 else 0
    avg_throughput = total_throughput / total_users if total_users > 0 else 0
    
    print(f"\nСредняя DL пропускная способность: {avg_throughput:.2f} бит/мс")
    print(f"Всего выделено RB: {stats['total_allocated_rbs']}/100")
    print(f"Общая утилизация RBG: {system_utilization:.1f}%")
    print(f"Всего передано данных: {total_throughput/1e6:.2f} Мбит")
    
    # 6. Статистика буфера BS
    bs_status = lte_grid.bs.GET_GLOBAL_BUFFER_STATUS(current_time)
    print("\nСтатус буфера BS:")
    print(f"Общий размер: {bs_status['total_size']/1e3:.1f} КБ")
    print(f"Средняя задержка: {bs_status['avg_delay']:.1f} мс")
    print(f"Макс. задержка: {bs_status['max_delay']} мс")
    
    print("[OK] Тест DL пройден успешно!")
    
def test_scheduler_grid():
    """Тест работы Round Robin планировщика с визуализацией полного фрейма"""
    print("\n=== Тест Proportional Fair (полный фрейм) ===")
    
    # Шаг 1: Инициализация компонентов
    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=2)  # 1 фрейм = 10 TTI
    visualizer = LTEGridVisualizer(lte_grid)
    bs = BaseStation(x=0, y=0, height=25.0, bandwidth=10)
    scheduler = ProportionalFairScheduler(lte_grid, bs)
    current_time = 0

    # Шаг 2: Создание пользователей
    ue1 = UserEquipment(UE_ID=1, x=300, y=300, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=700, y=700, ue_class="car")
    ue3 = UserEquipment(UE_ID=3, x=100, y=200, ue_class="car")

    # Настройка моделей
    ue1.SET_MOBILITY_MODEL(RandomWaypointModel(x_min=0, x_max=1000, y_min=0, y_max=1000, pause_time=10))
    ue1.SET_TRAFFIC_MODEL(PoissonModel(packet_rate=1000))
    ue1.SET_CH_MODEL(UMiModel(bs))
    
    ue2.SET_MOBILITY_MODEL(RandomWalkModel(x_min=0, x_max=1000, y_min=0, y_max=1000))
    ue2.SET_TRAFFIC_MODEL(PoissonModel(packet_rate=10000))
    ue2.SET_CH_MODEL(UMiModel(bs))
    
    ue3.SET_MOBILITY_MODEL(RandomWalkModel(x_min=0, x_max=1000, y_min=0, y_max=1000))
    ue3.SET_TRAFFIC_MODEL(PoissonModel(packet_rate=1000))
    ue3.SET_CH_MODEL(UMiModel(bs))
    
    bs.REG_UE(ue1)
    bs.REG_UE(ue2)
    bs.REG_UE(ue3)
    
    bs.ue_buffers[1].ADD_PACKET(Packet(size=50, ue_id=1, creation_time=current_time), current_time=current_time)
    bs.ue_buffers[2].ADD_PACKET(Packet(size=10000, ue_id=2, creation_time=current_time), current_time=current_time)
    bs.ue_buffers[1].ADD_PACKET(Packet(size=2000, ue_id=1, creation_time=current_time), current_time=current_time)
    bs.ue_buffers[2].ADD_PACKET(Packet(size=3000, ue_id=2, creation_time=current_time), current_time=current_time)
    bs.ue_buffers[3].ADD_PACKET(Packet(size=30000, ue_id=3, creation_time=current_time), current_time=current_time)

    sim_duration = 20 # Время симуляции (в мс)
    update_interval = 1 # Интервал обновления параметров пользователя (в мс)

    cqi_history = {
        1: [],
        2: [],
        3: []
    } 

    # Шаг 3: Основной цикл симуляции 
    for current_time in range(update_interval, sim_duration + 1, update_interval):  # 0-9 TTI (полный фрейм)
    
        # Генерация трафика
# =============================================================================
#         print("\n======== ГЕНЕРАЦИЯ ТРАФИКА ========")
#         ue1.GEN_TRFFC(current_time, update_interval)
#         ue2.GEN_TRFFC(current_time, update_interval)
#         ue3.GEN_TRFFC(current_time, update_interval)
#         print("===================================")
# =============================================================================
        
        # Обновление состояния
        ue1.UPD_POSITION(update_interval, bs.position, bs.height)
        ue2.UPD_POSITION(update_interval, bs.position, bs.height)
        ue3.UPD_POSITION(update_interval, bs.position, bs.height)
        ue1.UPD_CH_QUALITY()
        ue2.UPD_CH_QUALITY()
        ue3.UPD_CH_QUALITY()
        
        # Логирование CQI
        cqi_history[1].append(ue1.cqi)
        cqi_history[2].append(ue2.cqi)
        cqi_history[3].append(ue3.cqi)
        
        # Цикл по каждому TTI
        for tti in range(current_time - update_interval, current_time):
            # Подготовка данных для планировщика
            users = [
                {
                    'UE_ID': 1,
                    'buffer_size': bs.ue_buffers[1].sizes[1],
                    'cqi': ue1.cqi,
                    'ue': ue1
                },
                {
                    'UE_ID': 2,
                    'buffer_size': bs.ue_buffers[2].sizes[2],
                    'cqi': ue2.cqi,
                    'ue': ue2
                },
                {
                    'UE_ID': 3,
                    'buffer_size': bs.ue_buffers[3].sizes[3],
                    'cqi': ue3.cqi,
                    'ue': ue3
                }
            ]
            
            
            # Вывод параметров для каждого TTI
            print(f"\n[TTI {tti}]")
            print(f"CQI: UE1={ue1.cqi}, UE2={ue2.cqi}, UE3={ue3.cqi}")
            print(f"Buffer Size: UE1={bs.ue_buffers[1].sizes[1]}B, UE2={bs.ue_buffers[2].sizes[2]}B, UE3={bs.ue_buffers[3].sizes[3]}B")
            
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
    
    subframe = lte_grid.GET_SUBFRAME(0)
    slot0 = subframe.slots[0]
    slot1 = subframe.slots[1]
    slot0_rbs = {rb.freq_idx: rb.UE_ID for rb in slot0.GET_ALL_RES_BLCK()}
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
    print("[OK] Тест завершен с визуализацией")
    
    subframe = lte_grid.GET_SUBFRAME(1)
    slot0 = subframe.slots[0]
    slot1 = subframe.slots[1]
    slot0_rbs = {rb.freq_idx: rb.UE_ID for rb in slot0.GET_ALL_RES_BLCK()}
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
    print("[OK] Тест завершен с визуализацией")
    
    subframe = lte_grid.GET_SUBFRAME(2)
    slot0 = subframe.slots[0]
    slot1 = subframe.slots[1]
    slot0_rbs = {rb.freq_idx: rb.UE_ID for rb in slot0.GET_ALL_RES_BLCK()}
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
    print("[OK] Тест завершен с визуализацией")

def test_scheduler_with_metrics():
    
    sim_duration = 5000 # Время симуляции (в мс)
    update_interval = 5 # Интервал обновления параметров пользователя (в мс)
    num_frames = int(np.ceil(sim_duration / 10)) # Кол-во кадров (для ресурсной сетки)
    bandwidth = 10 # Ширина полосы (в МГц)
    inf = math.inf 

    # Настройка базовой станции и пользователей
    bs = BaseStation(x=1000, y=1000, height=25.0, bandwidth=bandwidth, global_max=inf, per_ue_max=inf)
    ue1 = UserEquipment(UE_ID=1, x=800, y=800, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=500, y=500, ue_class="car")
    ue3 = UserEquipment(UE_ID=3, x=100, y=100, ue_class="car")
    
    ue1.SET_MOBILITY_MODEL(RandomWaypointModel(x_min=0, x_max=2000, y_min=0, y_max=2000, pause_time=10))
    ue1.SET_CH_MODEL(UMiModel(bs))
    
    ue2.SET_MOBILITY_MODEL(RandomWalkModel(x_min=0, x_max=2000, y_min=0, y_max=2000))
    ue2.SET_CH_MODEL(UMiModel(bs))
    
    ue3.SET_MOBILITY_MODEL(RandomWalkModel(x_min=0, x_max=2000, y_min=0, y_max=2000))
    ue3.SET_CH_MODEL(UMiModel(bs))
    
    bs.REG_UE(ue1)
    bs.REG_UE(ue2)
    bs.REG_UE(ue3)
    
    # Имитация Full Buffer 
    bs.ue_buffers[1].ADD_PACKET(Packet(size=inf, ue_id=1, creation_time=0), current_time=0)
    bs.ue_buffers[2].ADD_PACKET(Packet(size=inf, ue_id=2, creation_time=0), current_time=0)
    bs.ue_buffers[3].ADD_PACKET(Packet(size=inf, ue_id=3, creation_time=0), current_time=0)
    
    # Настройка ресурсной сетки и планировщика
    lte_grid = RES_GRID_LTE(bandwidth=bandwidth, num_frames=num_frames)
    scheduler = ProportionalFairScheduler(lte_grid, bs)
    
    total_throughput_tti = []
    
    users_throughput = {
        1: [],
        2: [],
        3: []
    }
    
    # Основной цикл симуляции (обновление различных параметров у пользователей)
    for current_time in range(update_interval, sim_duration + 1, update_interval):
        
        # Обновление местоположения пользователей
        ue1.UPD_POSITION(update_interval, bs.position, bs.height)
        ue2.UPD_POSITION(update_interval, bs.position, bs.height)
        ue3.UPD_POSITION(update_interval, bs.position, bs.height)
        
        # Обновление состояния канала пользователей
        ue1.UPD_CH_QUALITY()
        ue2.UPD_CH_QUALITY()
        ue3.UPD_CH_QUALITY() 
        
        # Цикл для планирования ресурсов (по TTI)
        for tti in range(current_time - update_interval, current_time): 
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
            
            # Вывод параметров для каждого TTI
            print(f"\n[TTI {tti}]")
            print(f"CQI: UE1={ue1.cqi}, UE2={ue2.cqi}, UE3={ue3.cqi}")
            print(f"Buffer Size: UE1={bs.ue_buffers[1].sizes[1]}B, UE2={bs.ue_buffers[2].sizes[2]}B, UE3={bs.ue_buffers[3].sizes[3]}B")
            
            # Запуск планировщика
            scheduler.schedule(tti, users)
            
            total_throughput = 0
            for user in users:
                throughput = user['ue'].current_dl_throughput
                total_throughput += throughput
                
            # Пропускная способность на TTI         
            total_throughput = (total_throughput) / 1e6 # бит/с -> мбит/с
            total_throughput_tti.append(total_throughput)
    
            users_throughput[1].append(ue1.current_dl_throughput / 1e6)
            users_throughput[2].append(ue2.current_dl_throughput / 1e6)
            users_throughput[3].append(ue3.current_dl_throughput / 1e6)
                
            
    # Подготовка данных для построения графиков
    frame_throughput = [
    np.mean(total_throughput_tti[i:i+10]) 
    for i in range(0, len(total_throughput_tti), 10)
    ]
    
    users_frame_throughput = {
    ue_id: [
        np.mean(user_tti_throughputs[i:i+10]) 
        for i in range(0, len(user_tti_throughputs), 10)
    ]
    for ue_id, user_tti_throughputs in users_throughput.items()
    }
    
    average_throughput_per_user = {
    ue_id: np.mean(throughput) for ue_id, throughput in users_throughput.items()
    }
    
    # Расчет Jain's Fairness Index для каждого кадра
    fairness_per_frame = []
    for i in range(num_frames):
        frame_throughputs = [users_frame_throughput[ue_id][i] for ue_id in users_frame_throughput]
        frame_throughputs = np.array(frame_throughputs)
        fairness = (np.sum(frame_throughputs) ** 2) / (len(frame_throughputs) * np.sum(frame_throughputs ** 2))
        fairness_per_frame.append(fairness)
        
    # Расчет Jain's Fairness Index для всей симуляции
    throughputs = np.array(list(average_throughput_per_user.values()))
    fairness_index = (np.sum(throughputs) ** 2) / (len(throughputs) * np.sum(throughputs ** 2))

    # Расчёт спектральной эффективности соты
    spectral_efficiency_cell = np.array(frame_throughput) / bandwidth
    
    print(f"\nСредняя пропускная способность соты за симуляцию: {np.mean(total_throughput_tti):.4f} Мбит/с")
    print(f"Индекс справедливости Джайна: {fairness_index:.4f}")
    print(f"Средняя спектральная эффективность соты за симуляцию: {np.mean(spectral_efficiency_cell):.4f} бит/с/Гц\n")
    
    metrics = {
        "sim_duration": sim_duration,
        "cell_throughput": frame_throughput,
        "user_throughput": users_frame_throughput,
        "avg_user_throughput": average_throughput_per_user,
        "jain_index_per_frame": fairness_per_frame,
        "jain_index_overall": fairness_index,
        "spectral_efficiency": spectral_efficiency_cell.tolist()
    }

    save_scheduler_metrics(scheduler.__class__.__name__, metrics)
    print(f"Метрики для {scheduler.__class__.__name__} сохранены в файл.")

def save_scheduler_metrics(scheduler_name, metrics, filename='metrics_results.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[scheduler_name] = metrics

    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == "__main__":
    #test_scheduler_with_buffer()
    #test_visualize_lte_timeline()
    #test_scheduler_grid()
    test_scheduler_with_metrics()
    print("Все тесты успешно пройдены!")

