"""
#------------------------------------------------------------------------------
# Модуль: RES_GRID_LTE - Модель ресурсной сетки LTE
#------------------------------------------------------------------------------
# Описание:
#   Предоставляет классы и методы для моделирования частотно-временной ресурсной
#   сетки LTE. Позволяет создавать сетку с заданными параметрами, выделять
#   и освобождать ресурсные блоки, а также визуализировать состояние сетки.
#
# Версия: 1.0.1
# Дата последнего изменения: 2025-03-15
# Автор: Брагин Кирилл
# Версия Python kernel 3.12.9
№
# Используемые библиотеки:
#   - numpy: Для работы с массивами и матрицами
#   - matplotlib: Для визуализации данных
#   - typing: Для аннотации типов
#
# Изменения:
#   v1.0.0 - 2025-03-15:
#     - Исходная версия модуля с базовой функциональностью
#     - Создание классов RES_BLCK, Slot, Subframe, Frame, RES_GRID_LTE
#     - Реализация методов выделения и освобождения RB
#     - Добавлена визуализация пустой сетки
#     - Добавлен пример заполнения сетки ресурсами + визуализация
#
#   v1.0.1 - 2025-03-15:
#     - Добавлено [карточка]
#     - Сделал заготовку класса под планировщик
#     - allocate -> ASSIGN_RB
#     - release -> RELEASE_RB
#
#------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

class RES_BLCK:
    """
    Класс, представляющий ресурсный блок (RB) в сетке.
    """
    def __init__(self, id: str, time_idx: int, freq_idx: int):
        """
        Инициализация ресурсного блока.
        
        Args:
            id: Уникальный идентификатор блока
            time_idx: Временной индекс (в рамках TTI)
            freq_idx: Частотный индекс
        """
        self.id = id
        self.time_idx = time_idx
        self.freq_idx = freq_idx
        self.UE_ID = None  # ID пользователя, которому назначен RB
        self.status = "free"  # Статус: "free" или "assigned"
    
    def ASSIGN_RB(self, UE_ID: int) -> bool:
        """
        Назначить ресурсный блок пользователю.
        
        Args:
            UE_ID: Идентификатор пользователя
            
        Returns:
            bool: True, если блок успешно назначен, False в противном случае
        """
        if self.status == "free":
            self.UE_ID = UE_ID
            self.status = "assigned"
            return True
        return False
    
    def RELEASE_RB(self) -> bool:
        """
        Освободить ресурсный блок.
        
        Returns:
            bool: True, если блок успешно освобожден
        """
        self.UE_ID = None
        self.status = "free"
        return True
    
    def CHCK_RB(self) -> bool:
        """
        Проверить, свободен ли ресурсный блок.
        
        Returns:
            bool: True, если блок свободен, False в противном случае
        """
        return self.status == "free"
    
    def __repr__(self) -> str:
        """Строковое представление объекта"""
        return f"RB(id={self.id}, time={self.time_idx}, freq={self.freq_idx}, status={self.status}, user={self.UE_ID})"


class Slot:
    """
    Класс, представляющий слот в структуре LTE.
    Слот содержит набор ресурсных блоков по частоте.
    """
    def __init__(self, slot_id: int, num_rb: int):
        """
        Инициализация слота.
        
        Args:
            slot_id: Идентификатор слота
            num_rb: Количество ресурсных блоков по частоте
        """
        self.id = slot_id
        self.resource_blocks = {}
        
        # Создание ресурсных блоков для слота
        for rb_idx in range(num_rb):
            rb_id = f"RB_{slot_id}_{rb_idx}"
            self.resource_blocks[rb_id] = RES_BLCK(rb_id, slot_id, rb_idx)
    
    def GET_RES_BLCK(self, rb_idx: int) -> Optional[RES_BLCK]:
        """
        Получить ресурсный блок по частотному индексу.
        
        Args:
            rb_idx: Частотный индекс ресурсного блока
            
        Returns:
            RES_BLCK или None, если блок не найден
        """
        rb_id = f"RB_{self.id}_{rb_idx}"
        return self.resource_blocks.get(rb_id)
    
    def GET_ALL_RES_BLCK(self) -> List[RES_BLCK]:
        """
        Получить все ресурсные блоки в слоте.
        
        Returns:
            List[RES_BLCK]: Список всех ресурсных блоков
        """
        return list(self.resource_blocks.values())
    
    def GET_FREE_RES_BLCK(self) -> List[RES_BLCK]:
        """
        Получить все свободные ресурсные блоки в слоте.
        
        Returns:
            List[RES_BLCK]: Список свободных ресурсных блоков
        """
        return [rb for rb in self.resource_blocks.values() if rb.CHCK_RB()]


class Subframe:
    """
    Класс, представляющий подкадр (TTI) в структуре LTE.
    Подкадр состоит из 2 слотов.
    """
    def __init__(self, subframe_id: int, num_rb: int):
        """
        Инициализация подкадра.
        
        Args:
            subframe_id: Идентификатор подкадра
            num_rb: Количество ресурсных блоков по частоте
        """
        self.id = subframe_id
        # Слот 0 и слот 1 в подкадре
        self.slots = [
            Slot(f"{subframe_id}_0", num_rb),
            Slot(f"{subframe_id}_1", num_rb)
        ]
    
    def GET_SLOT(self, slot_idx: int) -> Optional[Slot]:
        """
        Получить слот по индексу.
        
        Args:
            slot_idx: Индекс слота (0 или 1)
            
        Returns:
            Slot или None, если слот не найден
        """
        if 0 <= slot_idx < len(self.slots):
            return self.slots[slot_idx]
        return None
    
    def GET_ALL_RES_BLCK(self) -> List[RES_BLCK]:
        """
        Получить все ресурсные блоки в подкадре.
        
        Returns:
            List[RES_BLCK]: Список всех ресурсных блоков
        """
        all_rbs = []
        for slot in self.slots:
            all_rbs.extend(slot.GET_ALL_RES_BLCK())
        return all_rbs
    
    def GET_FREE_RES_BLCK(self) -> List[RES_BLCK]:
        """
        Получить все свободные ресурсные блоки в подкадре.
        
        Returns:
            List[RES_BLCK]: Список свободных ресурсных блоков
        """
        free_rbs = []
        for slot in self.slots:
            free_rbs.extend(slot.GET_FREE_RES_BLCK())
        return free_rbs


class Frame:
    """
    Класс, представляющий кадр в структуре LTE.
    Кадр состоит из 10 подкадров.
    """
    def __init__(self, frame_id: int, num_rb: int):
        """
        Инициализация кадра.
        
        Args:
            frame_id: Идентификатор кадра
            num_rb: Количество ресурсных блоков по частоте
        """
        self.id = frame_id
        # 10 подкадров в кадре
        self.subframes = [Subframe(f"{frame_id}_{i}", num_rb) for i in range(10)]
    
    def GET_SUBFRAME(self, subframe_idx: int) -> Optional[Subframe]:
        """
        Получить подкадр по индексу.
        
        Args:
            subframe_idx: Индекс подкадра (0-9)
            
        Returns:
            Subframe или None, если подкадр не найден
        """
        if 0 <= subframe_idx < len(self.subframes):
            return self.subframes[subframe_idx]
        return None
    
    def GET_ALL_RES_BLCK(self) -> List[RES_BLCK]:
        """
        Получить все ресурсные блоки в кадре.
        
        Returns:
            List[RES_BLCK]: Список всех ресурсных блоков
        """
        all_rbs = []
        for subframe in self.subframes:
            all_rbs.extend(subframe.GET_ALL_RES_BLCK())
        return all_rbs
    
    def GET_FREE_RES_BLCK(self) -> List[RES_BLCK]:
        """
        Получить все свободные ресурсные блоки в кадре.
        
        Returns:
            List[RES_BLCK]: Список свободных ресурсных блоков
        """
        free_rbs = []
        for subframe in self.subframes:
            free_rbs.extend(subframe.GET_FREE_RES_BLCK())
        return free_rbs


class RES_GRID_LTE:
    """
    Основной класс для моделирования ресурсной сетки LTE.
    """
    # Словарь соответствия полосы частот и количества RB согласно стандарту LTE
    BANDWIDTH_TO_RB = {
        1.4: 6,
        3: 15,
        5: 25,
        10: 50,
        15: 75,
        20: 100
    }
    
    def __init__(self, bandwidth: float = 10, num_frames: int = 10, cp_type: str = "normal"):
        """
        Инициализация ресурсной сетки LTE.
        
        Args:
            bandwidth: Полоса частот в МГц (1.4, 3, 5, 10, 15, 20)
            num_frames: Количество кадров для симуляции
            cp_type: Тип циклического префикса ("normal" или "extended")
        """
        if bandwidth not in self.BANDWIDTH_TO_RB:
            raise ValueError(f"Недопустимая полоса частот. Допустимые значения: {list(self.BANDWIDTH_TO_RB.keys())}")
        
        self.bandwidth = bandwidth
        self.num_rb = self.BANDWIDTH_TO_RB[bandwidth]
        self.num_frames = num_frames
        self.cp_type = cp_type
        self.total_tti = num_frames * 10  # Общее количество TTI (10 TTI на кадр)
        
        # Создание кадров
        self.frames = [Frame(i, self.num_rb) for i in range(num_frames)]
        
        # Словарь для быстрого доступа к RB по TTI и частотному индексу
        self.rb_map = {}
        
        # Текущий TTI (счетчик от 0 до num_frames * 10 - 1)
        self.current_tti = 0
        
        # Статистика использования ресурсов
        self.stats = {
            "allocated_rbs": 0,
            "total_rbs": self.num_rb * self.total_tti,
            "allocation_by_user": {},
            "allocation_by_tti": {}
        }
        
        # Инициализация rb_map для быстрого доступа к ресурсным блокам
        self._init_rb_map()
    
    def _init_rb_map(self):
        """Инициализация карты ресурсных блоков для быстрого доступа"""
        for frame_idx, frame in enumerate(self.frames):
            for sf_idx, subframe in enumerate(frame.subframes):
                tti = frame_idx * 10 + sf_idx
                self.stats["allocation_by_tti"][tti] = 0
                for rb in subframe.GET_ALL_RES_BLCK():
                    key = (tti, rb.freq_idx)
                    self.rb_map[key] = rb
    
    def GET_RB(self, tti: int, freq_idx: int) -> Optional[RES_BLCK]:
        """
        Получить ресурсный блок по TTI и частотному индексу.
        
        Args:
            tti: Индекс TTI
            freq_idx: Частотный индекс
            
        Returns:
            RES_BLCK или None, если блок не найден
        """
        key = (tti, freq_idx)
        return self.rb_map.get(key)
    
    def ALLOCATE_RB(self, tti: int, freq_idx: int, UE_ID: int) -> bool:
        """
        Назначить ресурсный блок пользователю.
        
        Args:
            tti: Индекс TTI
            freq_idx: Частотный индекс
            UE_ID: Идентификатор пользователя
            
        Returns:
            bool: True, если блок успешно назначен, False в противном случае
        """
        rb = self.GET_RB(tti, freq_idx)
        if rb and rb.CHCK_RB():
            if rb.ASSIGN_RB(UE_ID):
                # Обновление статистики
                self.stats["allocated_rbs"] += 1
                self.stats["allocation_by_tti"][tti] = self.stats["allocation_by_tti"].get(tti, 0) + 1
                self.stats["allocation_by_user"][UE_ID] = self.stats["allocation_by_user"].get(UE_ID, 0) + 1
                return True
        return False
    
    def ALLOCATE_RB_GROUP(self, tti: int, freq_indices: List[int], UE_ID: int) -> int:
        """
        Назначить группу ресурсных блоков пользователю.
        
        Args:
            tti: Индекс TTI
            freq_indices: Список частотных индексов
            UE_ID: Идентификатор пользователя
            
        Returns:
            int: Количество успешно назначенных блоков
        """
        allocated_count = 0
        for freq_idx in freq_indices:
            if self.ALLOCATE_RB(tti, freq_idx, UE_ID):
                allocated_count += 1
        return allocated_count
    
    def GET_FRAME(self, frame_idx: int) -> Optional[Frame]:
        """
        Получить кадр по индексу.
        
        Args:
            frame_idx: Индекс кадра
            
        Returns:
            Frame или None, если кадр не найден
        """
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None
    
    def GET_SUBFRAME(self, tti: int) -> Optional[Subframe]:
        """
        Получить подкадр по TTI.
        
        Args:
            tti: Индекс TTI
            
        Returns:
            Subframe или None, если подкадр не найден
        """
        frame_idx = tti // 10
        sf_idx = tti % 10
        frame = self.GET_FRAME(frame_idx)
        if frame:
            return frame.GET_SUBFRAME(sf_idx)
        return None
    
    def GET_FREE_RB_FOR_TTI(self, tti: int) -> List[RES_BLCK]:
        """
        Получить список свободных RB для заданного TTI.
        Возможно, стоит использовать обозначение субкадра для правильности,
        но пока оставил временной интервал, это практически удобнее
        
        Args:
            tti: Индекс TTI
            
        Returns:
            List[RES_BLCK]: Список свободных ресурсных блоков
        """
        if tti >= self.total_tti:
            return []
        
        frame_idx = tti // 10
        sf_idx = tti % 10
        frame = self.frames[frame_idx]
        subframe = frame.subframes[sf_idx]
        return subframe.GET_FREE_RES_BLCK()
    
    def GET_TTI_STATUS(self, tti: int) -> Dict[int, Optional[int]]:
        """
        Получить статус распределения ресурсных блоков для заданного TTI.
        
        Args:
            tti: Индекс TTI
            
        Returns:
            Dict[int, Optional[int]]: Словарь {freq_idx: UE_ID}, где UE_ID=None для свободных блоков
        """
        result = {}
        if tti >= self.total_tti:
            return result
        
        for freq_idx in range(self.num_rb):
            rb = self.GET_RB(tti, freq_idx)
            if rb:
                result[freq_idx] = rb.UE_ID
        
        return result
    
    def NEXT_TTI(self) -> bool:
        """
        Переход к следующему TTI.
        
        Returns:
            bool: True, если есть еще TTI, False если достигнут конец симуляции
        """
        self.current_tti += 1
        return self.current_tti < self.total_tti
    
    def RESET_GRID(self):
        """Сброс всех назначений ресурсных блоков"""
        for frame in self.frames:
            for rb in frame.GET_ALL_RES_BLCK():
                rb.RELEASE_RB_RB()
        
        self.current_tti = 0
        self.stats = {
            "allocated_rbs": 0,
            "total_rbs": self.num_rb * self.total_tti,
            "allocation_by_user": {},
            "allocation_by_tti": {tti: 0 for tti in range(self.total_tti)}
        }
    
    def GET_GRID_STATUS(self) -> Dict[int, Dict[int, Dict[str, Union[str, Optional[int]]]]]:
        """
        Получить текущий статус всей ресурсной сетки.
        
        Returns:
            Dict: Словарь {tti: {freq_idx: {"status": str, "UE_ID": Optional[int]}}}
        """
        status = {}
        for tti in range(self.total_tti):
            status[tti] = {}
            for freq_idx in range(self.num_rb):
                rb = self.GET_RB(tti, freq_idx)
                if rb:
                    status[tti][freq_idx] = {
                        "status": rb.status,
                        "UE_ID": rb.UE_ID
                    }
        return status
    
    def visualize_grid(self, tti_start: int = 0, tti_end: Optional[int] = None, 
                       show_UE_IDs: bool = True, save_path: Optional[str] = None):
        """
        Визуализация ресурсной сетки.Живет тут для примера и проверки кода.
        Потом выпилить отсюда, и впилить в отдельный модуль.
        
        Args:
            tti_start: Начальный TTI для визуализации
            tti_end: Конечный TTI для визуализации (не включительно)
            show_UE_IDs: Показывать ли ID пользователей на графике
            save_path: Путь для сохранения изображения (если None, то изображение отображается)
        """
        if tti_end is None:
            tti_end = min(tti_start + 10, self.total_tti)
        
        # Создаем матрицу для отображения статуса RB
        # 0 для свободных, >0 для занятых блоков (значение = UE_ID)
        grid = np.zeros((self.num_rb, tti_end - tti_start))
        
        for tti in range(tti_start, tti_end):
            for freq_idx in range(self.num_rb):
                rb = self.GET_RB(tti, freq_idx)
                if rb:
                    if rb.CHCK_RB():
                        grid[freq_idx, tti - tti_start] = 0
                    else:
                        # Для визуализации используем UE_ID как цвет
                        grid[freq_idx, tti - tti_start] = rb.UE_ID if rb.UE_ID is not None else 0
        
        plt.figure(figsize=(15, 10))
        
        # Создаем маску для свободных блоков
        mask_free = (grid == 0)
        
        # Создаем кастомную цветовую карту
        cmap = plt.cm.jet
        cmap.set_bad('white', 1.0)  # Свободные блоки будут белыми
        
        # Создаем матрицу с NaN для свободных блоков
        grid_masked = np.ma.array(grid, mask=mask_free)
        
        # Отображаем матрицу
        plt.imshow(grid_masked, aspect='auto', cmap=cmap, interpolation='nearest')
        plt.colorbar(label='User ID')
        
        # Отдельно отображаем свободные блоки
        plt.imshow(mask_free, aspect='auto', cmap='binary', alpha=0.3, interpolation='nearest')
        
        plt.xlabel('TTI')
        plt.ylabel('Frequency (RB index)')
        plt.title(f'LTE Resource Grid (Bandwidth: {self.bandwidth} MHz, TTI: {tti_start}-{tti_end-1})')
        
        # Настройка осей
        plt.yticks(np.arange(0, self.num_rb, 5))
        plt.xticks(np.arange(0, tti_end - tti_start, 1), np.arange(tti_start, tti_end, 1))
        
        # Добавление линий для разделения кадров
        for i in range(tti_start, tti_end, 10):
            rel_i = i - tti_start
            if 0 <= rel_i < (tti_end - tti_start):
                plt.axvline(x=rel_i, color='red', linestyle='-', linewidth=1, label='Frame boundary' if i == tti_start else None)
        
        # Добавление линий для разделения подкадров
        for i in range(tti_start, tti_end):
            rel_i = i - tti_start
            if 0 <= rel_i < (tti_end - tti_start):
                plt.axvline(x=rel_i, color='gray', linestyle='--', linewidth=0.5)
        
        # Добавление текста с ID пользователей, если требуется
        if show_UE_IDs:
            for tti in range(tti_start, tti_end):
                for freq_idx in range(self.num_rb):
                    rb = self.GET_RB(tti, freq_idx)
                    if rb and not rb.CHCK_RB():
                        plt.text(tti - tti_start, freq_idx, str(rb.UE_ID), 
                                 ha='center', va='center', color='black', fontsize=8)
        
        plt.grid(True, color='black', linestyle='-', linewidth=0.2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.gca().invert_yaxis()
            plt.show()


class SchedulerInterface:
    """
    Интерфейс для планировщиков ресурсов.
    Все планировщики должны наследоваться от этого класса и реализовывать метод schedule.
    """
    def __init__(self, lte_grid: RES_GRID_LTE):
        """
        Инициализация интерфейса планировщика.
        
        Args:
            lte_grid: Объект RES_GRID_LTE для работы с ресурсной сеткой
        """
        self.lte_grid = lte_grid
    
    def schedule(self, tti: int, users: List[Dict]):
        """
        Метод планирования ресурсов для заданного TTI.
        
        Args:
            tti: Индекс TTI для планирования
            users: Список пользователей с их параметрами
            
        Returns:
            Dict: Результаты планирования
        """
        raise NotImplementedError("Этот метод должен быть переопределен в дочернем классе")


def example_usage():
    """Пример использования модели ресурсной сетки LTE"""
    # Создание ресурсной сетки с полосой 10 МГц (50 RB) и 1 кадром (10 TTI)
    lte_grid = RES_GRID_LTE(bandwidth=3, num_frames=1)
    
    print(f"Создана LTE сетка с полосой {lte_grid.bandwidth} МГц")
    print(f"Количество ресурсных блоков: {lte_grid.num_rb}")
    print(f"Количество TTI: {lte_grid.total_tti}")
    
    # Визуализация пустой сетки
    print("Визуализация пустой ресурсной сетки рис1")
    lte_grid.visualize_grid()
    
    # Назначение нескольких ресурсных блоков для проверки визуализации
    lte_grid.ALLOCATE_RB(0, 0, 1)  # TTI 0, RB 0, User 1
    lte_grid.ALLOCATE_RB(0, 1, 1)  # TTI 0, RB 1, User 1
    lte_grid.ALLOCATE_RB(1, 5, 2)  # TTI 1, RB 5, User 2
    lte_grid.ALLOCATE_RB_GROUP(2, [10, 11, 12], 3)  # TTI 2, RBs 10-12, User 3
    
    # Визуализация сетки с назначенными ресурсами
    print("Визуализация сетки с назначенными ресурсами рис2")
    lte_grid.visualize_grid()
    
    return lte_grid


if __name__ == "__main__":
    example_usage()
