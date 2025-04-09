"""
#------------------------------------------------------------------------------
# Модуль: RES_GRID_LTE - Модель ресурсной сетки LTE
#------------------------------------------------------------------------------
# Описание:
#   Предоставляет классы и методы для моделирования частотно-временной ресурсной
#   сетки LTE. Позволяет создавать сетку с заданными параметрами, выделять
#   и освобождать ресурсные блоки, а также визуализировать состояние сетки.
#
# Версия: 1.0.2
# Дата последнего изменения: 2025-04-09
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
#   v1.0.2 - 2025-04-15:
#     - Переписал весь код модуля ресурсной сетки. Теперь работает адекватно.
#     Ранее работали неадекватно методы распределения ресурсов. Неправильно
#     расчитывались слоты и не работали с планировщиком. Неправильно отрисовывалась сетка.
#     Неправильно индексировались ресурсные блоки.
#     - Добавлены тесты. Теперь будем знать, что работает не так.
#     - Добавлена визуализация ресурсной сетки. Так много я еще не страдал.
#     - На будущее. Убрать тесты в отдельный модуль.
#------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
# %matplotlib

class RES_BLCK:
    """
    Класс, представляющий ресурсный блок (RB) в сетке.
    """
    def __init__(self, id: str, slot_id: str, freq_idx: int):
        """
        Инициализация ресурсного блока.
        
        Args:
            id: Уникальный идентификатор блока
            slot_id: Временной индекс [0 или 1] (в рамках TTI)
            freq_idx: Частотный индекс
        """
        self.id = id
        self.slot_id = slot_id
        # self.tti_idx = tti_idx
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
    
    # def __repr__(self) -> str:
    #     """Строковое представление объекта"""
    #     return f"RB(id={self.id}, time={self.time_idx}, freq={self.freq_idx}, status={self.status}, user={self.UE_ID})"


class Slot:
    """
    Класс, представляющий слот в структуре LTE.
    Слот содержит набор ресурсных блоков по частоте.
    """
    def __init__(self, slot_id: str, rb_per_slot: int):
        """
        Инициализация слота.
        
        Args:
            slot_id: Идентификатор слота
            num_rb: Количество ресурсных блоков по частоте
        """
        self.id = slot_id
        self.resource_blocks: Dict[str, RES_BLCK] = {}
        
        for rb_idx in range(rb_per_slot):
            rb_id = f"RB_{slot_id}_{rb_idx}"  # Унифицированный формат
            self.resource_blocks[rb_id] = RES_BLCK(rb_id, slot_id, rb_idx)
    
    def GET_RES_BLCK(self, freq_idx: int) -> Optional[RES_BLCK]:
        """
        Получить ресурсный блок по частотному индексу.
        
        Args:
            rb_idx: Частотный индекс ресурсного блока
            
        Returns:
            RES_BLCK или None, если блок не найден
        """
       
        for rb in self.resource_blocks.values():
            if rb.freq_idx == freq_idx:
                return rb
        return None
    
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
    def __init__(self, subframe_id: int, rb_per_slot: int):
        """
        Инициализация подкадра.
        
        Args:
            subframe_id: Идентификатор подкадра
            rb_per_slot: Количество ресурсных блоков в слоте
        """
        self.id = subframe_id  
        self.slots = [
            Slot(f"sub_{subframe_id}_slot_0", rb_per_slot),
            Slot(f"sub_{subframe_id}_slot_1", rb_per_slot)
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
    def __init__(self, frame_id: int, rb_per_slot: int):
        """
        Инициализация кадра.
        
        Args:
            frame_id: Идентификатор кадра
            num_rb: Количество ресурсных блоков по частоте
        """
        self.id = frame_id
        self.subframes = [Subframe(i, rb_per_slot) for i in range(10)]
    
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
        1.4: 6,    # 6 RB на слот → 12 RB на TTI
        3: 15,     # 15 RB на слот → 30 RB на TTI
        5: 25,     # 25 RB на слот → 50 RB на TTI
        10: 50,    # 50 RB на слот → 100 RB на TTI
        15: 75,    # 75 RB на слот → 150 RB на TTI
        20: 100    # 100 RB на слот → 200 RB на TTI
    }
    
    def __init__(self, bandwidth: float = 10, num_frames: int = 10, cp_type: str = "normal"):
        """
        Инициализация ресурсной сетки LTE.
        
        Args:
            bandwidth: Полоса частот в МГц (1.4, 3, 5, 10, 15, 20)
            num_frames: Количество кадров для симуляции
            cp_type: Тип циклического префикса ("normal" или "extended")
        """
        self.bandwidth = bandwidth
        if bandwidth not in self.BANDWIDTH_TO_RB:
            raise ValueError(f"Недопустимая полоса частот. Допустимые значения: {list(self.BANDWIDTH_TO_RB.keys())}")
        
        self.rb_per_slot = self.BANDWIDTH_TO_RB[bandwidth]
        self.num_rb = self.rb_per_slot * 2  # 2 слота на TTI
        self.frames = [Frame(i, self.rb_per_slot) for i in range(num_frames)]
        self.total_tti = num_frames * 10
        
        # Словарь для быстрого доступа к RB по TTI и частотному индексу
        self.rb_map: Dict[tuple, RES_BLCK] = {}
        self._init_rb_map()
        
        # Текущий TTI (счетчик от 0 до num_frames * 10 - 1)
        self.current_tti = 0
        
        # Статистика использования ресурсов
        self.stats = {
            "allocated_rbs": 0,
            "total_rbs": self.num_rb * self.total_tti,
            "allocation_by_user": {},
            "allocation_by_tti": {tti: 0 for tti in range(self.total_tti)}  # Инициализация всех TTI
        }
        # Инициализация rb_map для быстрого доступа к ресурсным блокам
        self._init_rb_map()
    
    def _init_rb_map(self):
        """Инициализация карты ресурсных блоков для быстрого доступа"""
        for frame in self.frames:
            for subframe in frame.subframes:
                tti = frame.id * 10 + subframe.id
                for slot in subframe.slots:
                    for rb in slot.resource_blocks.values():
                        key = (tti, slot.id, rb.freq_idx)
                        self.rb_map[key] = rb
    
    def GET_RB(self, tti: int, slot_id: str, freq_idx: int) -> Optional[RES_BLCK]:
        """
        Получить ресурсный блок по TTI и частотному индексу.
        
        Args:
            tti: Индекс TTI
            freq_idx: Частотный индекс
            
        Returns:
            RES_BLCK или None, если блок не найден
        """
        return self.rb_map.get((tti, slot_id, freq_idx))
    
    def ALLOCATE_RB(self, tti: int, slot_id: str, freq_idx: int, UE_ID: int) -> bool:
        """
        Назначить ресурсный блок пользователю.
        
        Args:
            tti: Индекс TTI
            freq_idx: Частотный индекс
            UE_ID: Идентификатор пользователя
            slot_idx: Индекс слота (0 или 1)
        Returns:
            bool: True, если блок успешно назначен, False в противном случае
        """
        if freq_idx >= self.rb_per_slot:
            print(f"Предупреждение: индекс {freq_idx} выходит за границы допустимого диапазона (0-{self.rb_per_slot-1})")
            return False
        
        rb = self.GET_RB(tti, slot_id, freq_idx)
        if rb and rb.CHCK_RB():
            success = rb.ASSIGN_RB(UE_ID)
            if success:
                self.stats["allocated_rbs"] += 1
                self.stats["allocation_by_user"][UE_ID] = self.stats["allocation_by_user"].get(UE_ID, 0) + 1
                self.stats["allocation_by_tti"][tti] += 1
            return success
        return False
    
    def ALLOCATE_RB_GROUP(self, tti: int, freq_idx: int, UE_ID: int) -> bool:
        success = True
        for slot in ["sub_{}_slot_0", "sub_{}_slot_1"]:
            slot_id = slot.format(tti)
            if not self.ALLOCATE_RB(tti, slot_id, freq_idx, UE_ID):
                success = False
                self.RELEASE_RB(tti, slot_id, freq_idx)
        return success
    
    def RELEASE_RB(self, tti: int, slot_id: str, freq_idx: int) -> bool:
        rb = self.GET_RB(tti, slot_id, freq_idx)
        if rb and not rb.CHCK_RB():
            UE_ID = rb.UE_ID
            if rb.RELEASE_RB():
                self.stats["allocated_rbs"] -= 1
                
                # Корректное уменьшение счетчика пользователя
                current_user_count = self.stats["allocation_by_user"].get(UE_ID, 0)
                current_user_count -= 1
                if current_user_count <= 0:
                    if UE_ID in self.stats["allocation_by_user"]:
                        del self.stats["allocation_by_user"][UE_ID]
                else:
                    self.stats["allocation_by_user"][UE_ID] = current_user_count
                
                # Уменьшение счетчика TTI
                if tti in self.stats["allocation_by_tti"]:
                    self.stats["allocation_by_tti"][tti] -= 1
                return True
        return False
    
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
        """Получить все свободные RB для заданного TTI."""
        if tti >= self.total_tti:
            return []
        frame_idx = tti // 10
        sf_idx = tti % 10
        frame = self.frames[frame_idx]
        subframe = frame.subframes[sf_idx]
        return (
            subframe.slots[0].GET_FREE_RES_BLCK() +  # Слот 0
            subframe.slots[1].GET_FREE_RES_BLCK()    # Слот 1
        )
    
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

def test_rb_allocation():
    grid = RES_GRID_LTE(bandwidth=10)
    
    # Выделение RB в первом слоте
    assert grid.ALLOCATE_RB(0, "sub_0_slot_0", 10, 100), "Ошибка выделения"
    
    # Попытка повторного выделения
    assert not grid.ALLOCATE_RB(0, "sub_0_slot_0", 10, 200), "Ожидалась ошибка"
    
    # Выделение во втором слоте
    assert grid.ALLOCATE_RB(0, "sub_0_slot_1", 10, 100), "Ошибка выделения"
    # assert grid.stats["allocation_by_tti"][0] == 1, "Счетчик TTI не обновлен"
    
    # Проверка статуса
    rb1 = grid.GET_RB(0, "sub_0_slot_0", 10)
    rb2 = grid.GET_RB(0, "sub_0_slot_1", 10)
    assert rb1.UE_ID == 100 and rb2.UE_ID == 100, "Некорректное назначение"

def test_bandwidth_configuration():
    for bw, expected_rb in [(1.4, 6), (3, 15), (5, 25), (10, 50), (15, 75), (20, 100)]:
        grid = RES_GRID_LTE(bandwidth=bw)
        assert grid.rb_per_slot == expected_rb, f"Ошибка: {bw} МГц → {expected_rb} RB/слот"
        assert grid.num_rb == expected_rb * 2, "Некорректное число RB на TTI"

def test_frame_structure():
    grid = RES_GRID_LTE(num_frames=2)
    
    # Проверка количества кадров
    assert len(grid.frames) == 2, "Ошибка количества кадров"
    
    # Проверка структуры подкадров
    frame = grid.frames[0]
    assert len(frame.subframes) == 10, "Ошибка: 10 подкадров/кадр"
    
    # Проверка структуры слотов
    subframe = frame.subframes[0]
    assert len(subframe.slots) == 2, "Ошибка: 2 слота/подкадр"
    
    # Проверка длительности TTI
    assert grid.total_tti == 20, "Ошибка: 2 кадра × 10 TTI = 20"
    
def test_rb_allocation_semantics():
    grid = RES_GRID_LTE(bandwidth=10)
    
    # Выделение RB в первом слоте
    assert grid.ALLOCATE_RB(0, "sub_0_slot_0", 25, 100), "Ошибка выделения"
    
    # Проверка статуса RB
    rb = grid.GET_RB(0, "sub_0_slot_0", 25)
    assert rb.UE_ID == 100 and rb.status == "assigned", "Некорректное состояние"
    
    # Освобождение RB
    rb.RELEASE_RB()
    assert rb.CHCK_RB(), "Ошибка освобождения"

def test_rb_group_allocation():
    grid = RES_GRID_LTE(bandwidth=5)
    freq_idx = 10
    
    # Выделение группы
    assert grid.ALLOCATE_RB_GROUP(0, freq_idx, 200), "Ошибка группового выделения"
    
    # Проверка обоих слотов
    slot0_rb = grid.GET_RB(0, "sub_0_slot_0", freq_idx)
    slot1_rb = grid.GET_RB(0, "sub_0_slot_1", freq_idx)
    assert slot0_rb.UE_ID == slot1_rb.UE_ID == 200, "Несоответствие группового выделения"

def test_boundary_conditions():
    grid = RES_GRID_LTE(bandwidth=20, num_frames=1)
    
    # Выделение ресурса
    assert grid.ALLOCATE_RB(9, "sub_9_slot_1", 99, 400), "Ошибка выделения"
    # Проверка, что счетчик TTI обновился
    assert grid.stats["allocation_by_tti"][9] == 1, "Счетчик TTI не увеличен"
    
    # Освобождение ресурса
    assert grid.RELEASE_RB(9, "sub_9_slot_1", 99), "Ошибка освобождения"
    
    # Проверка статистики
    assert grid.stats["allocated_rbs"] == 0, "Счетчик RB не обнулился"
    assert 400 not in grid.stats["allocation_by_user"], "UE_ID не удален из статистики"
    assert grid.stats["allocation_by_tti"][9] == 0, "Счетчик TTI не уменьшен"


def test_3gpp_compliance():
    # TS 36.211 Section 6.2.3
    grid = RES_GRID_LTE(bandwidth=10, num_frames=1)
    
    # Проверка параметров слота
    slot = grid.frames[0].subframes[0].slots[0]
    assert len(slot.resource_blocks) == 50, "Ожидается 50 RB/слот для 10 МГц"
    
    # Проверка структуры RE (12 поднесущих × 7 символов)
    rb = next(iter(slot.resource_blocks.values()))
    assert rb.freq_idx >= 0 and rb.freq_idx < 50, "Некорректный частотный индекс"

def test_resource_utilization_stats():
    grid = RES_GRID_LTE()
    
    # Выделение 5 RB с разными UE_ID
    for i in range(5):
        grid.ALLOCATE_RB(0, f"sub_0_slot_{i%2}", i, 100 + i)
    
    # Проверка общей статистики
    assert grid.stats["allocated_rbs"] == 5, f"Ожидается 5, получено {grid.stats['allocated_rbs']}"
    
    # Проверка подсчета по пользователям
    for i in range(5):
        assert grid.stats["allocation_by_user"].get(100 + i) == 1, f"Ошибка для UE_ID={100+i}"


class LTEGridVisualizer:
    def __init__(self, lte_grid):
        self.lte_grid = lte_grid
    
    def visualize_timeline_grid(self, tti_start=0, tti_end=None):
        """Визуализирует ресурсную сетку LTE с правильным расположением слотов по временной оси"""
        if tti_end is None:
            tti_end = min(tti_start + 5, self.lte_grid.total_tti)
        
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
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Отображаем занятые ресурсные блоки
        for tti in range(tti_start, tti_end):
            # Вычисляем начальный индекс слота для данного TTI
            slot_start_idx = (tti - tti_start) * 2
            
            # Обрабатываем оба слота
            for slot_num in [0, 1]:
                slot_id = f"sub_{tti}_slot_{slot_num}"
                slot_idx = slot_start_idx + slot_num
                
                for freq_idx in range(rb_range):
                    rb = self.lte_grid.GET_RB(tti, slot_id, freq_idx)
                    if rb and not rb.CHCK_RB():
                        ue_id = rb.UE_ID
                        color_idx = ue_id % 10
                        
                        # Рисуем прямоугольник
                        rect = plt.Rectangle(
                            (slot_idx - 0.4, freq_idx - 0.4),
                            0.8, 0.8,
                            facecolor=colors[color_idx],
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
        
        return fig, ax

    
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
    
if __name__ == "__main__":
    test_rb_allocation()
    test_bandwidth_configuration()
    test_frame_structure()
    test_rb_allocation_semantics()
    test_rb_group_allocation()
    test_boundary_conditions()
    test_3gpp_compliance()
    test_resource_utilization_stats()
    test_visualize_lte_timeline()
    print("Все тесты успешно пройдены!")