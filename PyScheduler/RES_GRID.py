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
#
#   v1.1.0 - 2026-03-14:
#   Автор: Македон Никита
#   Новые классы и архитектура:
#   - Добавлен SlidingWindowCache — FIFO-кэш с вытеснением старых TTI, ограничивает память окном window_size (по умолчанию 100 TTI)
#   - Добавлен абстрактный GridInterface — интерфейс для типизации сетки в планировщике
#   - Добавлен новый класс RES_GRID_LTE_CACHED, реализующий GridInterface; 
#   - Cтарый RES_GRID_LTE оставлен для совместимости
#   Оптимизации
#   - Slot.GET_RES_BLCK() — O(n) цикл по всем RB заменён на O(1) lookup через новый словарь resource_blocks_by_freq
#   - get_rbg_indices() — результат кэшируется в _rbg_indices_cache, вычисляется максимум 16–25 раз вместо миллионов
#   - В allocate_rbg() / release_rbg() / generate_bitmap() — subframe достаётся из кэша один раз вместо повторных обращений
#   Тесты
#   - Убраны в tests/test_res_grid.py
#   - Добавлены новые для RES_GRID_LTE_CACHED
#   v1.1.1 - 2026-04-01:
#   Автор: Македон Никита
#   Версия Python Kernel: 3.12.9
#   - Продолжена оптимизация ресурсной сетки LTE
#   - Добавлено более компактное хранение назначений RBG
#   - Уменьшены накладные расходы на создание объектов TTI
#   - Реализовано ленивое создание детальной структуры subframe/slot
#   - Ускорена работа allocate/release и генерации bitmap
#   - Сохранена совместимость со старой реализацией сетки
#   - Добавлен новый тест test_res_grid_equivalence
#------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Union
from collections import OrderedDict
from BS_MODULE import BaseStation


# ============================================================================
# НОВОЕ в v1.1.0: Скользящее окно кэширования для оптимизации памяти
# ============================================================================

class SlidingWindowCache:
    """Кэш FIFO для оптимизации памяти (только последние N TTI в памяти)"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.cache: OrderedDict[int, Dict] = OrderedDict()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'current_size': 0}
    
    def get(self, tti: int) -> Optional[Dict]:
        """Попытка получить кэшированные данные для TTI"""
        if tti in self.cache:
            self.stats['hits'] += 1
            self.cache.move_to_end(tti)
            return self.cache[tti]
        self.stats['misses'] += 1
        return None
    
    def put(self, tti: int, data: Dict) -> None:
        """Добавить данные в кэш с автоматическим удалением старых"""
        if tti not in self.cache:
            self.stats['current_size'] += 1
        self.cache[tti] = data
        self.cache.move_to_end(tti)
        while len(self.cache) > self.window_size:
            self.cache.popitem(last=False)
            self.stats['current_size'] -= 1
            self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict:
        """Получить статистику кэша (для мониторинга)"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        return {
            'window_size': self.window_size,
            'current_size': self.stats['current_size'],
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.stats['evictions']
        }


class GridInterface:
    """Абстрактный интерфейс для работы с ресурсной сеткой (НОВОЕ в v1.1.0)"""
    def allocate_rbg(self, tti: int, rbg_idx: int, ue_id: int) -> bool: raise NotImplementedError
    def release_rbg(self, tti: int, rbg_idx: int) -> bool: raise NotImplementedError
    def get_rbg_indices(self, rbg_idx: int) -> List[int]: raise NotImplementedError
    def get_rbg_size(self) -> int: raise NotImplementedError
    def generate_bitmap(self, tti: int, ue_id: int) -> List[int]: raise NotImplementedError
    def get_window_stats(self) -> Dict: raise NotImplementedError


# ============================================================================
# БАЗОВЫЕ КЛАССЫ (v1.0.2) - С ОПТИМИЗАЦИЕЙ УРОВНЯ 2
# ============================================================================

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
        self.freq_idx = freq_idx
        self.UE_ID = None # ID пользователя, которому назначен RB
        self.status = "free" # Статус: "free" или "assigned"
    
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


class Slot:
    """
    Класс, представляющий слот в структуре LTE.
    Слот содержит набор ресурсных блоков по частоте.
    
    ОПТИМИЗАЦИЯ УРОВНЯ 2:
    - Добавлен resource_blocks_by_freq для O(1) доступа вместо O(n)
    """

    def __init__(self, slot_id: str, rb_per_slot: int):
        """
        Инициализация слота.

        Args:
            slot_id: Идентификатор слота
            rb_per_slot: Количество ресурсных блоков по частоте
        """
        self.id = slot_id
        self.resource_blocks: Dict[str, RES_BLCK] = {}
        self.resource_blocks_by_freq: Dict[int, RES_BLCK] = {}  # ← ОПТИМИЗАЦИЯ
        
        for rb_idx in range(rb_per_slot):
            rb_id = f"RB_{slot_id}_{rb_idx}" # Унифицированный формат
            rb = RES_BLCK(rb_id, slot_id, rb_idx)
            self.resource_blocks[rb_id] = rb
            self.resource_blocks_by_freq[rb_idx] = rb  # ← НОВОЕ: O(1) доступ
    
    def GET_RES_BLCK(self, freq_idx: int) -> Optional[RES_BLCK]:
        """
        Получить ресурсный блок по частотному индексу.

        Args:
            freq_idx: Частотный индекс ресурсного блока

        Returns:
            RES_BLCK или None, если блок не найден
        """
        # ОПТИМИЗИРОВАНО: O(1) dict lookup вместо O(n) цикла
        return self.resource_blocks_by_freq.get(freq_idx)
    
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
            rb_per_slot: Количество ресурсных блоков по частоте
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


# ============================================================================
# НОВОЕ в v1.1.0: Оптимизированная реализация с кэшем
# ============================================================================

class RES_GRID_LTE_CACHED(GridInterface):
    """
    Сетка LTE с кэшированием скользящего окна (100x экономия памяти)

    Основная оптимизация для профиля:
    - Ведём per-TTI массив rbg_alloc (RBG -> UE_ID/None) и строим bitmap из него за O(total_rbg),
      вместо сканирования всех RB в каждом bitmap (которое порождает миллионы GET_RES_BLCK).
    - Оставляем legacy-scan (fallback) для валидации/совместимости.
    """

    # Словарь соответствия полосы частот и количества RB согласно стандарту LTE
    BANDWIDTH_TO_RB = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
    
    # Словарь размеров ресурсных групп по TS 36.213
    RBG_SIZE_TABLE = {1.4: 1, 3: 2, 5: 2, 10: 3, 15: 4, 20: 4}

    def __init__(self, bandwidth: float = 10, window_size: int = 100,
                 cp_type: str = "normal", verbose: bool = False,
                 enable_bitmap_cache: bool = True,
                 enable_resgrid_numpy: bool = False,
                 fast_rbg_only: bool = True):
        if bandwidth not in self.BANDWIDTH_TO_RB:
            raise ValueError(f"Недопустимая полоса: {list(self.BANDWIDTH_TO_RB.keys())}")

        self.bandwidth = bandwidth
        self.rb_per_slot = self.BANDWIDTH_TO_RB[bandwidth]
        self.num_rb = self.rb_per_slot * 2
        self.cache = SlidingWindowCache(window_size)
        self.current_tti, self.bs, self.verbose = 0, None, verbose

        self._rbg_indices_cache: Dict[int, List[int]] = {}

        self.enable_bitmap_cache = enable_bitmap_cache
        self.enable_resgrid_numpy = enable_resgrid_numpy
        self.strict_rb_precheck = False   # fast mode по умолчанию
        self.lazy_subframe = True         # subframe создаём только при необходимости
        self.fast_rbg_only = fast_rbg_only

        self._rbg_size = self.RBG_SIZE_TABLE[self.bandwidth]
        self._total_rbg = (self.rb_per_slot + self._rbg_size - 1) // self._rbg_size

        try:
            import numpy as _np  # noqa: F401
            self._np_available = True
        except Exception:
            self._np_available = False
            self.enable_resgrid_numpy = False

    def _get_or_create_tti_state(self, tti: int) -> Dict:
        cached = self.cache.get(tti)
        if cached:
            return cached

        sf = None
        if not self.lazy_subframe:
            sf = Subframe(tti % 10, self.rb_per_slot)

        state = {
            'tti': tti,
            'subframe': sf,
            'rbg_alloc': [None] * self._total_rbg,
        }

        if self.enable_resgrid_numpy and self._np_available:
            import numpy as np
            state['rbg_alloc_np'] = np.full((self._total_rbg,), -1, dtype=np.int32)

        self.cache.put(tti, state)
        if self.verbose:
            print(f"[RES_GRID] Created TTI state for TTI {tti}")
        return state

    def _get_or_create_subframe(self, tti: int) -> Subframe:
        state = self._get_or_create_tti_state(tti)
        sf = state.get('subframe')

        if sf is None:
            sf = Subframe(tti % 10, self.rb_per_slot)
            state['subframe'] = sf

            alloc = state.get('rbg_alloc', [])
            for rbg_idx, owner in enumerate(alloc):
                if owner is None:
                    continue

                rb_indices = self.get_rbg_indices(rbg_idx)
                for slot in sf.slots:
                    rb_by_freq = slot.resource_blocks_by_freq
                    for freq in rb_indices:
                        rb = rb_by_freq.get(freq)
                        if rb:
                            rb.ASSIGN_RB(owner)

            if self.verbose:
                print(f"[RES_GRID] Materialized Subframe for TTI {tti}")

        return sf

    def allocate_rbg(self, tti: int, rbg_idx: int, ue_id: int) -> bool:
        """Выделить RBG (Resource Block Group) пользователю.

        Цели:
        - Быстрый O(1) pre-check по компактной карте rbg_alloc (и rbg_alloc_np при наличии).
        - В fast path не выполняем полный обход RB (и не материализуем Subframe), если в этом нет нужды.
        - RB-структуры обновляем только если Subframe уже существует, либо если включён strict_rb_precheck.

        Важно:
        - Источник истины в cached-версии: state['rbg_alloc'] (и state['rbg_alloc_np'] если включено).
        - При lazy_subframe=True subframe может быть None, пока не нужен legacy-scan/отладка.
        """
        state = self._get_or_create_tti_state(tti)

        # --- O(1) pre-check занятости RBG ---
        if 'rbg_alloc_np' in state:
            if int(state['rbg_alloc_np'][rbg_idx]) != -1:
                return False
        else:
            if state['rbg_alloc'][rbg_idx] is not None:
                return False

        strict = bool(getattr(self, 'strict_rb_precheck', False))

        # --- STRICT PATH: полная проверка RB (дороже, но максимально строго) ---
        if strict:
            sf = self._get_or_create_subframe(tti)
            rb_indices = self.get_rbg_indices(rbg_idx)

            for slot in sf.slots:
                rb_by_freq = slot.resource_blocks_by_freq
                for freq in rb_indices:
                    rb = rb_by_freq.get(freq)
                    if rb is None or not rb.CHCK_RB():
                        return False

            # Commit: обновляем компактную карту
            state['rbg_alloc'][rbg_idx] = ue_id
            if 'rbg_alloc_np' in state:
                state['rbg_alloc_np'][rbg_idx] = int(ue_id)

            # Commit: обновляем RB-структуры
            for slot in sf.slots:
                rb_by_freq = slot.resource_blocks_by_freq
                for freq in rb_indices:
                    rb = rb_by_freq.get(freq)
                    if not (rb and rb.ASSIGN_RB(ue_id)):
                        self.release_rbg(tti, rbg_idx)
                        return False

            return True

        # --- FAST PATH: только компактная карта, RB-слой трогаем только если он уже существует ---
        state['rbg_alloc'][rbg_idx] = ue_id
        if 'rbg_alloc_np' in state:
            state['rbg_alloc_np'][rbg_idx] = int(ue_id)

        sf = state.get('subframe')
        if sf is None:
            return True

        rb_indices = self.get_rbg_indices(rbg_idx)
        assigned = []

        for slot in sf.slots:
            rb_by_freq = slot.resource_blocks_by_freq
            for freq in rb_indices:
                rb = rb_by_freq.get(freq)
                if rb and rb.ASSIGN_RB(ue_id):
                    assigned.append(rb)
                else:
                    state['rbg_alloc'][rbg_idx] = None
                    if 'rbg_alloc_np' in state:
                        state['rbg_alloc_np'][rbg_idx] = -1
                    for r in assigned:
                        r.RELEASE_RB()
                    return False

        return True


    def release_rbg(self, tti: int, rbg_idx: int) -> bool:
        state = self._get_or_create_tti_state(tti)

        # Всегда чистим компактную карту
        state['rbg_alloc'][rbg_idx] = None
        if 'rbg_alloc_np' in state:
            state['rbg_alloc_np'][rbg_idx] = -1

        # RB трогаем только если subframe уже существует
        sf = state.get('subframe')
        if sf is None and self.strict_rb_precheck:
            sf = self._get_or_create_subframe(tti)

        if sf is not None:
            rb_indices = self.get_rbg_indices(rbg_idx)
            for slot in sf.slots:
                rb_by_freq = slot.resource_blocks_by_freq
                for freq in rb_indices:
                    rb = rb_by_freq.get(freq)
                    if rb:
                        rb.RELEASE_RB()

        return True

    def get_rbg_indices(self, rbg_idx: int) -> List[int]:
        cached = self._rbg_indices_cache.get(rbg_idx)
        if cached is None:
            start = rbg_idx * self._rbg_size
            cached = list(range(start, min(start + self._rbg_size, self.rb_per_slot)))
            self._rbg_indices_cache[rbg_idx] = cached
        return cached

    def get_rbg_size(self) -> int:
        return self._rbg_size

    def _generate_bitmap_legacy_scan(self, tti: int, ue_id: int) -> List[int]:
        sf = self._get_or_create_subframe(tti)
        bitmap = []
        for rbg_idx in range(self._total_rbg):
            allocated = False
            rb_indices = self.get_rbg_indices(rbg_idx)
            for slot in sf.slots:
                rb_by_freq = slot.resource_blocks_by_freq
                for freq in rb_indices:
                    rb = rb_by_freq.get(freq)
                    if rb and rb.UE_ID == ue_id:
                        allocated = True
                        break
                if allocated:
                    break
            bitmap.append(1 if allocated else 0)
        return bitmap

    def generate_bitmap(self, tti: int, ue_id: int) -> List[int]:
        if self.enable_bitmap_cache:
            state = self._get_or_create_tti_state(tti)
            if 'rbg_alloc_np' in state:
                arr = state['rbg_alloc_np']
                return (arr == ue_id).astype('int8').tolist()
            alloc = state['rbg_alloc']
            return [1 if v == ue_id else 0 for v in alloc]
        return self._generate_bitmap_legacy_scan(tti, ue_id)

    def get_window_stats(self) -> Dict:
        return self.cache.get_stats()

    def SET_BS(self, bs: BaseStation):
        self.bs = bs

    def ALLOCATE_RBG(self, tti: int, rbg_idx: int, UE_ID: int) -> bool:
        return self.allocate_rbg(tti, rbg_idx, UE_ID)

    def RELEASE_RBG(self, tti: int, rbg_idx: int) -> bool:
        return self.release_rbg(tti, rbg_idx)

    def GET_RBG_INDICES(self, rbg_idx: int) -> List[int]:
        return self.get_rbg_indices(rbg_idx)

    def GET_RBG_SIZE(self) -> int:
        return self.get_rbg_size()

    def GENERATE_BITMAP(self, tti: int, UE_ID: int) -> List[int]:
        return self.generate_bitmap(tti, UE_ID)

# ============================================================================
# ОРИГИНАЛЬНАЯ РЕАЛИЗАЦИЯ (v1.0.2) - ДЛЯ СОВМЕСТИМОСТИ
# ============================================================================

class RES_GRID_LTE:
    """
    Оригинальная сетка LTE (v1.0.2) - используйте RES_GRID_LTE_CACHED для новых проектов
    
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
    
    # Словарь размеров ресурсных групп по TS 36.213
    RBG_SIZE_TABLE = {
        1.4: 1, 3: 2, 5: 2, 10: 3, 15: 4, 20: 4
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
            "allocation_by_tti": {tti: 0 for tti in range(self.total_tti)} # Инициализация всех TTI
        }
        
        # Инициализация rb_map для быстрого доступа к ресурсным блокам
        self._init_rb_map()
        self.bs = None
    
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
    
    def ALLOCATE_RB_PAIR(self, tti: int, freq_idx: int, UE_ID: int) -> bool:
        subframe = tti % 10
        success = True
        for slot in [0, 1]:
            slot_id = f"sub_{subframe}_slot_{slot}"
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

    def GET_RBG_SIZE(self) -> int:
        return self.RBG_SIZE_TABLE[self.bandwidth]

    def GET_RBG_INDICES(self, rbg_idx: int) -> List[int]:
        size = self.GET_RBG_SIZE()
        start = rbg_idx * size
        end = start + size
        return list(range(start, min(end, self.rb_per_slot)))

    def ALLOCATE_RBG(self, tti: int, rbg_idx: int, UE_ID: int) -> bool:
        rb_indices = self.GET_RBG_INDICES(rbg_idx)
        subframe = tti % 10

        # ---------- PRE-CHECK ----------
        for slot in [0, 1]:
            slot_id = f"sub_{subframe}_slot_{slot}"
            for freq in rb_indices:
                rb = self.GET_RB(tti, slot_id, freq)
                if rb is None or not rb.CHCK_RB():
                    return False

        # ---------- COMMIT ----------
        allocated_pairs = []
        for slot in [0, 1]:
            slot_id = f"sub_{subframe}_slot_{slot}"
            for freq in rb_indices:
                if self.ALLOCATE_RB(tti, slot_id, freq, UE_ID):
                    allocated_pairs.append((slot_id, freq))
                else:
                    for sid, f in allocated_pairs:
                        self.RELEASE_RB(tti, sid, f)
                    return False

        return True

    def RELEASE_RBG(self, tti: int, rbg_idx: int) -> bool:
        subframe = tti % 10
        for slot in [0, 1]:
            slot_id = f"sub_{subframe}_slot_{slot}"
            for freq in self.GET_RBG_INDICES(rbg_idx):
                self.RELEASE_RB(tti, slot_id, freq)
        return True
    
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
    
    def GENERATE_BITMAP(self, tti: int, UE_ID: int) -> List[int]:
        """Генерирует bitmap для пользователя на заданном TTI"""
        subframe = tti % 10
        rbg_size = self.GET_RBG_SIZE()
        total_rbg = (self.rb_per_slot + rbg_size - 1) // rbg_size
        
        bitmap = []
        for rbg_idx in range(total_rbg):
            allocated = False
            # Проверка выделения в любом из слотов
            for slot in [0, 1]:
                slot_id = f"sub_{subframe}_slot_{slot}"
                for freq in self.GET_RBG_INDICES(rbg_idx):
                    rb = self.GET_RB(tti, slot_id, freq)
                    if rb and rb.UE_ID == UE_ID:
                        allocated = True
                        break
                if allocated:
                    break
            bitmap.append(1 if allocated else 0)
        return bitmap
    
    def SET_BS(self, bs: BaseStation):
        """Установка ссылки на базовую станцию"""
        self.bs = bs
