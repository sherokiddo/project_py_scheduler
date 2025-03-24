# SCHEDULER.py
from typing import List, Dict, Optional
from LTE_GRID_ver_alpha import RES_GRID_LTE, SchedulerInterface
#upd

class RoundRobinScheduler(SchedulerInterface):
    """
    Планировщик, использующий алгоритм Round Robin для распределения 
    ресурсных блоков LTE между пользователями.
    """
    def __init__(self, lte_grid: RES_GRID_LTE):
        """
        Инициализация планировщика Round Robin.
        
        Args:
            lte_grid: Объект RES_GRID_LTE для работы с ресурсной сеткой
        """
        super().__init__(lte_grid)
        self.active_users = []  # Список активных пользователей
        self.current_user_idx = 0  # Индекс текущего пользователя для Round Robin
    
    def add_user(self, user_id: int):
        """
        Добавить пользователя в список активных.
        
        Args:
            user_id: Идентификатор пользователя
        """
        if user_id not in self.active_users:
            self.active_users.append(user_id)
    
    def remove_user(self, user_id: int):
        """
        Удалить пользователя из списка активных.
        
        Args:
            user_id: Идентификатор пользователя
        """
        if user_id in self.active_users:
            self.active_users.remove(user_id)
            # Корректировка индекса текущего пользователя при необходимости
            if self.current_user_idx >= len(self.active_users) and len(self.active_users) > 0:
                self.current_user_idx = 0
    
    def schedule(self, tti: int, users: List[Dict] = None):
        """
        Метод планирования ресурсов для заданного TTI.
        Реализация алгоритма Round Robin.
        
        Args:
            tti: Индекс TTI для планирования
            users: Список пользователей с их параметрами (опционально)
            
        Returns:
            Dict: Результаты планирования
        """
        if not self.active_users:
            return {"allocated_count": 0, "message": "Нет активных пользователей"}
        
        # Получаем список свободных ресурсных блоков для данного TTI
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        
        if not free_rbs:
            return {"allocated_count": 0, "message": "Нет свободных ресурсных блоков"}
        
        # Сортируем ресурсные блоки по частотному индексу
        free_rbs.sort(key=lambda rb: rb.freq_idx)
        
        # Общее количество назначенных ресурсных блоков
        allocated_count = 0
        
        # Распределяем ресурсные блоки между пользователями
        for rb in free_rbs:
            # Определяем пользователя, которому назначим текущий RB
            if not self.active_users:
                break
            
            user_id = self.active_users[self.current_user_idx]
            
            # Назначаем ресурсный блок пользователю
            if self.lte_grid.ALLOCATE_RB(tti, rb.freq_idx, user_id):
                allocated_count += 1
            
            # Переходим к следующему пользователю (Round Robin)
            self.current_user_idx = (self.current_user_idx + 1) % len(self.active_users)
        
        return {
            "allocated_count": allocated_count,
            "message": f"Успешно назначено {allocated_count} ресурсных блоков"
        }
    
    def schedule_all_ttis(self):
        """
        Распределить ресурсные блоки для всех TTI.
        
        Returns:
            int: Общее количество назначенных ресурсных блоков
        """
        total_allocated = 0
        
        for tti in range(self.lte_grid.total_tti):
            result = self.schedule(tti)
            total_allocated += result["allocated_count"]
        
        return total_allocated

class EnhancedRoundRobinScheduler(RoundRobinScheduler):
    """
    Улучшенная версия планировщика Round Robin, которая распределяет 
    ресурсные блоки группами смежных RB.
    """
    def __init__(self, lte_grid: RES_GRID_LTE, min_rb_per_user=1, max_rb_per_user=None):
        """
        Инициализация улучшенного планировщика Round Robin.
        
        Args:
            lte_grid: Объект RES_GRID_LTE для работы с ресурсной сеткой
            min_rb_per_user: Минимальное число RB для одного пользователя за TTI
            max_rb_per_user: Максимальное число RB для одного пользователя за TTI
                           (None = без ограничения)
        """
        super().__init__(lte_grid)
        self.min_rb_per_user = min_rb_per_user
        self.max_rb_per_user = max_rb_per_user if max_rb_per_user is not None else float('inf')
    
    def schedule(self, tti: int, users: List[Dict] = None):
        """
        Метод планирования ресурсов для заданного TTI.
        Улучшенная версия Round Robin с группировкой ресурсов.
        
        Args:
            tti: Индекс TTI для планирования
            users: Список пользователей с их параметрами (опционально)
            
        Returns:
            Dict: Результаты планирования
        """
        if not self.active_users:
            return {"allocated_count": 0, "message": "Нет активных пользователей"}
        
        # Получаем список свободных ресурсных блоков для данного TTI
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        
        if not free_rbs:
            return {"allocated_count": 0, "message": "Нет свободных ресурсных блоков"}
        
        # Сортируем ресурсные блоки по частотному индексу
        free_rbs.sort(key=lambda rb: rb.freq_idx)
        
        # Группируем смежные ресурсные блоки
        rb_groups = self._group_adjacent_rbs(free_rbs)
        
        # Общее количество назначенных ресурсных блоков
        allocated_count = 0
        
        # Распределяем группы ресурсных блоков между пользователями
        while rb_groups and self.active_users:
            user_id = self.active_users[self.current_user_idx]
            
            # Выбираем подходящую группу RB для пользователя
            group_idx, group = self._select_rb_group(rb_groups, user_id)
            
            if group_idx is not None:
                # Ограничиваем количество RB для пользователя, если нужно
                if len(group) > self.max_rb_per_user:
                    # Разделяем группу на две части
                    to_allocate = group[:self.max_rb_per_user]
                    rb_groups[group_idx] = group[self.max_rb_per_user:]
                else:
                    to_allocate = group
                    # Удаляем группу из списка доступных
                    rb_groups.pop(group_idx)
                
                # Назначаем выбранные RB пользователю
                freq_indices = [rb.freq_idx for rb in to_allocate]
                allocated = self.lte_grid.ALLOCATE_RB_GROUP(tti, freq_indices, user_id)
                allocated_count += allocated
            
            # Переходим к следующему пользователю (Round Robin)
            self.current_user_idx = (self.current_user_idx + 1) % len(self.active_users)
        
        return {
            "allocated_count": allocated_count,
            "message": f"Успешно назначено {allocated_count} ресурсных блоков с группировкой"
        }
    
    def _group_adjacent_rbs(self, rbs):
        """
        Группировка смежных ресурсных блоков.
        
        Args:
            rbs: Список ресурсных блоков
            
        Returns:
            List[List[RES_BLCK]]: Список групп смежных ресурсных блоков
        """
        if not rbs:
            return []
        
        groups = []
        current_group = [rbs[0]]
        
        for i in range(1, len(rbs)):
            # Если текущий RB смежный с последним в группе
            if rbs[i].freq_idx == current_group[-1].freq_idx + 1:
                current_group.append(rbs[i])
            else:
                # Закрываем текущую группу и начинаем новую
                groups.append(current_group)
                current_group = [rbs[i]]
        
        # Добавляем последнюю группу
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _select_rb_group(self, rb_groups, user_id):
        """
        Выбор подходящей группы RB для пользователя.
        
        Args:
            rb_groups: Список групп ресурсных блоков
            user_id: Идентификатор пользователя
            
        Returns:
            Tuple[int, List[RES_BLCK]]: Индекс выбранной группы и сама группа,
                                       или (None, None), если подходящей группы нет
        """
        # Сначала ищем группу, размер которой точно соответствует min_rb_per_user
        for i, group in enumerate(rb_groups):
            if len(group) == self.min_rb_per_user:
                return i, group
        
        # Затем ищем группу, размер которой больше или равен min_rb_per_user
        for i, group in enumerate(rb_groups):
            if len(group) >= self.min_rb_per_user:
                return i, group
        
        # Если нет подходящих групп, берем самую большую из доступных
        if rb_groups:
            largest_group_idx = max(range(len(rb_groups)), key=lambda i: len(rb_groups[i]))
            return largest_group_idx, rb_groups[largest_group_idx]
        
        return None, None
