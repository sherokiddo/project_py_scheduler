"""
#------------------------------------------------------------------------------
# Модуль: SCHEDULER - Планировщик ресурсов для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для распределения ресурсных блоков между
# пользовательскими устройствами (UE) в сети LTE. Реализует алгоритм
# планирования Round Robin.
#
# Версия: 1.0.7
# Дата последнего изменения: 2025-04-09
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл, Норицин Иван
#
# Зависимости:
# - UE_MODULE.py (модели пользовательского оборудования)
# - RES_GRID (модель ресурсной сетки LTE)
#
# Изменения v1.0.5:
# - пофикшен баг с выделением ресурсов пользователям в ситуациях, когда количество
# юзеров намного больше чем количество RB в слоте. Выявлена ошибка в расчете последнего
# обслуженного UE last_served_user метода schedule
# - RES_GRID (модель ресурсной сетки LTE)
#
# Изменения v1.0.6:
# - Обновлен планировщик RR в связи с изменением принципа работы буфера
# и изменения UE_MODULE.
# - Добавлен модуль AMC
# - Добавлены тесты планировщика
# - По итогам тестов оказалось, что АМС не работает полноценно. Это печально.
# - Для дальнейшей коррекции работы, необходимо ввести фрагментацию пакетов.
#
# Изменения v1.0.7:
# - Доделан планировщик RR.Дебаг работы планировщика при работе со слотами и буфером
# - Исправлено некорректное распределение блоков по времени.
# - Расширены тесты планировщика, добавлены доп. валидации для проверки работы АМС.
# - Отложить фрагментацию пакетов. Подумать над принципом работы с буфером, корректное
# извлечение пакетов и его уменьшение. А затем перейти к формированию концепции
# транспортных блоков.
# - АМС - пофикшено, работает правильно, статистику считает правильно. Успех!
# - На будущее: тесты выпилить в отдельный блок, чтобы больше тут не жили.
#------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Union, Tuple
from RES_GRID import RES_GRID_LTE, SchedulerInterface
from BS_MODULE import BaseStation
from enum import Enum
from CHANNEL_MODEL import ChannelModel

class AdaptiveModulationAndCoding:
    """
    Класс для преобразования CQI в MCS и расчета бит на ресурсный блок.
    """
    
    # Таблица соответствия CQI → (Modulation Order, Code Rate)
    CQI_TO_MCS = {
        1: (2, 0.152),   # QPSK
        2: (2, 0.234),   # QPSK
        3: (2, 0.377),   # QPSK
        4: (2, 0.601),   # QPSK
        5: (4, 0.369),   # 16QAM
        6: (4, 0.479),   # 16QAM
        7: (4, 0.601),   # 16QAM
        8: (6, 0.455),   # 64QAM
        9: (6, 0.554),   # 64QAM
        10: (6, 0.650),  # 64QAM
        11: (6, 0.754),  # 64QAM
        12: (6, 0.852),  # 64QAM
        13: (6, 0.926),  # 64QAM
        14: (6, 0.953),  # 64QAM
        15: (6, 0.978)   # 64QAM
    }

    def GET_BITS_PER_RB(self, cqi: int) -> int:
        """
        Рассчитать количество бит на ресурсный блок (RB) для заданного CQI.
        """
        if cqi not in self.CQI_TO_MCS:
            raise ValueError(f"Invalid CQI: {cqi}. Must be 1-15.")
        
        modulation, code_rate = self.CQI_TO_MCS[cqi]
        symbols_per_rb = 12 * 7  # 84 символа в RB (с учетом слотов)
        return int(symbols_per_rb * modulation * code_rate)
	#@sherokiddo: "Добавить зависимость от CP"
    
    def calculate_throughput(self, allocation: Dict, users: List[Dict], tti: int, bs: BaseStation) -> Dict:
        """
        Рассчитывает фактическую пропускную способность с учетом:
        - Реальных данных из буфера
        - Адаптивной модуляции и кодирования (AMC)
    
        Args:
            allocation: Словарь распределения RB {UE_ID: список freq_indices}
            users: Список активных пользователей с параметрами CQI
    
        Returns:
            Словарь с метриками производительности:
            - total_allocated_rbs: Общее количество выделенных RB
            - user_throughput: Пропускная способность на пользователя (бит/с)
            - average_throughput: Средняя пропускная способность (бит/с)
            - total_effective_bits: Фактически переданные биты
        """
        all_users = {u['UE_ID']: u for u in users}
        stats = {
            'total_allocated_rbs': 0,
            'user_throughput': {ue_id: 0 for ue_id in all_users}, 
            'user_effective_throughput': {ue_id: 0 for ue_id in all_users},
            'user_max_throughput': {ue_id: 0 for ue_id in all_users},
            'average_dl_throughput': 0.0,
            'total_effective_bits': 0
        }
        
        total_effective_bits = 0
    
        for ue_id in all_users:
            user = all_users[ue_id]
            rb_count = len(allocation.get(ue_id, [])) * 2
            
            if rb_count == 0:
                # Обновление истории для неактивных пользователей
                user['ue'].UPD_DL_THROUGHPUT(0, 1)
                continue
    
            # 1. Расчет максимальной ёмкости RB для данного CQI
            bits_per_rb = self.GET_BITS_PER_RB(user['cqi'])

            # 2. Расчет эффективно использованных бит
            max_bits = rb_count * bits_per_rb
            
            # 3. Реальные переданные биты из буфера
            real_bits = user['ue'].current_dl_throughput
            effective_bits = min(real_bits, max_bits)
            
            # 4. Обновление статистики
            stats['user_throughput'][ue_id] = effective_bits
            total_effective_bits += effective_bits
            stats['total_allocated_rbs'] += rb_count
            stats['user_effective_throughput'][ue_id] = effective_bits
            stats['user_max_throughput'][ue_id] = max_bits
    
        # 5. Расчет средней пропускной способности
        active_users = [u for u in all_users.values() if stats['user_throughput'][u['UE_ID']] > 0]
        if active_users:
            stats['average_dl_throughput'] = total_effective_bits / len(active_users)
        
        stats['total_effective_bits'] = total_effective_bits
        return stats

class HARQState(Enum):
    """
    Состояния HARQ-процесса.
    """
    IDLE = 0           # Процесс свободен
    WAITING_ACK = 1    # Ожидание ACK/NACK
    RETRANSMIT = 2     # Требуется повторная передача


class HARQProcess:
    """
    Класс для управления одним HARQ-процессом.
    
    Attributes:
        process_id: Идентификатор процесса (0-7 для LTE)
        state: Текущее состояние процесса
        tx_count: Число попыток передачи (включая начальную)
        max_tx: Максимальное число передач
        tb_data: Буфер транспортного блока
        rv_sequence: Последовательность Redundancy Version
        cqi: Текущий CQI для передачи
        allocated_rbs: Список выделенных RB
    """
    
    def __init__(self, process_id: int, max_tx: int = 4):
        """
        Инициализация HARQ-процесса.
        
        Args:
            process_id: Идентификатор процесса (0-7)
            max_tx: Максимальное число передач (по умолчанию 4)
        """
        self.process_id = process_id
        self.state = HARQState.IDLE
        self.tx_count = 0
        self.max_tx = max_tx
        self.tb_data = None
        self.rv_sequence = [0, 2, 3, 1]  # Стандартная последовательность RV для LTE
        self.cqi = None
        self.allocated_rbs = []
        
    def start_transmission(self, tb_data: bytes, cqi: int, rbs: List[int]):
        """
        Начало новой передачи TB.
        
        Args:
            tb_data: Данные транспортного блока
            cqi: Channel Quality Indicator
            rbs: Список выделенных resource blocks
        """
        self.state = HARQState.WAITING_ACK
        self.tx_count = 1
        self.tb_data = tb_data
        self.cqi = cqi
        self.allocated_rbs = rbs
        
    def handle_ack(self):
        """
        Обработка положительного подтверждения (ACK).
        """
        self.reset()
        self.tx_count = 0
        self.tb_data = None # Процесс освобождается
        
    def handle_nack(self) -> bool:
        """
        Обработка отрицательного подтверждения (NACK).
        
        Returns:
            True если возможна повторная передача, False если достигнут лимит
        """
        if self.tx_count < self.max_tx:
            self.tx_count += 1
            self.state = HARQState.RETRANSMIT
            return True
        else:
            # Достигнут максимум передач, сбрасываем процесс
            self.reset()
            return False
            
    def get_current_rv(self) -> int:
        """
        Получение текущего Redundancy Version.
        
        Returns:
            Индекс RV для текущей передачи
        """
        return self.rv_sequence[(self.tx_count - 1) % len(self.rv_sequence)]
        
    def reset(self):
        """
        Сброс процесса в начальное состояние.
        """
        self.state = HARQState.IDLE
        self.tx_count = 0
        self.tb_data = None
        self.cqi = None
        self.allocated_rbs = []
        
    def is_idle(self) -> bool:
        """
        Проверка, свободен ли процесс.
        """
        return self.state == HARQState.IDLE
        
    def needs_retransmission(self) -> bool:
        """
        Проверка, требуется ли повторная передача.
        """
        return self.state == HARQState.RETRANSMIT


class HARQManager:
    """
    Менеджер HARQ-процессов для всех UE в системе.
    
    Attributes:
        num_processes: Число HARQ-процессов на UE (обычно 8 для LTE)
        max_tx: Максимальное число передач на процесс
        processes: Словарь {ue_id: [HARQProcess, ...]}
    """
    
    def __init__(self, num_processes: int = 8, max_tx: int = 4):
        """
        Инициализация менеджера HARQ.
        
        Args:
            num_processes: Число HARQ-процессов на UE (по умолчанию 8)
            max_tx: Максимальное число передач (по умолчанию 4)
        """
        self.num_processes = num_processes
        self.max_tx = max_tx
        self.processes: Dict[int, List[HARQProcess]] = {}
        
    def init_ue(self, ue_id: int):
        """
        Инициализация HARQ-процессов для нового UE.
        
        Args:
            ue_id: Идентификатор UE
        """
        if ue_id not in self.processes:
            self.processes[ue_id] = [
                HARQProcess(pid, self.max_tx) 
                for pid in range(self.num_processes)
            ]
            
    def get_idle_process(self, ue_id: int) -> Optional[HARQProcess]:
        """
        Получение свободного HARQ-процесса для UE.
        
        Args:
            ue_id: Идентификатор UE
            
        Returns:
            Свободный HARQProcess или None если все заняты
        """
        if ue_id not in self.processes:
            self.init_ue(ue_id)
            
        for process in self.processes[ue_id]:
            if process.is_idle():
                return process
        return None
        
    def get_retransmission_process(self, ue_id: int) -> Optional[HARQProcess]:
        """
        Получение процесса, требующего повторной передачи.
        
        Args:
            ue_id: Идентификатор UE
            
        Returns:
            HARQProcess требующий ретрансмиссии или None
        """
        if ue_id not in self.processes:
            return None
            
        for process in self.processes[ue_id]:
            if process.needs_retransmission():
                return process
        return None
        
    def handle_feedback(self, ue_id: int, process_id: int, ack: bool):
        """
        Обработка ACK/NACK от UE.
        
        Args:
            ue_id: Идентификатор UE
            process_id: Идентификатор HARQ-процесса
            ack: True для ACK, False для NACK
        """
        if ue_id not in self.processes:
            return
            
        process = self.processes[ue_id][process_id]
        
        if ack:
            process.handle_ack()
        else:
            success = process.handle_nack()
            if not success:
                # Достигнут лимит передач, TB потерян
                print(f"[HARQ] UE {ue_id} Process {process_id}: TB dropped after {self.max_tx} attempts")
                
    def get_statistics(self, ue_id: int) -> Dict[str, float]:
        """
        Получение статистики HARQ для UE.
        
        Args:
            ue_id: Идентификатор UE
            
        Returns:
            Словарь со статистикой {metric: value}
        """
        if ue_id not in self.processes:
            return {}
            
        total_tx = sum(p.tx_count for p in self.processes[ue_id] if not p.is_idle())
        active_processes = sum(1 for p in self.processes[ue_id] if not p.is_idle())
        
        return {
            'active_processes': active_processes,
            'avg_transmissions': total_tx / max(active_processes, 1),
            'idle_processes': self.num_processes - active_processes
        }

class RoundRobinScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE, bs: BaseStation):
        super().__init__(lte_grid)
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.last_served_ue_id = None 
        #теперь планировщик знает предыдущего обслуженного в tti прользователя
        #именно через этот метод
        self.amc = AdaptiveModulationAndCoding()
        self.harq_manager = HARQManager(num_processes=8, max_tx=4)
        for user_id in range(self.lte_grid.bs.num_ues):
            self.harq_manager.init_ue(user_id)
    
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
            """
            Планирование ресурсов с учётом данных в буфере и CQI.
            
            Args:
                tti: Индекс TTI
                users: Список пользователей с параметрами:
                    - 'UE_ID': Идентификатор
                    - 'buffer_size': Размер буфера в байтах
                    - 'cqi': Индекс качества канала
                    - 'ue': Объект UserEquipment
            
            Returns:
                Dict: Результаты распределения ресурсов
            """
            for user in users:
                user['ue'].current_dl_throughput = 0
            
            # 1. Фильтрация активных пользователей через буфер BS
            active_users = []
            for user in users:
                ue_id = user['UE_ID']
                bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
                if not bs_buffer:
                    continue
                    
                buffer_status = bs_buffer.GET_UE_STATUS(tti)['per_ue'].get(ue_id, {})
                buffer_size = buffer_status.get('size', 0)
                
                if buffer_size > 0 and 1 <= user['cqi'] <= 15:
                    active_users.append(user)
                    user['bs_buffer_size'] = buffer_size
            
            if not active_users:
                return {'allocation': {}, 'statistics': {}, 'bitmap': {}}
        
            # 2. Расчет параметров RBG
            rbg_size = self.lte_grid.GET_RBG_SIZE()
            total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size
        
            # 3. Инициализация структур данных
            allocation = {user['UE_ID']: [] for user in active_users}
            if self.last_served_ue_id is None:
                # Первый запуск, начинаем с 0
                current_idx = 0
            else:
                # Ищем последнего обслуженного пользователя в новом списке активных
                found_idx = -1
                for i, user in enumerate(active_users):
                    if user['UE_ID'] == self.last_served_ue_id:
                        found_idx = i
                        break
                        
                # Начинаем со следующего пользователя после последнего обслуженного
                if found_idx == -1:
                    # Если не нашли пользователя, начинаем с 0
                    current_idx = 0
                else:
                    # Если нашли, берем следующего
                    current_idx = (found_idx + 1) % len(active_users)
            
            #4. Определение количества бит на передачу каждому пользователю
            remaining_buffer = {user['UE_ID']: user['bs_buffer_size'] * 8 for user in active_users}
            
            last_allocated_ue_id = None

            for rbg_idx in range(total_rbg):
                if all(v <= 0 for v in remaining_buffer.values()):
                    break
        
                # 5. Поиск следующего пользователя с данными
                initial_idx = current_idx
                while remaining_buffer[active_users[current_idx]['UE_ID']] <= 0:
                    current_idx = (current_idx + 1) % len(active_users)
                    if current_idx == initial_idx:
                        break
                
                # Получаем текущего пользователя и выделяем RBG
                user = active_users[current_idx]
                ue_id = user['UE_ID']
                
                # 6. Основной цикл распределения RBG только если в буфере еще есть данные
                if remaining_buffer[ue_id] > 0:
                    if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                        #Получаем индексы ресурсных блоков в группе
                        rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                        allocation[ue_id].extend(rb_indices)
                        last_allocated_ue_id = ue_id
                    
                # Уменьшает размер буфера в соотв. с емкостью RBG
                        bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                        rbg_capacity = len(rb_indices) * bits_per_rb * 2
                        remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)

                        # --- HARQ: передача TB через канал ---
                        # пытаемся найти процесс для ретрансмиссии
                        harq_proc = self.harq_manager.get_retransmission_process(ue_id)
                        if harq_proc:
                            proc_id = harq_proc.process_id
                            tb_data = harq_proc.tb_data
                            is_retx = True
                        else:
                            # новый процесс
                            harq_proc = self.harq_manager.get_idle_process(ue_id)
                            proc_id = harq_proc.process_id

                            # расчёт максимального размера TB
                            max_bytes = (len(rb_indices) * self.amc.GET_BITS_PER_RB(user['cqi']) * 2) // 8
                            
                            # формируем TB из буфера UE
                            tb_data = user['ue'].buffer.GET_BYTES(max_bytes)  # или ваш pack_transport_block
                            harq_proc.start_transmission(tb_data, user['cqi'], rb_indices)
                            is_retx = False

                        # передаём TB и получаем ACK/NACK
                        ack = self.channel_model.receive_tb(
                            ue_id=ue_id,
                            rb_list=rb_indices,
                            cqi=user['cqi'],
                            process_id=proc_id,
                            is_retransmission=is_retx
                        )

                        # обрабатываем обратную связь
                        self.harq_manager.handle_feedback(ue_id, proc_id, ack)

                # 7. Переход к следующему пользователю
                current_idx = (current_idx + 1) % len(active_users)
                
                # 8. Обновление индекса последнего обслуженного UE
            if last_allocated_ue_id is not None:
                self.last_served_ue_id = last_allocated_ue_id
                
                # 9. Обработка буфера и статистики
            for user in users:
                ue = user['ue']
                ue_id = user['UE_ID'] #да, эта часть кода странная, но только после этого все заработало
                bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
                
                if not bs_buffer:
                    continue
                allocated_rb = len(allocation.get(user['UE_ID'], [])) * 2
                bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                max_bytes = (allocated_rb * bits_per_rb) // 8  #rbg_capacity // 8??

                #Если пользователь неактивен, передаём 0 байт
                packets, total = bs_buffer.GET_PACKETS(
                    ue_id=ue_id,
                    max_bytes=max_bytes,
                    bits_per_rb=bits_per_rb,
                    current_time=tti
                )               
            # Обновление метрик DL
                ue.UPD_DL_THROUGHPUT(total * 8, 1)                
    
        # 8. Формирование bitmap
            bitmap = {user['UE_ID']: self.lte_grid.GENERATE_BITMAP(tti, user['UE_ID']) for user in active_users}
            
            return {
                'allocation': allocation,
                'statistics': self.amc.calculate_throughput(allocation, users, tti, self.lte_grid.bs),
                'bitmap': bitmap#,
                #'dl_throughput': {user['UE_ID']: user['ue'].current_dl_throughput for user in users}
            }
        
    #@sherokiddo в рамках оптимизации можно разбить весь планировщик на несколько методов
    #для удобного логирования и подсчета времени. Например - подготовка данных один метод
    #затем идет непосредственно все планирование, и метод формирования статистики

class BestCQIScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE, bs: BaseStation):
        super().__init__(lte_grid)
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.amc = AdaptiveModulationAndCoding() 
    
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Планирование ресурсов с учётом данных в буфере и CQI.
        
        Args:
            tti: Индекс TTI
            users: Список пользователей с параметрами:
                - 'UE_ID': Идентификатор
                - 'buffer_size': Размер буфера в байтах
                - 'cqi': Индекс качества канала
                - 'ue': Объект UserEquipment
        
        Returns:
            Dict: Результаты распределения ресурсов
        """
        for user in users:
            user['ue'].current_dl_throughput = 0
        
        # Фильтрация активных пользователей
        active_users = []
        for user in users:
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            buffer_status = bs_buffer.GET_UE_STATUS(tti)['per_ue'].get(ue_id, {})
            buffer_size = buffer_status.get('size', 0)
            if buffer_size > 0 and 1 <= user['cqi'] <= 15:
                active_users.append(user)
                user['bs_buffer_size'] = buffer_size

        if not active_users:
            return {'allocation': {}, 'statistics': {}, 'bitmap': {}}

        # 2. Расчет RBG
        rbg_size = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        # 3. Инициализация структур данных
        allocation = {user['UE_ID']: [] for user in active_users}
       
        # 4. Определение количества бит на передачу каждому пользователю
        remaining_buffer = {user['UE_ID']: user['bs_buffer_size'] * 8 for user in active_users}

        for rbg_idx in range(total_rbg):
            if all(v <= 0 for v in remaining_buffer.values()):
                break
            #фильтрация пользователей с данными в буфере
            users_with_data = [u for u in active_users if remaining_buffer[u['UE_ID']] > 0]
            if not users_with_data:
                break

            # 5. Сортировка по CQI для выбора лучшего
            users_with_data.sort(key=lambda u: u['cqi'], reverse=True)
            best_user = users_with_data[0]  # Пользователь с наивысшим CQI
            ue_id = best_user['UE_ID']
            
            # 6. Основной цикл распределение RBG
            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                allocation[ue_id].extend(rb_indices)
                
                # Уменьшаем размер буфера в соотв. с емкостью RBG
                bits_per_rb = self.amc.GET_BITS_PER_RB(best_user['cqi'])
                rbg_capacity = len(rb_indices) * bits_per_rb * 2
                remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)
   
        # 8. Обработка буфера и статистики
        for user in users:
            ue = user['ue']
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            
            allocated_rb = len(allocation.get(user['UE_ID'], [])) * 2
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            max_bytes = (allocated_rb * bits_per_rb) // 8
            
            packets, total = bs_buffer.GET_PACKETS(
                ue_id=ue_id,
                max_bytes=max_bytes,
                bits_per_rb=bits_per_rb,
                current_time=tti
            )
            
            # Обновление метрик DL
            ue.UPD_DL_THROUGHPUT(total * 8, 1)
   
        # 8. Формирование bitmap
        bitmap = {user['UE_ID']: self.lte_grid.GENERATE_BITMAP(tti, user['UE_ID']) for user in active_users}
        
        return {
            'allocation': allocation,
            'statistics': self.amc.calculate_throughput(allocation, users, tti, self.lte_grid.bs),
            'bitmap': bitmap#,
            #'dl_throughput': {user['UE_ID']: user['ue'].current_dl_throughput for user in users}
        }
    
    #@sherokiddo в рамках оптимизации можно разбить весь планировщик на несколько методов
    #для удобного логирования и подсчета времени. Например - подготовка данных один метод
    #затем идет непосредственно все планирование, и метод формирования статистики

class ProportionalFairScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE, bs: BaseStation):
        super().__init__(lte_grid)
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.amc = AdaptiveModulationAndCoding()
        self.channel_model = ChannelModel()
        
    def calculate_pf_metric(self, users: List[Dict]):
        """
        Расчёт PF-метрики для каждого пользователя.
        
        Args:
            users: Список пользователей с параметрами:
                - 'UE_ID': Идентификатор
                - 'buffer_size': Размер буфера в байтах
                - 'cqi': Индекс качества канала
                - 'ue': Объект UserEquipment
        
        Returns:
            users: Список пользователей с добавленной PF-метрикой
        """
        for user in users:
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            rb_per_slot = self.lte_grid.rb_per_slot
            instant_throughput = rb_per_slot * bits_per_rb * 2 * 1000
            
            if user['ue'].average_throughput <= 0:
                user['ue'].average_throughput = 1e-6
            
            PF_metric = instant_throughput / user['ue'].average_throughput
            user['PF_metric'] = PF_metric
            user['ue'].PF_metric = PF_metric
            # print(f"PF metric: UE{user['UE_ID']} = {PF_metric}")
        return users
            
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Планирование ресурсов с учётом данных в буфере и CQI.
        
        Args:
            tti: Индекс TTI
            users: Список пользователей с параметрами:
                - 'UE_ID': Идентификатор
                - 'buffer_size': Размер буфера в байтах
                - 'cqi': Индекс качества канала
                - 'ue': Объект UserEquipment
        
        Returns:
            Dict: Результаты распределения ресурсов
        """
        for user in users:
            user['ue'].current_dl_throughput = 0
            
        # 1. Фильтрация активных пользователей
        active_users = []
        for user in users:
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            buffer_status = bs_buffer.GET_UE_STATUS(tti)['per_ue'].get(ue_id, {})
            buffer_size = buffer_status.get('size', 0)
            if buffer_size > 0 and 1 <= user['cqi'] <= 15:
                active_users.append(user)
                user['bs_buffer_size'] = buffer_size

        if not active_users:
            return {'allocation': {}, 'statistics': {}, 'bitmap': {}}
        
        # 2. Расчёт PF-метрики для каждого пользователя
        active_users = self.calculate_pf_metric(active_users)
        
        # 3. Расчет RBG
        rbg_size = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size
        
        # 4. Инициализация структур данных
        allocation = {user['UE_ID']: [] for user in active_users}

        # Инициализация для AMC результатов 
        amc_results = {user['UE_ID']: None for user in active_users}
        
        # 5. Определение количества бит на передачу каждому пользователю
        remaining_buffer = {user['UE_ID']: user['bs_buffer_size'] * 8 for user in active_users}
        
        for rbg_idx in range(total_rbg):
            if all(v <= 0 for v in remaining_buffer.values()):
                break
            #фильтрация пользователей с данными в буфере
            users_with_data = [u for u in active_users if remaining_buffer[u['UE_ID']] > 0]
            if not users_with_data:
                break
                
            # 7. Сортировка по PF-метрике для выбора лучшего
            users_with_data.sort(key=lambda u: u['PF_metric'], reverse=True)
            best_user = users_with_data[0]  # Пользователь с наивысшей PF-метрикой
            ue_id = best_user['UE_ID']
            
            # 8. Выделение RBG
            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                allocation[ue_id].extend(rb_indices)

                # Расчет AMC только один раз для пользователя
                if amc_results[ue_id] is None:
                    # Получить количество выделенных RB
                    num_rb = len(allocation[ue_id])
                    cqi = best_user['cqi']
                
                    # Вызов AMC функции
                    amc_results[ue_id] = self.channel_model.calculate_tbs(cqi=cqi, nprb=num_rb)
                
                # 9. Обновление буфера
                bits_per_rb = self.amc.GET_BITS_PER_RB(best_user['cqi'])
                rbg_capacity = len(rb_indices) * bits_per_rb * 2
                remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)
        
        # 10. Обработка буфера и статистики
        for user in users:
            ue = user['ue']
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            
            allocated_rb = len(allocation.get(user['UE_ID'], [])) * 2

            # Использование AMC результатов
            if amc_results[ue_id]:
                amc = amc_results[ue_id]
                mcs = amc['mcs']
                tbs_bits = amc['tbs_bits']
                tbs_bytes = amc['tbs_bytes']
                throughput = amc['throughput_mbps']
                bits_per_rb = amc['qm']
                
                # Логирование AMC результатов
                print(f"[TTI {tti}] UE{ue_id}: CQI={amc['cqi']} → MCS={mcs} → "
                    f"NPRB={amc['nprb']} → TBS={tbs_bits} бит → {throughput:.3f} Мбит/с")
                
                max_bytes = tbs_bytes  # ← Используем TBS из AMC
            else:
                # Fallback на старый метод
                bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                max_bytes = (allocated_rb * bits_per_rb) // 8
                
            packets, total = bs_buffer.GET_PACKETS(
                ue_id=ue_id,
                max_bytes=max_bytes,
                bits_per_rb=bits_per_rb,
                current_time=tti
            )
            
            # Обновление метрик DL
            ue.UPD_DL_THROUGHPUT(total * 8, 1)

            average_throughput_past = ue.average_throughput
            ue.average_throughput = (1 - 0.2) * average_throughput_past + 0.2 * ue.current_dl_throughput 

        # 11. Формирование bitmap по Resource Allocation 0
        bitmap = {user['UE_ID']: self.lte_grid.GENERATE_BITMAP(tti, user['UE_ID']) for user in active_users}
        
        return {
            'allocation': allocation,
            'statistics': self.amc.calculate_throughput(allocation, users, tti, self.lte_grid.bs),
            'bitmap': bitmap#,
            #'dl_throughput': {user['UE_ID']: user['ue'].current_dl_throughput for user in users}
        }
    
class ProportionalFairScheduler_v2(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE):
        super().__init__(lte_grid)
        self.amc = AdaptiveModulationAndCoding()
        self.history = {}  # Словарь для хранения истории throughput пользователей
        self.alpha = 0.1   # Коэффициент сглаживания для EMA

    def update_history(self, UE_ID: int, throughput: float):
        """Обновление истории пропускной способности пользователя"""
        if UE_ID not in self.history:
            self.history[UE_ID] = throughput
        else:
            self.history[UE_ID] = (1 - self.alpha) * self.history[UE_ID] + self.alpha * throughput

    def get_average_throughput(self, UE_ID: int) -> float:
        """Получение средней пропускной способности пользователя"""
        return self.history.get(UE_ID, 1e-6)  # Защита от деления на ноль
        
    def calculate_pf_metric(self, user: Dict) -> float:
        """Расчет PF-метрики с обработкой исключений"""
        try:
            UE_ID = user['UE_ID']
            cqi = user['cqi']
            
            bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)
            instant_throughput = 2 * bits_per_rb * 1000  # бит/с (с учетом двух слотов)
            
            avg_throughput = self.get_average_throughput(UE_ID)
            if avg_throughput <= 0:
                avg_throughput = 1e-6  # Защита от деления на ноль
                
            return instant_throughput / avg_throughput
        except (KeyError, ValueError) as e:
            # Объединенная обработка ошибок
            print(f"Ошибка расчета метрики для UE {user.get('UE_ID', 'неизвестен')}: {e}")
            return 0  # Возвращаем 0 вместо генерации исключения
    
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Планирование ресурсов с учётом данных в буфере и CQI.
        
        Args:
            tti: Индекс TTI
            users: Список пользователей с параметрами:
                - 'UE_ID': Идентификатор
                - 'buffer_size': Размер буфера в байтах
                - 'cqi': Индекс качества канала
                - 'ue': Объект UserEquipment
        
        Returns:
            Dict: Результаты распределения ресурсов
        """
        # Фильтрация активных пользователей
        active_users = []
        for user in users:
            try:
                if (user['buffer_size'] > 0 
                    and 1 <= user['cqi'] <= 15 
                    and 'UE_ID' in user 
                    and 'ue' in user):
                    active_users.append(user)
            except KeyError as e:
                print(f"Некорректные данные пользователя: отсутствует ключ {e}")
                continue
    
        if not active_users:
            return {'allocation': {}, 'statistics': {}}
    
        # Расчет PF-метрик с обработкой ошибок
        pf_metrics = {}
        for user in active_users:
            UE_ID = user['UE_ID']
            try:
                metric = self.calculate_pf_metric(user)
                pf_metrics[UE_ID] = metric
            except (KeyError, ValueError) as e:
                print(f"Ошибка расчета метрики для UE {UE_ID}: {e}")
                pf_metrics[UE_ID] = 0  # Значение по умолчанию

        # Получение уникальных частотных индексов
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        freq_indices = list({rb.freq_idx for rb in free_rbs})  # <-- Добавлено
    
        # Сортировка с гарантией числовых значений
        sorted_users = sorted(
            active_users,
            key=lambda u: pf_metrics.get(u['UE_ID'], 0),
            reverse=True
        )
        # Инициализация распределения
        allocation = {u['UE_ID']: [] for u in active_users}
        current_idx = 0

        # Распределение ресурсов
        for freq_idx in freq_indices:
            if current_idx >= len(sorted_users):
                current_idx = 0
            
            user = sorted_users[current_idx]
            if self.lte_grid.ALLOCATE_RB_PAIR(tti, freq_idx, user['UE_ID']):
                allocation[user['UE_ID']].append(freq_idx)
                current_idx += 1

        # Обработка буфера и обновление статистики
        for user in active_users:
            ue = user['ue']
            allocated_pairs = len(allocation[user['UE_ID']])
            
            bits_per_pair = 2 * self.amc.GET_BITS_PER_RB(user['cqi'])
            max_bytes = (allocated_pairs * bits_per_pair) // 8
            
            packets, total = ue.buffer.GET_PACKETS(max_bytes, bits_per_pair, current_time=tti)
            ue.UPD_THROUGHPUT(total * 8, 1)
            self.update_history(user['UE_ID'], total * 8)

        # Расчет итоговой статистики
        stats = self.amc.calculate_throughput(allocation, active_users)
        return {'allocation': allocation, 'statistics': stats}