"""
#------------------------------------------------------------------------------
# Модуль: AMC - Адаптивная модуляция и кодирование для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет выбор индекса MCS по CQI, расчёт ёмкости ресурсного блока
# и табличный расчёт размера транспортного блока (TBS) в битах.
# Содержит совместимую политику выбора MCS по широкополосному CQI (WB-CQI)
# и интерфейс для подключения других политик выбора MCS.
#
# Версия: 1.0.2
# Дата последнего изменения: 2026-11-08
#
# Зависимости:
# - GLOBALS.py (таблица TBS_TABLE и соответствие MCS -> I_TBS)
# - dataclasses, typing (структуры конфигурации и контракт политики MCS)
# Опосредованные зависимости:
# - SCHEDULER.py (состояние PDCCH и данные для legacy-статистики при передаче
#   экземпляра планировщика в конструктор; прямого импорта Scheduler нет)
#
# Основные компоненты:
# - AdaptiveModulationAndCoding: совместимый API расчёта ёмкости и статистики.
# - AmcConfig: неизменяемая конфигурация числа управляющих OFDM-символов.
# - McsSelectionInput: входные данные политики MCS, включая WB/SB-CQI
#   и сведения о выделенных PRB.
# - McsSelectionPolicy, LegacyWbMcsPolicy: контракт и стандартная WB-политика.
# - get_tbs_bits: проверка MCS/числа PRB и выбор размера TB из таблицы.
#
# Текущее состояние v1.0.2:
# - Класс AMC вынесен из SCHEDULER.py; прежний импорт через Scheduler сохранён.
# - Планировщики обращаются к расчёту ёмкости через интерфейс AmcView
#   и адаптер LegacyAmcView, определённые в SCHEDULER.py.
# - GET_BITS_PER_RB учитывает управляющие символы и число антенных портов.
# - Явный AmcConfig имеет приоритет; без него используется текущий PCFICH
#   планировщика либо значение 2 при работе без планировщика.
# - get_stats сохраняет сбор статистики из состояния планировщика.
# - Выбор MCS и табличный расчёт TBS разделены; TransportBlock использует
#   get_tbs_bits без дублирования таблиц и проверок входных параметров.
#
# Ограничения и дальнейшее развитие:
# - Рабочий бюджет передачи планировщиков пока рассчитывается по формуле
#   GET_BITS_PER_RB, а не по таблице TBS.
# - Новый select_mcs ещё не вызывается планировщиками. Стандартная политика
#   использует только WB-CQI; политика агрегации SB-CQI не реализована.
# - Legacy-преобразование CQI -> MCS является правилом данного проекта.
# - Формула ёмкости использует 7 OFDM-символов на слот; выбор CP не реализован.
#------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from typing import Dict, Protocol
import GLOBALS

SPECTRAL_EFFICIENCY_FOR_CQI = [
    0.0,
    0.15, 0.23, 0.38, 0.6, 0.88, 1.18, 1.48,
    1.91, 2.41, 2.73, 3.32, 3.9, 4.52, 5.12, 5.55,
]
 
# Индекс = MCS (0..31, валидны 0..28; 29..31 зарезервированы и равны 0).
SPECTRAL_EFFICIENCY_FOR_MCS = [
    0.15, 0.19, 0.23, 0.31, 0.38, 0.49, 0.6, 0.74,
    0.88, 1.03, 1.18, 1.33, 1.48, 1.7, 1.91, 2.16,
    2.41, 2.57, 2.73, 3.03, 3.32, 3.61, 3.9, 4.21,
    4.52, 4.82, 5.12, 5.33, 5.55, 0, 0, 0,
]

def get_tbs_bits(mcs: int, rb: int) -> int:
    if type(mcs) is not int or not 0 <= mcs <= 28:
        raise ValueError("MCS должен быть целым числом от 0 до 28")
    row = GLOBALS.TBS_TABLE[GLOBALS.mcs_to_itbs[mcs]]
    if type(rb) is not int or not 1 <= rb <= len(row):
        raise ValueError(f"Число PRB должно быть целым числом от 1 до {len(row)}")
    return row[rb - 1]


def legacy_mcs_selection(cqi: int) -> int:
    """Определение эффективности для унаследованных проектов, а не универсальное правило преобразования CQI в MCS для LTE."""
    if not (0 <= cqi <= 15):
        raise ValueError(f"CQI must be in [0..15] = {cqi}")
    efficiency = SPECTRAL_EFFICIENCY_FOR_CQI[cqi]
    mcs = 0
    while mcs < 28 and SPECTRAL_EFFICIENCY_FOR_MCS[mcs + 1] <= efficiency:
        mcs += 1
    return mcs


@dataclass(frozen=True)
class AmcConfig:
    control_symbols: int = 2


@dataclass(frozen=True)
class McsSelectionInput:
    """Значения для одного выделения; пары PRB-поддиапазон указаны явно.
        Все коллекции представляют собой кортежи, а не ссылки на планировщик или карты CQI в реальном времени.
        Политика выделения SB должна определять правила агрегирования и отчета об отсутствии данных.
    """
    wb_cqi: int
    sb_cqi: tuple[int, ...] = ()
    allocated_prbs: tuple[int, ...] = ()
    prb_to_subband: tuple[tuple[int, int], ...] = ()


class McsSelectionPolicy(Protocol):
    def select_mcs(self, context: McsSelectionInput) -> int: ...


class LegacyWbMcsPolicy:
    def select_mcs(self, context: McsSelectionInput) -> int:
        return legacy_mcs_selection(context.wb_cqi)


class AdaptiveModulationAndCoding:
    """
    Класс для преобразования CQI в MCS и расчета бит на ресурсный блок.
    """
    
    def get_mcs_from_cqi(self, cqi: int) -> int:
        """
        Совместимый интерфейс legacy-политики выбора MCS проекта.
 
        Берёт спектральную эффективность для CQI и ищет наибольший MCS,
        спектральная эффективность которого не превышает её.
        """
        return legacy_mcs_selection(cqi)

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

    def __init__(self, scheduler=None, *, config: AmcConfig | None = None,
                 mcs_policy: McsSelectionPolicy | None = None):
        if config is not None and not isinstance(config, AmcConfig):
            raise TypeError("config must be AmcConfig")
        self.scheduler = scheduler
        self.config = config
        self.mcs_policy = mcs_policy if mcs_policy is not None else LegacyWbMcsPolicy()

    def select_mcs(self, context: McsSelectionInput) -> int:
        """API с поддержкой распределения ресурсов (allocation-aware API); существующие механизмы планирования остаются в статусе «устаревших» (legacy)"""
        mcs = self.mcs_policy.select_mcs(context)
        if type(mcs) is not int or not 0 <= mcs <= 28:
            raise ValueError("MCS policy must return an integer in [0..28]")
        return mcs

    def GET_BITS_PER_RB(self, cqi: int, n_ports: int = 1) -> int:
        """
        Рассчитать количество бит на ресурсный блок (RB) для заданного CQI.
        """
        if cqi not in self.CQI_TO_MCS:
            raise ValueError(f"Invalid CQI {cqi}. Must be 1-15.")

        # Explicit config wins; otherwise preserve live legacy PDCCH reading.
        pcfich = (self.config.control_symbols if self.config is not None else
                  self.scheduler.pdcch_manager.pcfich if self.scheduler else 2)

        modulation, code_rate = self.CQI_TO_MCS[cqi]

        rs_per_slot           = n_ports * 4          # CRS: 4 RE/slot при 1 порте
        re_slot0              = (7 - pcfich) * 12 - rs_per_slot
        re_slot1              = 7 * 12 - rs_per_slot
        re_per_rb_tti         = re_slot0 + re_slot1

        return int(re_per_rb_tti * modulation * code_rate)

    # @sherokiddo: "Добавить зависимость от CP"
    # Legacy formula remains in use. Table sizing is separately available as
    # get_tbs_bits(mcs, rb); activating it in scheduling is a separate change.

    def get_stats(self) -> Dict:
        """
        Получить статистику AMC за последний TTI из кэша scheduler.
        Рассчитывает throughput на основе _last_allocation и _last_users.
        Вызывается StatsManager.

        Returns:
            Dict: {
                "total_throughput": float,     # Общий throughput в bps
                "avg_bits_per_rb": float,      # Средние биты на RB
                "ue_throughputs": Dict         # UEID -> throughput (bps)
            }
        """
        if not self.scheduler or not hasattr(self.scheduler, '_last_users'):
            return {
                "dl_capacity_bits_sum_tti": 0,
                "dl_transmitted_bits_sum_tti": 0,
                "dl_throughput_sum_kbps": 0.0,
                "dl_bits_per_rb_avg": 0.0,
                "dl_cqi_wb_avg_idx": 0.0,
                "dl_sinr_avg": 0.0,
                "dl_ue_throughputs": {},
                "ue_cqi": {},
                "ue_cqi_sbb": {},
                "ue_sinr": {},
                "ue_rb_allocated": {},
            }

        users = self.scheduler._last_users
        allocation = self.scheduler._last_allocation

        total_throughput_bps    = 0.0
        total_capacity_bits     = 0
        total_transmitted_bits  = 0
        total_allocated_rbs     = 0

        ue_throughputs = {user.get('UE_ID'): 0 for user in users}

        ue_cqi = {}
        ue_cqi_sbb = {}
        ue_sinr = {}
        ue_rb_allocated = {}
        cqi_values = []
        sinr_values = []

        for user in users:
            ue = user.get('ue')
            ue_id = user.get('UE_ID')

            if not ue or not ue_id:
                continue

            if ue_id in self.scheduler.cqi_map:
                cqi_entry = self.scheduler.cqi_map[ue_id]

                cqi = cqi_entry.wb_cqi
                ue_cqi[ue_id] = cqi

                if cqi_entry.sb_cqi and len(cqi_entry.sb_cqi) > 0:
                    ue_cqi_sbb[ue_id] = cqi_entry.sb_cqi.copy()
            else:
                # Fallback
                cqi = 0

            #TODO: вынести cqi из цикла

            throughput_bps        = ue.current_dl_throughput
            ue_throughputs[ue_id] = throughput_bps
            total_throughput_bps += throughput_bps

            allocated_rbs = len(allocation.get(ue_id, []))
            ue_rb_allocated[ue_id] = allocated_rbs
            
            if allocated_rbs > 0 and 1 <= cqi <= 15:
                bits_per_rb     = self.GET_BITS_PER_RB(cqi)
                capacity_bits   = bits_per_rb * allocated_rbs
                
                total_capacity_bits += capacity_bits

            transmitted_bits        = int(ue.last_transmitted_bits)
            total_transmitted_bits += transmitted_bits
            total_allocated_rbs    += allocated_rbs

            cqi_values.append(cqi)

            if hasattr(ue, 'SINR'):
                sinr_values.append(ue.SINR)
                ue_sinr[ue_id] = ue.SINR

        # Avg bits per RB (на основе actual)
        if total_allocated_rbs > 0:
            avg_bits_per_rb = total_transmitted_bits / (total_allocated_rbs)
        else:
            avg_bits_per_rb = 0.0

        if cqi_values:
            cqi_avg = sum(cqi_values) / len(cqi_values)
        else: cqi_avg = 0

        if sinr_values:
            sinr_avg = sum(sinr_values) / len(sinr_values)
        else: sinr_avg = 0

        stats = {
            "dl_capacity_bits_sum_tti": total_capacity_bits,
            "dl_transmitted_bits_sum_tti": total_transmitted_bits,
            "dl_throughput_sum_kbps": round(total_throughput_bps/1000, 2),
            "dl_bits_per_rb_avg": round(avg_bits_per_rb, 2),
            "dl_cqi_wb_avg_idx": round(cqi_avg, 2),
            "dl_sinr_avg": round(sinr_avg, 2),
            "dl_ue_throughputs": ue_throughputs,
            "ue_cqi": ue_cqi,
            "ue_cqi_sbb": ue_cqi_sbb,
            "ue_sinr": ue_sinr,
            "ue_rb_allocated": ue_rb_allocated,}

        return stats

