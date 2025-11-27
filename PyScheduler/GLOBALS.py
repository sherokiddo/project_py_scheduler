CURRENT_TIME = 0.0

SEED = None

MS_PER_SECOND = 1000    # милисекунд в секунду
SYMBOLS_PER_SLOT = 7    # OFDM символы в слоте (Normal CP)
SLOT_PER_TTI = 2        # слотов в TTI
BITS_PER_BYTE = 8       # бит в байте
RB_SIZE_KHZ = 180       # 1 поднесущия 15 khz, всего 12 поднесущих в RB

# ============================================================================
# ФУНКЦИИ КОНВЕРТАЦИИ ЕДИНИЦ ИЗМЕРЕНИЯ
# ============================================================================

def bits_to_bytes(bits: int) -> int:
    """
    Конвертировать биты в байты.
    Args:
        bits (int): Количество бит
    Returns:
        int: Количество байт (целое число)
    """
    if bits < 0:
        raise ValueError("Количество бит не может быть отрицательным")
    return bits // BITS_PER_BYTE

def bytes_to_bits(bytes_count: int) -> int:
    """
    Конвертировать байты в биты.
    Args:
        bytes_count (int): Количество байт
    Returns:
        int: Количество бит
    """
    if bytes_count < 0:
        raise ValueError("Количество байт не может быть отрицательным")
    return bytes_count * BITS_PER_BYTE

def bps_to_bpms(bps: float) -> float:
    """
    Конвертировать из бит/с в бит/мс.
    Args:
        bps (float): Пропускная способность в бит/с
    Returns:
        float: Пропускная способность в бит/мс
    """
    if bps < 0:
        raise ValueError("Пропускная способность не может быть отрицательной")
    return bps / MS_PER_SECOND

def bpms_to_bps(bpms: float) -> float:
    """
    Конвертировать из бит/мс в бит/с.
    Args:
        bpms (float): Пропускная способность в бит/мс
    Returns:
        float: Пропускная способность в бит/с
    """
    if bpms < 0:
        raise ValueError("Пропускная способность не может быть отрицательной")
    return bpms * MS_PER_SECOND

def bps_to_mbps(bps: float) -> float:
    """
    Конвертировать бит/с в мегабит/с (для пользователя).
    Args:
        bps (float): Пропускная способность в бит/с
    Returns:
        float: Пропускная способность в Мбит/с
    """
    return bps / 1000000

def mbps_to_bps(mbps: float) -> float:
    """
    Конвертировать мегабит/с в бит/с.
    Args:
        mbps (float): Пропускная способность в Мбит/с
    Returns:
        float: Пропускная способность в бит/с
    """
    return mbps * 1000000

# ============================================================================
# СПРАВОЧНАЯ ИНФОРМАЦИЯ
# ============================================================================

"""
ЕДИНИЦЫ ИЗМЕРЕНИЯ В ПРОЕКТЕ LTE SCHEDULER:

1. ДАННЫЕ:
   - Внутри кода: БИТЫ (BIT)
   - Буферы UE: БАЙТЫ (хранение), конвертируем в биты для расчетов

   Конверсия: 1 БАЙТ = 8 БИТ
   Функции: bits_to_bytes(), bytes_to_bits()

2. ПРОПУСКНАЯ СПОСОБНОСТЬ:
   - Внутри кода (TTI расчеты): БИТ/МИЛЛИСЕКУНДА (bpms)
   - На выходе (для пользователя): БИТ/СЕКУНДА (bps) или МЕГАБИТ/СЕКУНДА (Mbps)

   Конверсия: 1 bps = 1/1000 bpms
   Функции: bps_to_bpms(), bpms_to_bps(), bps_to_mbps()

3. АЛГОРИТМ КОНВЕРСИИ ДАННЫХ:
   - Из UE буфера (байты):  bytes_to_bits(buffer_size) → биты
   - В расчетах (биты):      используем везде биты

ПРИМЕР:
   buffer_size = 1024  # байты (в UE)
   bits_available = bytes_to_bits(buffer_size)  # 8192 бит
"""
