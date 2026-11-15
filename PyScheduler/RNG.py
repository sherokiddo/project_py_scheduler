"""
#------------------------------------------------------------------------------
# Модуль: RNG - Генераторы псевдослучайных чисел и случайных величин
#------------------------------------------------------------------------------
# Описание:
#   Математическая реализация генераторов псевдослучайных чисел (ПСЧ) и
#   статистических распределений для моделей трафика, мобильности и канала
#   в симуляторе планировщика беспроводной сети (PyScheduler).
#   Модуль обеспечивает детерминированную генерацию, побитовую совместимость
#   с эталонной реализацией MRG32k3a и поддержку независимых потоков
#   (streams) и прогонов (runs) для воспроизводимости экспериментов.
#
# Происхождение алгоритмов (Provenance):
#   - RngStream (MRG32k3a): независимая реализация на языке Python по
#     опубликованной математической спецификации P. L'Ecuyer, R. Simard,
#     E. J. Chen, W. D. Kelton, "An Object-Oriented Random-Number Package with
#     Many Long Streams and Substreams", Operations Research, 50(6), 2002.
#     Схема индексации потоков и подпотоков согласована с контрактом ns-3 для
#     обеспечения воспроизводимости кросс-валидационных сценариев.
#   - PhiloxStream (Philox4x64-10): независимая реализация на языке Python по
#     спецификации J. K. Salmon et al., "Parallel Random Numbers: As Easy as
#     1, 2, 3", SC'11, 2011.
#
# Версия: 2.0.1
# Дата последнего изменения: 2026-11-15
# Автор: Шаимов Богдан
# Версия Python Kernel: 3.12.9
# v.1.1.0:
# - Добавлен алгоритм генерации псевдослучайных чисел Philox4x64.
# v.2.0.0:
# - Введены статические контракты typing.Protocol (UniformEngine, RandomProvider).
# - Выбор математического ядра в RandomGenerator переведен на паттерн «Стратегия».
# - Нормализация в PhiloxStream приведена к стандарту IEEE-754: (val >> 11) * (2.0 ** -53).
# - В get_integer() усечение int() заменено на math.floor() для отрицательных границ.
# - Добавлена строгая валидация параметров распределений, индексов потоков и прогонов.
# - Вычисление стандартного отклонения в normal() вынесено до цикла генерации.
# v.2.0.1:
# - Добавлена защита от NaN/Inf во всех float-параметрах и блокировка bound <= 0
#   при variance > 0 в normal() для предотвращения бесконечного цикла отбраковки.
# - Установлены строгие верхние границы для seed (< m2), stream_idx и run_idx (<= 2^63 - 1).
# - Добавлен реестр и аллокатор потоков StreamAllocator по ключу (domain, purpose, entity_id).
#------------------------------------------------------------------------------
"""

import math
from typing import Dict, List, Optional, Protocol, Tuple, Type


# ==============================================================================
#                     КОНСТАНТЫ ГРАНИЦ МАТЕМАТИЧЕСКОГО КОНТРАКТА
# ==============================================================================

MRG_M1: int = 4294967087
MRG_M2: int = 4294944443

MAX_SEED: int = MRG_M2 - 1  # 4294944442

MAX_STREAM_IDX: int = (1 << 63) - 1
MAX_RUN_IDX: int = (1 << 63) - 1


# ==============================================================================
#                          КОНТРАКТЫ МОДУЛЯ (PROTOCOLS)
# ==============================================================================


class UniformEngine(Protocol):
    """
    Контракт математического ядра генератора псевдослучайных чисел.

    Определяет интерфейс базового источника равномерно распределенных чисел
    на полуинтервале [0.0, 1.0), используемого внутри фасада распределений.
    """

    def rand_u01(self) -> float:
        """
        Генерация очередного псевдослучайного числа на полуинтервале [0.0, 1.0).

        Returns:
            float: Псевдослучайное число из диапазона [0.0, 1.0).

        """
        pass


class RandomProvider(Protocol):
    """
    Внешний контракт источника случайных распределений для моделей симулятора.

    Используется для передачи генератора в модели мобильности, радиоканала
    и трафика через внедрение зависимостей (Dependency Injection).
    """

    def uniform(
        self, min_val: float = 0.0, max_val: float = 1.0, stream_offset: int = 0
    ) -> float:
        """
        Генерация случайного числа с равномерным распределением на [min_val, max_val).

        Args:
            min_val (float, optional): Нижняя граница диапазона. По умолчанию 0.0.
            max_val (float, optional): Верхняя граница диапазона. По умолчанию 1.0.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            float: Случайное число из диапазона [min_val, max_val).

        Raises:
            ValueError: Если параметры содержат NaN/Inf, min_val > max_val
                или stream_offset выходит за допустимые границы.

        """
        pass

    def normal(
            self,
            mean: float = 0.0,
            variance: float = 1.0,
            bound: float = float("inf"),
            stream_offset: int = 0,
    ) -> float:
        """
        Генерация случайного числа с нормальным распределением.

        Args:
            mean (float, optional): Математическое ожидание. По умолчанию 0.0.
            variance (float, optional): Дисперсия распределения (>= 0.0).
                По умолчанию 1.0.
            bound (float, optional): Максимальное допустимое отклонение от среднего
                (> 0.0 при variance > 0.0, либо >= 0.0 при variance == 0.0).
                По умолчанию inf.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            float: Случайное число из нормального распределения.

        Raises:
            ValueError: Если параметры содержат NaN/Inf, variance < 0,
                bound <= 0 (при variance > 0) или stream_offset некорректен.

        """
        pass

    def exponential(
        self, mean: float = 1.0, bound: float = 0.0, stream_offset: int = 0
    ) -> float:
        """
        Генерация случайного числа с экспоненциальным распределением.

        Args:
            mean (float, optional): Математическое ожидание (> 0.0). По умолчанию 1.0.
            bound (float, optional): Верхний предел значения (>= 0.0, при 0.0 не ограничен).
                По умолчанию 0.0.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            float: Случайное число из экспоненциального распределения.

        Raises:
            ValueError: Если параметры содержат NaN/Inf, mean <= 0, bound < 0
                или stream_offset некорректен.

        """
        pass

    def get_integer(
        self, min_val: int, max_val: int, stream_offset: int = 0
    ) -> int:
        """
        Генерация равновероятного целого числа в диапазоне [min_val, max_val] включительно.

        Args:
            min_val (int): Нижняя целочисленная граница диапазона.
            max_val (int): Верхняя целочисленная граница диапазона.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            int: Случайное целое число из отрезка [min_val, max_val].

        Raises:
            ValueError: Если min_val или max_val не int (или bool),
                min_val > max_val или stream_offset некорректен.

        """
        pass


# ==============================================================================
#                        ВСПОМОГАТЕЛЬНАЯ ВАЛИДАЦИЯ
# ==============================================================================


def _validate_int_param(
    value: int,
    name: str,
    min_allowed: int,
    max_allowed: Optional[int] = None,
) -> int:
    """
    Проверка целочисленного параметра на тип int (не bool) и нижнюю границу.

    Args:
        value (int): Проверяемое значение.
        name (str): Имя параметра для формирования сообщения об ошибке.
        min_allowed (int): Минимально допустимое значение параметра (включительно).
        max_allowed (Optional[int], optional): Максимально допустимое значение
            параметра (включительно). По умолчанию None.

    Returns:
        int: Проверенное целочисленное значение.

    Raises:
        ValueError: Если значение не является int (или является bool)
            либо выходит за границы [min_allowed, max_allowed].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Parameter '{name}' must be an integer (int). "
            f"The obtained value: {value!r}"
        )
    if value < min_allowed:
        raise ValueError(
            f"Parameter '{name}' must be >= {min_allowed}. "
            f"The obtained value: {value}"
        )
    if max_allowed is not None and value > max_allowed:
        raise ValueError(
            f"Parameter '{name}' must be <= {max_allowed}. "
            f"The obtained value: {value}"
        )
    return value


def _validate_float_param(
    value: float,
    name: str,
    allow_pos_inf: bool = False,
) -> float:
    """
    Проверка вещественного параметра на числовой тип и отсутствие NaN/Inf.

    Args:
        value (float): Проверяемое числовое значение.
        name (str): Имя параметра для сообщения об ошибке.
        allow_pos_inf (bool, optional): Разрешить ли значение +inf (например,
            для параметра bound в normal()). По умолчанию False.

    Returns:
        float: Проверенное значение, приведенное к float.

    Raises:
        ValueError: Если значение не является int/float (или является bool),
            равно NaN или недопустимой бесконечности.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Parameter '{name}' must be a real number (int or float). "
            f"The obtained value: {value!r}"
        )
    val = float(value)
    if math.isnan(val):
        raise ValueError(f"Parameter '{name}' cannot be NaN.")
    if math.isinf(val):
        if allow_pos_inf and val > 0.0:
            return val
        raise ValueError(
            f"Parameter '{name}' must be finite. "
            f"The obtained value: {value}"
        )
    return val


def _validate_seed(seed: int) -> int:
    """
    Проверка начального зерна генератора (seed).

    Запрещает отрицательные, булевы значения и значения >= MRG_M2 (4294944443),
    чтобы гарантировать невырожденность обоих компонент состояния MRG32k3a.
    При передаче 0 возвращает стандартное значение 12345.

    Args:
        seed (int): Начальное зерно генератора.

    Returns:
        int: Валидное ненулевое зерно генератора.

    Raises:
        ValueError: Если seed не является int (или является bool),
            меньше 0 или больше MAX_SEED (4294944442).

    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"Parameter 'seed' must be an integer (int). "
            f"The obtained value: {seed!r}"
        )
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(
            f"Parameter 'seed' must be in range [0, {MAX_SEED}] (< m2={MRG_M2}). "
            f"The obtained value: {seed}"
        )
    return seed if seed != 0 else 12345


# ==============================================================================
#                        МАТЕМАТИЧЕСКИЕ ЯДРА (STRATEGIES)
# ==============================================================================


class RngStream:
    """
    Математическое ядро генератора ПСЧ по алгоритму MRG32k3a (L'Ecuyer, 2002).

    Разделение последовательности на независимые потоки (шаг 2^127) и прогоны (шаг 2^76)
    выполняется с помощью предвычисленного кэша матриц перехода.

    Attributes:
        seed (int): Зерно генератора в диапазоне [1, 4294944442] (0 заменяется на 12345).
        stream_idx (int): Индекс независимого потока в диапазоне [0, 2^63 - 1].
        run_idx (int): Номер прогона внутри потока в диапазоне [1, 2^63 - 1].
        state (List[float]): Вектор текущего состояния из 6 элементов.

    """

    m1 = float(MRG_M1)
    m2 = float(MRG_M2)
    norm = 1.0 / (m1 + 1.0)
    a12 = 1403580.0
    a13n = 810728.0
    a21 = 527612.0
    a23n = 1370589.0

    A1p0 = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-810728.0, 1403580.0, 0.0],
    ]

    A2p0 = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1370589.0, 0.0, 527612.0],
    ]

    _cache_built = False
    _cache_A1: List[List[List[float]]] = []
    _cache_A2: List[List[List[float]]] = []

    @classmethod
    def _build_cache(cls) -> None:
        """
        Заполнение кэша двоичных степеней матриц перехода A1 и A2 при первом вызове.
        """
        if cls._cache_built:
            return
        curr_a1 = [[cls.A1p0[i][j] for j in range(3)] for i in range(3)]
        curr_a2 = [[cls.A2p0[i][j] for j in range(3)] for i in range(3)]
        for _ in range(190):
            curr_a1 = cls._mat_mat_mod_m(curr_a1, curr_a1, cls.m1)
            curr_a2 = cls._mat_mat_mod_m(curr_a2, curr_a2, cls.m2)
            cls._cache_A1.append(curr_a1)
            cls._cache_A2.append(curr_a2)
        cls._cache_built = True

    def __init__(self, seed: int = 12345, stream_idx: int = 0, run_idx: int = 1):
        """
        Инициализация состояния потока MRG32k3a и сдвиг к заданному потоку и прогону.

        Args:
            seed (int, optional): Начальное зерно генератора в [0, 4294944442]
                (0 заменяется на 12345). По умолчанию 12345.
            stream_idx (int, optional): Индекс потока в [0, 2^63 - 1]. По умолчанию 0.
            run_idx (int, optional): Номер прогона в [1, 2^63 - 1]. По умолчанию 1.

        Raises:
            ValueError: Если параметры имеют некорректный тип или выходят за допустимые границы.

        """
        self.seed = _validate_seed(seed)
        self.stream_idx = _validate_int_param(
            stream_idx, "stream_idx", min_allowed=0, max_allowed=MAX_STREAM_IDX
        )
        self.run_idx = _validate_int_param(
            run_idx, "run_idx", min_allowed=1, max_allowed=MAX_RUN_IDX
        )
        self.state = [float(self.seed)] * 6

        target_stream = (1 << 63) + self.stream_idx
        self._build_cache()
        self._advance_nth_by(target_stream, 127, self.state)
        self._advance_nth_by(self.run_idx, 76, self.state)

    def rand_u01(self) -> float:
        """
        Выполнение одного шага алгоритма MRG32k3a и возврат числа из [0.0, 1.0).

        Returns:
            float: Псевдослучайное число на полуинтервале [0.0, 1.0).

        """
        p1 = self.a12 * self.state[1] - self.a13n * self.state[0]
        k1 = int(p1 / self.m1)
        p1 -= k1 * self.m1
        if p1 < 0.0:
            p1 += self.m1
        self.state[0], self.state[1], self.state[2] = self.state[1], self.state[2], p1

        p2 = self.a21 * self.state[5] - self.a23n * self.state[3]
        k2 = int(p2 / self.m2)
        p2 -= k2 * self.m2
        if p2 < 0.0:
            p2 += self.m2
        self.state[3], self.state[4], self.state[5] = self.state[4], self.state[5], p2

        if p1 > p2:
            return (p1 - p2) * self.norm
        return (p1 - p2 + self.m1) * self.norm

    @staticmethod
    def _mult_mod_m(a: float, s: float, c: float, m: float) -> float:
        """
        Вычисление (a * s + c) mod m в точной целочисленной арифметике.
        """
        return float((int(a) * int(s) + int(c)) % int(m))

    @classmethod
    def _mat_vec_mod_m(
        cls, A: List[List[float]], s: List[float], m: float
    ) -> List[float]:
        """
        Умножение матрицы 3x3 на вектор-столбец длины 3 по модулю m.
        """
        v = [0.0] * 3
        for i in range(3):
            x = cls._mult_mod_m(A[i][0], s[0], 0.0, m)
            x = cls._mult_mod_m(A[i][1], s[1], x, m)
            x = cls._mult_mod_m(A[i][2], s[2], x, m)
            v[i] = x
        return v

    @classmethod
    def _mat_mat_mod_m(
        cls, A: List[List[float]], B: List[List[float]], m: float
    ) -> List[List[float]]:
        """
        Перемножение двух матриц 3x3 по модулю m.
        """
        C = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                v = 0.0
                for k in range(3):
                    v = cls._mult_mod_m(A[i][k], B[k][j], v, m)
                C[i][j] = v
        return C

    @classmethod
    def _power_of_two_matrix(
        cls, n: int
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Получение предвычисленных матриц перехода для сдвига на 2^n шагов.
        """
        return cls._cache_A1[n - 1], cls._cache_A2[n - 1]

    def _advance_nth_by(self, nth: int, by: int, state: List[float]) -> None:
        """
        Сдвиг вектора состояния state вперед на nth * (2^by) шагов.
        """
        for i in range(64):
            nbit = 63 - i
            bit = (int(nth) >> nbit) & 1
            if bit:
                matrix1, matrix2 = self._power_of_two_matrix(by + nbit)
                s1 = self._mat_vec_mod_m(matrix1, state[0:3], self.m1)
                s2 = self._mat_vec_mod_m(matrix2, state[3:6], self.m2)
                state[0:3] = s1
                state[3:6] = s2


class PhiloxStream:
    """
    Ядро Philox4x64: counter-based генератор, 10 раундов.

    За один инкремент счётчика выдаёт 4 псевдослучайных 64-битных числа.
    Приведение к [0.0, 1.0): берутся старшие 53 бита числа (размер мантиссы
    double в IEEE-754) и умножаются на 2**-53, поэтому результат всегда
    строго меньше 1.0.

    Attributes:
        seed (int): Зерно генератора в диапазоне [1, 4294944442] (0 заменяется на 12345).
        stream_idx (int): Индекс независимого потока в диапазоне [0, 2^63 - 1].
        run_idx (int): Номер прогона внутри потока в диапазоне [1, 2^63 - 1].
        key (List[int]): Ключ потока: [seed, stream_idx].
        counter (List[int]): Счётчик из четырёх 64-битных слов,
            начальное значение [0, 0, 0, run_idx].

    """

    PHILOX_M4x64_0 = 0xD2E7470EE14C6C93
    PHILOX_M4x64_1 = 0xCA5A826395121157
    PHILOX_W64_0 = 0x9E3779B97F4A7C15
    PHILOX_W64_1 = 0xBB67AE8584CAA73B
    _NORM_53 = 2.0 ** -53

    def __init__(self, seed: int = 12345, stream_idx: int = 0, run_idx: int = 1):
        """
        Инициализация ключа и счетчика потока Philox4x64.

        Args:
            seed (int, optional): Начальное зерно генератора в [0, 4294944442]
                (0 заменяется на 12345). По умолчанию 12345.
            stream_idx (int, optional): Индекс потока в [0, 2^63 - 1]. По умолчанию 0.
            run_idx (int, optional): Номер прогона в [1, 2^63 - 1]. По умолчанию 1.

        Raises:
            ValueError: Если параметры имеют некорректный тип или выходят за допустимые границы.

        """
        self.seed = _validate_seed(seed)
        self.stream_idx = _validate_int_param(
            stream_idx, "stream_idx", min_allowed=0, max_allowed=MAX_STREAM_IDX
        )
        self.run_idx = _validate_int_param(
            run_idx, "run_idx", min_allowed=1, max_allowed=MAX_RUN_IDX
        )

        self.key = [
            self.seed & 0xFFFFFFFFFFFFFFFF,
            self.stream_idx & 0xFFFFFFFFFFFFFFFF,
        ]
        self.counter = [0, 0, 0, self.run_idx & 0xFFFFFFFFFFFFFFFF]
        self._buffer: List[int] = []

    def _mulhilo(self, a: int, b: int) -> Tuple[int, int]:
        """
        Умножение двух 64-битных целых чисел с возвратом старших и младших 64 бит.
        """
        product = a * b
        return (product >> 64) & 0xFFFFFFFFFFFFFFFF, product & 0xFFFFFFFFFFFFFFFF

    def _philox_round(self) -> None:
        """
        Инкремент 256-битного счетчика, выполнение 10 раундов Philox и обновление буфера.
        """
        self.counter[0] = (self.counter[0] + 1) & 0xFFFFFFFFFFFFFFFF
        if self.counter[0] == 0:
            self.counter[1] = (self.counter[1] + 1) & 0xFFFFFFFFFFFFFFFF
            if self.counter[1] == 0:
                self.counter[2] = (self.counter[2] + 1) & 0xFFFFFFFFFFFFFFFF
                if self.counter[2] == 0:
                    self.counter[3] = (self.counter[3] + 1) & 0xFFFFFFFFFFFFFFFF

        c0, c1, c2, c3 = self.counter
        k0, k1 = self.key

        for _ in range(10):
            hi0, lo0 = self._mulhilo(self.PHILOX_M4x64_0, c0)
            hi1, lo1 = self._mulhilo(self.PHILOX_M4x64_1, c2)

            new_c0 = hi1 ^ c1 ^ k0
            new_c1 = lo1
            new_c2 = hi0 ^ c3 ^ k1
            new_c3 = lo0

            c0, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3

            k0 = (k0 + self.PHILOX_W64_0) & 0xFFFFFFFFFFFFFFFF
            k1 = (k1 + self.PHILOX_W64_1) & 0xFFFFFFFFFFFFFFFF

        self._buffer = [c3, c2, c1, c0]

    def rand_u01(self) -> float:
        """
        Извлечение 64-битного числа из буфера и нормализация к полуинтервалу [0.0, 1.0).

        Returns:
            float: Псевдослучайное число из диапазона [0.0, 1.0).

        """
        if not self._buffer:
            self._philox_round()
        val = self._buffer.pop()
        return (val >> 11) * self._NORM_53


# ==============================================================================
#                  ФАСАД МАТЕМАТИЧЕСКИХ РАСПРЕДЕЛЕНИЙ (PROVIDER)
# ==============================================================================


class RandomGenerator:
    """
    Фасад генерации случайных величин из математических распределений.

    Реализует контракт RandomProvider. Управляет созданием независимых подпотоков
    по смещению stream_offset и делегирует базовую генерацию выбранной стратегии
    из реестра _ENGINES ('MRG32k3a' или 'Philox').

    Attributes:
        seed (int): Зерно генератора в [1, 4294944442] (0 заменяется на 12345).
        base_stream_idx (int): Базовый индекс потока для данной подсистемы в [0, 2^63 - 1].
        run_idx (int): Номер независимого прогона симуляции в [1, 2^63 - 1].
        engine (str): Имя активной математической стратегии ('MRG32k3a' или 'Philox').

    """

    _ENGINES: Dict[str, Type[UniformEngine]] = {
        "MRG32k3a": RngStream,
        "Philox": PhiloxStream,
    }

    def __init__(
            self,
            seed: int = 12345,
            base_stream_idx: int = 0,
            run_idx: int = 1,
            engine: str = "MRG32k3a",
    ):
        """
        Инициализация фасада распределений и валидация параметров конфигурации.

        Args:
            seed (int, optional): Начальное зерно генератора в [0, 4294944442]
                (0 заменяется на 12345). По умолчанию 12345.
            base_stream_idx (int, optional): Базовый индекс потока в [0, 2^63 - 1].
                По умолчанию 0.
            run_idx (int, optional): Номер прогона симуляции в [1, 2^63 - 1].
                По умолчанию 1.
            engine (str, optional): Название стратегии ('MRG32k3a' или 'Philox').
                По умолчанию 'MRG32k3a'.

        Raises:
            ValueError: Если переданы некорректные параметры индексации или неизвестный движок.

        """
        self.seed = _validate_seed(seed)
        self.base_stream_idx = _validate_int_param(
            base_stream_idx,
            "base_stream_idx",
            min_allowed=0,
            max_allowed=MAX_STREAM_IDX,
        )
        self.run_idx = _validate_int_param(
            run_idx, "run_idx", min_allowed=1, max_allowed=MAX_RUN_IDX
        )

        if not isinstance(engine, str) or engine not in self._ENGINES:
            allowed = ", ".join(self._ENGINES.keys())
            raise ValueError(
                f"Unknown mathematical engine '{engine}'. "
                f"Available strategies: {allowed}"
            )

        self.engine = engine
        self._engine_cls: Type[UniformEngine] = self._ENGINES[engine]
        self._streams: Dict[int, UniformEngine] = {}
        self._normal_cache: Dict[int, Tuple[float, float]] = {}

    def _get_stream(self, stream_offset: int) -> UniformEngine:
        """
        Получение потока ПСЧ для заданного смещения.

        Args:
            stream_offset (int): Неотрицательное смещение подпотока (>= 0).

        Returns:
            UniformEngine: Экземпляр потока с индексом base_stream_idx + stream_offset.

        Raises:
            ValueError: Если stream_offset не является целым числом >= 0
                или сумма base_stream_idx + stream_offset превышает MAX_STREAM_IDX.

        """
        _validate_int_param(
            stream_offset,
            "stream_offset",
            min_allowed=0,
            max_allowed=MAX_STREAM_IDX,
        )
        target_stream_idx = self.base_stream_idx + stream_offset
        if target_stream_idx > MAX_STREAM_IDX:
            raise ValueError(
                f"Combined stream index (base_stream_idx + stream_offset = {target_stream_idx}) "
                f"exceeds MAX_STREAM_IDX ({MAX_STREAM_IDX})."
            )
        if stream_offset not in self._streams:
            self._streams[stream_offset] = self._engine_cls(
                seed=self.seed,
                stream_idx=target_stream_idx,
                run_idx=self.run_idx,
            )
        return self._streams[stream_offset]

    def uniform(
        self, min_val: float = 0.0, max_val: float = 1.0, stream_offset: int = 0
    ) -> float:
        """
        Генерация случайного числа с равномерным распределением на [min_val, max_val).

        Args:
            min_val (float, optional): Нижняя граница диапазона. По умолчанию 0.0.
            max_val (float, optional): Верхняя граница диапазона. По умолчанию 1.0.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            float: Случайное число из диапазона [min_val, max_val).

        Raises:
            ValueError: Если параметры содержат NaN/Inf, min_val > max_val
                или stream_offset выходит за допустимые границы.

        """
        min_val = _validate_float_param(min_val, "min_val")
        max_val = _validate_float_param(max_val, "max_val")
        if min_val > max_val:
            raise ValueError(
                f"Lower bound min_val ({min_val}) cannot be greater than "
                f"max_val ({max_val})"
            )
        stream = self._get_stream(stream_offset)
        return min_val + stream.rand_u01() * (max_val - min_val)

    def normal(
        self,
        mean: float = 0.0,
        variance: float = 1.0,
        bound: float = float("inf"),
        stream_offset: int = 0,
    ) -> float:
        """
        Генерация случайного числа с нормальным распределением (полярный метод Бокса — Мюллера).

        Args:
            mean (float, optional): Математическое ожидание. По умолчанию 0.0.
            variance (float, optional): Дисперсия распределения (>= 0.0).
                По умолчанию 1.0.
            bound (float, optional): Максимальное отклонение от mean (> 0.0 при variance > 0.0,
                либо >= 0.0 при variance == 0.0). По умолчанию inf.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            float: Случайное число из усеченного нормального распределения.

        Raises:
            ValueError: Если параметры содержат NaN/Inf, variance < 0,
                bound <= 0 (при variance > 0) или stream_offset некорректен.

        """
        mean = _validate_float_param(mean, "mean")
        variance = _validate_float_param(variance, "variance")
        bound = _validate_float_param(bound, "bound", allow_pos_inf=True)

        if variance < 0.0:
            raise ValueError(
                f"Variance must be >= 0. "
                f"The obtained value: {variance}"
            )
        if bound < 0.0:
            raise ValueError(
                f"Deviation bound must be >= 0. "
                f"The obtained value: {bound}"
            )
        if variance > 0.0 and bound == 0.0:
            raise ValueError(
                "Deviation bound cannot be 0.0 when variance > 0.0 "
                "(causes infinite rejection loop)."
            )

        stream = self._get_stream(stream_offset)
        if variance == 0.0:
            return mean

        std = math.sqrt(variance)

        if stream_offset in self._normal_cache:
            y, v2 = self._normal_cache.pop(stream_offset)
            x2 = mean + v2 * y * std
            if abs(x2 - mean) <= bound:
                return x2

        while True:
            u1 = stream.rand_u01()
            u2 = stream.rand_u01()
            v1 = 2.0 * u1 - 1.0
            v2 = 2.0 * u2 - 1.0
            w = v1 * v1 + v2 * v2

            if 0.0 < w <= 1.0:
                y = math.sqrt((-2.0 * math.log(w)) / w)
                x1 = mean + v1 * y * std
                x2 = mean + v2 * y * std

                if abs(x1 - mean) <= bound:
                    self._normal_cache[stream_offset] = (y, v2)
                    return x1
                elif abs(x2 - mean) <= bound:
                    return x2

    def exponential(
        self, mean: float = 1.0, bound: float = 0.0, stream_offset: int = 0
    ) -> float:
        """
        Генерация случайного числа с экспоненциальным распределением.

        Args:
            mean (float, optional): Математическое ожидание (> 0.0). По умолчанию 1.0.
            bound (float, optional): Верхний предел значения (>= 0.0, при 0.0 не ограничен).
                По умолчанию 0.0.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            float: Случайное число из экспоненциального распределения.

        Raises:
            ValueError: Если параметры содержат NaN/Inf, mean <= 0, bound < 0
                или stream_offset некорректен.

        """
        mean = _validate_float_param(mean, "mean")
        bound = _validate_float_param(bound, "bound")

        if mean <= 0.0:
            raise ValueError(
                f"Mean value for exponential distribution must be > 0. "
                f"The obtained value: {mean}"
            )
        if bound < 0.0:
            raise ValueError(
                f"Upper bound must be >= 0. "
                f"The obtained value: {bound}"
            )

        stream = self._get_stream(stream_offset)

        while True:
            v = stream.rand_u01()
            if v == 0.0:
                continue
            r = -mean * math.log(v)
            if bound == 0.0 or r <= bound:
                return r

    def get_integer(
        self, min_val: int, max_val: int, stream_offset: int = 0
    ) -> int:
        """
        Генерация равновероятного целого числа в диапазоне [min_val, max_val] включительно.

        Args:
            min_val (int): Нижняя целочисленная граница диапазона.
            max_val (int): Верхняя целочисленная граница диапазона.
            stream_offset (int, optional): Смещение индекса подпотока (>= 0).
                По умолчанию 0.

        Returns:
            int: Случайное целое число из отрезка [min_val, max_val].

        Raises:
            ValueError: Если min_val или max_val не int (или bool),
                min_val > max_val или stream_offset некорректен.

        """
        if isinstance(min_val, bool) or not isinstance(min_val, int):
            raise ValueError(
                f"Parameter 'min_val' must be an integer (int). "
                f"The obtained value: {min_val!r}"
            )
        if isinstance(max_val, bool) or not isinstance(max_val, int):
            raise ValueError(
                f"Parameter 'max_val' must be an integer (int). "
                f"The obtained value: {max_val!r}"
            )
        if min_val > max_val:
            raise ValueError(
                f"Lower bound min_val ({min_val}) cannot be greater than "
                f"max_val ({max_val})"
            )

        v = self.uniform(0.0, 1.0, stream_offset=stream_offset)
        return math.floor(min_val + v * (max_val - min_val + 1))


# ==============================================================================
#            РЕЕСТР И АЛЛОКАТОР ПОТОКОВ (STREAM REGISTRY / ALLOCATOR)
# ==============================================================================


class StreamAllocator:
    """
    Централизованный реестр и аллокатор независимых потоков ПСЧ.

    Выполняет детерминированную выдачу уникальных индексов потоков (stream_idx)
    по составному ключу (domain, purpose, entity_id), например:
        ("CHANNEL", "SHADOW_FADING", 17)
        ("MOBILITY", "WAYPOINT", 5)
    Гарантирует отсутствие коллизий между доменами, случайными процессами
    внутри одного модуля и пользовательскими сущностями без использования
    фиксированных «магических» диапазонов.
    """

    def __init__(
        self,
        seed: int = 12345,
        run_idx: int = 1,
        engine: str = "MRG32k3a",
    ):
        """
        Инициализация централизованного аллокатора потоков.

        Args:
            seed (int, optional): Глобальное зерно симуляции в [0, 4294944442].
                По умолчанию 12345.
            run_idx (int, optional): Номер прогона симуляции в [1, 2^63 - 1].
                По умолчанию 1.
            engine (str, optional): Название математического ядра ('MRG32k3a' или 'Philox').
                По умолчанию 'MRG32k3a'.

        """
        self.seed = _validate_seed(seed)
        self.run_idx = _validate_int_param(
            run_idx, "run_idx", min_allowed=1, max_allowed=MAX_RUN_IDX
        )
        if not isinstance(engine, str) or engine not in RandomGenerator._ENGINES:
            allowed = ", ".join(RandomGenerator._ENGINES.keys())
            raise ValueError(
                f"Unknown mathematical engine '{engine}'. "
                f"Available strategies: {allowed}"
            )
        self.engine = engine
        self._registry: Dict[Tuple[str, str, int], int] = {}
        self._next_stream_idx: int = 0

    def allocate_stream_idx(
        self, domain: str, purpose: str, entity_id: int = 0
    ) -> int:
        """
        Получить или зарегистрировать уникальный stream_idx для тройки (domain, purpose, entity_id).

        При повторном обращении с тем же ключом возвращает ранее выделенный индекс.

        Args:
            domain (str): Домен симулятора (например, 'MOBILITY', 'CHANNEL', 'TRAFFIC').
            purpose (str): Назначение случайного процесса внутри домена
            entity_id (int, optional): Идентификатор сущности (UE_ID, BS_ID или 0 для глобального).
                По умолчанию 0.

        Returns:
            int: Уникальный неотрицательный индекс потока stream_idx.

        Raises:
            ValueError: Если domain/purpose пусты или не являются строками,
                либо entity_id < 0.

        """
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("Parameter 'domain' must be a non-empty string.")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("Parameter 'purpose' must be a non-empty string.")
        _validate_int_param(
            entity_id, "entity_id", min_allowed=0, max_allowed=MAX_STREAM_IDX
        )

        key = (domain.strip().upper(), purpose.strip().upper(), entity_id)
        if key not in self._registry:
            if self._next_stream_idx > MAX_STREAM_IDX:
                raise ValueError("StreamAllocator exhausted MAX_STREAM_IDX.")
            self._registry[key] = self._next_stream_idx
            self._next_stream_idx += 1
        return self._registry[key]

    def get_provider(
        self, domain: str, purpose: str, entity_id: int = 0
    ) -> RandomGenerator:
        """
        Создать экземпляр RandomGenerator с уникальным потоком для (domain, purpose, entity_id).

        Args:
            domain (str): Домен симулятора ('MOBILITY', 'CHANNEL', 'TRAFFIC' и др.).
            purpose (str): Назначение случайного процесса внутри домена.
            entity_id (int, optional): Идентификатор сущности (>= 0). По умолчанию 0.

        Returns:
            RandomGenerator: Настроенный экземпляр провайдера для указанного ключа.

        """
        stream_idx = self.allocate_stream_idx(domain, purpose, entity_id)
        return RandomGenerator(
            seed=self.seed,
            base_stream_idx=stream_idx,
            run_idx=self.run_idx,
            engine=self.engine,
        )

    def get_registered_streams(self) -> Dict[Tuple[str, str, int], int]:
        """
        Получить копию текущей таблицы выделенных потоков для аудита и логирования.

        Returns:
            Dict[Tuple[str, str, int], int]: Словарь {(domain, purpose, entity_id): stream_idx}.

        """
        return self._registry.copy()