"""
#------------------------------------------------------------------------------
# Модуль: RNG - Генераторы псевдослучайных чисел и распределений
#------------------------------------------------------------------------------
# Описание:
# Математическая реализация генератора псевдослучайных чисел.
# Модуль обеспечивает побитовую совместимость генерации чисел и поддержку
# независимых потоков (streams) и прогонов (runs) для обеспечения повторяемости
# симуляций. Включает формальные контракты источников (UniformEngine,
# RandomProvider, SampleStream), генератор распределений (Uniform, Normal,
# Exponential, Integer) и поставщик данных из датасетов (DataStream).
#
# Версия: 2.0.0
# Дата последнего изменения: 2026-10-28
# Автор: Шаимов Богдан
# Версия Python Kernel: 3.12.9
#
# Изменения:
# v1.1.0 - 2026-09-25:
# - Добавлен алгоритм генерации псевдослучайных чисел - Philox4x64
# v1.2.0 - 2026-10-02:
# - Добавлен DataStream для чтения датасетов (.json, .csv)
# v2.0.0 - 2026-10-28:
# - Введены формальные контракты typing.Protocol (UniformEngine, RandomProvider, SampleStream)
# - Выбор математического ядра в RandomGenerator переведен на паттерн «Стратегия»
# - Режим чтения файлов (DataStream) отделен от математического фасада RandomGenerator
# - Исправлена нормализация Philox4x64 на (val >> 11) * (2.0 ** -53)
# - Исправлено округление в get_integer на math.floor() для отрицательных границ
# - Добавлена строгая валидация параметров распределений, индексов потоков и прогонов
# - Оптимизировано хранение буфера в DataStream и вычисление std в методе normal()
#------------------------------------------------------------------------------
"""

import csv
import json
import math
import os
from typing import Dict, Protocol, Tuple, Type, runtime_checkable


@runtime_checkable
class UniformEngine(Protocol):
    """
    Внутренний контракт математического ядра ПСЧ.
    Генерирует базовые псевдослучайные числа на полуинтервале [0.0, 1.0).
    """

    def rand_u01(self) -> float:
        ...


@runtime_checkable
class RandomProvider(Protocol):
    """
    Внешний контракт источника случайных распределений для моделей симулятора.
    """

    def uniform(self, min_val: float = 0.0, max_val: float = 1.0, stream_offset: int = 0) -> float:
        ...

    def normal(
        self,
        mean: float = 0.0,
        variance: float = 1.0,
        bound: float = float("inf"),
        stream_offset: int = 0,
    ) -> float:
        ...

    def exponential(self, mean: float = 1.0, bound: float = 0.0, stream_offset: int = 0) -> float:
        ...

    def get_integer(self, min_val: int, max_val: int, stream_offset: int = 0) -> int:
        ...


@runtime_checkable
class SampleStream(Protocol):
    """
    Внешний контракт поставщика заранее подготовленных числовых данных (датасетов).
    """

    def get_next_value(self) -> float:
        ...

    def reset(self) -> None:
        ...


def _validate_int_param(value: int, name: str, min_allowed: int) -> int:
    """Проверяет, что параметр является целым числом (не bool) и не меньше min_allowed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Параметр '{name}' должен быть целым числом (int). Получено: {value!r}")
    if value < min_allowed:
        raise ValueError(
            f"Параметр '{name}' должен быть >= {min_allowed}. Получено: {value}"
        )
    return value


def _validate_seed(seed: int) -> int:
    """Проверяет seed: запрещает отрицательные значения и заменяет 0 на 12345."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Параметр 'seed' должен быть целым числом (int). Получено: {seed!r}")
    if seed < 0:
        raise ValueError(f"Параметр 'seed' не может быть отрицательным. Получено: {seed}")
    return seed if seed != 0 else 12345


class RngStream:
    """
    Математическое ядро: Генератор MRG32k3a.
    Реализует контракт UniformEngine.
    """

    m1 = 4294967087.0
    m2 = 4294944443.0
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
    _cache_A1 = []
    _cache_A2 = []

    @classmethod
    def _build_cache(cls) -> None:
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
        self.seed = _validate_seed(seed)
        self.stream_idx = _validate_int_param(stream_idx, "stream_idx", min_allowed=0)
        self.run_idx = _validate_int_param(run_idx, "run_idx", min_allowed=1)
        self.state = [float(self.seed)] * 6

        target_stream = (1 << 63) + self.stream_idx
        self._build_cache()
        self._advance_nth_by(target_stream, 127, self.state)
        self._advance_nth_by(self.run_idx, 76, self.state)

    def rand_u01(self) -> float:
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
        return float((int(a) * int(s) + int(c)) % int(m))

    @classmethod
    def _mat_vec_mod_m(cls, A, s, m):
        v = [0.0] * 3
        for i in range(3):
            x = cls._mult_mod_m(A[i][0], s[0], 0.0, m)
            x = cls._mult_mod_m(A[i][1], s[1], x, m)
            x = cls._mult_mod_m(A[i][2], s[2], x, m)
            v[i] = x
        return v

    @classmethod
    def _mat_mat_mod_m(cls, A, B, m):
        C = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                v = 0.0
                for k in range(3):
                    v = cls._mult_mod_m(A[i][k], B[k][j], v, m)
                C[i][j] = v
        return C

    @classmethod
    def _power_of_two_matrix(cls, n: int):
        return cls._cache_A1[n - 1], cls._cache_A2[n - 1]

    def _advance_nth_by(self, nth: int, by: int, state: list) -> None:
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
    Математическое ядро: Генератор Philox4x64.
    Реализует контракт UniformEngine.
    """

    PHILOX_M4x64_0 = 0xD2E7470EE14C6C93
    PHILOX_M4x64_1 = 0xCA5A826395121157
    PHILOX_W64_0 = 0x9E3779B97F4A7C15
    PHILOX_W64_1 = 0xBB67AE8584CAA73B
    _NORM_53 = 2.0 ** -53

    def __init__(self, seed: int = 12345, stream_idx: int = 0, run_idx: int = 1):
        self.seed = _validate_seed(seed)
        self.stream_idx = _validate_int_param(stream_idx, "stream_idx", min_allowed=0)
        self.run_idx = _validate_int_param(run_idx, "run_idx", min_allowed=1)

        self.key = [
            self.seed & 0xFFFFFFFFFFFFFFFF,
            self.stream_idx & 0xFFFFFFFFFFFFFFFF,
        ]
        self.counter = [0, 0, 0, self.run_idx & 0xFFFFFFFFFFFFFFFF]
        self._buffer = []

    def _mulhilo(self, a: int, b: int) -> Tuple[int, int]:
        """Умножение двух 64-битных чисел."""
        product = a * b
        return (product >> 64) & 0xFFFFFFFFFFFFFFFF, product & 0xFFFFFFFFFFFFFFFF

    def _philox_round(self) -> None:
        """Генерация 4-х псевдослучайных 64-битных чисел."""
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
        Нормализует 64-битное число до строгого полуинтервала [0.0, 1.0)
        через сдвиг до 53 старших бит мантиссы IEEE-754.
        """
        if not self._buffer:
            self._philox_round()
        val = self._buffer.pop()
        return (val >> 11) * self._NORM_53


class DataStream:
    """
    Чтение заранее известных данных из .json или .csv.
    Реализует контракт SampleStream. Работает в обход математического генератора.
    """

    def __init__(self, filepath: str, loop: bool = True):
        if not filepath or not isinstance(filepath, str):
            raise ValueError("Необходимо указать корректный путь к файлу (filepath).")

        self.filepath = filepath
        self.loop = bool(loop)
        self._data: Tuple[float, ...] = ()
        self._cursor: int = 0

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Файл с датасетом не найден: {self.filepath}")

        self._load_data()

    def _load_data(self) -> None:
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext == ".json":
            loaded = self._load_from_json()
        elif ext == ".csv":
            loaded = self._load_from_csv()
        else:
            raise ValueError(
                f"Неподдерживаемый формат файла: {ext}. Требуется .json или .csv"
            )

        if not loaded:
            raise ValueError(
                f"Файл {self.filepath} пуст или не содержит числовых данных."
            )

        self._data = tuple(loaded)
        self._cursor = 0

    def _load_from_json(self) -> list:
        items = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    items.append(float(item))
            elif isinstance(data, dict):
                for _, val in data.items():
                    if isinstance(val, list):
                        for item in val:
                            items.append(float(item))
                        break
        return items

    def _load_from_csv(self) -> list:
        items = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for item in row:
                    try:
                        items.append(float(item.strip()))
                    except ValueError:
                        continue
        return items

    def get_next_value(self) -> float:
        """Возвращает следующее значение из датасета без дублирования буфера в памяти."""
        if self._cursor >= len(self._data):
            if self.loop:
                self._cursor = 0
            else:
                raise StopIteration(f"Данные в файле {self.filepath} закончились.")

        val = self._data[self._cursor]
        self._cursor += 1
        return val

    def reset(self) -> None:
        """Сбрасывает указатель чтения в начало датасета."""
        self._cursor = 0


class RandomGenerator:
    """
    Единая точка входа для генерации случайных величин из математических распределений.
    Реализует контракт RandomProvider.
    Поддерживает генераторы: 'MRG32k3a' и 'Philox'.
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
        self.seed = _validate_seed(seed)
        self.base_stream_idx = _validate_int_param(
            base_stream_idx, "base_stream_idx", min_allowed=0
        )
        self.run_idx = _validate_int_param(run_idx, "run_idx", min_allowed=1)

        if not isinstance(engine, str) or engine not in self._ENGINES:
            allowed = ", ".join(self._ENGINES.keys())
            raise ValueError(
                f"Неизвестный математический движок '{engine}'. "
                f"Доступные стратегии: {allowed}. "
                f"Для чтения файлов используйте отдельный класс DataStream."
            )

        self.engine = engine
        self._engine_cls: Type[UniformEngine] = self._ENGINES[engine]
        self._streams: Dict[int, UniformEngine] = {}
        self._normal_cache: Dict[int, Tuple[float, float]] = {}

    def _get_stream(self, stream_offset: int) -> UniformEngine:
        _validate_int_param(stream_offset, "stream_offset", min_allowed=0)
        if stream_offset not in self._streams:
            self._streams[stream_offset] = self._engine_cls(
                seed=self.seed,
                stream_idx=self.base_stream_idx + stream_offset,
                run_idx=self.run_idx,
            )
        return self._streams[stream_offset]

    def uniform(
        self, min_val: float = 0.0, max_val: float = 1.0, stream_offset: int = 0
    ) -> float:
        if min_val > max_val:
            raise ValueError(
                f"Нижняя граница min_val ({min_val}) не может быть больше max_val ({max_val})."
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
        if variance < 0.0:
            raise ValueError(
                f"Дисперсия variance должна быть >= 0. Получено: {variance}"
            )
        if bound < 0.0:
            raise ValueError(
                f"Ограничение отклонения bound должно быть >= 0. Получено: {bound}"
            )

        stream = self._get_stream(stream_offset)
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
        if mean <= 0.0:
            raise ValueError(
                f"Среднее значение mean для экспоненциального распределения должно быть > 0. Получено: {mean}"
            )
        if bound < 0.0:
            raise ValueError(
                f"Верхний предел bound должен быть >= 0. Получено: {bound}"
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
        Генерирует равновероятное целое число в диапазоне [min_val, max_val] включительно.
        """
        if isinstance(min_val, bool) or not isinstance(min_val, int):
            raise ValueError(f"Параметр 'min_val' должен быть int. Получено: {min_val!r}")
        if isinstance(max_val, bool) or not isinstance(max_val, int):
            raise ValueError(f"Параметр 'max_val' должен быть int. Получено: {max_val!r}")
        if min_val > max_val:
            raise ValueError(
                f"Нижняя граница min_val ({min_val}) не может быть больше max_val ({max_val})."
            )

        v = self.uniform(0.0, 1.0, stream_offset=stream_offset)
        return math.floor(min_val + v * (max_val - min_val + 1))
