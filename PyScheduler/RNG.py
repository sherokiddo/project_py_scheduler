"""
#------------------------------------------------------------------------------
# Модуль: RNG - Генераторы псевдослучайных чисел и распределений
#------------------------------------------------------------------------------
# Описание:
# Математическая реализация генератора псевдослучайных чисел
# Модуль обеспечивает
# побитовую совместимость генерации чисел и поддержку независимых
# потоков (streams) и прогонов (runs) для обеспечения повторяемости
# симуляций. Также включает классы случайных величин (Uniform, Normal,
# Exponential).
#
# Версия: 1.0.1
# Дата последнего изменения: 2026-10-02
# Автор: Шаимов Богдан
# Версия Python Kernel: 3.12.9
#
# Изменения:
# v1.0.1 - 2026-10-02:
# - Добавлен алгоритм генерации псевдослучайных чисел - Philox4x64
#------------------------------------------------------------------------------
"""

import math
class RngStream:
    """
    Математическое ядро: Генератор MRG32k3a.
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
        [-810728.0, 1403580.0, 0.0]
    ]

    A2p0 = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1370589.0, 0.0, 527612.0]
    ]

    _cache_built = False
    _cache_A1 = []
    _cache_A2 = []

    @classmethod
    def _build_cache(cls):
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

    def __init__(self, seed=12345, stream_idx=0, run_idx=1):
        self.seed = seed if seed != 0 else 12345
        self.stream_idx = stream_idx
        self.run_idx = run_idx
        self.state = [float(self.seed)] * 6

        target_stream = (1 << 63) + self.stream_idx if self.stream_idx >= 0 else 0
        self._build_cache()
        self._advance_nth_by(target_stream, 127, self.state)
        self._advance_nth_by(self.run_idx, 76, self.state)

    def rand_u01(self):
        p1 = self.a12 * self.state[1] - self.a13n * self.state[0]
        k1 = int(p1 / self.m1)
        p1 -= k1 * self.m1
        if p1 < 0.0: p1 += self.m1
        self.state[0], self.state[1], self.state[2] = self.state[1], self.state[2], p1

        p2 = self.a21 * self.state[5] - self.a23n * self.state[3]
        k2 = int(p2 / self.m2)
        p2 -= k2 * self.m2
        if p2 < 0.0: p2 += self.m2
        self.state[3], self.state[4], self.state[5] = self.state[4], self.state[5], p2

        if p1 > p2: return (p1 - p2) * self.norm
        else: return (p1 - p2 + self.m1) * self.norm

    @staticmethod
    def _mult_mod_m(a, s, c, m):
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
    def _power_of_two_matrix(cls, n):
        return cls._cache_A1[n - 1], cls._cache_A2[n - 1]

    def _advance_nth_by(self, nth, by, state):
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
    """
    PHILOX_M4x64_0 = 0xD2E7470EE14C6C93
    PHILOX_M4x64_1 = 0xCA5A826395121157
    PHILOX_W64_0 = 0x9E3779B97F4A7C15
    PHILOX_W64_1 = 0xBB67AE8584CAA73B

    def __init__(self, seed=12345, stream_idx=0, run_idx=1):
        self.seed = seed if seed != 0 else 12345
        self.stream_idx = stream_idx
        self.run_idx = run_idx

        self.key = [self.seed & 0xFFFFFFFFFFFFFFFF, self.stream_idx & 0xFFFFFFFFFFFFFFFF]

        self.counter = [0, 0, 0, self.run_idx & 0xFFFFFFFFFFFFFFFF]

        self._buffer = []

    def _mulhilo(self, a, b):
        """Умножение двух 64-битных чисел."""
        product = a * b
        return (product >> 64) & 0xFFFFFFFFFFFFFFFF, product & 0xFFFFFFFFFFFFFFFF

    def _philox_round(self):
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

    def rand_u01(self):
        """Нормализует 64-битное число до [0.0, 1.0) делением на 2^64"""
        if not self._buffer:
            self._philox_round()
        val = self._buffer.pop()
        return val / 18446744073709551616.0

class RandomGenerator:
    """
    Единая точка входа для генерации случайных распределений.
    Поддерживает алгоритмы: 'MRG32k3a' и 'Philox'
    """

    def __init__(self, seed=12345, base_stream_idx=0, run_idx=1, engine='MRG32k3a'):
        self.seed = seed if seed != 0 else 12345
        self.base_stream_idx = base_stream_idx
        self.run_idx = run_idx
        self.engine = engine
        self._streams = {}
        self._normal_cache = {}

    def _get_stream(self, stream_offset):
        if stream_offset not in self._streams:
            if self.engine == 'Philox':
                self._streams[stream_offset] = PhiloxStream(
                    seed=self.seed,
                    stream_idx=self.base_stream_idx + stream_offset,
                    run_idx=self.run_idx
                )
            else:
                self._streams[stream_offset] = RngStream(
                    seed=self.seed,
                    stream_idx=self.base_stream_idx + stream_offset,
                    run_idx=self.run_idx
                )
        return self._streams[stream_offset]

    def uniform(self, min_val=0.0, max_val=1.0, stream_offset=0):
        stream = self._get_stream(stream_offset)
        return min_val + stream.rand_u01() * (max_val - min_val)

    def normal(self, mean=0.0, variance=1.0, bound=float('inf'), stream_offset=0):
        stream = self._get_stream(stream_offset)

        if stream_offset in self._normal_cache:
            y, v2 = self._normal_cache.pop(stream_offset)
            x2 = mean + v2 * y * math.sqrt(variance)
            if abs(x2 - mean) <= bound:
                return x2

        while True:
            u1 = stream.rand_u01()
            u2 = stream.rand_u01()
            v1 = 2 * u1 - 1
            v2 = 2 * u2 - 1
            w = v1 * v1 + v2 * v2

            if 0 < w <= 1.0:
                y = math.sqrt((-2 * math.log(w)) / w)
                x1 = mean + v1 * y * math.sqrt(variance)
                x2 = mean + v2 * y * math.sqrt(variance)

                if abs(x1 - mean) <= bound:
                    self._normal_cache[stream_offset] = (y, v2)
                    return x1
                elif abs(x2 - mean) <= bound:
                    return x2

    def exponential(self, mean=1.0, bound=0.0, stream_offset=0):
        stream = self._get_stream(stream_offset)
        while True:
            v = stream.rand_u01()
            if v == 0:
                continue
            r = -mean * math.log(v)
            if bound == 0 or r <= bound:
                return r

    def get_integer(self, min_val, max_val, stream_offset=0):
        """
        Генерирует целое число в диапазоне [min_val, max_val] включительно.
        """
        v = self.uniform(0.0, 1.0, stream_offset=stream_offset)
        return int(min_val + v * (max_val - min_val + 1))
