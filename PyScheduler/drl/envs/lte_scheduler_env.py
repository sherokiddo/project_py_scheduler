"""
Gymnasium-среда для обучения прямого DRL-планировщика LTE.

Агент последовательно принимает решение по каждому RBG в пределах одного TTI.
Состояние включает признаки per-UE и глобальный контекст текущего шага.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


CQI_BITS_PER_RE = {
    1: 0.1523,
    2: 0.2344,
    3: 0.3770,
    4: 0.6016,
    5: 0.8770,
    6: 1.1758,
    7: 1.4766,
    8: 1.9141,
    9: 2.4063,
    10: 2.7305,
    11: 3.3223,
    12: 3.9023,
    13: 4.5234,
    14: 5.1152,
    15: 5.5547,
}

RE_PER_RB_TTI = 132
LTE_N_RB_DL_TO_BANDWIDTH_HZ = {
    6: 1.4e6,
    15: 3e6,
    25: 5e6,
    50: 10e6,
    75: 15e6,
    100: 20e6,
}

_P_STAY = 0.7
_P_STEP = 0.15


def _build_cqi_transition_matrix() -> np.ndarray:
    n = 15
    transition = np.zeros((n, n))
    for i in range(n):
        transition[i, i] = _P_STAY
        if i > 0:
            transition[i, i - 1] = _P_STEP
        else:
            transition[i, i] += _P_STEP
        if i < n - 1:
            transition[i, i + 1] = _P_STEP
        else:
            transition[i, i] += _P_STEP
    return transition / transition.sum(axis=1, keepdims=True)


CQI_TRANSITION = _build_cqi_transition_matrix()
MAX_BUFFER_BYTES = 1_000_000
MAX_AVG_TPUT = 100e6

TRAFFIC_MODE_FULL_BUFFER = "full_buffer"
TRAFFIC_MODE_ON_OFF = "on_off"
TRAFFIC_MODE_BURSTY = "bursty"
SUPPORTED_TRAFFIC_MODES = (
    TRAFFIC_MODE_FULL_BUFFER,
    TRAFFIC_MODE_ON_OFF,
    TRAFFIC_MODE_BURSTY,
)


def _normalize_traffic_profile(
    traffic_profile: str | list[str] | tuple[str, ...],
    n_ue: int,
) -> list[str]:
    if isinstance(traffic_profile, str):
        modes = [traffic_profile] * n_ue
    else:
        modes = list(traffic_profile)
        if len(modes) != n_ue:
            raise ValueError(
                f"traffic_profile length must match n_ue={n_ue}, got {len(modes)}"
            )

    for mode in modes:
        if mode not in SUPPORTED_TRAFFIC_MODES:
            raise ValueError(
                f"Unsupported traffic mode: {mode}. Supported modes: {SUPPORTED_TRAFFIC_MODES}"
            )

    return modes


def _build_rat0_rbg_rb_sizes(n_rb_dl: int) -> np.ndarray:
    p = _get_rat0_rbg_size(n_rb_dl)
    n_full = n_rb_dl // p
    remainder = n_rb_dl % p

    sizes = [p] * n_full
    if remainder > 0:
        sizes.append(remainder)

    return np.array(sizes, dtype=np.int32)


def _get_bandwidth_hz_from_n_rb_dl(n_rb_dl: int) -> float:
    if n_rb_dl not in LTE_N_RB_DL_TO_BANDWIDTH_HZ:
        raise ValueError(f"Unsupported n_rb_dl={n_rb_dl}")
    return float(LTE_N_RB_DL_TO_BANDWIDTH_HZ[n_rb_dl])


def _get_rat0_rbg_size(n_rb_dl: int) -> int:
    if n_rb_dl <= 10:
        return 1
    if 11 <= n_rb_dl <= 26:
        return 2
    if 27 <= n_rb_dl <= 63:
        return 3
    return 4


class LTESchedulerEnv(gym.Env):
    """
    Gymnasium-среда для обучения прямого LTE scheduler агента.

    Один `step()` соответствует одному решению для текущего RBG.
    После выдачи всех RBG внутри TTI среда обновляет буферы, CQI и reward.
    """

    metadata = {"render_modes": []}
    N_UE_FEATURES = 6
    N_CONTEXT_FEATURES = 3

    def __init__(
        self,
        n_ue: int = 3,
        n_rb_dl: int = 50,
        episode_len: int = 500,
        reward_mode: str = "delayed",
        reward_window: int = 10,
        alpha: float = 0.5,
        beta: float = 0.5,
        cqi_markov: bool = True,
        wb_cqi_report_period_tti: int = 5,
        wb_cqi_report_with_random_offset: bool = True,
        traffic_lambda: float = 5000.0,
        traffic_profile: str | list[str] | tuple[str, ...] = TRAFFIC_MODE_FULL_BUFFER,
        on_lambda_bytes: float = 5000.0,
        off_lambda_bytes: float = 0.0,
        on_off_p_on_to_off: float = 0.05,
        on_off_p_off_to_on: float = 0.10,
        burst_lambda_bytes: float = 12000.0,
        burst_start_prob: float = 0.08,
        burst_mean_tti: int = 5,
        rate_scale_bps: float = 1e6,
        jfi_target: float = 0.7,
        lambda_jfi: float = 1.0,
        seed: Optional[int] = None,
        bandwidth_hz: Optional[int] = None,
    ):
        super().__init__()

        if reward_mode not in ("per_tti", "delayed"):
            raise ValueError("reward_mode должен быть 'per_tti' или 'delayed'")
        if n_ue < 1:
            raise ValueError("n_ue должен быть >= 1")

        self.n_ue = n_ue
        self.episode_len = episode_len
        self.reward_mode = reward_mode
        self.reward_window = reward_window
        self.alpha = alpha
        self.beta = beta
        expected_bandwidth_hz = _get_bandwidth_hz_from_n_rb_dl(n_rb_dl)

        if bandwidth_hz is None:
            self.bandwidth_hz = expected_bandwidth_hz
        elif not np.isclose(float(bandwidth_hz), expected_bandwidth_hz):
            raise ValueError(
                f"bandwidth_hz={bandwidth_hz} does not match n_rb_dl={n_rb_dl} "
                f"(expected {expected_bandwidth_hz})"
            )
        else:
            self.bandwidth_hz = float(bandwidth_hz)

        self.cqi_markov = cqi_markov
        self.wb_cqi_report_period_tti = max(int(wb_cqi_report_period_tti), 1)
        self.wb_cqi_report_with_random_offset = bool(
            wb_cqi_report_with_random_offset
        )

        self.traffic_lambda = traffic_lambda
        self.traffic_profile = _normalize_traffic_profile(traffic_profile, n_ue)
        self.on_lambda_bytes = on_lambda_bytes
        self.off_lambda_bytes = off_lambda_bytes
        self.on_off_p_on_to_off = on_off_p_on_to_off
        self.on_off_p_off_to_on = on_off_p_off_to_on
        self.burst_lambda_bytes = burst_lambda_bytes
        self.burst_start_prob = burst_start_prob
        self.burst_mean_tti = max(int(burst_mean_tti), 1)

        self.rate_scale_bps = rate_scale_bps
        self.jfi_target = jfi_target
        self.lambda_jfi = lambda_jfi
        self._last_rbg_alloc = None
        self.grid_history = []
        self.n_rb_dl = n_rb_dl
        self.rbg_size_rb = _get_rat0_rbg_size(self.n_rb_dl)
        self.rbg_rb_sizes = _build_rat0_rbg_rb_sizes(self.n_rb_dl)
        self.n_rbg = int(len(self.rbg_rb_sizes))
        self.last_rbg_size_rb = int(self.rbg_rb_sizes[-1])

        obs_dim = n_ue * self.N_UE_FEATURES + self.N_CONTEXT_FEATURES
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_ue)

        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._true_wb_cqi: np.ndarray = None
        self._reported_wb_cqi: np.ndarray = None
        self._wb_cqi_age: np.ndarray = None
        self._wb_cqi_report_offset: np.ndarray = None
        self._buffer: np.ndarray = None
        self._avg_tput: np.ndarray = None
        self._rbg_alloc: np.ndarray = None
        self._current_rbg: int = 0
        self._current_tti: int = 0
        self._reward_accum: float = 0.0
        self._window_count: int = 0
        self._ema_alpha = 0.01

        self._traffic_mode: np.ndarray = None
        self._traffic_on: np.ndarray = None
        self._burst_timer: np.ndarray = None
        self._active_flag: np.ndarray = None
        self._tti_active_mask: np.ndarray = None

        self.history: Dict[str, list] = {
            "throughput": [],
            "se": [],
            "jfi": [],
            "jfi_all": [],
            "reward": [],
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._true_wb_cqi = self._rng.integers(4, 13, size=self.n_ue)
        self._reported_wb_cqi = self._true_wb_cqi.copy()
        self._wb_cqi_age = np.zeros(self.n_ue, dtype=np.int32)
        if self.wb_cqi_report_with_random_offset and self.wb_cqi_report_period_tti > 1:
            self._wb_cqi_report_offset = self._rng.integers(
                0,
                self.wb_cqi_report_period_tti,
                size=self.n_ue,
                dtype=np.int32,
            )
        else:
            self._wb_cqi_report_offset = np.zeros(self.n_ue, dtype=np.int32)

        self._buffer = self._rng.uniform(10_000, 50_000, size=self.n_ue)
        self._avg_tput = np.zeros(self.n_ue, dtype=np.float64)

        self._traffic_mode = np.array(self.traffic_profile, dtype=object)
        self._traffic_on = np.zeros(self.n_ue, dtype=bool)
        self._burst_timer = np.zeros(self.n_ue, dtype=np.int32)

        for i, mode in enumerate(self._traffic_mode):
            if mode == TRAFFIC_MODE_FULL_BUFFER:
                self._traffic_on[i] = True
            elif mode == TRAFFIC_MODE_ON_OFF:
                self._traffic_on[i] = bool(self._rng.random() < 0.5)
            elif mode == TRAFFIC_MODE_BURSTY:
                self._traffic_on[i] = False
                self._burst_timer[i] = 0

        self._active_flag = self._buffer > 0.0
        self._tti_active_mask = self._active_flag.copy()

        self._rbg_alloc = np.full(self.n_rbg, -1, dtype=np.int32)
        self._current_rbg = 0
        self._current_tti = 0
        self._reward_accum = 0.0
        self._window_count = 0

        self._last_rbg_alloc = None
        self.grid_history = []
        self.history = {
            "throughput": [],
            "se": [],
            "jfi": [],
            "jfi_all": [],
            "reward": [],
        }
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Недопустимое действие: {action}")

        invalid_action = not bool(self._get_action_mask()[action])
        self._rbg_alloc[self._current_rbg] = action
        self._current_rbg += 1

        reward = 0.0
        terminated = False
        truncated = False

        if self._current_rbg >= self.n_rbg:
            reward = self._process_tti()
            self._current_tti += 1

            self._last_rbg_alloc = self._rbg_alloc.copy()
            self.grid_history.append(self._last_rbg_alloc.copy())

            self._current_rbg = 0
            self._rbg_alloc[:] = -1

            if self._current_tti >= self.episode_len:
                terminated = True

        info = self._get_info()
        info["invalid_action"] = invalid_action
        return self._get_obs(), reward, terminated, truncated, info

    def _process_tti(self) -> float:
        tti_active_mask = self._tti_active_mask.copy()
        bits = self._compute_bits_delivered()
        self._update_buffers(bits)

        prev_avg_tput = self._avg_tput.copy()
        tti_tput_bps = bits * 1000.0
        self._avg_tput = (
            (1.0 - self._ema_alpha) * self._avg_tput
            + self._ema_alpha * tti_tput_bps
        )

        if self.cqi_markov:
            self._update_true_wb_cqi()
        self._update_wb_cqi_reports()

        throughput_bps = float(np.sum(tti_tput_bps))
        se_bps_hz = throughput_bps / self.bandwidth_hz
        jfi_all = self._compute_jfi(self._avg_tput)
        jfi_active = self._compute_jfi_masked(self._avg_tput, tti_active_mask)

        self.history["throughput"].append(throughput_bps / 1e6)
        self.history["se"].append(se_bps_hz)
        self.history["jfi"].append(jfi_active)
        self.history["jfi_all"].append(jfi_all)

        tti_reward = self._compute_reward(
            se_bps_hz=se_bps_hz,
            jfi=jfi_active,
            prev_avg_tput=prev_avg_tput,
            tti_tput_bps=tti_tput_bps,
        )
        self._tti_active_mask = self._active_flag.copy()
        if self.reward_mode == "per_tti":
            self.history["reward"].append(tti_reward)
            return tti_reward

        self._reward_accum += tti_reward
        self._window_count += 1
        if self._window_count >= self.reward_window:
            avg_reward = self._reward_accum / self.reward_window
            self.history["reward"].append(avg_reward)
            self._reward_accum = 0.0
            self._window_count = 0
            return avg_reward
        return 0.0

    def _compute_bits_delivered(self) -> np.ndarray:
        raw_bits = np.zeros(self.n_ue, dtype=np.float64)

        for rbg_idx in range(self.n_rbg):
            ue = self._rbg_alloc[rbg_idx]
            if ue < 0:
                continue

            rb_count = self.rbg_rb_sizes[rbg_idx]
            bits_per_re = CQI_BITS_PER_RE.get(int(self._true_wb_cqi[ue]), 0.0)
            raw_bits[ue] += rb_count * RE_PER_RB_TTI * bits_per_re

        buf_bits = self._buffer * 8.0
        return np.minimum(raw_bits, buf_bits)

    def _sample_new_bytes(self) -> np.ndarray:
        new_bytes = np.zeros(self.n_ue, dtype=np.float64)

        for i, mode in enumerate(self._traffic_mode):
            if mode == TRAFFIC_MODE_FULL_BUFFER:
                new_bytes[i] = float(self._rng.poisson(self.traffic_lambda))

            elif mode == TRAFFIC_MODE_ON_OFF:
                if self._traffic_on[i]:
                    if self._rng.random() < self.on_off_p_on_to_off:
                        self._traffic_on[i] = False
                else:
                    if self._rng.random() < self.on_off_p_off_to_on:
                        self._traffic_on[i] = True

                lam = self.on_lambda_bytes if self._traffic_on[i] else self.off_lambda_bytes
                new_bytes[i] = float(self._rng.poisson(lam))

            elif mode == TRAFFIC_MODE_BURSTY:
                if self._burst_timer[i] > 0:
                    self._burst_timer[i] -= 1
                    self._traffic_on[i] = True
                    new_bytes[i] = float(self._rng.poisson(self.burst_lambda_bytes))
                else:
                    self._traffic_on[i] = False
                    if self._rng.random() < self.burst_start_prob:
                        self._burst_timer[i] = max(
                            1,
                            int(self._rng.poisson(self.burst_mean_tti)),
                        )
                        self._traffic_on[i] = True
                        new_bytes[i] = float(self._rng.poisson(self.burst_lambda_bytes))

        return new_bytes

    def _update_buffers(self, bits_delivered: np.ndarray) -> None:
        bytes_out = bits_delivered / 8.0
        new_bytes = self._sample_new_bytes()

        self._buffer = np.clip(
            self._buffer - bytes_out + new_bytes,
            0.0,
            MAX_BUFFER_BYTES,
        )

        self._active_flag = self._buffer > 0.0

    def _update_true_wb_cqi(self) -> None:
        for i in range(self.n_ue):
            cqi_idx = int(self._true_wb_cqi[i]) - 1
            self._true_wb_cqi[i] = self._rng.choice(15, p=CQI_TRANSITION[cqi_idx]) + 1

    def _update_wb_cqi_reports(self) -> None:
        next_tti = self._current_tti + 1
        for i in range(self.n_ue):
            due = (
                self.wb_cqi_report_period_tti <= 1
                or (
                    next_tti + int(self._wb_cqi_report_offset[i])
                ) % self.wb_cqi_report_period_tti == 0
            )

            if due:
                self._reported_wb_cqi[i] = self._true_wb_cqi[i]
                self._wb_cqi_age[i] = 0
            else:
                self._wb_cqi_age[i] += 1

    def _get_wb_cqi_age_norm(self) -> np.ndarray:
        age_denom = max(self.wb_cqi_report_period_tti - 1, 1)
        return np.clip(self._wb_cqi_age.astype(np.float32) / age_denom, 0.0, 1.0)

    def _get_alloc_count_per_ue(self) -> np.ndarray:
        counts = np.zeros(self.n_ue, dtype=np.int32)
        for ue in range(self.n_ue):
            counts[ue] = int(np.sum(self._rbg_alloc == ue))
        return counts

    def _get_alloc_frac_per_ue(self) -> np.ndarray:
        alloc_counts = self._get_alloc_count_per_ue().astype(np.float32)
        return alloc_counts / max(self.n_rbg, 1)

    def _get_action_mask(self) -> np.ndarray:
        mask = self._active_flag.astype(bool).copy()
        if not np.any(mask):
            mask[:] = True
        return mask

    def _compute_jfi(self, bits: np.ndarray) -> float:
        total = float(np.sum(bits))
        total_sq = float(np.sum(bits**2))
        if total_sq < 1e-12:
            return 0.0
        return (total**2) / (self.n_ue * total_sq)

    def _compute_jfi_masked(self, bits: np.ndarray, mask: np.ndarray) -> float:
        active_mask = np.asarray(mask, dtype=bool)
        active_bits = np.asarray(bits, dtype=np.float64)[active_mask]
        n_active = int(np.sum(active_mask))

        if n_active == 0:
            return 1.0

        total = float(np.sum(active_bits))
        total_sq = float(np.sum(active_bits**2))
        if total_sq < 1e-12:
            return 1.0
        return (total**2) / (n_active * total_sq)

    def _compute_reward(
        self,
        se_bps_hz: float,
        jfi: float,
        prev_avg_tput: np.ndarray,
        tti_tput_bps: np.ndarray,
    ) -> float:
        rate_scale_bps = max(float(self.rate_scale_bps), 1.0)
        prev_u = np.mean(np.log1p(prev_avg_tput / rate_scale_bps))
        curr_u = np.mean(np.log1p(self._avg_tput / rate_scale_bps))
        pf_delta = curr_u - prev_u

        se_term = np.log1p(se_bps_hz)
        jfi_penalty = self.lambda_jfi * max(0.0, self.jfi_target - jfi) ** 2

        return float(self.alpha * se_term + self.beta * pf_delta - jfi_penalty)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(
            self.n_ue * self.N_UE_FEATURES + self.N_CONTEXT_FEATURES,
            dtype=np.float32,
        )
        wb_cqi_age_norm = self._get_wb_cqi_age_norm()
        alloc_frac_per_ue = self._get_alloc_frac_per_ue()
        for i in range(self.n_ue):
            b = i * self.N_UE_FEATURES
            obs[b + 0] = float(self._reported_wb_cqi[i]) / 15.0
            obs[b + 1] = float(wb_cqi_age_norm[i])
            obs[b + 2] = float(self._active_flag[i])
            obs[b + 3] = float(self._buffer[i]) / MAX_BUFFER_BYTES
            obs[b + 4] = float(self._avg_tput[i]) / MAX_AVG_TPUT
            obs[b + 5] = float(alloc_frac_per_ue[i])

        c = self.n_ue * self.N_UE_FEATURES
        obs[c + 0] = self._current_rbg / self.n_rbg
        obs[c + 1] = self._current_tti / max(self.episode_len, 1)
        obs[c + 2] = np.sum(self._rbg_alloc >= 0) / self.n_rbg

        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        alloc_count_per_ue = self._get_alloc_count_per_ue()
        alloc_frac_per_ue = self._get_alloc_frac_per_ue()
        action_mask = self._get_action_mask()
        return {
            "tti": self._current_tti,
            "rbg_step": self._current_rbg,
            "cqi": self._reported_wb_cqi.copy(),
            "true_wb_cqi": self._true_wb_cqi.copy(),
            "reported_wb_cqi": self._reported_wb_cqi.copy(),
            "wb_cqi_age": self._wb_cqi_age.copy(),
            "wb_cqi_report_period_tti": self.wb_cqi_report_period_tti,
            "wb_cqi_report_offset": self._wb_cqi_report_offset.copy(),
            "buffer_bytes": self._buffer.copy(),
            "avg_tput_bps": self._avg_tput.copy(),
            "rbg_alloc": self._rbg_alloc.copy(),
            "n_rb_dl": self.n_rb_dl,
            "n_rbg": self.n_rbg,
            "rbg_size_rb": self.rbg_size_rb,
            "rbg_rb_sizes": self.rbg_rb_sizes.copy(),
            "active_flag": self._active_flag.copy(),
            "tti_active_flag": self._tti_active_mask.copy(),
            "alloc_rbg_count_tti": alloc_count_per_ue,
            "alloc_rbg_frac_tti": alloc_frac_per_ue,
            "action_mask": action_mask,
            "traffic_mode": self._traffic_mode.copy(),
            "traffic_on": self._traffic_on.copy(),
            "burst_timer": self._burst_timer.copy(),
        }

    def get_episode_summary(self) -> Dict[str, float]:
        if not self.history["se"]:
            return {}
        throughput = np.array(self.history["throughput"])
        se = np.array(self.history["se"])
        jfi_active = np.array(self.history["jfi"])
        jfi_all = np.array(self.history["jfi_all"])
        reward = np.array(self.history["reward"])

        return {
            "mean_throughput_mbps": float(np.mean(throughput)) if len(throughput) > 0 else 0.0,
            "mean_se_bps_hz": float(np.mean(se)) if len(se) > 0 else 0.0,
            "mean_jfi": float(np.mean(jfi_active)) if len(jfi_active) > 0 else 0.0,
            "mean_jfi_active": float(np.mean(jfi_active)) if len(jfi_active) > 0 else 0.0,
            "mean_jfi_all": float(np.mean(jfi_all)) if len(jfi_all) > 0 else 0.0,
            "mean_reward": float(np.mean(reward)) if len(reward) > 0 else 0.0,
            "min_jfi": float(np.min(jfi_active)) if len(jfi_active) > 0 else 0.0,
            "min_jfi_active": float(np.min(jfi_active)) if len(jfi_active) > 0 else 0.0,
            "min_jfi_all": float(np.min(jfi_all)) if len(jfi_all) > 0 else 0.0,
            "max_se_bps_hz": float(np.max(se)) if len(se) > 0 else 0.0,
        }

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return

        tti = self._current_tti - 1
        print(f"\n[RENDER] TTI {tti}")
        print(f"  TrueCQI:  {self._true_wb_cqi.astype(int).tolist()}")
        print(f"  RepCQI:   {self._reported_wb_cqi.astype(int).tolist()}")
        print(f"  CQIAge:   {self._wb_cqi_age.astype(int).tolist()}")
        print(f"  BufferKB: {[round(b / 1024, 1) for b in self._buffer]}")
        print(f"  Active:   {self._active_flag.astype(int).tolist()}")
        print(f"  TTIActive:{self._tti_active_mask.astype(int).tolist()}")
        print(f"  ActMask:  {self._get_action_mask().astype(int).tolist()}")
        print(f"  Traffic:  {self._traffic_mode.tolist()}")
        print(f"  AvgTput:  {[round(r / 1e6, 3) for r in self._avg_tput]} Мбит/с")
        if self.history["jfi"]:
            print(f"  JFI(act): {self.history['jfi'][-1]:.3f}")
        if self.history["jfi_all"]:
            print(f"  JFI(all): {self.history['jfi_all'][-1]:.3f}")
        if self.history["se"]:
            print(f"  SE:       {self.history['se'][-1]:.3f} бит/с/Гц")
        if self._last_rbg_alloc is not None:
            counts = [int(np.sum(self._last_rbg_alloc == i)) for i in range(self.n_ue)]
            print(f"  RBG per UE: {counts} (из {self.n_rbg})")
