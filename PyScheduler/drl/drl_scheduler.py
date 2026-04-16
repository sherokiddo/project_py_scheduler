"""
Базовый runtime-планировщик для DRL-моделей внутри PyScheduler.
"""

from typing import Any, Dict, List, Optional

from SCHEDULER import SchedulerInterface
from drl.pdsch_allocation_session import PDSCHAllocationSession
from drl.playground_adapter import DRLPlaygroundObservationAdapter, MODE_CURRENT_STEP
from drl.simulation_bridge import (
    DRLPlaygroundCompatibilityReport,
    DRLPlaygroundSimulationConfig,
    DRLPlaygroundSnapshot,
    DRLPlaygroundStepSnapshot,
    DRLPlaygroundUEState,
)


class DrlScheduler(SchedulerInterface):
    """
    Общий runtime-layer для DRL-планировщиков с per-RBG inference.

    Класс инкапсулирует общий pipeline, который одинаков для DQN/PPO и других
    policy-based / value-based алгоритмов:
    - инициализация observation adapter;
    - стабильный порядок кандидатов;
    - per-RBG цикл принятия решений;
    - преобразование action -> UE_ID;
    - сбор диагностических метрик по инференсу.

    Конкретные алгоритмы должны определить только:
    - способ построения policy runner;
    - namespaced параметры (`dqn_*`, `ppo_*`, ...);
    - prefix для runtime-статистики.
    """

    def __init__(
        self,
        lte_grid,
        bs,
        *,
        model_path: Optional[str] = None,
        max_n_ue: Optional[int] = None,
        wb_cqi_report_period_tti: int = 5,
        episode_len_tti: Optional[int] = None,
        strict_observation: bool = False,
        deterministic: bool = True,
        inference_device: Optional[Any] = "cpu",
        policy_runner: Optional[Any] = None,
        stats_prefix: str = "drl",
        **kwargs,
    ):
        resolved_inference_device = self._normalize_inference_device(
            inference_device
        )

        super().__init__(lte_grid, bs, **kwargs)

        resolved_episode_len_tti = episode_len_tti
        if resolved_episode_len_tti is None:
            resolved_episode_len_tti = self.simulation_context.get(
                "sim_duration_tti"
            )

        self.stats_prefix = str(stats_prefix)
        self.policy_runner = self._build_policy_runner(
            model_path=model_path,
            provided_runner=policy_runner,
            deterministic=deterministic,
            device=resolved_inference_device,
        )
        resolved_max_n_ue = self._resolve_max_n_ue(
            configured_max_n_ue=max_n_ue,
            policy_runner=self.policy_runner,
        )
        self.observation_adapter = DRLPlaygroundObservationAdapter(
            max_n_ue=resolved_max_n_ue,
            episode_len_tti=resolved_episode_len_tti,
            wb_cqi_report_period_tti=wb_cqi_report_period_tti,
            strict_mode=bool(strict_observation),
            ensure_nonempty_action_mask=False,
        )

        self.model_path = model_path
        self.max_n_ue = resolved_max_n_ue
        self.wb_cqi_report_period_tti = int(max(wb_cqi_report_period_tti, 1))
        self.episode_len_tti = resolved_episode_len_tti
        self.strict_observation = bool(strict_observation)
        self.deterministic = bool(deterministic)
        self.inference_device = self._describe_policy_device(
            policy_runner=self.policy_runner,
            fallback_device=resolved_inference_device,
        )

        self._last_invalid_action_count = 0
        self._last_selected_ue_ids: List[int] = []
        self._last_raw_actions: List[int] = []
        self._last_step_count = 0
        self._priority_rotation_offset = 0

    def _calculate_priorities(
        self,
        windowed_ues: List[Dict],
        tti: int,
    ) -> List[Dict]:
        """
        Сформировать стабильный порядок UE для этапов PDCCH и DRL inference.
        """

        num_ues = len(windowed_ues)
        if num_ues == 0:
            return windowed_ues

        if self.max_dl_ue_tti and self.max_dl_ue_tti < num_ues:
            ues_to_plan = self.max_dl_ue_tti
        else:
            ues_to_plan = num_ues

        start_idx = self._priority_rotation_offset % num_ues
        rotated_ues = windowed_ues[start_idx:] + windowed_ues[:start_idx]

        for idx, user in enumerate(rotated_ues):
            user["priority"] = ues_to_plan - idx

        if self.verbose:
            top_ue = rotated_ues[0]["UE_ID"]
            print(
                f"[SCHEDULER.{self.__class__.__name__} TTI {tti}] Priority order: "
                f"rotated start UE {top_ue}, offset={self._priority_rotation_offset}"
            )

        return rotated_ues

    def _apply_pdsch_estimation(
        self,
        priority_list: List[Dict],
        tti: int,
    ) -> List[Dict]:
        """
        Не применять greedy PDSCH-estimation.
        """

        if self.verbose:
            print(
                f"[SCHEDULER.{self.__class__.__name__} TTI {tti}] "
                f"PDSCH estimation: SKIPPED (per-RBG policy loop)"
            )

        return priority_list

    def _allocate_pdsch(
        self,
        tti: int,
        ues_with_pdcch: List[Dict],
        eligible_ues: List[Dict],
    ) -> Dict[int, List[int]]:
        """
        Выполнить per-RBG распределение через policy runner.
        """

        allocation = {user["UE_ID"]: [] for user in eligible_ues}
        if not ues_with_pdcch:
            self._reset_step_trace()
            return allocation

        total_rbg = self._get_total_rbg()
        if total_rbg <= 0:
            self._reset_step_trace()
            return allocation

        session = PDSCHAllocationSession(
            tti=tti,
            eligible_ues=eligible_ues,
            ues_with_pdcch=ues_with_pdcch,
            lte_grid=self.lte_grid,
            get_reported_wb_cqi=self._get_wb_cqi,
            get_cqi_for_rbg=self._get_cqi_for_rbg,
            bits_per_rb_fn=self.amc.GET_BITS_PER_RB,
            build_step_snapshot_fn=lambda **kwargs: self._build_current_step_snapshot(
                tti=tti,
                eligible_ues=eligible_ues,
                eligible_ue_ids=kwargs["eligible_ue_ids"],
                remaining_buffer_bits=kwargs["remaining_buffer_bits"],
                alloc_rbg_counts=kwargs["alloc_rbg_counts"],
                action_mask=kwargs["action_mask"],
                current_rbg_index=kwargs["current_rbg_index"],
                total_rbg=kwargs["total_rbg"],
            ),
        )

        while not session.is_done():
            action_mask = session.get_action_mask()
            if not any(action_mask):
                break

            snapshot = session.build_step_snapshot(action_mask)
            adapted = self.observation_adapter.build(
                snapshot=snapshot,
                mode=MODE_CURRENT_STEP,
                ue_ids=session.eligible_ue_ids,
            )

            raw_action = int(
                self.policy_runner.predict(
                    adapted.observation,
                    adapted.action_mask,
                    deterministic=self.deterministic,
                )
            )
            chosen_ue_id, invalid_action = self._resolve_model_action(
                raw_action=raw_action,
                adapted_action_mask=adapted.action_mask,
                ue_ids_in_order=adapted.ue_ids_in_order,
            )
            session.record_raw_action(raw_action, invalid_action)
            session.apply_selected_ue(chosen_ue_id)

        if (
            session.last_allocated_ue_id is not None
            and session.last_allocated_ue_id in session.eligible_ue_ids
        ):
            last_served_idx = session.eligible_ue_ids.index(session.last_allocated_ue_id)
            self._priority_rotation_offset = (last_served_idx + 1) % len(
                session.eligible_ue_ids
            )

        self._last_invalid_action_count = session.invalid_action_count
        self._last_selected_ue_ids = list(session.selected_ue_ids)
        self._last_raw_actions = list(session.raw_actions)
        self._last_step_count = len(session.raw_actions)
        allocation = session.allocation

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(
                f"[SCHEDULER.{self.__class__.__name__} TTI {tti}] "
                f"PDSCH: {allocated_ues} UE, {total_rb} RB total, "
                f"invalid_actions={session.invalid_action_count}"
            )

        return allocation

    def get_stats(self) -> Dict:
        """
        Расширить базовую статистику диагностикой runtime-инференса.
        """

        stats = super().get_stats()
        prefix = self.stats_prefix
        stats.update(
            {
                f"{prefix}_invalid_action_count": self._last_invalid_action_count,
                f"{prefix}_step_count": self._last_step_count,
                f"{prefix}_invalid_action_rate": (
                    self._last_invalid_action_count / self._last_step_count
                    if self._last_step_count > 0
                    else 0.0
                ),
                f"{prefix}_selected_ue_ids": list(self._last_selected_ue_ids),
                f"{prefix}_raw_actions": list(self._last_raw_actions),
                f"{prefix}_model_path": self.model_path,
                f"{prefix}_inference_device": self.inference_device,
            }
        )
        return stats

    def _build_current_step_snapshot(
        self,
        *,
        tti: int,
        eligible_ues: List[Dict],
        eligible_ue_ids: List[int],
        remaining_buffer_bits: Dict[int, int],
        alloc_rbg_counts: Dict[int, int],
        action_mask: List[int],
        current_rbg_index: int,
        total_rbg: int,
    ) -> DRLPlaygroundSnapshot:
        """
        Построить current-step snapshot для текущего RBG.
        """

        allocated_rbg_total = sum(alloc_rbg_counts.values())
        allocated_fraction_progress = (
            allocated_rbg_total / total_rbg if total_rbg > 0 else 0.0
        )

        ue_states: List[DRLPlaygroundUEState] = []
        for user in eligible_ues:
            ue_id = int(user["UE_ID"])
            ue = user["ue"]
            reported_wb_cqi, wb_cqi_age_tti = self._get_reported_wb_cqi_and_age(
                ue_id=ue_id,
                current_tti=tti,
                fallback_cqi=int(user.get("cqi", 0) or 0),
            )

            ue_states.append(
                DRLPlaygroundUEState(
                    ue_id=ue_id,
                    reported_wb_cqi=reported_wb_cqi,
                    true_wb_cqi=int(getattr(ue, "cqi", user.get("cqi", 0)) or 0),
                    wb_cqi_age_tti=wb_cqi_age_tti,
                    active_flag=remaining_buffer_bits.get(ue_id, 0) > 0,
                    buffer_bytes=int(remaining_buffer_bits.get(ue_id, 0) // 8),
                    average_throughput_bps=float(
                        getattr(ue, "average_throughput", 0.0) or 0.0
                    ),
                    current_dl_throughput_bps=float(
                        getattr(ue, "current_dl_throughput", 0.0) or 0.0
                    ),
                    alloc_rbg_count_tti=int(alloc_rbg_counts.get(ue_id, 0)),
                    alloc_rbg_frac_tti=(
                        alloc_rbg_counts.get(ue_id, 0) / total_rbg
                        if total_rbg > 0
                        else 0.0
                    ),
                    sinr_db=float(getattr(ue, "SINR", 0.0) or 0.0),
                    reported_sb_cqi=self._get_reported_sb_cqi(
                        ue_id=ue_id,
                        fallback_sb_cqi=list(user.get("sbb_cqi", []) or []),
                    ),
                )
            )

        compatibility = DRLPlaygroundCompatibilityReport(
            exact_per_rbg_step_supported=True,
            reported_vs_true_wb_cqi_supported=True,
            wb_cqi_age_supported=True,
            alloc_frac_this_tti_supported=True,
            current_rbg_index_supported=True,
            scheduler_eligibility_mask_supported=True,
            notes=[
                "current_rbg_index и allocated_rbg_fraction_progress формируются прямо внутри runtime per-RBG loop.",
                "action_mask дополнительно учитывает PDCCH доступность текущего TTI.",
            ],
        )

        return DRLPlaygroundSnapshot(
            simulation_config=DRLPlaygroundSimulationConfig(
                sim_duration_tti=self.episode_len_tti,
                update_interval_tti=1,
                mobility_update_interval_tti=0,
                channel_update_interval_tti=self.wb_cqi_upd_interval,
                traffic_mode=(
                    "legacy_simple_buffer"
                    if getattr(self.lte_grid.bs, "use_simple_buffer", True)
                    else "layered_buffer"
                ),
                traffic_generator="",
                buffer_mode=(
                    "simple_buffer"
                    if getattr(self.lte_grid.bs, "use_simple_buffer", True)
                    else "layered_buffer"
                ),
                scheduler_algorithm=self.__class__.__name__,
                scheduler_max_dl_ue_tti=self.max_dl_ue_tti,
                scheduler_window_size=self.window_size,
                scheduler_window_enabled=self.enable_window,
                stats_enabled=True,
                bandwidth_mhz=float(getattr(self.lte_grid, "bandwidth", 0.0) or 0.0),
                frequency_ghz=float(
                    getattr(self.lte_grid.bs, "frequency_GHz", 0.0) or 0.0
                ),
                n_rb_dl=int(getattr(self.lte_grid, "rb_per_slot", 0) or 0),
                rbg_size_rb=int(self.lte_grid.GET_RBG_SIZE()),
                n_rbg=int(total_rbg),
                channel_model_type=str(
                    getattr(self.lte_grid.bs, "ch_model_type", "") or ""
                ),
                enable_tdl=bool(getattr(self.lte_grid.bs, "enable_tdl", False)),
            ),
            step_snapshot=DRLPlaygroundStepSnapshot(
                current_time=tti,
                current_tti=tti,
                current_rbg_index=current_rbg_index,
                allocated_rbg_fraction_progress=allocated_fraction_progress,
                allocated_rbg_fraction_final_tti=allocated_fraction_progress,
                n_rb_dl=int(getattr(self.lte_grid, "rb_per_slot", 0) or 0),
                rbg_size_rb=int(self.lte_grid.GET_RBG_SIZE()),
                n_rbg=int(total_rbg),
            ),
            ue_states=ue_states,
            action_mask=[int(value) for value in action_mask],
            compatibility=compatibility,
            scheduler_result=None,
        )

    def _resolve_model_action(
        self,
        *,
        raw_action: int,
        adapted_action_mask,
        ue_ids_in_order: List[int],
    ) -> tuple[int, bool]:
        """
        Преобразовать action модели в реальный UE_ID.
        """

        if 0 <= raw_action < len(ue_ids_in_order) and adapted_action_mask[raw_action]:
            return int(ue_ids_in_order[raw_action]), False

        valid_actions = [
            idx for idx, is_valid in enumerate(adapted_action_mask[: len(ue_ids_in_order)])
            if bool(is_valid)
        ]
        if not valid_actions:
            raise RuntimeError(
                f"{self.__class__.__name__} получил invalid action при пустом наборе valid actions."
            )

        fallback_action = valid_actions[0]
        return int(ue_ids_in_order[fallback_action]), True

    def _get_total_rbg(self) -> int:
        """
        Получить количество RBG в одном TTI.
        """

        rbg_size = int(self.lte_grid.GET_RBG_SIZE())
        if rbg_size <= 0:
            return 0
        return int((self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size)

    def _get_cqi_for_rbg(self, ue_id: int, rbg_idx: int) -> int:
        """
        Получить CQI для конкретного RBG с fallback на WB CQI.
        """

        sb_cqi = self._get_sb_cqi(ue_id)
        if sb_cqi and rbg_idx < len(sb_cqi) and sb_cqi[rbg_idx] > 0:
            return int(sb_cqi[rbg_idx])
        return int(self._get_wb_cqi(ue_id))

    def _get_reported_wb_cqi_and_age(
        self,
        *,
        ue_id: int,
        current_tti: int,
        fallback_cqi: int,
    ) -> tuple[int, Optional[int]]:
        """
        Получить reported WB CQI и его возраст.
        """

        if ue_id in self.cqi_map:
            cqi_entry = self.cqi_map[ue_id]
            age = max(0, current_tti - int(getattr(cqi_entry, "last_wb_update", 0) or 0))
            return int(getattr(cqi_entry, "wb_cqi", 0) or 0), age
        return int(fallback_cqi), None

    def _get_reported_sb_cqi(
        self,
        *,
        ue_id: int,
        fallback_sb_cqi: List[int],
    ) -> List[int]:
        """
        Получить reported SB CQI из CQI map.
        """

        if ue_id in self.cqi_map:
            return [int(cqi) for cqi in list(getattr(self.cqi_map[ue_id], "sb_cqi", []) or [])]
        return [int(cqi) for cqi in list(fallback_sb_cqi or [])]

    def _reset_step_trace(self) -> None:
        """
        Сбросить внутренний trace последнего inference TTI.
        """

        self._last_invalid_action_count = 0
        self._last_selected_ue_ids = []
        self._last_raw_actions = []
        self._last_step_count = 0

    @staticmethod
    def _build_policy_runner(
        *,
        model_path: Optional[str],
        provided_runner: Optional[Any],
        deterministic: bool,
        device: Optional[Any],
    ) -> Any:
        """
        Построить runner для конкретной policy.
        """

        raise NotImplementedError(
            "DrlScheduler._build_policy_runner() должен быть реализован в подклассе."
        )

    @staticmethod
    def _normalize_inference_device(device: Optional[Any]) -> Optional[Any]:
        """
        Нормализовать конфигурацию устройства для инференса.
        """

        if device is None:
            return "cpu"

        if isinstance(device, str):
            normalized = device.strip().lower()
            if normalized in {"", "default"}:
                return "cpu"
            if normalized == "auto":
                return None
            return normalized

        return device

    @staticmethod
    def _describe_policy_device(
        *,
        policy_runner: Any,
        fallback_device: Optional[Any],
    ) -> str:
        """
        Вернуть строковое описание устройства инференса для диагностики.
        """

        runner_device = getattr(policy_runner, "inference_device", None)
        if runner_device is not None:
            return str(runner_device)

        if fallback_device is None:
            return "auto"

        return str(fallback_device)

    @staticmethod
    def _resolve_max_n_ue(
        *,
        configured_max_n_ue: Optional[int],
        policy_runner: Any,
    ) -> Optional[int]:
        """
        Определить max_n_ue для observation adapter.
        """

        if configured_max_n_ue is not None:
            return int(configured_max_n_ue)

        runner_max_n_ue = getattr(policy_runner, "max_n_ue", None)
        if runner_max_n_ue is not None:
            return int(runner_max_n_ue)

        return None
