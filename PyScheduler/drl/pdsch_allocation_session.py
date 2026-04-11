"""
Состояние и шаги внутренней PDSCH/RBG allocation-сессии для DRL-планировщиков.
"""

from typing import Callable, Dict, List


class PDSCHAllocationSession:
    """
    Состояние per-RBG allocation внутри одного TTI.

    Сессия не знает ничего о модели или observation adapter. Она отвечает только
    за общую механику распределения RBG:
    - хранение оставшегося буфера;
    - action-mask в порядке eligible UE;
    - применение выбранного UE к текущему RBG;
    - накопление trace последнего inference-прохода.

    Благодаря этому один и тот же per-RBG loop можно использовать как для
    inference через `DqnScheduler`, так и для будущего simulation-backed DRL env.
    """

    def __init__(
        self,
        *,
        tti: int,
        eligible_ues: List[Dict],
        ues_with_pdcch: List[Dict],
        lte_grid,
        get_reported_wb_cqi: Callable[[int], int],
        get_cqi_for_rbg: Callable[[int, int], int],
        bits_per_rb_fn: Callable[[int], int],
        build_step_snapshot_fn: Callable[..., object],
    ) -> None:
        self.tti = int(tti)
        self.eligible_ues = list(eligible_ues)
        self.eligible_ue_ids = [int(user["UE_ID"]) for user in self.eligible_ues]
        self.pdcch_ue_ids = {int(user["UE_ID"]) for user in ues_with_pdcch}
        self.lte_grid = lte_grid

        self.get_reported_wb_cqi = get_reported_wb_cqi
        self.get_cqi_for_rbg = get_cqi_for_rbg
        self.bits_per_rb_fn = bits_per_rb_fn
        self.build_step_snapshot_fn = build_step_snapshot_fn

        self.remaining_buffer_bits = {
            int(user["UE_ID"]): int(user.get("bs_buffer_size", 0) * 8)
            for user in self.eligible_ues
        }
        self.alloc_rbg_counts = {ue_id: 0 for ue_id in self.eligible_ue_ids}
        self.allocation = {ue_id: [] for ue_id in self.eligible_ue_ids}

        self.current_rbg_index = 0
        self.total_rbg = self._calculate_total_rbg()
        self.last_allocated_ue_id = None

        self.invalid_action_count = 0
        self.raw_actions: List[int] = []
        self.selected_ue_ids: List[int] = []

    def _calculate_total_rbg(self) -> int:
        rbg_size = int(self.lte_grid.GET_RBG_SIZE())
        if rbg_size <= 0:
            return 0
        return int((self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size)

    def is_done(self) -> bool:
        return self.current_rbg_index >= self.total_rbg

    def get_action_mask(self) -> List[int]:
        """
        Вернуть mask действий для текущего RBG в порядке eligible UE.
        """
        mask: List[int] = []
        for ue_id in self.eligible_ue_ids:
            reported_cqi = int(self.get_reported_wb_cqi(ue_id))
            is_valid = (
                ue_id in self.pdcch_ue_ids
                and self.remaining_buffer_bits.get(ue_id, 0) > 0
                and 1 <= reported_cqi <= 15
            )
            mask.append(int(is_valid))
        return mask

    def build_step_snapshot(self, action_mask: List[int]):
        """
        Построить snapshot текущего состояния per-RBG allocation.
        """
        return self.build_step_snapshot_fn(
            eligible_ue_ids=self.eligible_ue_ids,
            remaining_buffer_bits=self.remaining_buffer_bits,
            alloc_rbg_counts=self.alloc_rbg_counts,
            action_mask=action_mask,
            current_rbg_index=self.current_rbg_index,
            total_rbg=self.total_rbg,
        )

    def record_raw_action(self, raw_action: int, invalid_action: bool) -> None:
        """
        Сохранить trace решения модели до применения allocation.
        """
        self.raw_actions.append(int(raw_action))
        self.invalid_action_count += int(bool(invalid_action))

    def apply_selected_ue(self, ue_id: int) -> bool:
        """
        Применить выбранного UE к текущему RBG и перейти к следующему шагу.
        """
        rbg_idx = self.current_rbg_index
        self.current_rbg_index += 1

        chosen_ue_id = int(ue_id)
        cqi = int(self.get_cqi_for_rbg(chosen_ue_id, rbg_idx))
        if cqi <= 0:
            return False

        if not self.lte_grid.ALLOCATE_RBG(self.tti, rbg_idx, chosen_ue_id):
            return False

        rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
        self.allocation[chosen_ue_id].extend(rb_indices)

        bits_per_rb = int(self.bits_per_rb_fn(cqi))
        rbg_capacity_bits = len(rb_indices) * bits_per_rb
        self.remaining_buffer_bits[chosen_ue_id] = max(
            0,
            self.remaining_buffer_bits[chosen_ue_id] - rbg_capacity_bits,
        )
        self.alloc_rbg_counts[chosen_ue_id] += 1
        self.selected_ue_ids.append(chosen_ue_id)
        self.last_allocated_ue_id = chosen_ue_id
        return True
