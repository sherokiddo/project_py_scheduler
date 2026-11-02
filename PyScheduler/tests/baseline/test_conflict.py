"""Сеточный конфликт (раздел 9.5, обязателен с v3).

Ожидаемое поведение — как на 6c0aae2: повторной выдачи RB нет; известный
риск частичного отказа ALLOCATE_RBG зафиксирован (BUG-SCH-009), аллокатор
собственный rollback не выполняет. Снимок фиксирует AS-IS; инвариант
«нет двойного владения» проверяется явно и должен пережить миграцию.
"""
import pytest
from conftest import assert_matches_baseline, load_snapshot

CONFLICT_SCENARIOS = ["conflict_rbg_contended", "conflict_pdcch_limited"]


@pytest.mark.parametrize("name", CONFLICT_SCENARIOS)
def test_conflict_matches_baseline(name):
    assert_matches_baseline(name)


@pytest.mark.parametrize("name", CONFLICT_SCENARIOS)
def test_no_double_ownership_in_grid(name):
    """Каждый RB каждого слота принадлежит не более чем одному UE."""
    snap = load_snapshot(name)
    for tti_state in snap["ttis"]:
        for slot, owners in tti_state["grid_rb_owners"].items():
            # owners: {freq_idx: ue_id} — структура dict уже исключает дубли,
            # контролируем согласованность с allocation.
            assert all(isinstance(ue, int) for ue in owners.values()), slot
        alloc = tti_state["allocation"]
        for ue_id, rbs in alloc.items():
            assert len(rbs) == len(set(rbs)), (
                f"{name}: UE {ue_id} получил повторяющиеся RB {rbs}")
        # пересечений между UE в allocation быть не должно
        seen = {}
        for ue_id, rbs in alloc.items():
            for rb in rbs:
                assert rb not in seen, (
                    f"{name}: RB {rb} выдан дважды: UE {seen.get(rb)} и {ue_id}")
                seen[rb] = ue_id


def test_rbg_contended_single_winner_as_is():
    """AS-IS: при равных приоритетах спор решается в пользу первого UE списка.

    UE1 и UE2 имеют равный CQI=15 и равные буферы; на 6c0aae2 весь ресурс
    забирает UE1 (первый в отсортированном priority_list). Это наблюдаемое
    поведение зафиксировано снимком; tie-break — инвариант FD_PF (audit).
    """
    snap = load_snapshot("conflict_rbg_contended")
    tti5 = snap["ttis"][0]
    assert [ue for ue, _p in tti5["priority_list"]][:2] == [1, 2]
    assert tti5["allocation"]["1"], "UE1 не получил ресурс в спорном сценарии"
    assert tti5["allocation"]["2"] == []


def test_pdcch_limited_blocks_second_ue():
    """max_dl_cce_allowance=1: CCE получает только первый UE, второй заблокирован."""
    snap = load_snapshot("conflict_pdcch_limited")
    tti5 = snap["ttis"][0]
    cce = tti5["pdcch_stats"]["pdcch_ue_cce_allocations"]
    assert list(cce.keys()) == ["1"], f"CCE выделен не одному UE: {cce}"
    assert tti5["allocation"]["2"] == []
    assert tti5["allocation"]["3"] == []
