# python -m pytest -q tests/test_res_grid_equivalence.py из PyScheduler
import pytest

from RES_GRID import RES_GRID_LTE, RES_GRID_LTE_CACHED


BANDWIDTHS = [1.4, 3, 5, 10, 15, 20]


def old_rbg_owner_map(grid_old: RES_GRID_LTE, tti: int) -> list[int | None]:
    """
    Эталонная карта владельцев RBG из старой реализации.
    Если хотя бы один RB в RBG назначен UE, считаем владельцем этого UE.
    Для нормального сценария вся RBG должна принадлежать одному UE.
    """
    owners = []
    subframe = grid_old.GET_SUBFRAME(tti)
    assert subframe is not None, f"Old grid has no subframe for TTI={tti}"

    for rbg_idx in range((grid_old.rb_per_slot + grid_old.GET_RBG_SIZE() - 1) // grid_old.GET_RBG_SIZE()):
        ue_id = None
        rb_indices = grid_old.GET_RBG_INDICES(rbg_idx)

        for slot in subframe.slots:
            for freq in rb_indices:
                rb = slot.GET_RES_BLCK(freq)
                if rb is not None and rb.UE_ID is not None:
                    if ue_id is None:
                        ue_id = rb.UE_ID
                    else:
                        # Внутри одной RBG не должно быть смешанных UE
                        assert ue_id == rb.UE_ID, (
                            f"Inconsistent RBG ownership in old grid: "
                            f"TTI={tti}, RBG={rbg_idx}, owners={ue_id} and {rb.UE_ID}"
                        )
        owners.append(ue_id)

    return owners


def cached_rbg_owner_map(grid_new: RES_GRID_LTE_CACHED, tti: int) -> list[int | None]:
    """
    Карта владельцев RBG из новой реализации по внутреннему состоянию.
    """
    state = grid_new._get_or_create_tti_state(tti)
    return list(state["rbg_alloc"])


def old_bitmap_for_ue(grid_old: RES_GRID_LTE, tti: int, ue_id: int) -> list[int]:
    """
    Эталонный bitmap пользователя из старой реализации.
    """
    owners = old_rbg_owner_map(grid_old, tti)
    return [1 if owner == ue_id else 0 for owner in owners]


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
def test_rbg_indices_equivalent(bandwidth):
    """
    Старый и новый RES_GRID должны одинаково разбивать полосу на RBG.
    """
    old = RES_GRID_LTE(bandwidth=bandwidth, num_frames=2)
    new = RES_GRID_LTE_CACHED(bandwidth=bandwidth, window_size=20)

    total_rbg = (old.rb_per_slot + old.GET_RBG_SIZE() - 1) // old.GET_RBG_SIZE()

    assert old.GET_RBG_SIZE() == new.get_rbg_size()

    for rbg_idx in range(total_rbg):
        assert old.GET_RBG_INDICES(rbg_idx) == new.get_rbg_indices(rbg_idx)


@pytest.mark.parametrize("bandwidth", [3, 5, 10, 20])
def test_single_allocation_equivalent(bandwidth):
    """
    Одна и та же аллокация должна дать одинаковую карту владельцев RBG.
    """
    old = RES_GRID_LTE(bandwidth=bandwidth, num_frames=2)
    new = RES_GRID_LTE_CACHED(bandwidth=bandwidth, window_size=20)

    tti = 3
    ue_id = 101
    total_rbg = (old.rb_per_slot + old.GET_RBG_SIZE() - 1) // old.GET_RBG_SIZE()

    for rbg_idx in range(total_rbg):
        ok_old = old.ALLOCATE_RBG(tti, rbg_idx, ue_id)
        ok_new = new.ALLOCATE_RBG(tti, rbg_idx, ue_id)

        assert ok_old == ok_new
        assert old_rbg_owner_map(old, tti) == cached_rbg_owner_map(new, tti)


@pytest.mark.parametrize("bandwidth", [5, 10])
def test_multiple_users_allocation_equivalent(bandwidth):
    """
    После серии аллокаций нескольким UE карты распределения должны совпадать.
    """
    old = RES_GRID_LTE(bandwidth=bandwidth, num_frames=3)
    new = RES_GRID_LTE_CACHED(bandwidth=bandwidth, window_size=20)

    tti = 7
    sequence = [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
    ]

    for rbg_idx, ue_id in sequence:
        ok_old = old.ALLOCATE_RBG(tti, rbg_idx, ue_id)
        ok_new = new.ALLOCATE_RBG(tti, rbg_idx, ue_id)

        assert ok_old == ok_new

    assert old_rbg_owner_map(old, tti) == cached_rbg_owner_map(new, tti)

    for ue_id in [1, 2, 3, 999]:
        assert old_bitmap_for_ue(old, tti, ue_id) == new.GENERATE_BITMAP(tti, ue_id)


@pytest.mark.parametrize("bandwidth", [5, 10])
def test_release_equivalent(bandwidth):
    """
    Освобождение RBG не должно расходиться между старой и новой реализацией.
    """
    old = RES_GRID_LTE(bandwidth=bandwidth, num_frames=3)
    new = RES_GRID_LTE_CACHED(bandwidth=bandwidth, window_size=20)

    tti = 5
    ops = [(0, 11), (1, 11), (2, 12), (3, 13)]

    for rbg_idx, ue_id in ops:
        assert old.ALLOCATE_RBG(tti, rbg_idx, ue_id)
        assert new.ALLOCATE_RBG(tti, rbg_idx, ue_id)

    assert old.RELEASE_RBG(tti, 1) is True
    assert new.RELEASE_RBG(tti, 1) is True

    assert old.RELEASE_RBG(tti, 3) is True
    assert new.RELEASE_RBG(tti, 3) is True

    assert old_rbg_owner_map(old, tti) == cached_rbg_owner_map(new, tti)

    for ue_id in [11, 12, 13]:
        assert old_bitmap_for_ue(old, tti, ue_id) == new.GENERATE_BITMAP(tti, ue_id)


@pytest.mark.parametrize("bandwidth", [3, 10])
def test_double_allocate_same_rbg_fails_and_rolls_back(bandwidth):
    """
    Повторная аллокация уже занятой RBG должна завершаться одинаково:
    возврат False и отсутствие порчи состояния.
    """
    old = RES_GRID_LTE(bandwidth=bandwidth, num_frames=2)
    new = RES_GRID_LTE_CACHED(bandwidth=bandwidth, window_size=20)

    tti = 2
    rbg_idx = 0

    assert old.ALLOCATE_RBG(tti, rbg_idx, 1) is True
    assert new.ALLOCATE_RBG(tti, rbg_idx, 1) is True

    before_old = old_rbg_owner_map(old, tti)
    before_new = cached_rbg_owner_map(new, tti)

    # Повторное назначение тому же/другому UE должно не пройти
    assert old.ALLOCATE_RBG(tti, rbg_idx, 2) is False
    assert new.ALLOCATE_RBG(tti, rbg_idx, 2) is False

    after_old = old_rbg_owner_map(old, tti)
    after_new = cached_rbg_owner_map(new, tti)

    assert after_old == before_old
    assert after_new == before_new
    assert after_old == after_new


@pytest.mark.parametrize("bandwidth", [1.4, 3, 20])
def test_last_partial_rbg_indices_correct(bandwidth):
    """
    Последняя RBG на границе полосы должна корректно обрезаться.
    Это важный граничный случай для GET_RBG_INDICES().
    """
    old = RES_GRID_LTE(bandwidth=bandwidth, num_frames=1)
    new = RES_GRID_LTE_CACHED(bandwidth=bandwidth, window_size=10)

    total_rbg = (old.rb_per_slot + old.GET_RBG_SIZE() - 1) // old.GET_RBG_SIZE()
    last = total_rbg - 1

    old_indices = old.GET_RBG_INDICES(last)
    new_indices = new.GET_RBG_INDICES(last)

    assert old_indices == new_indices
    assert min(old_indices) >= 0
    assert max(old_indices) < old.rb_per_slot
    assert len(old_indices) >= 1
    assert len(old_indices) <= old.GET_RBG_SIZE()


def test_generate_bitmap_matches_legacy_scan_when_cache_enabled():
    """
    В новой реализации быстрый bitmap через rbg_alloc должен совпадать
    с legacy scan по RB.
    """
    new = RES_GRID_LTE_CACHED(bandwidth=10, window_size=10, enable_bitmap_cache=True)

    tti = 1
    assert new.ALLOCATE_RBG(tti, 0, 10)
    assert new.ALLOCATE_RBG(tti, 2, 10)
    assert new.ALLOCATE_RBG(tti, 3, 20)

    for ue_id in [10, 20, 30]:
        fast = new.generate_bitmap(tti, ue_id)
        slow = new._generate_bitmap_legacy_scan(tti, ue_id)
        assert fast == slow


def test_cache_window_eviction_does_not_break_recent_tti():
    """
    При маленьком window_size старые TTI должны вытесняться,
    а новые состояния оставаться корректными.
    """
    new = RES_GRID_LTE_CACHED(bandwidth=10, window_size=2, enable_bitmap_cache=True)

    assert new.ALLOCATE_RBG(0, 0, 1)
    assert new.ALLOCATE_RBG(1, 1, 2)
    assert new.ALLOCATE_RBG(2, 2, 3)  # должен вытеснить TTI=0

    stats = new.get_window_stats()
    assert stats["window_size"] == 2
    assert stats["current_size"] <= 2
    assert stats["evictions"] >= 1

    # Актуальные TTI должны быть корректны
    assert new.GENERATE_BITMAP(1, 2)[1] == 1
    assert new.GENERATE_BITMAP(2, 3)[2] == 1


@pytest.mark.parametrize("bandwidth", [5, 10])
def test_same_sequence_on_several_tti(bandwidth):
    """
    Эквивалентность не только на одном TTI, но и на серии TTI.
    """
    old = RES_GRID_LTE(bandwidth=bandwidth, num_frames=5)
    new = RES_GRID_LTE_CACHED(bandwidth=bandwidth, window_size=50)

    sequences = {
        0: [(0, 1), (1, 2)],
        1: [(0, 2), (2, 2)],
        7: [(1, 3), (3, 1)],
        11: [(0, 5)],
    }

    for tti, seq in sequences.items():
        for rbg_idx, ue_id in seq:
            ok_old = old.ALLOCATE_RBG(tti, rbg_idx, ue_id)
            ok_new = new.ALLOCATE_RBG(tti, rbg_idx, ue_id)
            assert ok_old == ok_new

        assert old_rbg_owner_map(old, tti) == cached_rbg_owner_map(new, tti)

        ue_ids = {ue for _, ue in seq} | {999}
        for ue_id in ue_ids:
            assert old_bitmap_for_ue(old, tti, ue_id) == new.GENERATE_BITMAP(tti, ue_id)