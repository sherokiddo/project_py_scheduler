"""RR-сценарий: ротация rr_ue_offset/rr_rbg_offset, 3+ TTI (раздел 9.3)."""

from conftest import assert_matches_baseline, load_snapshot


def test_rr_multitti_matches_baseline():
    assert_matches_baseline("rr_multitti")


def test_rr_offsets_sequence_fixed():
    """Последовательность offsets TTI 5..8 зафиксирована в снимке AS-IS.

    Значения сняты ПОСЛЕ schedule() соответствующего TTI. Любое изменение
    ротации на шагах M2a/M5b обязано сохранить эту последовательность.
    """
    snap = load_snapshot("rr_multitti")
    seq = [(t["tti"],
            t["scheduler_extra"]["rr_ue_offset"],
            t["scheduler_extra"]["rr_rbg_offset"]) for t in snap["ttis"]]
    baseline = [(5, 0, 1), (6, 0, 0), (7, 0, 0), (8, 0, 0)]
    assert seq == baseline, f"последовательность offsets изменилась: {seq}"


def test_rr_rotation_changes_allocation():
    """Sanity: ротация реальна — стартовый RBG первого TTI отличается от следующих."""
    snap = load_snapshot("rr_multitti")
    allocs = [t["allocation"] for t in snap["ttis"]]
    assert allocs[0] != allocs[1], "TTI5 и TTI6 идентичны — ротации нет"
