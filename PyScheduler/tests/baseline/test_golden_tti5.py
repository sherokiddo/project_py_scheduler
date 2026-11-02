"""Золотой сценарий TTI 5 (README §4, SPEC-FEAT-SCH-008 §9.1).

Ожидаемые значения зафиксированы в документации и воспроизведены побитово
на ae9db40 и 6c0aae2:
- allocation = {1: [6..17], 2: [], 3: [0..5]}
- pdcch_ue_cce_allocations = {3: 1, 1: 1} (2/25 CCE)
- извлечено 8004/2404 бита (UE1/UE3), EWMA 16008/4808
"""
from conftest import assert_matches_baseline, load_snapshot


def test_golden_tti5_matches_baseline():
    assert_matches_baseline("golden_tti5")


def test_golden_tti5_matches_readme_values():
    """Независимая проверка: значения из README §4, а не из снимка."""
    snap = load_snapshot("golden_tti5")
    tti5 = snap["ttis"][0]
    exp = snap["meta"]["expected_readme"]

    assert tti5["allocation"] == exp["allocation"]
    assert (tti5["pdcch_stats"]["pdcch_ue_cce_allocations"]
            == exp["pdcch_ue_cce_allocations"])
    assert tti5["pdcch_stats"]["pdcch_cce_allocated_count"] == 2
    assert tti5["pdcch_stats"]["pdcch_cce_total_count"] == 25
    assert tti5["last_ue_transmitted_bits"] == exp["last_ue_transmitted_bits"]

    avg = {u["ue_id"]: u["average_throughput"] for u in tti5["ues"]}
    assert avg == {int(k): v for k, v in exp["average_throughput"].items()}


def test_golden_dev_revision_pinned():
    """Снимок привязан к ревизии dev, на которой валидированы якоря."""
    snap = load_snapshot("golden_tti5")
    assert snap["meta"]["dev_revision"] == "6c0aae2"
