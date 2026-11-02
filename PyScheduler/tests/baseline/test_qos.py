"""QoS-сценарии: flow grants + per-QCI stats (раздел 9.4).

Эталон полного прогона — sim_with_manager_qos (TEST_MODULES.py :234) —
вынесен в контрольные точки RUNBOOK.md (full-SIM); здесь — детерминированные
сценарии на уровне schedule(): те же наблюдаемые поля (allocation_flow,
гранты через буферы, per-QCI throughput QciThroughputTracker).
"""

from conftest import assert_matches_baseline, load_snapshot


def test_qos_basic_single_bearer_matches_baseline():
    assert_matches_baseline("qos_basic_single_bearer")


def test_qos_aware_multiflow_matches_baseline():
    assert_matches_baseline("qos_aware_multiflow")


def test_qos_aware_allocation_flow_present_and_nonzero():
    """allocation_flow (:3087/:3149) заполнен и содержит ненулевые гранты."""
    snap = load_snapshot("qos_aware_multiflow")
    flow = snap["ttis"][0]["scheduler_extra"]["allocation_flow"]
    assert flow, "allocation_flow пуст"
    assert any(v > 0 for v in flow.values()), f"все гранты нулевые: {flow}"
    # ключи — потоки (ue_id:qci) из профилей QOS_MULTIFLOW
    assert set(flow) == {"1:1", "1:9", "2:4", "3:6", "3:9"}


def test_qos_per_qci_stats_recorded():
    """Per-QCI throughput tracker обновлён для обоих QoS-алгоритмов."""
    for name in ("qos_basic_single_bearer", "qos_aware_multiflow"):
        snap = load_snapshot(name)
        tracker = snap["ttis"][0]["qci_tracker"]
        assert tracker, f"{name}: qci_tracker пуст"
        for ue_id, per_qci in tracker.items():
            assert per_qci, f"{name}: UE {ue_id} без QCI-статистики"
            for qci, st in per_qci.items():
                assert st["current_throughput"] >= 0, (
                    f"{name}: UE {ue_id} QCI {qci}: current_throughput < 0")
                assert st["average_throughput"] >= 0, (
                    f"{name}: UE {ue_id} QCI {qci}: average_throughput < 0")


def test_qos_layered_buffer_status_has_qci_and_hol():
    """Sanity: сценарии построены на LayeredBufferManager — qci/hol_delay не None."""
    for name in ("qos_basic_single_bearer", "qos_aware_multiflow"):
        snap = load_snapshot(name)
        status = snap["ttis"][0]["buffer_status"]
        entries = [e for lst in status.values() for e in lst]
        assert entries, f"{name}: buffer_status пуст"
        for e in entries:
            assert e["qci"] is not None, f"{name}: qci=None — не layered-буфер"
            assert e["hol_delay"] is not None, f"{name}: hol_delay=None"
