"""Characterization, not proof of LTE conformance. Baseline: 2026-10-30."""
import hashlib
import json
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('cqi,expected', list(enumerate(
    [0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])))
def test_cqi_to_mcs(amc_factory, cqi, expected):
    assert amc_factory().get_mcs_from_cqi(cqi) == expected


@pytest.mark.parametrize('cqi', [-1, 16])
def test_cqi_range(amc_factory, cqi):
    with pytest.raises(ValueError):
        amc_factory().get_mcs_from_cqi(cqi)


@pytest.mark.parametrize('cqi,expected', list(enumerate(
    [41, 63, 102, 163, 200, 260, 326, 371, 452, 530, 615, 695, 755, 777, 798], 1)))
def test_legacy_capacity(amc_factory, cqi, expected):
    assert amc_factory().GET_BITS_PER_RB(cqi) == expected


@pytest.mark.parametrize('symbols,ports,expected', [
    (1, 1, [44, 577, 868]), (1, 2, [42, 546, 821]), (1, 4, [37, 483, 727]),
    (2, 1, [41, 530, 798]), (2, 2, [38, 499, 751]), (2, 4, [34, 436, 657]),
    (3, 1, [37, 483, 727]), (3, 2, [35, 452, 680]), (3, 4, [30, 390, 586]),
])
def test_control_symbols_and_ports(amc_factory, symbols, ports, expected):
    assert [amc_factory(symbols).GET_BITS_PER_RB(c, ports) for c in (1, 10, 15)] == expected


@pytest.mark.parametrize('cqi', [-1, 0, 16])
def test_capacity_invalid_cqi(amc_factory, cqi):
    with pytest.raises(ValueError):
        amc_factory().GET_BITS_PER_RB(cqi)


@pytest.mark.parametrize('mcs,rb,expected', [
    (0, 1, 16), (0, 10, 256), (10, 10, 1544), (17, 10, 3112),
    (18, 10, 3240), (28, 10, 7480), (6, 1, 88), (6, 2, 176),
])
def test_tbs_controls(tbs, tb_class, mcs, rb, expected):
    assert tbs(mcs, rb) == expected
    block = tb_class(7, 0, mcs, rb)
    assert block.determination_of_size_TB() == block.tbs == expected


@pytest.mark.parametrize('field,value', [
    ('mcs', -1), ('mcs', 29), ('mcs', True), ('mcs', 1.0), ('mcs', '1'), ('mcs', None),
    ('rb', 0), ('rb', 111), ('rb', False), ('rb', 1.0), ('rb', '1'), ('rb', None),
])
def test_tbs_rejection_is_atomic(tbs, tb_class, field, value):
    args = {'mcs': 18, 'rb': 10, field: value}
    block = tb_class(7, 123, **args)
    with pytest.raises(ValueError):
        tbs(**args)
    with pytest.raises(ValueError):
        block.determination_of_size_TB()
    assert block.tbs == 123


def test_all_3190_tbs_values_match_frozen_baseline(tbs):
    values = [tbs(m, r) for m in range(29) for r in range(1, 111)]
    assert all(type(v) is int and v > 0 and v % 8 == 0 for v in values)
    assert values[6 * 110] == 88
    assert tbs(5, 4) == 328  # Do not correct the unrelated I_TBS=5, N_PRB=4 cell.
    # Restore ONLY the approved anomaly to compare all other cells with A0.
    values[6 * 110] = 328
    digest = hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
    assert digest == 'b6036b0ac52d9996d6a627ecd39b4147c358cd30de3edf757230b080152c9f6c'


def test_tb_recalculation_preserves_payload_and_harq(tb_class):
    block = tb_class(7, 0, 18, 10, buffer=[1, 2], ndi=0, rv=2, harq_id=3)
    assert block.determination_of_size_TB() == 3240
    assert block.determination_of_size_TB() == 3240
    block.mcs = 28
    assert block.determination_of_size_TB() == 7480
    assert (block.ue_id, block.buffer, block.ndi, block.rv, block.harq_id) == (7, [1, 2], 0, 2, 3)
    other = tb_class(8, 0, 0, 1)
    other.buffer.append(9)
    assert block.buffer == [1, 2]


def test_stats_values_and_query_time(read_stats, amc_factory):
    from SCHEDULER import SchedulerInterface
    scheduler = SchedulerInterface.__new__(SchedulerInterface)
    scheduler.pdcch_manager = NS(pcfich=2)
    scheduler.amc = amc_factory()
    ue1 = NS(current_dl_throughput=12000.0, last_transmitted_bits=240, SINR=10.0)
    ue2 = NS(current_dl_throughput=5000.0, last_transmitted_bits=80, SINR=4.0)
    scheduler._last_users = [{'UE_ID': 1, 'ue': ue1}, {'UE_ID': 2, 'ue': ue2}]
    scheduler._last_allocation = {1: [0, 1], 2: [2]}
    scheduler.cqi_map = {1: NS(wb_cqi=10, sb_cqi=[3, 15])}
    expected = {
        'dl_capacity_bits_sum_tti': 1060, 'dl_transmitted_bits_sum_tti': 320,
        'dl_throughput_sum_kbps': 17.0, 'dl_bits_per_rb_avg': 106.67,
        'dl_cqi_wb_avg_idx': 5.0, 'dl_sinr_avg': 7.0,
        'dl_ue_throughputs': {1: 12000.0, 2: 5000.0}, 'ue_cqi': {1: 10},
        'ue_cqi_sbb': {1: [3, 15]}, 'ue_sinr': {1: 10.0, 2: 4.0},
        'ue_rb_allocated': {1: 2, 2: 1},
    }
    first = read_stats(scheduler)
    assert first == expected
    first['ue_cqi_sbb'][1].append(7)
    assert scheduler.cqi_map[1].sb_cqi == [3, 15]
    ue1.last_transmitted_bits = 264
    assert read_stats(scheduler)['dl_transmitted_bits_sum_tti'] == 344


def test_empty_completed_stats(read_stats, amc_factory):
    from SCHEDULER import SchedulerInterface
    scheduler = SchedulerInterface.__new__(SchedulerInterface)
    scheduler.amc = amc_factory()
    scheduler._last_users, scheduler._last_allocation, scheduler.cqi_map = [], {}, {}
    assert read_stats(scheduler) == {
        'dl_capacity_bits_sum_tti': 0, 'dl_transmitted_bits_sum_tti': 0,
        'dl_throughput_sum_kbps': 0.0, 'dl_bits_per_rb_avg': 0.0,
        'dl_cqi_wb_avg_idx': 0, 'dl_sinr_avg': 0,
        'dl_ue_throughputs': {}, 'ue_cqi': {}, 'ue_cqi_sbb': {},
        'ue_sinr': {}, 'ue_rb_allocated': {},
    }
