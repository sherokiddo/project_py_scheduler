"""Explicit contract checks revised after curator review, 2026-11-01."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS

import pytest
import AMC
import TRANSPORT_BLOCK
import SCHEDULER


def test_legacy_imports_and_separate_tb():
    assert SCHEDULER.AdaptiveModulationAndCoding is AMC.AdaptiveModulationAndCoding
    assert SCHEDULER.TransportBlock is TRANSPORT_BLOCK.TransportBlock
    assert not hasattr(AMC, 'TransportBlock')
    assert callable(AMC.AdaptiveModulationAndCoding().get_stats)


def test_immutable_config_and_legacy_dynamic_configuration():
    config = AMC.AmcConfig(control_symbols=2)
    with pytest.raises(FrozenInstanceError):
        config.control_symbols = 3
    scheduler = NS(pdcch_manager=NS(pcfich=2))
    legacy = AMC.AdaptiveModulationAndCoding(scheduler)
    explicit = AMC.AdaptiveModulationAndCoding(scheduler, config=config)
    assert legacy.GET_BITS_PER_RB(10) == explicit.GET_BITS_PER_RB(10) == 530
    scheduler.pdcch_manager.pcfich = 3
    assert legacy.GET_BITS_PER_RB(10) == 483
    assert explicit.GET_BITS_PER_RB(10) == 530
    standalone = AMC.AdaptiveModulationAndCoding(config=AMC.AmcConfig(3))
    assert standalone.GET_BITS_PER_RB(10) == 483


def test_tb_delegates_to_calculator(monkeypatch):
    calls = []
    def calculate(mcs, rb):
        calls.append((mcs, rb))
        return 800
    monkeypatch.setattr(AMC, 'get_tbs_bits', calculate)
    block = TRANSPORT_BLOCK.TransportBlock(1, 0, 18, 10)
    assert block.determination_of_size_TB() == block.tbs == 800
    assert calls == [(18, 10)]


@pytest.mark.parametrize('cqi', range(16))
def test_default_policy_preserves_legacy(cqi):
    amc = AMC.AdaptiveModulationAndCoding()
    context = AMC.McsSelectionInput(cqi, (1, 15), (10, 11), ((10, 1), (11, 1)))
    assert amc.select_mcs(context) == amc.get_mcs_from_cqi(cqi)


def test_custom_policy_receives_allocation_without_changing_legacy_calls():
    received = []
    class Policy:
        def select_mcs(self, context):
            received.append(context)
            return 24
    context = AMC.McsSelectionInput(3, (2, 13), (7,), ((7, 1),))
    amc = AMC.AdaptiveModulationAndCoding(mcs_policy=Policy())
    assert amc.select_mcs(context) == 24
    assert received == [context]
    assert amc.get_mcs_from_cqi(3) == 4
    assert amc.GET_BITS_PER_RB(10) == 530


@pytest.mark.parametrize('result', [-1, 29, True, 2.5, None])
def test_invalid_policy_result_rejected(result):
    class Policy:
        def select_mcs(self, context):
            return result
    with pytest.raises(ValueError):
        AMC.AdaptiveModulationAndCoding(mcs_policy=Policy()).select_mcs(AMC.McsSelectionInput(10))
