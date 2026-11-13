"""Run unchanged numerical expectations before and after AMC extraction."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pytest_addoption(parser):
    parser.addoption('--amc-target', choices=('legacy', 'extracted'), default='legacy')


@pytest.fixture(scope='session')
def target(request):
    return request.config.getoption('--amc-target')


@pytest.fixture
def amc_factory(target):
    if target == 'legacy':
        from SCHEDULER import AdaptiveModulationAndCoding
        return lambda symbols=2: AdaptiveModulationAndCoding(
            SimpleNamespace(pdcch_manager=SimpleNamespace(pcfich=symbols)))
    module = importlib.import_module('AMC')  # Missing implementation must FAIL.
    return lambda symbols=2: module.AdaptiveModulationAndCoding(
        config=module.AmcConfig(control_symbols=symbols))


@pytest.fixture
def tb_class():
    # The old import remains supported after extraction.
    from SCHEDULER import TransportBlock
    return TransportBlock


@pytest.fixture
def tbs(target, tb_class):
    if target == 'legacy':
        return lambda mcs, rb: tb_class(1, 0, mcs, rb).determination_of_size_TB()
    return importlib.import_module('AMC').get_tbs_bits


@pytest.fixture
def read_stats(target):
    if target == 'legacy':
        from SCHEDULER import AdaptiveModulationAndCoding
        return lambda scheduler: AdaptiveModulationAndCoding(scheduler).get_stats()
    from AMC import AdaptiveModulationAndCoding
    return lambda scheduler: AdaptiveModulationAndCoding(scheduler).get_stats()
