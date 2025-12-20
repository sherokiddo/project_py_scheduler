import sys
from pathlib import Path

import pytest

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from TRAFFIC_MODEL import ITrafficModel, MMPPModel, OnOffModel, PoissonModel, TrafficModelFactory


def test_factory_creates_poisson():
    model = TrafficModelFactory.create_model("Poisson", packet_rate=10)
    assert isinstance(model, PoissonModel)
    assert isinstance(model, ITrafficModel)


def test_factory_creates_onoff():
    model = TrafficModelFactory.create_model("OnOff", duration_on=2, duration_off=3, packet_rate=25)
    assert isinstance(model, OnOffModel)


def test_factory_creates_mmpp():
    model = TrafficModelFactory.create_model("MMPP", packet_rates=[5, 20, 40])
    assert isinstance(model, MMPPModel)


def test_factory_raises_on_unknown_model():
    with pytest.raises(ValueError) as exc:
        TrafficModelFactory.create_model("UnknownModel")

    assert "Unknown model type" in str(exc.value)
    assert "Available" in str(exc.value)


def test_get_available_models():
    models = TrafficModelFactory.get_available_models()
    assert "Poisson" in models
    assert "OnOff" in models
    assert "MMPP" in models
