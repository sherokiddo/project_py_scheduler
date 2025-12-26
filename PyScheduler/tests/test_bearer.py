import sys
from pathlib import Path

import pytest

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from TRAFFIC_MODEL import Bearer, PoissonModel, TrafficType


def test_bearer_creation():
    model = PoissonModel(packet_rate=10)
    bearer = Bearer(bearer_id=1, model=model, qci=1, traffic_type=TrafficType.VOIP)
    assert bearer.bearer_id == 1
    assert bearer.enabled is True
    assert bearer.weight == 1.0


def test_bearer_validation_qci():
    model = PoissonModel(packet_rate=10)
    with pytest.raises(ValueError):
        Bearer(bearer_id=1, model=model, qci=10, traffic_type=TrafficType.VOIP)
