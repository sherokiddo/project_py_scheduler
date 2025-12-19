import sys
from pathlib import Path

import pytest

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from TRAFFIC_MODEL import ITrafficModel, Packet


def test_interface_is_abstract():
    """ITrafficModel нельзя создать напрямую"""
    with pytest.raises(TypeError):
        # Должна быть ошибка: Can't instantiate abstract class
        model = ITrafficModel()


def test_interface_requires_methods():
    """Наследник должен реализовать все абстрактные методы"""

    # Неполная реализация
    class IncompleteModel(ITrafficModel):
        def generate_traffic(self, ue_id, current_time, update_interval):
            return []

        # Забыли get_model_name()!

    with pytest.raises(TypeError):
        model = IncompleteModel()


def test_interface_complete_implementation():
    """Полная реализация интерфейса"""

    class MockModel(ITrafficModel):
        def generate_traffic(self, ue_id, current_time, update_interval):
            return [Packet(size=1000, ue_id=ue_id, creation_time=current_time)]

        def get_model_name(self):
            return "Mock"

    # Должно создаться без ошибок
    model = MockModel()
    assert model.get_model_name() == "Mock"

    packets = model.generate_traffic(ue_id=1, current_time=100, update_interval=10)
    assert len(packets) == 1
    assert isinstance(packets[0], Packet)
