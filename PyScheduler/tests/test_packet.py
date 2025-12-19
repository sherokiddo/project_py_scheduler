import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from TRAFFIC_MODEL import Packet, TrafficType


def test_packet_creation():
    """Создание пакета с минимальными параметрами"""
    pkt = Packet(size=1000, ue_id=1, creation_time=100.0)
    assert pkt.size == 1000
    assert pkt.ue_id == 1
    assert pkt.creation_time == 100.0
    assert pkt.qci == 9  # default
    assert pkt.priority == 0  # default


def test_packet_with_traffic_type():
    """Пакет с типом трафика автоматически получает QCI и deadline"""
    pkt = Packet(size=500, ue_id=2, creation_time=1000.0, traffic_type=TrafficType.VOIP)
    assert pkt.qci == 9  # Не перезаписывается автоматически
    assert pkt.deadline == 1000.0 + 100  # VOIP delay budget = 100ms


def test_packet_to_dict():
    """Конвертация в dict"""
    pkt = Packet(size=800, ue_id=3, creation_time=500.0, qci=1)
    d = pkt.to_dict()
    assert isinstance(d, dict)
    assert d["size"] == 800
    assert d["ue_id"] == 3
    assert d["qci"] == 1


def test_packet_from_dict():
    """Создание из dict (legacy)"""
    data = {"size": 1200, "creation_time": 300.0, "priority": 1}
    pkt = Packet.from_dict(data, ue_id=5)
    assert pkt.size == 1200
    assert pkt.ue_id == 5
    assert pkt.creation_time == 300.0
    assert pkt.priority == 1
