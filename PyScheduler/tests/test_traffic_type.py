import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from TRAFFIC_MODEL import TrafficType


def test_traffic_type_enum():
    """Проверка что все типы существуют"""
    assert TrafficType.VOIP.value == "voip"
    assert TrafficType.WEB.value == "web"


def test_get_qci():
    """Проверка маппинга QCI"""
    assert TrafficType.VOIP.get_qci() == 1
    assert TrafficType.VIDEO_CALL.get_qci() == 2
    assert TrafficType.WEB.get_qci() == 9


def test_get_delay_budget():
    """Проверка delay budget"""
    assert TrafficType.VOIP.get_delay_budget() == 100
    assert TrafficType.GAMING.get_delay_budget() == 50
    assert TrafficType.WEB.get_delay_budget() == 300
