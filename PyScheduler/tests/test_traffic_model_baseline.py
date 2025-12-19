"""
Baseline тесты для ТЕКУЩЕЙ версии TRAFFIC_MODEL.py
Эти тесты должны проходить ДО и ПОСЛЕ рефакторинга!
"""

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from TRAFFIC_MODEL import MMPPModel, OnOffModel, PoissonModel


class TestPoissonModelBaseline:
    def test_generates_packets(self):
        model = PoissonModel(packet_rate=10, min_packet_size=150, max_packet_size=1500)
        # Текущая сигнатура (БЕЗ ue_id)
        packets = model.generate_traffic(current_time=1000, update_interval=100)
        assert isinstance(packets, list)
        # Сейчас возвращает List[Dict]
        if packets:
            assert isinstance(packets, list)
            assert "size" in packets[0]
            assert "creation_time" in packets[0]


class TestOnOffModelBaseline:
    def test_generates_packets(self):
        model = OnOffModel(
            duration_on=2.0,
            duration_off=3.0,
            packet_rate=25,
            min_packet_size=150,
            max_packet_size=1500,
        )
        # Текущая сигнатура (С ue_id)
        packets = model.generate_traffic(UE_ID=1, current_time=1000, update_interval=100)
        assert isinstance(packets, list)


class TestMMPPModelBaseline:
    def test_generates_packets(self):
        model = MMPPModel(packet_rates=[5, 20, 40], min_packet_size=150, max_packet_size=1500)
        packets = model.generate_traffic(UE_ID=1, current_time=1000, update_interval=100)
        assert isinstance(packets, list)
