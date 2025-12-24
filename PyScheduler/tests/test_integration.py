import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from TRAFFIC_MODEL import Packet, SimpleGenerator


def test_full_pipeline():
    """Полный pipeline: Factory → Model → Generator → Packets"""
    gen = SimpleGenerator(default_qci=7)
    gen.set_model(ue_id=1, model_type="Poisson", packet_rate=10)

    packets = gen.generate_packets(ue_id=1, current_time=1000, update_interval=100)

    assert all(isinstance(pkt, Packet) for pkt in packets)
    assert all(pkt.ue_id == 1 for pkt in packets)
    assert all(pkt.qci == 7 for pkt in packets)
