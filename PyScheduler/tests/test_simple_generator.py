import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from TRAFFIC_MODEL import ITrafficGeneratorInterface, Packet, SimpleGenerator


class TestSimpleGenerator:
    def test_implements_interface(self):
        gen = SimpleGenerator()
        assert isinstance(gen, ITrafficGeneratorInterface)

    def test_set_and_generate(self):
        gen = SimpleGenerator(default_qci=5)
        gen.set_model(ue_id=1, model_type="Poisson", packet_rate=10)

        packets = gen.generate_packets(ue_id=1, current_time=1000, update_interval=100)

        assert isinstance(packets, list)
        for pkt in packets:
            assert isinstance(pkt, Packet)
            assert pkt.qci == 5  # default_qci

    def test_reset_ue_clears_state(self):
        gen = SimpleGenerator()
        gen.set_model(ue_id=1, model_type="OnOff", duration_on=2, duration_off=3, packet_rate=25)

        # Генерация создаст состояние
        gen.generate_packets(ue_id=1, current_time=1000, update_interval=100)

        model = gen.models[1]
        assert 1 in model._device_states

        # ✅ reset должен очистить
        gen.reset_ue(ue_id=1)

        # Модель удалена
        assert 1 not in gen.models
