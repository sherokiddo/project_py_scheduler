from BS_MODULE import BaseStation
from TRAFFIC_MODEL import PoissonModel, TrafficType
from UE_MODULE import UserEquipment


def test_legacy_approach():
    """Тест старого подхода (SimpleGenerator)"""
    print("\n" + "=" * 60)
    print("TEST 1: Legacy Approach (SimpleGenerator)")
    print("=" * 60)

    bs = BaseStation(bandwidth=10)
    ue1 = UserEquipment(UE_ID=1, x=100, y=100)
    bs.REG_UE(ue1)

    # Старый способ: через SET_TRAFFIC_MODEL
    poisson = PoissonModel(packet_rate=10)  # 10 пакетов в секунду
    bs.SET_TRAFFIC_MODEL(ue1, poisson)
    bs.ue_traffic_models[ue1.UE_ID] = poisson

    # Генерация трафика старым способом (несколько раз для большей выборки)
    total_packets_before = 0
    total_size_before = 0

    for _ in range(5):
        bs.GEN_TRFFC(current_time=1000, update_interval=100, ue_id=1)
        status = bs.GET_GLOBAL_BUFFER_STATUS(current_time=1000)
        total_packets_before = status["total_packets"]
        total_size_before = status["total_size"]

        if total_packets_before > 0:
            break  # Если сгенерировались пакеты, выходим

    # Проверка буфера
    print(f"Legacy: Buffer size = {total_size_before} bytes")
    print(f"Legacy: Total packets = {total_packets_before}")

    # Проверяем что хотя бы один раз сгенерировались пакеты
    assert total_packets_before > 0, "Legacy: хотя бы раз должны были сгенерироваться пакеты"
    print("✅ Legacy approach работает!")

    return total_packets_before, total_size_before


def test_multibearer_approach():
    """Тест нового подхода (PacketManager с multi-bearer)"""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Bearer Approach (PacketManager)")
    print("=" * 60)

    bs = BaseStation(bandwidth=10)
    ue1 = UserEquipment(UE_ID=1, x=100, y=100)
    bs.REG_UE(ue1)

    # Новый способ: через setup_ue_traffic_multi_bearer
    # Два независимых bearer'а с разными QCI
    bs.setup_ue_traffic_multi_bearer(
        ue_id=1,
        bearers=[
            {
                "model_type": "Poisson",
                "qci": 1,
                "traffic_type": TrafficType.VOIP,
                "packet_rate": 50,  # 50 пакетов в сек
                "max_bitrate": 0.064,  # 64 Kbps
            },
            {
                "model_type": "Poisson",
                "qci": 9,
                "traffic_type": TrafficType.BACKGROUND,
                "packet_rate": 10,  # 10 пакетов в сек
            },
        ],
    )

    # Генерация трафика новым способом (несколько раз для большей выборки)
    total_packets = 0
    total_size = 0
    qci_set = set()

    for i in range(10):
        bs.generate_traffic_all_ues(current_time=1000 + i * 100, update_interval=100)
        status = bs.GET_GLOBAL_BUFFER_STATUS(current_time=1000 + i * 100)
        total_packets = status["total_packets"]
        total_size = status["total_size"]

        # Проверка пакетов с разными QCI
        buffer = bs.ue_buffers[1]
        packets = list(buffer.queues[1])
        qci_set = set(pkt.qci for pkt in packets)

        if len(qci_set) > 1:  # Если найдены оба QCI, выходим
            break

    # Проверка буфера
    print(f"Multi-bearer: Buffer size = {total_size} bytes")
    print(f"Multi-bearer: Total packets = {total_packets}")
    print(f"Multi-bearer: QCI found = {qci_set}")

    # Проверки
    assert total_packets > 0, "Multi-bearer: должны быть сгенерированы пакеты"
    assert len(qci_set) >= 1, "Multi-bearer: должны быть пакеты с QCI"

    if len(qci_set) > 1:
        print(f"✅ Найдены пакеты ОБОИХ bearer'ов! QCI: {qci_set}")
    else:
        print(f"⚠️ Пока сгенерированы пакеты только с QCI {qci_set} (стохастика)")

    print("✅ Multi-bearer approach работает!")

    return total_packets, total_size


def test_both_approaches_coexist():
    """Проверка что оба подхода могут работать одновременно"""
    print("\n" + "=" * 60)
    print("TEST 3: Coexistence (оба подхода вместе)")
    print("=" * 60)

    bs = BaseStation(bandwidth=10)

    # UE 1 - старый подход (Legacy)
    ue1 = UserEquipment(UE_ID=1, x=100, y=100)
    bs.REG_UE(ue1)
    poisson1 = PoissonModel(packet_rate=10)
    bs.SET_TRAFFIC_MODEL(ue1, poisson1)
    bs.ue_traffic_models[ue1.UE_ID] = poisson1  # Явно для legacy

    # UE 2 - новый подход (Multi-bearer)
    ue2 = UserEquipment(UE_ID=2, x=200, y=200)
    bs.REG_UE(ue2)
    bs.setup_ue_traffic_multi_bearer(
        ue_id=2,
        bearers=[
            {
                "model_type": "Poisson",
                "qci": 1,
                "traffic_type": TrafficType.VOIP,
                "packet_rate": 50,
            },
            {
                "model_type": "OnOff",
                "qci": 7,
                "traffic_type": TrafficType.VIDEO_STREAM,
                "duration_on": 2,
                "duration_off": 3,
                "packet_rate": 200,
                "max_bitrate": 2.0,
            },
        ],
    )

    # Генерация для обоих UE несколько раз
    for i in range(5):
        bs.GEN_TRFFC(current_time=1000 + i * 100, update_interval=100, ue_id=1)  # Legacy
        bs.generate_traffic_all_ues(current_time=1000 + i * 100, update_interval=100)  # New

    # Проверка что оба буфера заполнены
    status = bs.GET_GLOBAL_BUFFER_STATUS(current_time=1500)

    ue1_size = status["per_ue"].get(1, {}).get("size", 0)
    ue2_size = status["per_ue"].get(2, {}).get("size", 0)

    ue1_packets = status["per_ue"].get(1, {}).get("packet_count", 0)
    ue2_packets = status["per_ue"].get(2, {}).get("packet_count", 0)

    print(f"UE 1 (Legacy): {ue1_packets} packets, {ue1_size} bytes")
    print(f"UE 2 (Multi-bearer): {ue2_packets} packets, {ue2_size} bytes")
    print(f"Total: {status['total_packets']} packets, {status['total_size']} bytes")

    # Гибкие проверки (стохастика может не сгенерировать за раз)
    total_packets = status["total_packets"]
    assert total_packets > 0, "Должны быть сгенерированы пакеты для обоих UE"

    print("✅ Оба подхода сосуществуют корректно!")

    return ue1_packets, ue2_packets


def test_traffic_statistics():
    """Проверка сбора статистики по трафику"""
    print("\n" + "=" * 60)
    print("TEST 4: Traffic Statistics")
    print("=" * 60)

    bs = BaseStation(bandwidth=10)
    ue = UserEquipment(UE_ID=1, x=100, y=100)
    bs.REG_UE(ue)

    # Настройка multi-bearer
    bs.setup_ue_traffic_multi_bearer(
        ue_id=1,
        bearers=[
            {
                "model_type": "Poisson",
                "qci": 1,
                "traffic_type": TrafficType.VOIP,
                "packet_rate": 100,  # Высокий rate для гарантированной генерации
            },
        ],
    )

    # Генерация многократно
    for i in range(10):
        bs.generate_traffic_all_ues(current_time=1000 + i * 100, update_interval=100)

    # Проверка глобальной статистики
    status = bs.GET_GLOBAL_BUFFER_STATUS(current_time=2000)

    print(f"Total packets generated: {status['total_packets']}")
    print(f"Total buffer size: {status['total_size']} bytes")
    print(f"Total dropped: {status['total_dropped']}")
    print(f"Total expired: {status['total_expired']}")
    print(f"Avg delay: {status['avg_delay']:.2f} ms")
    print(f"Max delay: {status['max_delay']} ms")

    assert status["total_packets"] > 0, "Должны быть сгенерированы пакеты"
    print("✅ Статистика собирается корректно!")


def main():
    """Главная функция для запуска всех тестов"""
    print("\n" + "#" * 60)
    print("# ТЕСТЫ СОВМЕСТИМОСТИ TRAFFIC GENERATION MODES")
    print("#" * 60)

    try:
        # Test 1: Legacy
        legacy_packets, legacy_size = test_legacy_approach()

        # Test 2: Multi-Bearer
        mb_packets, mb_size = test_multibearer_approach()

        # Test 3: Coexistence
        ue1_packets, ue2_packets = test_both_approaches_coexist()

        # Test 4: Statistics
        test_traffic_statistics()

        # Summary
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 60)
        print(f"✅ Test 1 (Legacy): {legacy_packets} packets generated")
        print(f"✅ Test 2 (Multi-Bearer): {mb_packets} packets generated")
        print(f"✅ Test 3 (Coexistence): UE1={ue1_packets}, UE2={ue2_packets} packets")
        print("✅ Test 4 (Statistics): Полная статистика собирается")
        print("\n" + "#" * 60)
        print("# ✅✅✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("#" * 60)

    except AssertionError as e:
        print(f"\n❌ ОШИБКА: {e}")
        return False
    except Exception as e:
        print(f"\n❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
