from math import hypot

from MOBILITY_MODEL import MapBorders
from RES_GRID import RES_GRID_LTE
from UE_MODULE import UserEquipment
from BS_MODULE import BaseStation
from TRAFFIC_MODEL import Packet
from SCHEDULER import SchedulerInterface


# from unittest.mock import MagicMock
# def make_test_bs(x=0.0, y=0.0, height=25.0):
#     bs = MagicMock()
#     bs.position = (x, y)
#     bs.height = height
#     return bs
# тут нужно уточнить, т.к. не обязательно весь объект тянуть  

def _in_bounds(position, borders) -> bool:
    x_min, x_max, y_min, y_max = borders
    x, y = float(position[0]), float(position[1])
    return x_min <= x <= x_max and y_min <= y <= y_max


def _point_to_json(position) -> dict:
    return {
        "x": float(position[0]),
        "y": float(position[1]),
    }


def _point_to_text(position) -> str:
    point = _point_to_json(position)
    return f"({point['x']:.3f}, {point['y']:.3f})"


def _get_ue_buffer_size(bs: BaseStation, ue_id: int) -> int:
    statuses = bs.buffer_manager.get_buffer_status(ue_id)
    return int(sum(status.buffer_size for status in statuses))


def _allocation_sizes(allocation: dict, ue_ids: list[int]) -> dict[int, int]:
    return {
        int(ue_id): int(len(allocation.get(ue_id, [])))
        for ue_id in ue_ids
    }


def run_mobility_test() -> dict:
    """
        Тест mobility-модуля для вывода результата на фронтенд.

        Здесь проверяется базовая работоспособность движения UE.
        Мы не запускаем всю симуляцию и не трогаем scheduler, traffic или buffer.
        Проверяется только один маленький участок системы: может ли пользовательское
        устройство получить модель движения и корректно обновлять свои координаты.

        Логика теста простая:
        создается UE, задаются границы карты, создается mock/base station,
        UE получает модель RandomWalk, после чего несколько раз вызывается
        UPD_POSITION(). После каждого шага мы смотрим, что координаты остались
        внутри карты и что состояние UE обновилось корректно.

        Этот тест нужен как быстрый API-check для UI:
        фронт нажимает кнопку, backend выполняет маленький сценарий и возвращает
        понятный JSON с результатами проверок.
    """
    test_name = "mobility"
    steps = 10
    update_interval_ms = 500
    borders = (-500.0, 500.0, -500.0, 500.0)

    checks = []

    try:
        MapBorders.reset()
        MapBorders(*borders)

        bs = BaseStation(
            x=0.0,
            y=0.0,
            height=25.0,
            bandwidth=10,
            ch_model_type=None,
            enable_tdl=False,
        )

        ue = UserEquipment(
            UE_ID=1,
            x=100.0,
            y=100.0,
            ue_class="pedestrian",
        )

        bs.REG_UE(ue)

        position_before = ue.position
        coordinates_len_before = len(ue.coordinates)

        ue.SET_MOBILITY_MODEL("RandomWalk")

        positions = []

        for _ in range(steps):
            ue.UPD_POSITION(update_interval_ms)
            positions.append(ue.position)

        position_after = ue.position
        coordinates_len_after = len(ue.coordinates)

        expected_dist = float(hypot(
            float(position_after[0]) - float(bs.position[0]),
            float(position_after[1]) - float(bs.position[1]),
        ))
        actual_dist = float(ue.dist_to_BS_2D)

        position_changed = any(
            _point_to_json(pos) != _point_to_json(position_before)
            for pos in positions
        )
        all_positions_in_bounds = all(_in_bounds(pos, borders) for pos in positions)

        checks.append({
            "name": "UE зарегистрирован на базовой станции",
            "passed": bool(ue.serving_bs is bs),
            "details": "REG_UE должен проставить ue.serving_bs",
        })

        checks.append({
            "name": "UE получил mobility model",
            "passed": bool(ue.mobility_model is not None),
            "details": ue.mobility_model.__class__.__name__,
        })

        checks.append({
            "name": "История координат увеличилась",
            "passed": bool(coordinates_len_after == coordinates_len_before + steps),
            "details": f"{coordinates_len_before} -> {coordinates_len_after}",
        })

        checks.append({
            "name": "UE изменил позицию",
            "passed": bool(position_changed),
            "details": f"{_point_to_text(position_before)} -> {_point_to_text(position_after)}",
        })

        checks.append({
            "name": "Все координаты остались внутри карты",
            "passed": bool(all_positions_in_bounds),
            "details": f"borders={borders}",
        })

        checks.append({
            "name": "Расстояние до BS пересчиталось",
            "passed": bool(abs(actual_dist - expected_dist) < 1e-6),
            "details": f"dist_to_BS_2D={actual_dist:.3f}",
        })

        passed = all(check["passed"] for check in checks)

        return {
            "test_name": test_name,
            "title": "Mobility module test",
            "passed": passed,
            "summary": {
                "users_count": 1,
                "steps": steps,
                "update_interval_ms": update_interval_ms,
                "model": "RandomWalk",
                "position_before": _point_to_json(position_before),
                "position_after": _point_to_json(position_after),
                "distance_to_bs_2d": actual_dist,
            },
            "checks": checks,
            "output": "Mobility test completed",
        }

    except Exception as exc:
        return {
            "test_name": test_name,
            "title": "Mobility module test",
            "passed": False,
            "summary": {
                "users_count": 1,
                "steps": steps,
                "model": "RandomWalk",
            },
            "checks": checks + [{
                "name": "Тест выполнился без исключений",
                "passed": False,
                "details": f"{type(exc).__name__}: {exc}",
            }],
            "output": "Mobility test failed",
        }

def run_buffer_test(scheduler_type: str = "RoundRobin") -> dict:
    """
        Тест связки buffer manager + scheduler для вывода результата на фронтенд.

        Мы не запускаем всю симуляцию целиком. Сценарий специально маленький:
        создается ресурсная сетка, базовая станция, три UE и буферы на стороне BS.
        В буферы UE1 и UE2 кладутся пакеты, а UE3 остается с пустым буфером.
        После этого выбранный scheduler делает один шаг планирования.

        Что проверяем:
        - выбранная модель планировщика реально создается через SchedulerInterface;
        - scheduler видит данные в buffer_manager;
        - UE с пустым буфером не получает ресурсные блоки;
        - после планирования буферы не растут сами по себе;
        - хотя бы один непустой буфер уменьшается, то есть данные реально извлекаются.
    """
    test_name = "buffer"
    tti = 0
    bandwidth_mhz = 5
    packet_sizes = {
        1: 5000,
        2: 2000,
        3: 0,
    }
    cqi_by_ue = {
        1: 12,
        2: 7,
        3: 5,
    }
    checks = []

    try:
        available_schedulers = SchedulerInterface.available_algorithms()
        scheduler_is_known = scheduler_type in available_schedulers

        checks.append({
            "name": "Переданная модель планировщика существует",
            "passed": bool(scheduler_is_known),
            "details": f"scheduler={scheduler_type}",
        })

        if not scheduler_is_known:
            return {
                "test_name": test_name,
                "title": "Buffer + scheduler test",
                "passed": False,
                "summary": {
                    "scheduler": scheduler_type,
                    "available_schedulers": available_schedulers,
                },
                "users": [],
                "checks": checks,
                "output": "Unknown scheduler type",
            }

        lte_grid = RES_GRID_LTE(bandwidth=bandwidth_mhz, num_frames=1)
        bs = BaseStation(
            x=0.0,
            y=0.0,
            height=25.0,
            bandwidth=bandwidth_mhz,
            ch_model_type=None,
            enable_tdl=False,
        )
        scheduler = SchedulerInterface.create(
            scheduler_type,
            lte_grid,
            bs,
            enable_window=False,
        )

        users = [
            UserEquipment(UE_ID=1, x=100.0, y=100.0),
            UserEquipment(UE_ID=2, x=250.0, y=250.0),
            UserEquipment(UE_ID=3, x=400.0, y=400.0),
        ]

        for ue in users:
            bs.REG_UE(ue)
            bs.buffer_manager.create_ue_buffer(ue.UE_ID)

        for ue_id, packet_size in packet_sizes.items():
            if packet_size <= 0:
                continue

            bs.buffer_manager.add_packet(
                ue_id,
                Packet(
                    size=packet_size,
                    ue_id=ue_id,
                    creation_time=tti,
                ),
            )

        buffer_before = {
            ue.UE_ID: _get_ue_buffer_size(bs, ue.UE_ID)
            for ue in users
        }

        scheduler_users = []
        for ue in users:
            cqi = cqi_by_ue[ue.UE_ID]
            ue.cqi = cqi
            scheduler_users.append({
                "UE_ID": ue.UE_ID,
                "cqi": cqi,
                "ue": ue,
            })

        scheduler_result = scheduler.schedule(tti, scheduler_users)
        allocation = scheduler_result.get("allocation", {})
        ue_ids = [ue.UE_ID for ue in users]
        allocated_rbs = _allocation_sizes(allocation, ue_ids)

        buffer_after = {
            ue.UE_ID: _get_ue_buffer_size(bs, ue.UE_ID)
            for ue in users
        }

        transmitted_bits = {
            ue_id: int(scheduler.last_ue_transmitted_bits.get(ue_id, 0))
            for ue_id in ue_ids
        }

        total_rbs_allocated = int(sum(allocated_rbs.values()))
        non_empty_ue_ids = [
            ue_id
            for ue_id, size in buffer_before.items()
            if size > 0
        ]

        checks.extend([
            {
                "name": "Scheduler выделил хотя бы один RB",
                "passed": bool(total_rbs_allocated > 0),
                "details": f"allocated_rbs={total_rbs_allocated}",
            },
            {
                "name": "UE с пустым буфером не получил RB",
                "passed": bool(allocated_rbs[3] == 0 and transmitted_bits[3] == 0),
                "details": f"UE3 allocated_rbs={allocated_rbs[3]}, transmitted_bits={transmitted_bits[3]}",
            },
            {
                "name": "Буферы после scheduler не выросли",
                "passed": bool(all(buffer_after[ue_id] <= buffer_before[ue_id] for ue_id in ue_ids)),
                "details": f"before={buffer_before}, after={buffer_after}",
            },
            {
                "name": "Хотя бы один непустой буфер уменьшился",
                "passed": bool(any(buffer_after[ue_id] < buffer_before[ue_id] for ue_id in non_empty_ue_ids)),
                "details": f"non_empty_ue_ids={non_empty_ue_ids}",
            },
            {
                "name": "Передача данных была зафиксирована",
                "passed": bool(any(transmitted_bits[ue_id] > 0 for ue_id in non_empty_ue_ids)),
                "details": f"transmitted_bits={transmitted_bits}",
            },
        ])

        users_result = []
        for ue in users:
            ue_id = ue.UE_ID
            users_result.append({
                "ue_id": int(ue_id),
                "cqi": int(cqi_by_ue[ue_id]),
                "buffer_before_bytes": int(buffer_before[ue_id]),
                "allocated_rbs": int(allocated_rbs[ue_id]),
                "transmitted_bits": int(transmitted_bits[ue_id]),
                "buffer_after_bytes": int(buffer_after[ue_id]),
                "buffer_delta_bytes": int(buffer_before[ue_id] - buffer_after[ue_id]),
            })

        passed = all(check["passed"] for check in checks)

        return {
            "test_name": test_name,
            "title": "Buffer + scheduler test",
            "passed": bool(passed),
            "summary": {
                "scheduler": scheduler_type,
                "tti": tti,
                "bandwidth_mhz": bandwidth_mhz,
                "users_count": len(users),
                "total_rbs_allocated": total_rbs_allocated,
                "buffer_before_total_bytes": int(sum(buffer_before.values())),
                "buffer_after_total_bytes": int(sum(buffer_after.values())),
            },
            "users": users_result,
            "checks": checks,
            "output": "Buffer test completed",
        }

    except Exception as exc:
        return {
            "test_name": test_name,
            "title": "Buffer + scheduler test",
            "passed": False,
            "summary": {
                "scheduler": scheduler_type,
            },
            "users": [],
            "checks": checks + [{
                "name": "Тест выполнился без исключений",
                "passed": False,
                "details": f"{type(exc).__name__}: {exc}",
            }],
            "output": "Buffer test failed",
        }
