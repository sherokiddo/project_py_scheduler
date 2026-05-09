from RES_GRID import RES_GRID_LTE
from BS_MODULE import BaseStation, Packet
from UE_MODULE import UserEquipment
from SCHEDULER import SchedulerInterface


def run_scheduler_with_buffer_test() -> dict:
    tti = 0

    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=1)

    bs = BaseStation(
        x=0,
        y=0,
        height=25.0,
        bandwidth=10,
        ch_model_type="UMi",
    )

    scheduler = SchedulerInterface.create(
        "RoundRobin",
        lte_grid,
        bs,
    )

    ue1 = UserEquipment(UE_ID=1, x=300, y=300)
    ue2 = UserEquipment(UE_ID=2, x=700, y=700)
    ue3 = UserEquipment(UE_ID=3, x=100, y=100)

    users = [ue1, ue2, ue3]

    for ue in users:
        bs.REG_UE(ue)

    bs.ue_buffers[1].ADD_PACKET(
        Packet(size=5000, ue_id=1, creation_time=0),
        current_time=0,
    )

    bs.ue_buffers[2].ADD_PACKET(
        Packet(size=2000, ue_id=2, creation_time=0),
        current_time=0,
    )

    buffer_before = {
        ue.UE_ID: bs.ue_buffers[ue.UE_ID].sizes[ue.UE_ID]
        for ue in users
    }

    ue1.cqi = 12
    ue2.cqi = 7
    ue3.cqi = 5

    scheduler_users = [
        {
            "UE_ID": ue.UE_ID,
            "buffer_size": buffer_before[ue.UE_ID],
            "cqi": ue.cqi,
            "ue": ue,
        }
        for ue in users
    ]

    scheduler.schedule(tti, scheduler_users)

    buffer_after = {
        ue.UE_ID: bs.ue_buffers[ue.UE_ID].sizes[ue.UE_ID]
        for ue in users
    }

    allocations = collect_grid_allocations(lte_grid, tti)

    result = {
        "test_name": "scheduler_with_buffer",
        "passed": True,
        "summary": {
            "scheduler": scheduler.__class__.__name__,
            "tti": tti,
            "users_count": len(users),
        },
        "users": [],
        "checks": [],
    }

    for ue in users:
        ue_id = ue.UE_ID
        result["users"].append({
            "ue_id": ue_id,
            "cqi": ue.cqi,
            "buffer_before": buffer_before[ue_id],
            "allocated_rbs": allocations.get(ue_id, 0),
            "buffer_after": buffer_after[ue_id],
        })

    result["checks"] = [
        {
            "name": "UE с пустым буфером не получил RB",
            "passed": allocations.get(3, 0) == 0,
        },
        {
            "name": "UE1 buffer не вырос после scheduler",
            "passed": buffer_after[1] <= buffer_before[1],
        },
        {
            "name": "UE2 buffer не вырос после scheduler",
            "passed": buffer_after[2] <= buffer_before[2],
        },
    ]

    result["passed"] = all(check["passed"] for check in result["checks"])
    result["output"] = build_output(result)

    return result

def collect_grid_allocations(lte_grid, tti: int) -> dict:
    subframe = lte_grid.GET_SUBFRAME(tti)
    allocations = {}

    for slot in subframe.slots:
        for rb in slot.GET_ALL_RES_BLCK():
            ue_id = rb.UE_ID

            if ue_id is None:
                continue

            allocations[ue_id] = allocations.get(ue_id, 0) + 1

    return allocations

def build_output(result: dict) -> str:
    lines = []

    lines.append("=== Тест планировщика с буфером ===")
    lines.append(f"Статус: {'PASSED' if result['passed'] else 'FAILED'}")
    lines.append(f"Планировщик: {result['summary']['scheduler']}")
    lines.append("")

    lines.append("Пользователи:")
    for user in result["users"]:
        lines.append(
            f"UE{user['ue_id']}: "
            f"CQI={user['cqi']}, "
            f"buffer_before={user['buffer_before']} B, "
            f"allocated_rbs={user['allocated_rbs']}, "
            f"buffer_after={user['buffer_after']} B"
        )

    lines.append("")
    lines.append("Проверки:")
    for check in result["checks"]:
        status = "OK" if check["passed"] else "FAIL"
        lines.append(f"[{status}] {check['name']}")

    return "\n".join(lines)

