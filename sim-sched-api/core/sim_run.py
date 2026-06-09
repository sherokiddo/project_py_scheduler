import asyncio
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from SIMULATION_MANAGER import SimulationManager
from UE_MODULE import UECollection, UserEquipment
from BS_MODULE import BaseStation
from MOBILITY_MODEL import MapBorders

simulations: Dict[str, dict] = {}
EVENT_QUEUE_MAXLEN = 256
API_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = API_ROOT / "metrics"

""" 
режимы: 
basic - отправка полноценного json на фронтенд
batch - отправка батчами по 50 json  
"""
DELIVERY_MODE = "basic"

def mode_valid(mode: str) -> bool:
    mode = mode.strip().lower()
    return mode in {"basic", "batch"}

def append_event(run_id: str, event: Dict[str, Any]) -> None:
    """Положить событие в bounded queue и обновить latest snapshot."""
    sim_data = simulations.get(run_id)
    if not sim_data:
        return

    events = sim_data.get("events")
    if events is not None:
        events.append(event)

    sim_data["latest_event"] = event

async def send_progress(ws_clients: list, data: dict):
    """Отправка прогресса всем WebSocket клиентам."""
    payload = data if "type" in data else {"type": "progress", **data}
    dead_clients = []
    for ws in list(ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead_clients.append(ws)
    for ws in dead_clients:
        if ws in ws_clients:
            ws_clients.remove(ws)


async def send_completion(ws_clients: list, results: dict):
    """Отправка уведомления о завершении симуляции."""
    for ws in ws_clients:
        try:
            await ws.send_json({"type": "completed", **results})
        except Exception:
            pass


async def send_failure(ws_clients: list, run_id: str, error: str):
    """Отправка уведомления о завершении с ошибкой."""
    payload = {"type": "failed", "run_id": run_id, "error": error}
    for ws in list(ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            if ws in ws_clients:
                ws_clients.remove(ws)


def build_users_payload(snapshot: dict) -> list[dict]:
    ue_throughputs = snapshot.get("dl_ue_throughputs", {})
    users = []

    for ue_id, throughput_bps in ue_throughputs.items():
        users.append({
            "ue_id": int(ue_id),
            "throughput_kbps": round(float(throughput_bps) / 1000.0, 2),
        })

    return users


def build_resource_grid_payload(snapshot: dict) -> list[dict]:
    allocation = snapshot.get("resource_grid", [])
    resource_grid = []

    for item in allocation:
        resource_grid.append({
            "rb": int(item.get("rb", 0)),
            "ue_id": item.get("ue_id"),
        })

    return resource_grid


def build_tti_event(run_id: str, snapshot: dict) -> dict:
    return {
        "type": "tti",
        "run_id": run_id,
        "data": {
            "tti": int(snapshot.get("tti", 0)),
            "cell": {
                "throughput_kbps": snapshot.get("dl_throughput_sum_kbps", 0.0),
                "fairness_jain": snapshot.get("dl_fairness_jain_index", 0.0),
                "sinr_avg_db": snapshot.get("dl_sinr_avg", 0.0),
                "spectral_efficiency_bps_hz": snapshot.get(
                    "dl_spectral_efficiency_avg_ue",
                    0.0,
                ),
                "rb_utilization_pct": snapshot.get("dl_rb_utilization_pct", 0.0),
            },
            "users": build_users_payload(snapshot),
            "resource_grid": build_resource_grid_payload(snapshot),
        },
    }


def build_json_filename(prefix: str) -> str:
    clean_prefix = str(prefix).strip() or "sim_stats"
    if clean_prefix.lower().endswith(".json"):
        return clean_prefix
    return f"{clean_prefix}.json"


def build_positions_filename(stats_prefix: str) -> str:
    clean_prefix = str(stats_prefix).strip() or "sim_stats"
    if clean_prefix.lower().endswith(".json"):
        clean_prefix = clean_prefix[:-5]

    if clean_prefix.endswith("_stats"):
        clean_prefix = f"{clean_prefix[:-6]}_positions"
    else:
        clean_prefix = f"{clean_prefix}_positions"

    return f"{clean_prefix}.json"


def build_position_item(
        tti: int,
        base_station: BaseStation,
        ue_collection: UECollection,
        bounds: dict,
        coordinate_step: int,
) -> dict:
    users = []

    for ue in ue_collection.GET_ALL_USERS():
        coordinates = getattr(ue, "coordinates", [])
        if coordinates:
            coord_idx = min(coordinate_step, len(coordinates) - 1)
            x, y = coordinates[coord_idx]
        else:
            x, y = ue.position

        users.append({
            "ue_id": int(ue.UE_ID),
            "x": float(x),
            "y": float(y),
        })

    return {
        "tti": int(tti),
        "positions": {
            "base_station": {
                "x": float(base_station.position[0]),
                "y": float(base_station.position[1]),
            },
            "users": users,
            "bounds": {
                "x_min": float(bounds["x_min"]),
                "x_max": float(bounds["x_max"]),
                "y_min": float(bounds["y_min"]),
                "y_max": float(bounds["y_max"]),
            },
        },
    }


def build_positions_history(
        base_station: BaseStation,
        ue_collection: UECollection,
        bounds: dict,
        sim_duration: int,
        update_interval: int,
) -> list[dict]:
    safe_update_interval = max(int(update_interval), 1)

    return [
        build_position_item(
            tti=tti,
            base_station=base_station,
            ue_collection=ue_collection,
            bounds=bounds,
            coordinate_step=(tti // safe_update_interval) + 1,
        )
        for tti in range(sim_duration)
    ]


async def run_simulation(run_id: str, config: dict):
    """Запуск симуляции в background thread pool."""
    sim_data = simulations.setdefault(run_id, {})
    sim_data.setdefault("events", deque(maxlen=EVENT_QUEUE_MAXLEN))
    sim_data.setdefault("latest_event", None)
    sim_data.setdefault("ws_clients", [])
    sim_data.setdefault("stats_ws_clients", [])
    sim_data.setdefault("positions_ws_clients", [])
    sim_data.setdefault("latest_stats_event", None)
    sim_data.setdefault("latest_positions_event", None)

    delivery_mode = (DELIVERY_MODE or "").strip().lower()
    if not mode_valid(delivery_mode):
        delivery_mode = "basic"
    
    sim_data["delivery_mode"] = delivery_mode

    sim_manager = SimulationManager()

    sim_duration = config.get("sim_duration", 1000)
    sim_manager.set_sim_duration(sim_duration)

    update_interval = config.get("update_interval", 1)
    sim_manager.set_upd_interval(update_interval)
    realtime_batch_size = config.get("realtime_batch_size", 50)
    if not isinstance(realtime_batch_size, int) or realtime_batch_size <= 0:
        raise ValueError(
            f"realtime_batch_size должен быть положительным целым числом. "
            f"Получено: {realtime_batch_size} ({type(realtime_batch_size).__name__})"
        )
    if hasattr(sim_manager, "set_realtime_emit_interval"):
        sim_manager.set_realtime_emit_interval(config.get("realtime_emit_interval", 1))
    if hasattr(sim_manager, "set_realtime_payload_mode"):
        sim_manager.set_realtime_payload_mode(config.get("realtime_payload_mode", "light"))

    if config.get("verbose", False):
        sim_manager.enable_verbose_log(to_file=config.get("log_to_file", False))

    if config.get("stats_log", False) or delivery_mode == "basic":
        sim_manager.set_stats_manager(
            enabled=True,
            collect_interval=config.get("stats_collect_interval", 1),
            scheduler_level=config.get("stats_scheduler_level", "advanced"),
            amc_level=config.get("stats_amc_level", "advanced"),
            pdcch_level=config.get("stats_pdcch_level", "basic"),
        )
        sim_manager.stats_config.file_prefix = config.get("stats_file_prefix", "sim_stats")

    bs = BaseStation(
        x=config["bs_coords"]["x"],
        y=config["bs_coords"]["y"],
        bandwidth=config["bs_bw_mhz"],
        ch_model_type=config.get("bs_ch_type", "UMa"),
        enable_tdl=config.get("enable_tdl", True),
    )

    sim_manager.set_base_station(bs)

    ue_collection = UECollection()

    # Mobility-модели опираются на глобальные границы карты через singleton MapBorders.
    # В API задаем их из запроса и принудительно обновляем на каждый запуск, чтобы
    # параллельные/повторные run'ы не работали со старыми (или None) границами.
    map_borders = MapBorders(
        config["ue_coords"]["x_min"],
        config["ue_coords"]["x_max"],
        config["ue_coords"]["y_min"],
        config["ue_coords"]["y_max"],
    )
    map_borders.x_min = config["ue_coords"]["x_min"]
    map_borders.x_max = config["ue_coords"]["x_max"]
    map_borders.y_min = config["ue_coords"]["y_min"]
    map_borders.y_max = config["ue_coords"]["y_max"]

    for ue_id in config["ue_ids"]:
        ue = UserEquipment(
            UE_ID=ue_id,
            x=config["ue_coords"]["x_min"],
            y=config["ue_coords"]["y_min"],
        )
        ue_collection.ADD_USER(ue)

    mobility_kwargs = {}
    if config.get("ue_pause") is not None:
        mobility_kwargs["pause_time"] = config["ue_pause"]
    if "bs" not in mobility_kwargs:
        mobility_kwargs["bs"] = bs

    ue_collection.SET_MOBILITY_MODEL(config["ue_move_pattern"], **mobility_kwargs)

    ue_collection.REG_USERS_TO_BS(bs)

    sim_manager.set_ue_collection(ue_collection)

    sim_manager.set_scheduler(algorithm=config["bs_scheduler"])

    traffic_pattern_map = {
        "PoissonModel": "Poisson",
        "OnOffModel": "OnOff",
        "MMPPModel": "MMPP",
    }
    traffic_model_type = traffic_pattern_map.get(config["ue_traffic_pattern"])
    if traffic_model_type is None:
        raise ValueError(
            f"Неподдерживаемая модель трафика: {config['ue_traffic_pattern']}. "
            f"Доступные: PoissonModel, OnOffModel, MMPPModel"
        )

    packet_rate = config.get("sim_packet_rate", 1000)
    traffic_params = {"packet_rate": packet_rate}
    if traffic_model_type == "OnOff":
        traffic_params.update({
            "duration_on": config.get("traffic_duration_on", 1.0),
            "duration_off": config.get("traffic_duration_off", 1.0),
        })
    elif traffic_model_type == "MMPP":
        traffic_params = {
            "packet_rates": config.get("traffic_packet_rates", [packet_rate, packet_rate * 2]),
        }

    for ue_id in config["ue_ids"]:
        sim_manager.setup_ue_traffic(
            ue_id=ue_id,
            model_type=traffic_model_type,
            **traffic_params,
        )

    # Запускаем симуляцию в отдельном потоке (blocking call)
    loop = asyncio.get_running_loop()
    batch_lock = threading.Lock()
    batch_seq = 0
    pending_snapshots: List[Dict[str, Any]] = []

    def build_batch_event(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal batch_seq
        batch_seq += 1
        return {
            "type": "tti_batch",
            "run_id": run_id,
            "batch_seq": batch_seq,
            "count": len(items),
            "tti_from": items[0].get("tti"),
            "tti_to": items[-1].get("tti"),
            "items": items,
        }

    def pop_ready_batch(force: bool = False) -> Dict[str, Any] | None:
        nonlocal pending_snapshots
        with batch_lock:
            if not pending_snapshots:
                return None
            if not force and len(pending_snapshots) < realtime_batch_size:
                return None
            batch_items = pending_snapshots
            pending_snapshots = []
        return build_batch_event(batch_items)

    async def flush_pending_batch() -> None:
        if delivery_mode != "batch":
            return
        batch_event = pop_ready_batch(force=True)
        if not batch_event:
            return
        append_event(run_id, batch_event)
        ws_clients = simulations.get(run_id, {}).get("ws_clients", [])
        if ws_clients:
            await send_progress(ws_clients, batch_event)

    def on_progress_update(progress_data: dict):
        # Эта функция будет вызываться внутри sim_manager (в фоновом потоке).
        if delivery_mode == "basic":
            return

        if progress_data.get("type") != "tti_snapshot":
            append_event(run_id, progress_data)
            ws_clients = simulations.get(run_id, {}).get("ws_clients", [])
            if ws_clients:
                asyncio.run_coroutine_threadsafe(
                    send_progress(ws_clients, progress_data),
                    loop
                )
            return

        with batch_lock:
            pending_snapshots.append(progress_data)

        batch_event = pop_ready_batch(force=False)
        if not batch_event:
            return

        append_event(run_id, batch_event)
        ws_clients = simulations.get(run_id, {}).get("ws_clients", [])
        if ws_clients:
            asyncio.run_coroutine_threadsafe(
                send_progress(ws_clients, batch_event),
                loop
            )
    if hasattr(sim_manager, "set_progress_callback"):
        sim_manager.set_progress_callback(on_progress_update)

    simulations[run_id]["status"] = "running"
    simulations[run_id]["manager"] = sim_manager
    
    try:
        await loop.run_in_executor(None, sim_manager.start_simulation)
    except Exception as e:
        await flush_pending_batch()
        error_msg = str(e)
        simulations[run_id]["status"] = "failed"
        simulations[run_id]["error"] = error_msg
        append_event(run_id, {"type": "failed", "run_id": run_id, "error": error_msg})
        ws_clients = simulations.get(run_id, {}).get("ws_clients", [])
        if ws_clients:
            await send_failure(ws_clients, run_id, error_msg)
        loop.create_task(cleanup_simulation(run_id, delay_seconds=300))
        return

    # Симуляция завершена — собираем результаты
    await flush_pending_batch()

    stats_len = 0
    if sim_manager.stats_manager and sim_manager.stats_config.enabled:
        stats_len = len(sim_manager.stats_manager.history)

    ws_clients = simulations.get(run_id, {}).get("ws_clients", [])
    stats_ws_clients = simulations.get(run_id, {}).get("stats_ws_clients", [])
    positions_ws_clients = simulations.get(run_id, {}).get("positions_ws_clients", [])

    if delivery_mode == "basic":
        full_items = []
        if sim_manager.stats_manager and sim_manager.stats_config.enabled:
            raw_items = list(sim_manager.stats_manager.history)
            full_items = [
                build_tti_event(run_id, snapshot)
                for snapshot in raw_items
            ]

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        stats_prefix = str(config.get("stats_file_prefix", "sim_stats")).strip() or "sim_stats"
        metrics_filename = build_json_filename(stats_prefix)
        positions_filename = build_positions_filename(stats_prefix)
        metrics_path = METRICS_DIR / metrics_filename
        positions_path = METRICS_DIR / positions_filename

        positions_items = build_positions_history(
            base_station=sim_manager.base_station,
            ue_collection=sim_manager.ue_collection,
            bounds=config["ue_coords"],
            sim_duration=sim_duration,
            update_interval=update_interval,
        )

        full_json_event = {
            "type": "full_json",
            "run_id": run_id,
            "count": len(full_items),
            "metrics_file": f"metrics/{metrics_filename}",
            "positions_file": f"metrics/{positions_filename}",
            "items": full_items,
        }

        positions_json_event = {
            "type": "positions_history",
            "run_id": run_id,
            "count": len(positions_items),
            "positions_file": f"metrics/{positions_filename}",
            "items": positions_items,
        }

        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(full_json_event, metrics_file, ensure_ascii=False, indent=2)

        with positions_path.open("w", encoding="utf-8") as positions_file:
            json.dump(positions_json_event, positions_file, ensure_ascii=False, indent=2)

        simulations[run_id]["metrics_file"] = f"metrics/{metrics_filename}"
        simulations[run_id]["positions_file"] = f"metrics/{positions_filename}"
        simulations[run_id]["latest_stats_event"] = full_json_event
        simulations[run_id]["latest_positions_event"] = positions_json_event

        append_event(run_id, full_json_event)
        if ws_clients:
            await send_progress(ws_clients, full_json_event)
        if stats_ws_clients:
            await send_progress(stats_ws_clients, full_json_event)
        if positions_ws_clients:
            await send_progress(positions_ws_clients, positions_json_event)

    simulations[run_id]["status"] = "completed"
    simulations[run_id]["stats_history_len"] = stats_len
    simulations[run_id]["tti_total"] = sim_duration
    completed_event = {
        "type": "completed",
        "run_id": run_id,
        "tti_total": sim_duration,
        "stats_history_len": stats_len,
        "delivery_mode": delivery_mode,
        "metrics_file": simulations[run_id].get("metrics_file"),
        "positions_file": simulations[run_id].get("positions_file"),
    }
    append_event(run_id, completed_event)
    if ws_clients:
        await send_completion(
            ws_clients,
            {
                "run_id": run_id,
                "tti_total": sim_duration,
                "stats_history_len": stats_len,
                "delivery_mode": delivery_mode,
            },
        )
    if stats_ws_clients:
        await send_completion(stats_ws_clients, completed_event)
    if positions_ws_clients:
        await send_completion(positions_ws_clients, completed_event)

    loop = asyncio.get_running_loop()
    loop.create_task(cleanup_simulation(run_id, delay_seconds=300))

async def cleanup_simulation(run_id: str, delay_seconds: int = 300):
    """Удаляет данные симуляции через 5 минут после завершения, чтобы не забивать ОЗУ."""
    await asyncio.sleep(delay_seconds)
    if run_id in simulations:
        # Закрываем оставшиеся соединения перед удалением
        for ws in simulations[run_id].get("ws_clients", []):
            try:
                await ws.close()
            except:
                pass
        del simulations[run_id]
        print(f"Симуляция {run_id} удалена из кэша.")
