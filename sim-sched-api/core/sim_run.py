import asyncio
import threading
from collections import deque
from typing import Any, Dict, List

from SIMULATION_MANAGER import SimulationManager
from UE_MODULE import UECollection, UserEquipment
from BS_MODULE import BaseStation
from TRAFFIC_MODEL import PoissonModel, OnOffModel, MMPPModel
from MOBILITY_MODEL import MapBorders

simulations: Dict[str, dict] = {}
EVENT_QUEUE_MAXLEN = 256


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


async def run_simulation(run_id: str, config: dict):
    """Запуск симуляции в background thread pool."""
    sim_data = simulations.setdefault(run_id, {})
    sim_data.setdefault("events", deque(maxlen=EVENT_QUEUE_MAXLEN))
    sim_data.setdefault("latest_event", None)
    sim_data.setdefault("ws_clients", [])

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

    if config.get("stats_log", False):
        sim_manager.set_stats_manager(enabled=True)
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

    packet_rate = config.get("sim_packet_rate", 1000)
    traffic_model = None

    if config["ue_traffic_pattern"] == "PoissonModel":
        traffic_model = PoissonModel(packet_rate=packet_rate)
    elif config["ue_traffic_pattern"] == "OnOffModel":
        traffic_model = OnOffModel(packet_rate=packet_rate)
    elif config["ue_traffic_pattern"] == "MMPPModel":
        traffic_model = MMPPModel(packet_rate=packet_rate)
    else:
        raise ValueError(
            f"Неподдерживаемая модель трафика: {config['ue_traffic_pattern']}. "
            f"Доступные: PoissonModel, OnOffModel, MMPPModel"
        )

    ue_collection.SET_TRAFFIC_MODEL(traffic_model)

    ue_collection.REG_USERS_TO_BS(bs)

    sim_manager.set_ue_collection(ue_collection)

    sim_manager.set_scheduler(algorithm=config["bs_scheduler"])

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
        batch_event = pop_ready_batch(force=True)
        if not batch_event:
            return
        append_event(run_id, batch_event)
        ws_clients = simulations.get(run_id, {}).get("ws_clients", [])
        if ws_clients:
            await send_progress(ws_clients, batch_event)

    def on_progress_update(progress_data: dict):
        # Эта функция будет вызываться внутри sim_manager (в фоновом потоке).
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

    simulations[run_id]["status"] = "completed"
    simulations[run_id]["stats_history_len"] = stats_len
    simulations[run_id]["tti_total"] = sim_duration
    append_event(
        run_id,
        {
            "type": "completed",
            "run_id": run_id,
            "tti_total": sim_duration,
            "stats_history_len": stats_len,
        },
    )

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
