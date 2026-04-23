from .              import router_v1
from .              import basic
from core.imports   import *
from SCHEDULER      import SchedulerInterface
from UE_MODULE      import UECollection, UserEquipment
from BS_MODULE      import BaseStation
from TRAFFIC_MODEL  import PoissonModel, OnOffModel, MMPPModel
import json
import uuid
from core.sim_run import run_simulation, simulations, send_completion

from typing import Optional, Any, Dict, List, Union
import csv
import asyncio
from collections import deque
from pydantic import BaseModel
from fastapi import WebSocket, WebSocketDisconnect

@router_v1.get("/test")
async def test_route():
    return {"message": "Test route"}

@router_v1.get("/all/scheduler_type", response_model=basic.BaseScheme_Response)
async def get_scheduler_type():
    algorithms = SchedulerInterface.available_algorithms()
    return {"status": "success", "code": 200, "data": {"algorithms": algorithms}}
# NO

class dot(BaseModel):
    x: float
    y: float

class coordinates_range(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float

class SimulationConfig(BaseModel):
    ue_cnt:             int                   # кол-во ue
    ue_ids:             List[int]             # id ue
    ue_move_pattern:    str                   # модель передвижения UE        см. available_ue_move_pattern
    ue_coords:          coordinates_range     # тут определённая структура    (x_min x_max),(y_min ymax)
    ue_traffic_pattern: str                   # Модель трафика                (id-key) см. available_ue_move_pattern
    ue_pause:           Optional[int] = None  
    bs_scheduler:       str                   # модель планировщика           (id-key) см. get_scheduler_type()
    bs_coords:          dot 
    bs_bw_mhz:          float     
    sim_packet_rate:    int             
    sim_duration:       int = 1000          # длительность в TTI
    update_interval:    int = 1             # интервал обновления
    verbose:           bool = False         # подробный вывод
    stats_log:         bool = False         # сохранение статистики
    bs_ch_type:        str = "UMa"          # тип канала
    enable_tdl:        bool = True          # TDL модель
    log_to_file:       bool = False         # логи в файл
    stats_file_prefix: str = "sim_stats"    # префикс файлов статистики
    realtime_emit_interval: int = 1         # частота realtime-событий
    realtime_payload_mode: str = "light"    # light/full payload
    realtime_batch_size: int = 50           # размер batched-пакета для WS


available_ue_traffic_pattern = ["PoissonModel", "OnOffModel", "MMPPModel"]
""" 
Модели трафика UE:
- PoissonModel:    пуассоновский поток пакетов (экспоненциальные интервалы)
- OnOffModel:      периодическая передача (вкл/выкл)
- MMPPModel:       модулированный пуассоновский процесс (burst-трафик)
"""

available_ue_move_pattern = ["RandomWalk", "RandomWaypoint", "RandomDirection", "GaussMarkov", "DiagonalWalk"]
"""
Модели перемещения UE:
- RandomWalk:        случайное блуждание (шаг + случайное направление)
- RandomWaypoint:    случайные цели и паузы между перемещениями
- RandomDirection:   движение до границы + смена направления
"""

available_bs_ch_types = ["UMa", "UMi", "RMa"]
""" 
Типы каналов между BS и UE (3GPP TR 38.901):
- UMa: Urban Macro    — выше крыш (30-50 м), плотная застройка
- UMi: Urban Micro    — ниже крыш (10-20 м), уличные сценарии  
- RMa: Rural Macro    — сельская местность, большие расстояния (~500-1500 м)
"""

@router_v1.post("/sim/run", response_model=basic.BaseScheme_Response)
async def sim_run(param: SimulationConfig):

    """
    Запуск симуляции
    Получает данные от фронта, валидирует, преобразует и передает в sim_run
    """

    if (param.ue_cnt <= 0):
        return {"status": "error", "code": 400, "data": {"message": "invalid ue_cnt param"}}
    
    if (len(param.ue_ids) != param.ue_cnt):
        return {"status": "error", "code": 400, "data": {"message": "len(param.ue_ids) not equal ue_cnt param"}}

    if param.ue_move_pattern not in available_ue_move_pattern:
        return {"status": "error", "code": 400, "data": {"message": f"invalid ue_move_pattern. available: {available_ue_move_pattern}"}}
    
    if param.ue_traffic_pattern not in available_ue_traffic_pattern:
        return {"status": "error", "code": 400, "data": {"message": f"invalid ue_traffic_pattern. available: {available_ue_traffic_pattern}"}}

    available_bs_scheduler_algorithms = SchedulerInterface.available_algorithms()

    if param.bs_scheduler not in available_bs_scheduler_algorithms:
        return {"status": "error", "code": 400, "data": {"message": f"invalid bs_scheduler. available: {available_bs_scheduler_algorithms}"}}

    if param.realtime_payload_mode not in ("light", "full"):
        return {"status": "error", "code": 400, "data": {"message": "invalid realtime_payload_mode. available: ['light', 'full']"}}

    if param.realtime_batch_size <= 0:
        return {"status": "error", "code": 400, "data": {"message": "invalid realtime_batch_size. must be > 0"}}
    
    run_id = str(uuid.uuid4())
    
    simulations[run_id] = {
        "status": "starting",
        "ws_clients": [],
        "events": deque(maxlen=256),
        "latest_event": None,
    }

    config_dict = {
        "sim_duration": param.sim_duration,
        "update_interval": param.update_interval,
        "verbose": param.verbose,
        "stats_log": param.stats_log,
        "log_to_file": param.log_to_file,
        "stats_file_prefix": param.stats_file_prefix,
        "realtime_emit_interval": param.realtime_emit_interval,
        "realtime_payload_mode": param.realtime_payload_mode,
        "realtime_batch_size": param.realtime_batch_size,
        
        # Параметры базовой станции
        "bs_coords": {
            "x": param.bs_coords.x,
            "y": param.bs_coords.y
        },
        "bs_bw_mhz": param.bs_bw_mhz,
        "bs_ch_type": param.bs_ch_type,
        "enable_tdl": param.enable_tdl,
        "bs_scheduler": param.bs_scheduler,
        
        # Параметры UE
        "ue_cnt": param.ue_cnt,
        "ue_ids": param.ue_ids,
        "ue_move_pattern": param.ue_move_pattern,
        "ue_coords": {
            "x_min": param.ue_coords.x_min,
            "x_max": param.ue_coords.x_max,
            "y_min": param.ue_coords.y_min,
            "y_max": param.ue_coords.y_max
        },
        "ue_traffic_pattern": param.ue_traffic_pattern,
        "ue_pause": param.ue_pause,
        "sim_packet_rate": param.sim_packet_rate,
    }

    asyncio.create_task(
        run_simulation(run_id, config_dict)
    )

    ws_url = f"/api/v1/ws/simulation/{run_id}"

    return {
        "status": "success",
        "code": 200,
        "data": {
            "run_id": run_id,
            "ws_url": ws_url,
            "message": "Симуляция запущена. Подключитесь к WebSocket для получения прогресса"
        }
    }

@router_v1.websocket("/ws/simulation/{run_id}")
async def simulation_websocket(websocket: WebSocket, run_id: str):
    """WebSocket для получения прогресса симуляции в реальном времени."""

    # Проверяем, существует ли симуляция
    if run_id not in simulations:
        await websocket.close(code=1008, reason=f"Simulation {run_id} not found")
        return

    await websocket.accept()

    # Добавляем клиента в список
    sim_data = simulations[run_id]
    ws_clients = sim_data["ws_clients"]
    ws_clients.append(websocket)

    try:
        latest_event = sim_data.get("latest_event")
        if latest_event is not None:
            await websocket.send_json(latest_event)

        status = sim_data.get("status")
        if status == "completed":
            await send_completion([websocket], {"run_id": run_id, "tti_total": sim_data.get("tti_total")})
            return
        if status == "failed":
            await websocket.send_json({
                "type": "failed",
                "run_id": run_id,
                "error": sim_data.get("error", "unknown error"),
            })
            return

        while sim_data.get("status") in ("starting", "running"):
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

        if sim_data.get("status") == "completed":
            await send_completion([websocket], {"run_id": run_id, "tti_total": sim_data.get("tti_total")})
        elif sim_data.get("status") == "failed":
            await websocket.send_json({
                "type": "failed",
                "run_id": run_id,
                "error": sim_data.get("error", "unknown error"),
            })

    except WebSocketDisconnect:
        pass

    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)
        try:
            await websocket.close()
        except:
            pass

@router_v1.websocket("/ws/csv_stream")
async def csv_stream_websocket(websocket: WebSocket):
    await websocket.accept()
    
    csv_path = "./handlers/emp_stats.csv"

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            
            headers = next(csv_reader, None)
            if headers:
                await websocket.send_json({
                    "type": "headers",
                    "data": headers
                })
                await asyncio.sleep(0.5)
            
            row_number = 1
            for row in csv_reader:
                await websocket.send_json({
                    "type": "row",
                    "row_number": row_number,
                    "data": row
                })
                await asyncio.sleep(0.5)
                row_number += 1
        
        await websocket.send_json({
            "type": "completed",
            "total_rows": row_number - 1
        })
        
    except FileNotFoundError:
        await websocket.send_json({
            "type": "error",
            "message": f"CSV file not found: {csv_path}"
        })
    except PermissionError:
        await websocket.send_json({
            "type": "error",
            "message": f"Permission denied: {csv_path}"
        })
    except csv.Error as e:
        await websocket.send_json({
            "type": "error",
            "message": f"CSV parsing error: {str(e)}"
        })
    except WebSocketDisconnect:
        print(f"Client disconnected from CSV stream: {csv_path}")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Unexpected error: {str(e)}"
        })
    finally:
        try:
            await websocket.close()
        except:
            pass

@router_v1.websocket("/ws/json_stream")
async def json_stream_websocket(websocket: WebSocket):
    await websocket.accept()
    
    json_path = "./handlers/data.json"
    
    try:
        with open(json_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
        
        # Определяем структуру JSON
        if isinstance(data, list):
            # Если JSON содержит массив объектов
            if len(data) > 0 and isinstance(data[0], dict):
                headers = list(data[0].keys())
                await websocket.send_json({
                    "type": "headers",
                    "data": headers
                })
                await asyncio.sleep(0.5)
            
            # Отправляем каждый элемент массива
            for index, item in enumerate(data, start=1):
                await websocket.send_json({
                    "type": "row",
                    "row_number": index,
                    "data": item if isinstance(item, (dict, list)) else [item]
                })
                await asyncio.sleep(0.5)
            
            total_items = len(data)
            
        elif isinstance(data, dict):
            # Если JSON содержит объект
            # Для объектов можно отправлять пары ключ-значение как строки
            headers = ["Key", "Value"]
            await websocket.send_json({
                "type": "headers",
                "data": headers
            })
            await asyncio.sleep(0.5)
            
            # Отправляем каждую пару ключ-значение как строку
            for index, (key, value) in enumerate(data.items(), start=1):
                await websocket.send_json({
                    "type": "row",
                    "row_number": index,
                    "data": [str(key), str(value)]
                })
                await asyncio.sleep(0.5)
            
            total_items = len(data)
        else:
            # Для простых типов данных
            await websocket.send_json({
                "type": "row",
                "row_number": 1,
                "data": [str(data)]
            })
            total_items = 1
        
        await websocket.send_json({
            "type": "completed",
            "total_rows": total_items
        })
        
    except FileNotFoundError:
        await websocket.send_json({
            "type": "error",
            "message": f"JSON file not found: {json_path}"
        })
    except PermissionError:
        await websocket.send_json({
            "type": "error",
            "message": f"Permission denied: {json_path}"
        })
    except json.JSONDecodeError as e:
        await websocket.send_json({
            "type": "error",
            "message": f"JSON parsing error: {str(e)}"
        })
    except WebSocketDisconnect:
        print(f"Client disconnected from JSON stream: {json_path}")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Unexpected error: {str(e)}"
        })
    finally:
        try:
            await websocket.close()
        except:
            pass
