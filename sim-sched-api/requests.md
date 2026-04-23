# API Requests Examples

## 1. GET — Главная

```
GET http://localhost:8000/
```

**Response:**
```json
{
  "message": "Schedule API is running"
}
```

---

## 2. GET — Тестовый роут

```
GET http://localhost:8000/api/v1/test
```

**Response:**
```json
{
  "message": "Test route"
}
```

---

## 3. GET — Доступные алгоритмы планировщика

```
GET http://localhost:8000/api/v1/all/scheduler_type
```

**Response:**
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "algorithms": ["BestCQI", "ProportionalFair", "RoundRobin"]
  }
}
```

---

## 4. POST — Запуск симуляции

```
POST http://localhost:8000/api/v1/sim/run
Content-Type: application/json
```

**Body:**
```json
{
  "ue_cnt": 5,
  "ue_ids": [1, 2, 3, 4, 5],
  "ue_move_pattern": "RandomWalk",
  "ue_coords": { "x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100 },
  "ue_traffic_pattern": "PoissonModel",
  "ue_pause": null,
  "bs_scheduler": "RoundRobin",
  "bs_coords": { "x": 50, "y": 50 },
  "bs_bw_mhz": 20,
  "sim_packet_rate": 1000,
  "sim_duration": 100,
  "update_interval": 1,
  "verbose": false,
  "stats_log": false,
  "bs_ch_type": "UMa",
  "enable_tdl": true,
  "log_to_file": false,
  "stats_file_prefix": "sim_stats"
}
```

**Response:**
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "run_id": "596fcf5d-0dca-4330-9eb3-e94e8be8b4a8",
    "ws_url": "/ws/simulation/596fcf5d-0dca-4330-9eb3-e94e8be8b4a8",
    "message": "Симуляция запущена. Подключитесь к WebSocket для получения прогресса"
  }
}
```

---

## 5. WebSocket — Подключение к симуляции

```
WS ws://localhost:8000/ws/simulation/{run_id}
```

Замени `{run_id}` на значение из ответа на запрос `POST /api/v1/sim/run`.

**Пример:**
```
ws://localhost:8000/ws/simulation/596fcf5d-0dca-4330-9eb3-e94e8be8b4a8
```

---

## 6. WebSocket — CSV стрим

```
WS ws://localhost:8000/ws/csv_stream
```

---

## 7. WebSocket — JSON стрим

```
WS ws://localhost:8000/ws/json_stream
```

---

## Значения параметров

### `ue_move_pattern`
- `RandomWalk`
- `RandomWaypoint`
- `RandomDirection`
- `GaussMarkov`
- `DiagonalWalk`

### `ue_traffic_pattern`
- `PoissonModel`
- `OnOffModel`
- `MMPPModel`

### `bs_scheduler`
- `BestCQI`
- `ProportionalFair`
- `RoundRobin`

### `bs_ch_type`
- `UMa`
- `UMi`
- `RMa`
