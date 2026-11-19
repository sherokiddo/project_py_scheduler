# Диаграммы модуля TRAFFIC_MODEL

> Ответственный: Кутенков Андрей | Проверено по коду: `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b` | Обновлено: 2026-10-17

Исходные диаграммы и все их узлы, связи и подписи сохранены. Исходные схемы сохранены как авторские материалы; ниже добавлены результаты сверки с кодом. SVG не перерисовывались, поэтому сверка маршрутов не означает автоматической проверки каждой связи внутри SVG.

## 1. Текущая структура (AS-IS)

### 1.1. Исходная UML-диаграмма
![Исходная UML-диаграмма](assets/traf.svg)


### 1.2. Исходное дерево операций
```mermaid
flowchart TD
SM[Менеджер] --> PM[PacketManager] --> PROFILE[UeTrafficProfile / Bearer]
PROFILE --> FACT[TrafficModelFactory]
FACT --> P[PoissonModel]
FACT --> O[OnOffModel]
FACT --> MM[MMPPModel]
P --> PACK[Packet]
O --> PACK
MM --> PACK
PACK --> STAT[TrafficStatistics]
```


Сверка дерева: PacketManager → UeTrafficProfile → bearer-модель → Packet → TrafficStatistics подтверждено. Фабрика вызывается при настройке set_model/add_bearer; на каждом шаге готовые модели используются повторно. Исходная схема сохранена.

### 1.3. Все исходные графические материалы

- [traf.svg](assets/traf.svg) — сохранён без изменения содержимого.

![traffic_states.svg](assets/traffic_states.svg)

- [traffic_states.svg](assets/traffic_states.svg) — сохранён без изменения содержимого.

## 2. Типовой сценарий (AS-IS)

Исходная концептуальная последовательность сохранена без изменения. Надписи в ней отражают прежнюю редакцию; точный маршрут проверен по коду проекта и показан дополнительно ниже.
```mermaid
sequenceDiagram
participant S as SimulationManager (контракт не проверен)
participant P as PacketManager
participant M as ITrafficModel
S->>P: generate_packets(...)
P->>M: generate_traffic(...)
M-->>P: пакеты
P-->>S: пакеты
```


### 2.1. Уточнённая последовательность по коду проекта

```mermaid
sequenceDiagram
    participant SM as SimulationManager
    participant PM as PacketManager
    participant P as UeTrafficProfile
    participant M as Модель bearer
    participant ST as TrafficStatistics
    SM->>PM: generate_packets(ue_id, current_time, update_interval)
    PM->>P: generate_all_traffic(current_time, update_interval)
    loop Активные bearer
        P->>M: generate_traffic(ue_id, current_time, update_interval)
        M-->>P: list[Packet]
        P->>P: Назначить bearer, QCI, priority, deadline
    end
    P-->>PM: Все пакеты UE
    PM->>ST: update(packets)
    PM-->>SM: list[Packet]
    SM->>SM: buffer_manager.add_packet, затем scheduler
```

## 3. Целевая структура (TO-BE) — только если согласована

Не согласовано, см. [features.md](features.md). Ниже сохранена ваша исходная обзорная схема предложения. Она иллюстрирует развилку источников, а не вводит отдельную спецификацию. Единственный подробный контракт — [INT-MF-001](../../integration_report.md). В его file-режиме математическая модель и RNG не создаются и не вызываются.
```mermaid
flowchart TD
SM[SimulationManager] --> CH{Есть внешний файл для модуля?}
CH -->|Да| DS[Источник: индекс UE и время]
CH -->|Нет| MODEL[Математическая модель + RNG]
DS --> APPLY[Применение результата к UE]
MODEL --> APPLY
```



## 4. Ссылки на исходники

- `PyScheduler/TRAFFIC_MODEL.py::PacketManager.set_model` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/TRAFFIC_MODEL.py::PacketManager.generate_packets` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.generate_all_traffic` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/TRAFFIC_MODEL.py::ITrafficModel.generate_traffic` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.update` — наличие и сигнатура подтверждены по указанной версии кода.

## 5. Сохранённые ветви для возможного распараллеливания
- **Между UE:** потенциально независимые операции при раздельном состоянии и RNG.
- **Внутри UE:** порядок обновления состояния обязателен; зависимые шаги нельзя распараллеливать без проверки.
- **Между модулями:** Mobility → Channel по координатам; Traffic потенциально параллелен расчёту Mobility при фиксированном снимке входов.
- Это кандидаты, а не доказательство потокобезопасности или ускорения.


### Связанные вызывающие компоненты

- `PyScheduler/SIMULATION_MANAGER.py::SimulationManager.run_tti` — физика, Traffic, буферы и scheduler.
- `PyScheduler/UE_MODULE.py::UECollection.UPDATE_ALL_USERS` — интервалы Mobility/Channel.
- `PyScheduler/UE_MODULE.py::UserEquipment.UPD_POSITION` и `UPD_CH_QUALITY` — применение результата.
