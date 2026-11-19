# Диаграммы модуля MOBILITY_MODEL

> Ответственный: Кутенков Андрей | Проверено по коду: `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b` | Обновлено: 2026-10-17

Исходные диаграммы и все их узлы, связи и подписи сохранены. Исходные схемы сохранены как авторские материалы; ниже добавлены результаты сверки с кодом. SVG не перерисовывались, поэтому сверка маршрутов не означает автоматической проверки каждой связи внутри SVG.

## 1. Текущая структура (AS-IS)

### 1.1. Исходная UML-диаграмма
![Исходная UML-диаграмма](assets/mobil.svg)


### 1.2. Исходное дерево операций
```mermaid
flowchart TD
SM[Менеджер или UE] --> I[MobilityInterface] --> F{Выбор модели}
F --> RW[RandomWalkModel]
F --> RWP[RandomWaypointModel]
F --> RD[RandomDirectionModel]
F --> GM[GaussMarkovModel]
F --> DW[DiagonalWalkModel]
RW --> B[MapBorders / проверка границ]
RWP --> B
RD --> B
GM --> B
DW --> B
B --> OUT[позиция, скорость, направление]
```


Сверка дерева: набор пяти моделей и MapBorders подтверждён. Менеджер вызывает модель через UECollection → UserEquipment; фабрика выбирает экземпляр при настройке, а не делегирует каждый update. Эти уточнения показаны в дополнительной проверенной последовательности; исходная схема сохранена.

### 1.3. Все исходные графические материалы

- [mobil.svg](assets/mobil.svg) — сохранён без изменения содержимого.

![mobility_states.svg](assets/mobility_states.svg)

- [mobility_states.svg](assets/mobility_states.svg) — сохранён без изменения содержимого.

## 2. Типовой сценарий (AS-IS)

Исходная концептуальная последовательность сохранена без изменения. Надписи в ней отражают прежнюю редакцию; точный маршрут проверен по коду проекта и показан дополнительно ниже.
```mermaid
sequenceDiagram
participant U as UE_MODULE
participant M as MobilityInterface
participant A as Модель движения
U->>M: update(параметры шага)
M->>A: update(...)
A-->>M: позиция, скорость, направление
M-->>U: результат
```


### 2.1. Уточнённая последовательность по коду проекта

```mermaid
sequenceDiagram
    participant SM as SimulationManager
    participant UC as UECollection
    participant UE as UserEquipment
    participant M as Конкретная модель движения
    SM->>UC: UPDATE_ALL_USERS(current_time, интервалы)
    UC->>UE: UPD_POSITION(mob_interval), если наступил срок
    UE->>M: update(time_ms=mob_interval)
    M-->>UE: position, velocity, direction
    UE->>UE: Применить поля и пересчитать расстояния
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

- `PyScheduler/MOBILITY_MODEL.py::MobilityInterface.create` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/MOBILITY_MODEL.py::MobilityInterface.update` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/MOBILITY_MODEL.py::MapBorders.get_borders` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/MOBILITY_MODEL.py::RandomWalkModel.update` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/MOBILITY_MODEL.py::RandomWaypointModel.update` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/MOBILITY_MODEL.py::RandomDirectionModel.update` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/MOBILITY_MODEL.py::GaussMarkovModel.update` — наличие и сигнатура подтверждены по указанной версии кода.
- `PyScheduler/MOBILITY_MODEL.py::DiagonalWalkModel.update` — наличие и сигнатура подтверждены по указанной версии кода.

## 5. Сохранённые ветви для возможного распараллеливания
- **Между UE:** потенциально независимые операции при раздельном состоянии и RNG.
- **Внутри UE:** порядок обновления состояния обязателен; зависимые шаги нельзя распараллеливать без проверки.
- **Между модулями:** Mobility → Channel по координатам; Traffic потенциально параллелен расчёту Mobility при фиксированном снимке входов.
- Это кандидаты, а не доказательство потокобезопасности или ускорения.


### Связанные вызывающие компоненты

- `PyScheduler/SIMULATION_MANAGER.py::SimulationManager.run_tti` — физика, Traffic, буферы и scheduler.
- `PyScheduler/UE_MODULE.py::UECollection.UPDATE_ALL_USERS` — интервалы Mobility/Channel.
- `PyScheduler/UE_MODULE.py::UserEquipment.UPD_POSITION` и `UPD_CH_QUALITY` — применение результата.
