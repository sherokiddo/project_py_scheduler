# TRAFFIC_MODEL

> Ответственный: Кутенков Андрей  
> Состояние: сверено с кодом ветки dev; симуляция запускалась автором
> Проверено по коду: ветка dev, `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b` | Обновлено: 2026-10-17

## 1. Назначение
Генерация пакетов и управление профилями QoS: Poisson, OnOff и MMPP; фасад PacketManager и bearer-профили UE.

## 2. Границы ответственности
**Делает:** Генерация пакетов и управление профилями QoS: Poisson, OnOff и MMPP; фасад PacketManager и bearer-профили UE.

**Не делает:** не управляет временем симуляции, выбором model/file и жизненным циклом UE. Реестра model/file, файлового адаптера готовых результатов и отдельного RNG API в просмотренных исходниках нет; они относятся к проектируемой интеграции.

## 3. Входы, выходы и зависимости
| Вход | Откуда | Формат / смысл |
|---|---|---|
| Параметры и состояние | SimulationManager → traffic_gen | `ue_id`, тип модели и её параметры, QCI; `current_time` — конец окна генерации, `update_interval` — его длительность, оба в мс. |

| Выход | Куда | Формат / смысл |
|---|---|---|
| Результат | Вызывающий код | Список `Packet`; обновляемая статистика пакетов и байтов. Размеры — в байтах, время создания — в мс. |

Зависимости: `numpy`, `GLOBALS`, `abc`, `collections.defaultdict`, `dataclasses`, `enum.IntEnum`, `typing`. Импорты и QCI_PROFILES в GLOBALS проверены.

## 4. Как проверить работу

Из каталога `PyScheduler`, Python:

```python
import numpy as np
from TRAFFIC_MODEL import PacketManager
np.random.seed(42)
m=PacketManager()
m.set_model(ue_id=1,model_type="Poisson",qci=9,packet_rate=100)
packets=m.generate_packets(ue_id=1,current_time=100,update_interval=100)
print(len(packets),m.statistics.get_global_stats()["total_bytes"])
```

Выполнено на NumPy 2.3.5: **17 пакетов, 14189 байт**; все относятся к UE=1, времена 4.6926808998..81.5699259953 мс. Это демонстрация одного Poisson/bearer-сценария. Отдельно воспроизведены граничные ошибки audit; полный тестовый набор Traffic не запускался.

## 5. Основные компоненты

| Класс | Для чего нужен |
|---|---|
| [`QCI`](#qci) | Класс обслуживания с таблицами допустимой задержки и приоритета. |
| [`Packet`](#packet) | Объект сетевого пакета: размер, UE, время создания, QCI, срок жизни и bearer. |
| [`ITrafficModel`](#itrafficmodel) | Базовый контракт для статистических генераторов пакетов. |
| [`PoissonModel`](#poissonmodel) | Независимые поступления пакетов с экспоненциальными интервалами. |
| [`OnOffModel`](#onoffmodel) | Поток с чередованием активности (ON) и молчания (OFF) для каждого UE. |
| [`MMPPModel`](#mmppmodel) | Пуассоновская генерация с интенсивностью, зависящей от состояния марковской цепи. |
| [`TrafficModelFactory`](#trafficmodelfactory) | Фабрика выбора Poisson, OnOff или MMPP. |
| [`ITrafficGeneratorInterface`](#itrafficgeneratorinterface) | Общий интерфейс управления генерацией трафика для UE. |
| [`SimpleGenerator`](#simplegenerator) | Облегчённый генератор: одна статистическая модель на UE, без нескольких bearer. |
| [`BearerInfo`](#bearerinfo) | Структура для выдачи свойств одного bearer. |
| [`Bearer`](#bearer) | Один логический поток трафика с моделью генерации и классом QoS. |
| [`UeBearersInfo`](#uebearersinfo) | Структура для выдачи сводки bearer одного UE. |
| [`UeTrafficProfile`](#uetrafficprofile) | Набор bearer конкретного UE, генерация и управление каждым потоком. |
| [`TrafficStatistics`](#trafficstatistics) | Счётчики пакетов и байтов по UE, QCI, bearer и всей симуляции. |
| [`PacketManager`](#packetmanager) | Основной фасад генерации трафика с профилями UE и накоплением статистики. |


Ключевые объекты кода: `TRAFFIC_MODEL.py::Packet`, `TRAFFIC_MODEL.py::ITrafficModel`, `TRAFFIC_MODEL.py::TrafficModelFactory`, `TRAFFIC_MODEL.py::UeTrafficProfile`, `TRAFFIC_MODEL.py::TrafficStatistics`, `TRAFFIC_MODEL.py::PacketManager`. Путь исходников относительно корня проекта: `PyScheduler/TRAFFIC_MODEL.py`. Подробные параметры и результаты прежнего описания сохранены в разделе 5.1 ниже.

### 5.1. Классы, методы, аргументы и результаты

Подробные описания и особенности сохранены из вашего README. Сигнатуры ниже извлечены из исходников проекта; результаты ревизии — в audit.md.


Указаны назначение, фактические параметры, результат и особенности реализации. Методы с подчёркиванием (`_...`) предназначены для внутреннего использования.

#### QCI

Класс обслуживания с таблицами допустимой задержки и приоритета.

##### `get_delay_budget()`

**Сигнатура по коду:** `get_delay_budget(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::QCI.get_delay_budget`.

**Роль:** Вспомогательная операция.

**Как работает:** По значению QCI выбирает максимальную задержку пакета из словаря профилей.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Бюджет задержки в миллисекундах (`int`).

##### `get_priority()`

**Сигнатура по коду:** `get_priority(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::QCI.get_priority`.

**Роль:** Вспомогательная операция.

**Как работает:** По значению QCI выбирает приоритет обслуживания из таблицы.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Приоритет (`int`): меньшее число означает более высокий приоритет.

#### Packet

Объект сетевого пакета: размер, UE, время создания, QCI, срок жизни и bearer.

##### `__post_init__()`

**Сигнатура по коду:** `__post_init__(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::Packet.__post_init__`.

**Роль:** Внутренний метод.

**Как работает:** После создания `dataclass` присваивает сроку отправки значение `creation_time + ttl_ms`.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; заполняет `deadline`.

**Особенность:** Переданное в конструктор значение `deadline` перезаписывается: устанавливается `creation_time + ttl_ms`.

##### `age()`

**Сигнатура по коду:** `age(self, current_time: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::Packet.age`.

**Роль:** Вспомогательная операция.

**Как работает:** Вычитает время создания пакета из переданного текущего времени.

**Принимает:** `current_time` — Текущее время, мс (int).

**Возвращает:** Возраст пакета в миллисекундах.

##### `split_packet()`

**Сигнатура по коду:** `split_packet(self, fragment_size: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::Packet.split_packet`.

**Роль:** Вспомогательная операция.

**Как работает:** Проверяет допустимость длины фрагмента, уменьшает размер исходного пакета и создаёт отдельный `Packet` с теми же метаданными.

**Принимает:** `fragment_size` — Размер отделяемого фрагмента, байт (int).

**Возвращает:** Новый объект `Packet` размером `fragment_size`; размер исходного уменьшается.

##### `to_dict()`

**Сигнатура по коду:** `to_dict(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::Packet.to_dict`.

**Роль:** Совместимость / упрощённый интерфейс.

**Как работает:** Копирует основные поля пакета в обычный словарь для старого интерфейса.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `dict` с размером, идентификаторами, временем, приоритетом, QCI и признаками фрагмента.

##### `from_dict()`

**Сигнатура по коду:** `from_dict(data: Dict, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::Packet.from_dict`.

**Роль:** Совместимость / упрощённый интерфейс.

**Как работает:** Читает данные из словаря и создаёт пакет с переданным `ue_id`, подставляя значения по умолчанию для необязательных полей.

**Принимает:** `data` — Словарь полей пакета (Dict); `ue_id` — Идентификатор UE (int).

**Возвращает:** Объект `Packet`.

#### ITrafficModel

Базовый контракт для статистических генераторов пакетов.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, min_packet_size: int=150, max_packet_size: int=1500)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Сохраняет минимальный и максимальный размер пакета.

**Принимает:** `min_packet_size` — Минимальный размер пакета, байт (int); по умолчанию `150`; `max_packet_size` — Максимальный размер пакета, байт (int); по умолчанию `1500`.

**Возвращает:** `None`; подготавливает параметры генерации.

##### `generate_traffic()`

**Сигнатура по коду:** `generate_traffic(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficModel.generate_traffic`.

**Роль:** Интерфейсный метод.

**Как работает:** Определяет обязательный метод для всех статистических моделей; конкретный расчёт находится в наследниках.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Список `Packet` из реализации модели.

##### `get_model_name()`

**Сигнатура по коду:** `get_model_name(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficModel.get_model_name`.

**Роль:** Интерфейсный метод.

**Как работает:** Определяет обязательный метод выдачи имени конкретной модели.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Строка с именем модели из реализации.

##### `get_model_info()`

**Сигнатура по коду:** `get_model_info(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficModel.get_model_info`.

**Роль:** Вспомогательная операция.

**Как работает:** Собирает имя модели и границы размера пакета.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Словарь `name`, `min_packet_size`, `max_packet_size`.

#### PoissonModel

Независимые поступления пакетов с экспоненциальными интервалами.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, packet_rate: float, min_packet_size: int=150, max_packet_size: int=1500)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PoissonModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Сохраняет интенсивность поступления в пакетах/сек и допустимый диапазон размеров.

**Принимает:**

- `packet_rate` — Интенсивность пакетов, пакетов/сек (float).
- `min_packet_size` — Минимальный размер пакета, байт (int); по умолчанию `150`.
- `max_packet_size` — Максимальный размер пакета, байт (int); по умолчанию `1500`.

**Возвращает:** `None`; инициализирует параметры.

##### `generate_traffic()`

**Сигнатура по коду:** `generate_traffic(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PoissonModel.generate_traffic`.

**Роль:** Основная операция.

**Как работает:** На отрезке `[current_time - update_interval, current_time]` последовательно генерирует экспоненциальные промежутки между пакетами и случайные размеры.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Список `Packet` с временем создания каждого пакета.

**Особенность:** Если `packet_rate=0`, вычисление `1000 / packet_rate` вызовет деление на ноль.

##### `get_model_name()`

**Сигнатура по коду:** `get_model_name(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PoissonModel.get_model_name`.

**Роль:** Вспомогательная операция.

**Как работает:** Возвращает фиксированное название используемой статистической модели.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `"Poisson"`.

##### `get_model_info()`

**Сигнатура по коду:** `get_model_info(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PoissonModel.get_model_info`.

**Роль:** Вспомогательная операция.

**Как работает:** Дополняет общие сведения величиной `packet_rate`.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Словарь параметров модели.

#### OnOffModel

Поток с чередованием активности (ON) и молчания (OFF) для каждого UE.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, duration_on: float, duration_off: float, packet_rate: float, min_packet_size: int=150, max_packet_size: int=1500)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Сохраняет средние длительности ON/OFF в секундах и интенсивность активного потока; создаёт словарь состояний по UE.

**Принимает:**

- `duration_on` — Средняя длительность фазы ON, с (float).
- `duration_off` — Средняя длительность фазы OFF, с (float).
- `packet_rate` — Интенсивность пакетов, пакетов/сек (float).
- `min_packet_size` — Минимальный размер пакета, байт (int); по умолчанию `150`.
- `max_packet_size` — Максимальный размер пакета, байт (int); по умолчанию `1500`.

**Возвращает:** `None`; готовит генератор.

##### `generate_traffic()`

**Сигнатура по коду:** `generate_traffic(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.generate_traffic`.

**Роль:** Основная операция.

**Как работает:** Для UE перебирает фазы активности в пределах интервала: в ON генерирует пакеты как пуассоновский поток, в OFF пропускает время без пакетов.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Список `Packet` за интервал.

**Особенность:** Длительности ON/OFF в конструкторе трактуются как секунды и умножаются на 1000 при генерации; `current_time` и `update_interval` передаются в мс.

##### `_initialize_state()`

**Сигнатура по коду:** `_initialize_state(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel._initialize_state`.

**Роль:** Внутренний метод.

**Как работает:** Случайно выбирает начальную фазу UE и время её завершения по экспоненциальному распределению.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** `None`; создаёт запись состояния UE.

##### `_switch_to_off()`

**Сигнатура по коду:** `_switch_to_off(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel._switch_to_off`.

**Роль:** Внутренний метод.

**Как работает:** Переводит состояние UE в OFF и назначает следующую длительность бездействия.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `None`; изменяет фазу и время завершения.

##### `_switch_to_on()`

**Сигнатура по коду:** `_switch_to_on(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel._switch_to_on`.

**Роль:** Внутренний метод.

**Как работает:** Переводит состояние UE в ON и назначает следующую длительность активности.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `None`; изменяет фазу и время завершения.

##### `clear_state()`

**Сигнатура по коду:** `clear_state(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.clear_state`.

**Роль:** Вспомогательная операция.

**Как работает:** Удаляет историю фаз конкретного UE из внутреннего словаря.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `None`; освобождает состояние UE.

##### `get_model_name()`

**Сигнатура по коду:** `get_model_name(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.get_model_name`.

**Роль:** Вспомогательная операция.

**Как работает:** Возвращает имя реализации.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `"OnOff"`.

##### `get_model_info()`

**Сигнатура по коду:** `get_model_info(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.get_model_info`.

**Роль:** Вспомогательная операция.

**Как работает:** Выдаёт длительности фаз, интенсивность и число UE с сохранённым состоянием.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Словарь параметров и `active_devices`.

#### MMPPModel

Пуассоновская генерация с интенсивностью, зависящей от состояния марковской цепи.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, packet_rates: List[float], transition_matrix: np.ndarray, min_packet_size: float=150, max_packet_size: float=1500)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Сохраняет интенсивности по состояниям и матрицу интенсивностей переходов; создаёт словарь состояний UE.

**Принимает:**

- `packet_rates` — Интенсивность пакетов отдельно для каждого состояния, пакетов/сек (List[float]).
- `transition_matrix` — Матрица интенсивностей переходов между состояниями (np.ndarray).
- `min_packet_size` — Минимальный размер пакета, байт (float); по умолчанию `150`.
- `max_packet_size` — Максимальный размер пакета, байт (float); по умолчанию `1500`.

**Возвращает:** `None`; задаёт параметры марковского потока.

##### `_get_next_state()`

**Сигнатура по коду:** `_get_next_state(self, current_state: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel._get_next_state`.

**Роль:** Внутренний метод.

**Как работает:** По строке матрицы переходов получает суммарную интенсивность, выбирает следующую фазу по весам и случайное время до перехода.

**Принимает:** `current_state` — Текущее состояние марковской цепи (int).

**Возвращает:** `(следующее состояние, время до перехода в мс)`; при нулевой сумме — текущее состояние и бесконечность.

##### `generate_traffic()`

**Сигнатура по коду:** `generate_traffic(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.generate_traffic`.

**Роль:** Основная операция.

**Как работает:** Хранит для UE текущую фазу, следующую фазу и момент перехода; на каждом участке между переходами генерирует пуассоновские пакеты с интенсивностью соответствующей фазы.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Список `Packet` за интервал; состояние цепи UE сохраняется.

**Особенность:** Функция возвращает объекты `Packet`, хотя старый текст docstring описывает словари.

##### `clear_state()`

**Сигнатура по коду:** `clear_state(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.clear_state`.

**Роль:** Вспомогательная операция.

**Как работает:** Удаляет состояние марковской цепи для выбранного UE.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `None`; запись UE удаляется.

##### `get_model_name()`

**Сигнатура по коду:** `get_model_name(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.get_model_name`.

**Роль:** Вспомогательная операция.

**Как работает:** Возвращает имя статистической модели.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `"MMPP"`.

##### `get_model_info()`

**Сигнатура по коду:** `get_model_info(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.get_model_info`.

**Роль:** Вспомогательная операция.

**Как работает:** Добавляет список интенсивностей, число состояний, матрицу переходов и число активных UE.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Словарь с параметрами MMPP.

#### TrafficModelFactory

Фабрика выбора Poisson, OnOff или MMPP.

##### `create_model()`

**Сигнатура по коду:** `create_model(model_type: str, **kwargs)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_model`.

**Роль:** Основная операция.

**Как работает:** По строке типа создаёт Poisson, OnOff или MMPP; для MMPP при отсутствии `transition_matrix` задаёт матрицу по умолчанию.

**Принимает:** `model_type` — Имя модели (str); `**kwargs` — дополнительные параметры конкретной модели.

**Возвращает:** Экземпляр `ITrafficModel`; при неизвестном типе — `ValueError`.

##### `get_available_models()`

**Сигнатура по коду:** `get_available_models()`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.get_available_models`.

**Роль:** Вспомогательная операция.

**Как работает:** Возвращает перечень моделей, которые умеет создавать фабрика.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `["Poisson", "OnOff", "MMPP"]`.

##### `create_poisson_model()`

**Сигнатура по коду:** `create_poisson_model(packet_rate: float, **kwargs)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_poisson_model`.

**Роль:** Вспомогательная операция.

**Как работает:** Оборачивает вызов конструктора `PoissonModel`.

**Принимает:** `packet_rate` — Интенсивность пакетов, пакетов/сек (float); `**kwargs` — дополнительные параметры конкретной модели.

**Возвращает:** Экземпляр `PoissonModel`.

##### `create_onoff_model()`

**Сигнатура по коду:** `create_onoff_model(duration_on: float, duration_off: float, packet_rate: float, **kwargs)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_onoff_model`.

**Роль:** Вспомогательная операция.

**Как работает:** Оборачивает вызов конструктора `OnOffModel`.

**Принимает:**

- `duration_on` — Средняя длительность фазы ON, с (float).
- `duration_off` — Средняя длительность фазы OFF, с (float).
- `packet_rate` — Интенсивность пакетов, пакетов/сек (float).
- `**kwargs` — дополнительные параметры конкретной модели.

**Возвращает:** Экземпляр `OnOffModel`.

##### `create_mmpp_model()`

**Сигнатура по коду:** `create_mmpp_model(packet_rates: List[float], transition_matrix: Optional[np.ndarray]=None, **kwargs)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_mmpp_model`.

**Роль:** Вспомогательная операция.

**Как работает:** Передаёт список интенсивностей и матрицу в конструктор `MMPPModel`.

**Принимает:**

- `packet_rates` — Интенсивность пакетов отдельно для каждого состояния, пакетов/сек (List[float]).
- `transition_matrix` — Матрица интенсивностей переходов между состояниями (Optional[np.ndarray]); по умолчанию `None`.
- `**kwargs` — дополнительные параметры конкретной модели.

**Возвращает:** Экземпляр `MMPPModel`.

**Особенность:** По умолчанию передаёт `transition_matrix=None` непосредственно в `MMPPModel`; автоматическая матрица создаётся только через `create_model`.

#### ITrafficGeneratorInterface

Общий интерфейс управления генерацией трафика для UE.

##### `generate_packets()`

**Сигнатура по коду:** `generate_packets(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficGeneratorInterface.generate_packets`.

**Роль:** Интерфейсный метод.

**Как работает:** Задаёт метод выдачи пакетов UE через единый интерфейс генератора.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Список `Packet` конкретной реализации.

##### `set_model()`

**Сигнатура по коду:** `set_model(self, ue_id: int, model_type: str, qci: Optional[QCI | int]=None, **params)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficGeneratorInterface.set_model`.

**Роль:** Интерфейсный метод.

**Как работает:** Задаёт операцию назначения генератора трафика пользователю.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `model_type` — Имя модели (str).
- `qci` — Класс качества обслуживания (Optional[QCI | int]); по умолчанию `None`.
- `**params` — дополнительные параметры конкретной модели.

**Возвращает:** Результат не определён контрактом; реализации изменяют настройки UE.

##### `get_statistics()`

**Сигнатура по коду:** `get_statistics(self, ue_id: Optional[int]=None)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficGeneratorInterface.get_statistics`.

**Роль:** Интерфейсный метод.

**Как работает:** Задаёт операцию получения статистики генератора.

**Принимает:** `ue_id` — Идентификатор UE (Optional[int]); по умолчанию `None`.

**Возвращает:** Словарь метрик.

##### `reset_ue()`

**Сигнатура по коду:** `reset_ue(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::ITrafficGeneratorInterface.reset_ue`.

**Роль:** Интерфейсный метод.

**Как работает:** Задаёт операцию удаления контекста генератора конкретного UE.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** Реализации изменяют внутренние словари; возвращаемое значение не задано.

#### SimpleGenerator

Облегчённый генератор: одна статистическая модель на UE, без нескольких bearer.

##### `__init__()`

**Сигнатура по коду:** `__init__(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.__init__`.

**Роль:** Инициализация.

**Как работает:** Создаёт словарь `ue_id → модель` и счётчики выпущенных пакетов.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; состояние генератора.

##### `generate_packets()`

**Сигнатура по коду:** `generate_packets(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.generate_packets`.

**Роль:** Совместимость / упрощённый интерфейс.

**Как работает:** Вызывает `generate_traffic` назначенной модели UE и увеличивает счётчики. Без модели возвращает пустой список.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Список `Packet`.

##### `set_model()`

**Сигнатура по коду:** `set_model(self, ue_id: int, model_type: str, qci: Optional[QCI | int]=None, **params)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.set_model`.

**Роль:** Совместимость / упрощённый интерфейс.

**Как работает:** Создаёт одну модель для UE и заменяет ею предыдущую. Параметр QCI не используется — выводится предупреждение.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `model_type` — Имя модели (str).
- `qci` — Класс качества обслуживания (Optional[QCI | int]); по умолчанию `None`.
- `**params` — дополнительные параметры конкретной модели.

**Возвращает:** `None`; изменяет назначение модели UE.

**Особенность:** QoS/QCI в упрощённом генераторе не применяется.

##### `get_statistics()`

**Сигнатура по коду:** `get_statistics(self, ue_id: Optional[int]=None)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.get_statistics`.

**Роль:** Совместимость / упрощённый интерфейс.

**Как работает:** Возвращает глобальные счётчики или статистику одного UE.

**Принимает:** `ue_id` — Идентификатор UE (Optional[int]); по умолчанию `None`.

**Возвращает:** Словарь с количеством пакетов и информацией о модели.

##### `reset_ue()`

**Сигнатура по коду:** `reset_ue(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.reset_ue`.

**Роль:** Совместимость / упрощённый интерфейс.

**Как работает:** Очищает состояние модели UE, удаляет назначение и его индивидуальный счётчик.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `None`; удаляет контекст UE.

#### BearerInfo

Структура для выдачи свойств одного bearer.

Это структура данных: поля описаны в объявлении `dataclass`, отдельных методов у неё нет.

#### Bearer

Один логический поток трафика с моделью генерации и классом QoS.

##### `__post_init__()`

**Сигнатура по коду:** `__post_init__(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::Bearer.__post_init__`.

**Роль:** Внутренний метод.

**Как работает:** Преобразует числовой QCI в enum, применяет параметры GBR/MBR из профиля и отвергает неположительные ограничения скорости.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; нормализует поля bearer.

##### `get_info()`

**Сигнатура по коду:** `get_info(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::Bearer.get_info`.

**Роль:** Вспомогательная операция.

**Как работает:** Преобразует bearer в структуру данных без внутреннего объекта генератора.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Объект `BearerInfo`.

#### UeBearersInfo

Структура для выдачи сводки bearer одного UE.

Это структура данных: поля описаны в объявлении `dataclass`, отдельных методов у неё нет.

#### UeTrafficProfile

Набор bearer конкретного UE, генерация и управление каждым потоком.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.__init__`.

**Роль:** Инициализация.

**Как работает:** Создаёт пустой набор bearer UE и счётчик для их идентификаторов.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `None`; сохраняет `ue_id` и словарь bearer.

##### `add_bearer()`

**Сигнатура по коду:** `add_bearer(self, model: ITrafficModel, qci: QCI | int, bearer_id: Optional[int]=None)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.add_bearer`.

**Роль:** Вспомогательная операция.

**Как работает:** Контролирует лимит bearer, уникальность QCI и ID, создаёт `Bearer` и добавляет в профиль.

**Принимает:**

- `model` — Экземпляр статистической модели, генерирующей пакеты этого bearer (ITrafficModel).
- `qci` — Класс качества обслуживания (QCI | int).
- `bearer_id` — Идентификатор логического потока (bearer) (Optional[int]); по умолчанию `None`.

**Возвращает:** Числовой идентификатор созданного bearer.

**Особенность:** Повторный QCI запрещён, а число bearer ограничено восемью.

##### `remove_bearer()`

**Сигнатура по коду:** `remove_bearer(self, bearer_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.remove_bearer`.

**Роль:** Вспомогательная операция.

**Как работает:** Удаляет bearer по ID, предварительно очищая сохранённое состояние статистической модели этого UE.

**Принимает:** `bearer_id` — Идентификатор логического потока (bearer) (int).

**Возвращает:** `None`; указанный bearer удаляется.

##### `generate_all_traffic()`

**Сигнатура по коду:** `generate_all_traffic(self, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.generate_all_traffic`.

**Роль:** Вспомогательная операция.

**Как работает:** Запрашивает пакеты у каждой активной модели bearer; каждому пакету назначает bearer ID, QCI, приоритет и deadline с учётом бюджета задержки.

**Принимает:** `current_time` — Текущее время, мс (int); `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Общий список `Packet` всех активных bearer.

##### `get_active_bearers()`

**Сигнатура по коду:** `get_active_bearers(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.get_active_bearers`.

**Роль:** Вспомогательная операция.

**Как работает:** Отбирает bearer с флагом `enabled=True`.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Список объектов `Bearer`.

##### `set_bearer_enabled()`

**Сигнатура по коду:** `set_bearer_enabled(self, bearer_id: int, enabled: bool)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.set_bearer_enabled`.

**Роль:** Вспомогательная операция.

**Как работает:** Меняет флаг активности указанного bearer; неизвестный ID вызывает `ValueError`.

**Принимает:** `bearer_id` — Идентификатор логического потока (bearer) (int); `enabled` — Флаг активности bearer (bool).

**Возвращает:** `None`; меняет `enabled`.

##### `get_bearers_info()`

**Сигнатура по коду:** `get_bearers_info(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.get_bearers_info`.

**Роль:** Вспомогательная операция.

**Как работает:** Создаёт сводку профиля: число bearer, число активных и описание каждого.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Объект `UeBearersInfo`.

##### `clear_all()`

**Сигнатура по коду:** `clear_all(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::UeTrafficProfile.clear_all`.

**Роль:** Вспомогательная операция.

**Как работает:** Последовательно вызывает удаление для всех bearer UE.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; профиль становится пустым.

#### TrafficStatistics

Счётчики пакетов и байтов по UE, QCI, bearer и всей симуляции.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, window_ms: int=1000)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.__init__`.

**Роль:** Инициализация.

**Как работает:** Создаёт словари-счётчики пакетов и байтов отдельно по UE, QCI и bearer, а также глобальные счётчики.

**Принимает:** `window_ms` — Настройка окна статистики, мс (int); по умолчанию `1000`.

**Возвращает:** `None`; создаёт накопители метрик.

##### `update()`

**Сигнатура по коду:** `update(self, packets: List[Packet])`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.update`.

**Роль:** Основная операция.

**Как работает:** Для каждого пакета увеличивает глобальные и индивидуальные счётчики, запоминает первое и последнее время генерации UE.

**Принимает:** `packets` — Список сформированных пакетов (List[Packet]).

**Возвращает:** `None`; обновляет статистику.

##### `get_ue_stats()`

**Сигнатура по коду:** `get_ue_stats(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.get_ue_stats`.

**Роль:** Вспомогательная операция.

**Как работает:** Вычисляет число пакетов/байтов, средний размер, длительность генерации и bitrate пользователя.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** Словарь `packets`, `bytes`, `avg_packet_size`, `bitrate_bps`, `bitrate_mbps`, `duration_ms`.

##### `get_qci_stats()`

**Сигнатура по коду:** `get_qci_stats(self, qci: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.get_qci_stats`.

**Роль:** Вспомогательная операция.

**Как работает:** Выбирает накопленные счётчики конкретного QCI и средний размер пакета.

**Принимает:** `qci` — Класс качества обслуживания (int).

**Возвращает:** Словарь `qci`, `packets`, `bytes`, `avg_packet_size`.

##### `get_bearer_stats()`

**Сигнатура по коду:** `get_bearer_stats(self, ue_id: int, bearer_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.get_bearer_stats`.

**Роль:** Вспомогательная операция.

**Как работает:** Выбирает счётчики по паре `(ue_id, bearer_id)` и средний размер пакета.

**Принимает:** `ue_id` — Идентификатор UE (int); `bearer_id` — Идентификатор логического потока (bearer) (int).

**Возвращает:** Словарь `ue_id`, `bearer_id`, `packets`, `bytes`, `avg_packet_size`.

##### `get_global_stats()`

**Сигнатура по коду:** `get_global_stats(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.get_global_stats`.

**Роль:** Вспомогательная операция.

**Как работает:** Собирает общие количества пакетов, байтов, мегабитов, UE, QCI и средний размер.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Словарь глобальных метрик.

##### `get_qci_distribution()`

**Сигнатура по коду:** `get_qci_distribution(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.get_qci_distribution`.

**Роль:** Вспомогательная операция.

**Как работает:** Делит количество байтов каждого QCI на общий объём трафика, переводит в проценты.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Словарь `{qci: процент байтов}`.

##### `reset()`

**Сигнатура по коду:** `reset(self)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.reset`.

**Роль:** Вспомогательная операция.

**Как работает:** Очищает все счётчики и начальный момент времени.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; обнуляет статистику.

#### PacketManager

Основной фасад генерации трафика с профилями UE и накоплением статистики.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bitrate_window_ms: int=1000)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.__init__`.

**Роль:** Инициализация.

**Как работает:** Создаёт словарь профилей `ue_id → UeTrafficProfile` и счётчик `TrafficStatistics`.

**Принимает:** `bitrate_window_ms` — Окно статистики битрейта, мс (int); по умолчанию `1000`.

**Возвращает:** `None`; подготавливает менеджер.

##### `generate_packets()`

**Сигнатура по коду:** `generate_packets(self, ue_id: int, current_time: int, update_interval: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.generate_packets`.

**Роль:** Основная операция.

**Как работает:** Находит профиль UE, вызывает его генерацию по bearer и передаёт пакеты в `TrafficStatistics.update`.

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `current_time` — Текущее время, мс (int).
- `update_interval` — Длительность интервала генерации трафика, мс (int).

**Возвращает:** Список `Packet`; для UE без профиля — `[]`.

##### `set_model()`

**Сигнатура по коду:** `set_model(self, ue_id: int, model_type: str, qci: Optional[QCI | int]=None, **params)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.set_model`.

**Роль:** Основная операция.

**Как работает:** Создаёт профиль UE при необходимости, выбирает модель через фабрику и добавляет bearer с QCI (по умолчанию 9).

**Принимает:**

- `ue_id` — Идентификатор UE (int).
- `model_type` — Имя модели (str).
- `qci` — Класс качества обслуживания (Optional[QCI | int]); по умолчанию `None`.
- `**params` — дополнительные параметры конкретной модели.

**Возвращает:** `None`; добавляет новый поток трафика UE.

##### `remove_bearer()`

**Сигнатура по коду:** `remove_bearer(self, ue_id: int, bearer_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.remove_bearer`.

**Роль:** Вспомогательная операция.

**Как работает:** Находит профиль UE и передаёт удаление bearer в `UeTrafficProfile`.

**Принимает:** `ue_id` — Идентификатор UE (int); `bearer_id` — Идентификатор логического потока (bearer) (int).

**Возвращает:** `None`; отсутствующий UE или bearer вызывает `ValueError`.

##### `get_statistics()`

**Сигнатура по коду:** `get_statistics(self, ue_id: Optional[int]=None)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.get_statistics`.

**Роль:** Вспомогательная операция.

**Как работает:** Возвращает общую статистику либо сводку по переданному UE.

**Принимает:** `ue_id` — Идентификатор UE (Optional[int]); по умолчанию `None`.

**Возвращает:** Словарь метрик.

##### `get_qos_statistics()`

**Сигнатура по коду:** `get_qos_statistics(self, ue_id: int, qci: Optional[int]=None)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.get_qos_statistics`.

**Роль:** Вспомогательная операция.

**Как работает:** При наличии QCI возвращает сводку этого QCI; иначе совмещает статистику UE с общим распределением байтов по QCI.

**Принимает:** `ue_id` — Идентификатор UE (int); `qci` — Класс качества обслуживания (Optional[int]); по умолчанию `None`.

**Возвращает:** Словарь с QoS-метриками.

**Особенность:** Значение `ue_id` не ограничивает выборку в ветви с заданным `qci`: `get_qci_stats(qci)` агрегирует все UE. `qci_distribution` при `qci=None` также глобальное.

##### `reset_ue()`

**Сигнатура по коду:** `reset_ue(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.reset_ue`.

**Роль:** Вспомогательная операция.

**Как работает:** Очищает все bearer профиля UE и удаляет сам профиль.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `None`; назначенные генераторы UE удаляются.

**Особенность:** Удаляются профили и bearer, но накопленные в `TrafficStatistics` исторические счётчики UE остаются.

##### `get_bearers_info()`

**Сигнатура по коду:** `get_bearers_info(self, ue_id: int)`.
**Источник:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.get_bearers_info`.

**Роль:** Вспомогательная операция.

**Как работает:** Читает сводку bearer UE из профиля.

**Принимает:** `ue_id` — Идентификатор UE (int).

**Возвращает:** `UeBearersInfo` или `None`, если UE отсутствует.


### Связь с другими частями симулятора

`SIMULATION_MANAGER.py` настраивает `PacketManager` и вызывает `generate_packets` в ходе работы симуляции. Дальше объекты `Packet` используются буферами сетевого планировщика в `BS_MODULE.py`.

## 6. Один типичный сценарий

1. `SimulationManager` выбирает `SimpleGenerator` для legacy либо `PacketManager` для layered-buffer режима.
2. `setup_ue_traffic` / `setup_ue_traffic_profiles` вызывает `traffic_gen.set_model`.
3. В `run_tti(current_time)` менеджер вызывает `generate_packets(ue_id,current_time,update_interval)` для UE с настроенным профилем.
4. PacketManager вызывает `UeTrafficProfile.generate_all_traffic`; bearer-модель формирует Packet, профиль назначает bearer/QCI/priority/deadline, затем PacketManager обновляет TrafficStatistics.
5. Менеджер передаёт каждый пакет в `base_station.buffer_manager.add_packet`, после чего запускает scheduler.

**Время AS-IS:** генераторы начинают с `current_time-update_interval` и заканчивают на current_time, а не после него. `creation_time` фактически float из экспоненциальных интервалов, несмотря на аннотацию int в Packet. Условие отсечения `> end` допускает точное попадание на правую границу. Внешний file/model-контракт пока не реализован.

## 7. Ограничения и открытые вопросы
- Статусы замечаний уточнены в audit.md: различаются воспроизведённые ошибки, подтверждённые конструкции кода и несогласованные архитектурные решения. Наличие метода не означает прохождение всех его ветвей.
- Сигнатуры и импорты сверены с кодом ветки dev. Симуляция запускалась автором; результаты дополнительных проверок приведены в разделе 4 и audit.md.
- File-режим, DTO и интерфейс RNG остаются предложениями в общей спецификации; исходники подтверждают текущую реализацию, но не согласование будущих изменений.

## 8. Связанные материалы
- [Диаграммы](docs/diagrams.md)
- [Ревизия](docs/audit.md)
- [План развития](docs/features.md)
- [Единая спецификация model/file](../integration_report.md)
