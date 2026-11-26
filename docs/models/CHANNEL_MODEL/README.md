# CHANNEL_MODEL

> Ответственный: Кутенков Андрей  
> Состояние: сверено с кодом ветки dev; симуляция запускалась автором
> Проверено по коду: ветка dev, `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b` | Обновлено: 2026-10-17

## 1. Назначение
Расчёт потерь и качества радиоканала для сценариев RMa, UMa, UMi и описанных SC-вариантов; TDL отвечает за частотно-селективные замирания.

## 2. Границы ответственности
**Делает:** Расчёт потерь и качества радиоканала для сценариев RMa, UMa, UMi и описанных SC-вариантов; TDL отвечает за частотно-селективные замирания.

**Не делает:** не управляет временем симуляции, выбором model/file и жизненным циклом UE. Реестра model/file, файлового адаптера готовых результатов и отдельного RNG API в просмотренных исходниках нет; они относятся к проектируемой интеграции.

## 3. Входы, выходы и зависимости
| Вход | Откуда | Формат / смысл |
|---|---|---|
| Параметры и состояние | SimulationManager → UECollection → UserEquipment | Объект BS, геометрия и состояние UE, параметры радиосценария; `calculate_SINR(UE_ID, displacement, d_2D, d_2D_in, d_3D, UE_height, ue_class)`. |

| Выход | Куда | Формат / смысл |
|---|---|---|
| Результат | Вызывающий код | SINR; при включённом TDL — значения по ресурсным блокам согласно исходной документации. При выключенном TDL возвращается скаляр, при включённом — ndarray по RB; аннотация float не описывает второй режим. |

Зависимости: `numpy`, `GLOBALS`, `BS_MODULE.BaseStation`; GLOBALS.TDL_PARAMS_TABLE, TDL_DELAY_SPREAD_TABLE, RB_BANDWIDTH, CURRENT_TIME; зарегистрированные UE у BS для Доплера. Импорты проверены.

## 4. Как проверить работу

Из каталога `PyScheduler`, Python:

```python
from types import SimpleNamespace
import numpy as np
from CHANNEL_MODEL import ChannelInterface, ChannelModel
np.random.seed(42)
ChannelModel.CHANNEL_COND_INFO.clear()
ChannelModel.SHADOW_FADING_INFO.clear()
ChannelModel.O2I_INFO.clear()
bs=SimpleNamespace(frequency_Hz=2e9,frequency_GHz=2.,bandwidth=20,
    rb_per_slot=100,height=25,enable_tdl=False,antenna_gain=15,
    MACROCELL_TX_POWER={20:46},registered_ues={})
m=ChannelInterface(bs).create("UMa")
print(m.calculate_SINR(1,1.,100.,0.,103.,1.5,"pedestrian"))
```

Выполнено: **59.47370433632662 дБ**. Использована минимальная BS-фикстура, реальная модель UMa, TDL выключен; результат демонстрирует исполнение, не подтверждает физическую точность формул. Отдельно проверены SC-конструкторы и O2I, см. audit.

## 5. Основные компоненты

| Класс | Для чего нужен |
|---|---|
| [`ChannelInterface`](#channelinterface) | Выбирает радиосценарий и создаёт его экземпляр для базовой станции. |
| [`ChannelState`](#channelstate) | Хранит предыдущее положение, LOS/NLOS, shadow fading и O2I для пространственно-согласованного режима. |
| [`ChannelModel`](#channelmodel) | Общие расчёты крупномасштабного радиоканала: условия распространения, затухание, затенение, проникновение и SINR. |
| [`RMaModel`](#rmamodel) | Формулы сценария Rural Macro: сельская макросота. |
| [`UMaModel`](#umamodel) | Формулы сценария Urban Macro: городская макросота. |
| [`UMiModel`](#umimodel) | Формулы сценария Urban Micro: городская микросота. |
| [`TDLModel`](#tdlmodel) | Частотно-селективные замирания по профилям TDL на ресурсных блоках. |
| [`RMaModel_SC`](#rmamodel_sc) | Вариант Rural Macro с дополнительными методами пространственной согласованности. |
| [`UMaModel_SC`](#umamodel_sc) | Вариант Urban Macro с дополнительными методами пространственной согласованности. |
| [`UMiModel_SC`](#umimodel_sc) | Вариант Urban Micro с дополнительными методами пространственной согласованности. |


Ключевые объекты кода: `CHANNEL_MODEL.py::ChannelInterface`, `CHANNEL_MODEL.py::ChannelState`, `CHANNEL_MODEL.py::ChannelModel`, `CHANNEL_MODEL.py::RMaModel`, `CHANNEL_MODEL.py::UMaModel`, `CHANNEL_MODEL.py::UMiModel`, `CHANNEL_MODEL.py::TDLModel`. Путь исходников относительно корня проекта: `PyScheduler/CHANNEL_MODEL.py`. Подробные параметры и результаты прежнего описания сохранены в разделе 5.1 ниже.

### 5.1. Классы, методы, аргументы и результаты

Подробные описания и особенности сохранены из вашего README. Сигнатуры ниже извлечены из исходников проекта; результаты ревизии — в audit.md.


Указаны назначение, фактические параметры, результат и особенности реализации. Методы с подчёркиванием (`_...`) предназначены для внутреннего использования.

#### ChannelInterface

Выбирает радиосценарий и создаёт его экземпляр для базовой станции.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.__init__`.

**Роль:** Инициализация.

**Как работает:** Сохраняет базовую станцию, для которой будут создаваться модели канала.

**Принимает:** `bs` — Объект базовой станции (BaseStation).

**Возвращает:** `None`; сохраняет ссылку `bs`.

##### `get_conditions()`

**Сигнатура по коду:** `get_conditions()`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_conditions`.

**Роль:** Вспомогательная операция.

**Как работает:** Возвращает имена каналов, доступных через фабрику, и соответствующие классы.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Словарь для `RMa`, `UMa`, `UMi`, `RMa-SC`.

**Особенность:** Фабрика не содержит ключей `UMa-SC` и `UMi-SC`, хотя соответствующие классы определены.

##### `validate_parameters()`

**Сигнатура по коду:** `validate_parameters(model: str, cond_update_period: float, freq_fad_nlos_model: str, freq_fad_los_model: str, ds_profile: str, los_arrival_angle: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.validate_parameters`.

**Роль:** Вспомогательная операция.

**Как работает:** Контролирует параметры периода, угла прихода, названия TDL и профиля разброса задержек.

**Принимает:**

- `model` — Имя сценария распространения (str).
- `cond_update_period` — Период обновления состояния канала, мс (float).
- `freq_fad_nlos_model` — TDL-профиль для NLOS (str).
- `freq_fad_los_model` — TDL-профиль для LOS (str).
- `ds_profile` — Профиль разброса задержек (str).
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float).

**Возвращает:** `None`; при ошибках — `ValueError` или `TypeError`.

##### `create()`

**Сигнатура по коду:** `create(self, cond: str, cond_update_period: float=0.0, freq_fad_nlos_model: str='TDL-A', freq_fad_los_model: str='TDL-D', ds_profile: str='normal', los_arrival_angle: float=np.pi / 4, o2i_model: str='low', W: float=20.0, h: float=5.0)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.create`.

**Роль:** Основная операция.

**Как работает:** По имени сценария создаёт соответствующий класс, передаёт настройки TDL и параметры сельского/городского распространения.

**Принимает:**

- `cond` — Сценарий радиоканала (str).
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `freq_fad_nlos_model` — TDL-профиль для NLOS (str); по умолчанию `'TDL-A'`.
- `freq_fad_los_model` — TDL-профиль для LOS (str); по умолчанию `'TDL-D'`.
- `ds_profile` — Профиль разброса задержек (str); по умолчанию `'normal'`.
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float); по умолчанию `np.pi / 4`.
- `o2i_model` — Тип модели потерь при проникновении в здание (str); по умолчанию `'low'`.
- `W` — Средняя ширина улицы, м (float); по умолчанию `20.0`.
- `h` — Средняя высота зданий, м (float); по умолчанию `5.0`.

**Возвращает:** Экземпляр модели канала.

**Особенность:** Для `RMa-SC` передаёт аргументы `freq_fad_nlos_model`, `freq_fad_los_model`, `ds_profile`, `los_arrival_angle`, которых нет в конструкторе `RMaModel_SC` (ошибка при создании).

##### `get_available_conds()`

**Сигнатура по коду:** `get_available_conds()`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_available_conds`.

**Роль:** Вспомогательная операция.

**Как работает:** Выводит список сценариев, доступных в фабрике.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; печатает список.

#### ChannelState

Хранит предыдущее положение, LOS/NLOS, shadow fading и O2I для пространственно-согласованного режима.

##### `__init__()`

**Сигнатура по коду:** `__init__(self)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelState.__init__`.

**Роль:** Инициализация.

**Как работает:** Обнуляет прошлое положение, shadow fading, LOS/NLOS и фиксированные значения O2I.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; устанавливает начальное состояние.

##### `reset()`

**Сигнатура по коду:** `reset(self)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelState.reset`.

**Роль:** Вспомогательная операция.

**Как работает:** Повторно вызывает инициализацию, стирая предыдущую историю канала.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** `None`; состояние очищено.

#### ChannelModel

Общие расчёты крупномасштабного радиоканала: условия распространения, затухание, затенение, проникновение и SINR.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, cond_update_period: float=0.0, freq_fad_nlos_model: str='TDL-A', freq_fad_los_model: str='TDL-D', ds_profile: str='normal', los_arrival_angle: float=np.pi / 4, spatial_consistency=False)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Сохраняет BS, интервал обновления, параметры пространственной корреляции и создаёт `TDLModel`. При включённом SC также создаёт `ChannelState`.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `freq_fad_nlos_model` — TDL-профиль для NLOS (str); по умолчанию `'TDL-A'`.
- `freq_fad_los_model` — TDL-профиль для LOS (str); по умолчанию `'TDL-D'`.
- `ds_profile` — Профиль разброса задержек (str); по умолчанию `'normal'`.
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float); по умолчанию `np.pi / 4`.
- `spatial_consistency` — Признак пространственно-согласованного режима; по умолчанию `False`.

**Возвращает:** `None`; заполняет параметры канала.

**Особенность:** Поле `channel_state` назначается два раза подряд; второе присваивание дублирует первое.

##### `_get_sigma_sf()`

**Сигнатура по коду:** `_get_sigma_sf(self, channel_condition: str, d_2D: float=None, UE_height: float=None)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._get_sigma_sf`.

**Роль:** Внутренний метод.

**Как работает:** Выбирает `sigma_SF_LOS` или `sigma_SF_NLOS` по режиму видимости.

**Принимает:**

- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float); по умолчанию `None`.
- `UE_height` — Высота UE над землёй, м (float); по умолчанию `None`.

**Возвращает:** Стандартное отклонение SF в дБ либо `None` для неизвестного состояния.

##### `_calculate_path_loss()`

**Сигнатура по коду:** `_calculate_path_loss(self, UE_ID: int, displacement: float, d_2D: float, d_2D_in: float, d_3D: float, UE_height: float, ue_class: str)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss`.

**Роль:** Внутренний метод.

**Как работает:** Определяет LOS/NLOS, рассчитывает shadow fading, вызывает формулу затухания конкретного сценария и добавляет O2I потери для UE в здании или машине.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_2D_in` — Длина пути сигнала внутри здания, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `ue_class` — Тип размещения UE (например, indoor/car) (str).

**Возвращает:** Суммарные потери на трассе в дБ (`float`).

##### `_calculate_channel_condition()`

**Сигнатура по коду:** `_calculate_channel_condition(self, UE_ID: int, cond_update_period: float, UE_height: float, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_channel_condition`.

**Роль:** Внутренний метод.

**Как работает:** Устанавливает или обновляет LOS/NLOS по вероятности сценария; учитывает период пересчёта и сохранённое состояние UE.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `cond_update_period` — Период обновления состояния канала, мс (float).
- `UE_height` — Высота UE над землёй, м (float).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Строка `"LOS"` или `"NLOS"`.

##### `_calculate_shadow_fading()`

**Сигнатура по коду:** `_calculate_shadow_fading(self, UE_ID: int, displacement: float, channel_condition: str, sigma_SF: float, correlation_dist: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_shadow_fading`.

**Роль:** Внутренний метод.

**Как работает:** Генерирует логнормальное затенение через нормальную случайную составляющую в дБ и использует сохранённое значение между обновлениями.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `sigma_SF` — Sigma sf (float).
- `correlation_dist` — Корреляционное расстояние, м (float).

**Возвращает:** Величина shadow fading в дБ.

##### `_calculate_o2i_building_loss()`

**Сигнатура по коду:** `_calculate_o2i_building_loss(self, UE_ID: int, channel_condition: str, model_type: str, d_2D_in: float, fc_GHz: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss`.

**Роль:** Внутренний метод.

**Как работает:** Складывает потери через внешнюю стену, внутреннюю дистанцию и случайное отклонение; сохранённое значение берёт по UE.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `channel_condition` — Состояние LOS/NLOS (str).
- `model_type` — Тип стенового проникновения (low/high) (str).
- `d_2D_in` — Длина пути сигнала внутри здания, м (float).
- `fc_GHz` — Несущая частота, ГГц (float).

**Возвращает:** Потери проникновения в здание (дБ).

##### `_calculate_o2i_car_loss()`

**Сигнатура по коду:** `_calculate_o2i_car_loss(self, UE_ID: int)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss`.

**Роль:** Внутренний метод.

**Как работает:** Для автомобиля назначает и сохраняет случайные потери с нормальным законом `N(9, 5²)`.

**Принимает:** `UE_ID` — Идентификатор UE (int).

**Возвращает:** Потери проникновения в автомобиль (дБ).

##### `calculate_SINR()`

**Сигнатура по коду:** `calculate_SINR(self, UE_ID: int, displacement: float, d_2D: float, d_2D_in: float, d_3D: float, UE_height: float, ue_class: str)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel.calculate_SINR`.

**Роль:** Основная операция.

**Как работает:** Получает суммарное затухание, считает мощность полезного сигнала и отношение к фиксированной интерференции и тепловому шуму. При `enable_tdl` добавляет массив RB-замираний.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_2D_in` — Длина пути сигнала внутри здания, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `ue_class` — Тип размещения UE (например, indoor/car) (str).

**Возвращает:** Скаляр SINR в дБ или массив SINR по RB при включённом TDL.

**Особенность:** Формула использует фиксированную мощность помех `-95 dBm` и запас 2 dB; ветвь spatial consistency не вызывается из этого метода.

##### `_calculate_correlated_shadow_fading_sc()`

**Сигнатура по коду:** `_calculate_correlated_shadow_fading_sc(self, current_position: np.ndarray, sigma_sf: float, is_los: bool)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_correlated_shadow_fading_sc`.

**Роль:** Внутренний метод.

**Как работает:** Для последующих позиций оценивает корреляцию `exp(−Δd/d_corr)` и обновляет fading авторегрессионной смесью со случайным отклонением.

**Принимает:**

- `current_position` — Координаты UE на плоскости (np.ndarray).
- `sigma_sf` — Стандартное отклонение shadow fading, дБ (float).
- `is_los` — Признак прямой видимости (bool).

**Возвращает:** Пространственно-коррелированное SF в дБ.

**Особенность:** При смещении меньше `0.01 м` возвращается предыдущее значение; это порог обновления в коде.

##### `_calculate_path_loss_sc()`

**Сигнатура по коду:** `_calculate_path_loss_sc(self, d_2D: float, d_2D_in: float, d_3D: float, UE_height: float, ue_class: str, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** Определяет пространственно-согласованный LOS, вызывает SC-формулу затухания сценария, добавляет O2I для помещений и машин.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_2D_in` — Длина пути сигнала внутри здания, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `ue_class` — Тип размещения UE (например, indoor/car) (str).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** Суммарное затухание SC в дБ.

##### `_calculate_spatially_consistent_los()`

**Сигнатура по коду:** `_calculate_spatially_consistent_los(self, current_position: np.ndarray, base_los_prob: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_spatially_consistent_los`.

**Роль:** Внутренний метод.

**Как работает:** Сохраняет LOS при малом смещении; иначе с вероятностью пространственной корреляции удерживает предыдущий режим либо переигрывает его по `base_los_prob`.

**Принимает:** `current_position` — Координаты UE на плоскости (np.ndarray); `base_los_prob` — Базовая вероятность прямой видимости (float).

**Возвращает:** `True` для LOS, `False` для NLOS.

**Особенность:** При первом вызове LOS определяется сравнением `base_los_prob > 0.5`, без случайного выбора. Поле `first_call` не сбрасывается, поэтому ветвь повторного обновления LOS остаётся недоступной.

##### `_calculate_o2i_building_loss_sc()`

**Сигнатура по коду:** `_calculate_o2i_building_loss_sc(self, model_type: str, d_2D_in: float, fc_GHz: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** При первом обращении рассчитывает потери через стену и внутри здания; затем возвращает сохранённое значение.

**Принимает:**

- `model_type` — Тип стенового проникновения (low/high) (str).
- `d_2D_in` — Длина пути сигнала внутри здания, м (float).
- `fc_GHz` — Несущая частота, ГГц (float).

**Возвращает:** O2I-потери в дБ.

##### `_calculate_o2i_car_loss_sc()`

**Сигнатура по коду:** `_calculate_o2i_car_loss_sc(self)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** При первом обращении создаёт случайные потери автомобиля, потом переиспользует их.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** O2I-потери в дБ.

##### `_calculate_los_path_loss_sc()`

**Сигнатура по коду:** `_calculate_los_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_los_path_loss_sc`.

**Роль:** Интерфейсный метод.

**Как работает:** Объявляет SC-формулу прямой видимости, которую определяет конкретный сценарий.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** В базовом классе выбрасывает `NotImplementedError`.

##### `_calculate_nlos_path_loss_sc()`

**Сигнатура по коду:** `_calculate_nlos_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_nlos_path_loss_sc`.

**Роль:** Интерфейсный метод.

**Как работает:** Объявляет SC-формулу непрямой видимости для реализаций-наследников.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** В базовом классе выбрасывает `NotImplementedError`.

#### RMaModel

Формулы сценария Rural Macro: сельская макросота.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, W: float=20.0, h: float=5.0, cond_update_period: float=0.0, freq_fad_nlos_model: str='TDL-A', freq_fad_los_model: str='TDL-D', ds_profile: str='normal', los_arrival_angle: float=np.pi / 4)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Задаёт коэффициенты и геометрические параметры данного сценария, затем инициализирует общую модель канала.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `W` — Средняя ширина улицы, м (float); по умолчанию `20.0`.
- `h` — Средняя высота зданий, м (float); по умолчанию `5.0`.
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `freq_fad_nlos_model` — TDL-профиль для NLOS (str); по умолчанию `'TDL-A'`.
- `freq_fad_los_model` — TDL-профиль для LOS (str); по умолчанию `'TDL-D'`.
- `ds_profile` — Профиль разброса задержек (str); по умолчанию `'normal'`.
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float); по умолчанию `np.pi / 4`.

**Возвращает:** `None`; инициализирует объект сценария.

##### `_get_sigma_sf()`

**Сигнатура по коду:** `_get_sigma_sf(self, channel_condition: str, d_2D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel._get_sigma_sf`.

**Роль:** Внутренний метод.

**Как работает:** Возвращает стандартное отклонение shadow fading, выбранное по состоянию канала и параметрам сценария.

**Принимает:**

- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Стандартное отклонение в дБ.

##### `_calculate_breakpoint_distance()`

**Сигнатура по коду:** `_calculate_breakpoint_distance(self, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel._calculate_breakpoint_distance`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает расстояние излома, отделяющее участки формулы распространения LOS.

**Принимает:** `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Расстояние breakpoint в метрах.

##### `_calculate_los_probability()`

**Сигнатура по коду:** `_calculate_los_probability(self, UE_height: float, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel._calculate_los_probability`.

**Роль:** Внутренний метод.

**Как работает:** По расстоянию BS–UE и параметрам сценария вычисляет вероятность прямой видимости.

**Принимает:** `UE_height` — Высота UE над землёй, м (float); `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Вероятность LOS от 0 до 1.

##### `_calculate_los_path_loss()`

**Сигнатура по коду:** `_calculate_los_path_loss(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel._calculate_los_path_loss`.

**Роль:** Внутренний метод.

**Как работает:** Подставляет расстояния, частоту и высоты в выражение затухания для прямой видимости данного сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss()`

**Сигнатура по коду:** `_calculate_nlos_path_loss(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel._calculate_nlos_path_loss`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает затухание для непрямой видимости с учётом формулы LOS и условий сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание NLOS в дБ.

#### UMaModel

Формулы сценария Urban Macro: городская макросота.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, cond_update_period: float=0.0, o2i_model: str='low', freq_fad_nlos_model: str='TDL-A', freq_fad_los_model: str='TDL-D', ds_profile: str='normal', los_arrival_angle: float=np.pi / 4)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Задаёт коэффициенты и геометрические параметры данного сценария, затем инициализирует общую модель канала.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `o2i_model` — Тип модели потерь при проникновении в здание (str); по умолчанию `'low'`.
- `freq_fad_nlos_model` — TDL-профиль для NLOS (str); по умолчанию `'TDL-A'`.
- `freq_fad_los_model` — TDL-профиль для LOS (str); по умолчанию `'TDL-D'`.
- `ds_profile` — Профиль разброса задержек (str); по умолчанию `'normal'`.
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float); по умолчанию `np.pi / 4`.

**Возвращает:** `None`; инициализирует объект сценария.

##### `_calculate_breakpoint_distance()`

**Сигнатура по коду:** `_calculate_breakpoint_distance(self, UE_height: float, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel._calculate_breakpoint_distance`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает расстояние излома, отделяющее участки формулы распространения LOS.

**Принимает:** `UE_height` — Высота UE над землёй, м (float); `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Расстояние breakpoint в метрах.

##### `_calculate_los_probability()`

**Сигнатура по коду:** `_calculate_los_probability(self, UE_height: float, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel._calculate_los_probability`.

**Роль:** Внутренний метод.

**Как работает:** По расстоянию BS–UE и параметрам сценария вычисляет вероятность прямой видимости.

**Принимает:** `UE_height` — Высота UE над землёй, м (float); `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Вероятность LOS от 0 до 1.

##### `_calculate_los_path_loss()`

**Сигнатура по коду:** `_calculate_los_path_loss(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel._calculate_los_path_loss`.

**Роль:** Внутренний метод.

**Как работает:** Подставляет расстояния, частоту и высоты в выражение затухания для прямой видимости данного сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss()`

**Сигнатура по коду:** `_calculate_nlos_path_loss(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel._calculate_nlos_path_loss`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает затухание для непрямой видимости с учётом формулы LOS и условий сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание NLOS в дБ.

#### UMiModel

Формулы сценария Urban Micro: городская микросота.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, cond_update_period: float=0.0, o2i_model: str='low', freq_fad_nlos_model: str='TDL-A', freq_fad_los_model: str='TDL-D', ds_profile: str='normal', los_arrival_angle: float=np.pi / 4)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Задаёт коэффициенты и геометрические параметры данного сценария, затем инициализирует общую модель канала.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `o2i_model` — Тип модели потерь при проникновении в здание (str); по умолчанию `'low'`.
- `freq_fad_nlos_model` — TDL-профиль для NLOS (str); по умолчанию `'TDL-A'`.
- `freq_fad_los_model` — TDL-профиль для LOS (str); по умолчанию `'TDL-D'`.
- `ds_profile` — Профиль разброса задержек (str); по умолчанию `'normal'`.
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float); по умолчанию `np.pi / 4`.

**Возвращает:** `None`; инициализирует объект сценария.

##### `_calculate_breakpoint_distance()`

**Сигнатура по коду:** `_calculate_breakpoint_distance(self, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel._calculate_breakpoint_distance`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает расстояние излома, отделяющее участки формулы распространения LOS.

**Принимает:** `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Расстояние breakpoint в метрах.

##### `_calculate_los_probability()`

**Сигнатура по коду:** `_calculate_los_probability(self, UE_height: float, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel._calculate_los_probability`.

**Роль:** Внутренний метод.

**Как работает:** По расстоянию BS–UE и параметрам сценария вычисляет вероятность прямой видимости.

**Принимает:** `UE_height` — Высота UE над землёй, м (float); `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Вероятность LOS от 0 до 1.

##### `_calculate_los_path_loss()`

**Сигнатура по коду:** `_calculate_los_path_loss(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel._calculate_los_path_loss`.

**Роль:** Внутренний метод.

**Как работает:** Подставляет расстояния, частоту и высоты в выражение затухания для прямой видимости данного сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss()`

**Сигнатура по коду:** `_calculate_nlos_path_loss(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel._calculate_nlos_path_loss`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает затухание для непрямой видимости с учётом формулы LOS и условий сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание NLOS в дБ.

#### TDLModel

Частотно-селективные замирания по профилям TDL на ресурсных блоках.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, nlos_model: str='TDL-A', los_model: str='TDL-D', delay_spread_profile: str='normal', los_arrival_angle: float=np.pi / 4)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel.__init__`.

**Роль:** Инициализация.

**Как работает:** Загружает таблицы TDL, задержки и мощности лучей, вычисляет частоты RB и длину волны, создаёт состояние коэффициентов по UE.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `nlos_model` — TDL-профиль для NLOS (str); по умолчанию `'TDL-A'`.
- `los_model` — TDL-профиль для LOS (str); по умолчанию `'TDL-D'`.
- `delay_spread_profile` — Профиль задержек (str); по умолчанию `'normal'`.
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float); по умолчанию `np.pi / 4`.

**Возвращает:** `None`; подготавливает профиль многолучевого канала.

##### `_validate_parameters()`

**Сигнатура по коду:** `_validate_parameters(self, nlos_model: str, los_model: str, delay_spread_profile: str, los_arrival_angle: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._validate_parameters`.

**Роль:** Внутренний метод.

**Как работает:** Проверяет совместимость LOS/NLOS TDL-профилей, имя профиля задержек и диапазон угла.

**Принимает:**

- `nlos_model` — TDL-профиль для NLOS (str).
- `los_model` — TDL-профиль для LOS (str).
- `delay_spread_profile` — Профиль задержек (str).
- `los_arrival_angle` — Угол прихода прямой компоненты, рад (float).

**Возвращает:** `None`; при ошибке — исключение.

##### `_get_params_value()`

**Сигнатура по коду:** `_get_params_value(self, model: str, key: str)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._get_params_value`.

**Роль:** Внутренний метод.

**Как работает:** Берёт параметр по ключу из таблицы модели TDL.

**Принимает:** `model` — Имя модели, класс модели или объект генератора — по сигнатуре метода (str); `key` — Имя поля в таблице (str).

**Возвращает:** Значение `params_table[model].get(key)`.

##### `_get_delay_spread()`

**Сигнатура по коду:** `_get_delay_spread(self, channel_model: str, ue_class: str)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._get_delay_spread`.

**Роль:** Внутренний метод.

**Как работает:** Выбирает таблицу разброса задержек по сценарию, типу UE и профилю short/normal/long.

**Принимает:** `channel_model` — Название крупномасштабного сценария (str); `ue_class` — Тип размещения UE (например, indoor/car) (str).

**Возвращает:** Разброс задержек в наносекундах.

##### `_get_rb_frequencies()`

**Сигнатура по коду:** `_get_rb_frequencies(self)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._get_rb_frequencies`.

**Роль:** Внутренний метод.

**Как работает:** Строит относительные частоты RB с шагом `RB_BANDWIDTH` вокруг центральной частоты.

**Принимает:** дополнительных аргументов нет.

**Возвращает:** Массив `numpy.ndarray` частот в Гц.

##### `_jakes_sum_of_sinusoids()`

**Сигнатура по коду:** `_jakes_sum_of_sinusoids(self, f_doppler: float, alpha: np.ndarray, phi: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._jakes_sum_of_sinusoids`.

**Роль:** Внутренний метод.

**Как работает:** Суммирует комплексные гармоники Джейкса по фазам, направлениям и доплеровской частоте; время берёт из `GLOBALS.CURRENT_TIME`.

**Принимает:**

- `f_doppler` — Доплеровская частота, Гц (float).
- `alpha` — Углы гармоник, рад (np.ndarray).
- `phi` — Начальные фазы гармоник, рад (np.ndarray).

**Возвращает:** Комплексные коэффициенты каждого луча.

**Особенность:** Время берётся из глобальной переменной `GLOBALS.CURRENT_TIME`, а не из аргумента.

##### `_calculate_freq_response()`

**Сигнатура по коду:** `_calculate_freq_response(self, UE_ID: int, channel_cond: str, channel_model: str, ue_class: str)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._calculate_freq_response`.

**Роль:** Внутренний метод.

**Как работает:** Хранит фазы лучей по UE; выбирает LOS/NLOS задержки и мощности, учитывает доплер по скорости UE и суммирует отклики лучей для каждого RB.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `channel_cond` — Состояние LOS/NLOS (str).
- `channel_model` — Название крупномасштабного сценария (str).
- `ue_class` — Тип размещения UE (например, indoor/car) (str).

**Возвращает:** Комплексный массив `H(f)` по ресурсным блокам.

##### `calculate_channel_gain()`

**Сигнатура по коду:** `calculate_channel_gain(self, UE_ID: int, channel_cond: str, channel_model: str, ue_class: str)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::TDLModel.calculate_channel_gain`.

**Роль:** Основная операция.

**Как работает:** Берёт квадрат модуля `H(f)`, нормирует на среднее и переводит результат в дБ.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `channel_cond` — Состояние LOS/NLOS (str).
- `channel_model` — Название крупномасштабного сценария (str).
- `ue_class` — Тип размещения UE (например, indoor/car) (str).

**Возвращает:** Массив относительных коэффициентов усиления по RB (дБ).

**Особенность:** Нормировка делит на среднюю мощность массива RB, поэтому возвращается относительное изменение по частоте.

#### RMaModel_SC

Вариант Rural Macro с дополнительными методами пространственной согласованности.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, W: float=20.0, h: float=5.0, cond_update_period: float=0.0, spatial_consistency: bool=True)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC.__init__`.

**Роль:** Инициализация.

**Как работает:** Задаёт коэффициенты и геометрические параметры данного сценария, затем инициализирует общую модель канала.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `W` — Средняя ширина улицы, м (float); по умолчанию `20.0`.
- `h` — Средняя высота зданий, м (float); по умолчанию `5.0`.
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `spatial_consistency` — Включить сохранение состояния SC (bool); по умолчанию `True`.

**Возвращает:** `None`; инициализирует объект сценария.

##### `_get_sigma_sf()`

**Сигнатура по коду:** `_get_sigma_sf(self, channel_condition: str, d_2D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._get_sigma_sf`.

**Роль:** Внутренний метод.

**Как работает:** Возвращает стандартное отклонение shadow fading, выбранное по состоянию канала и параметрам сценария.

**Принимает:**

- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Стандартное отклонение в дБ.

##### `_calculate_breakpoint_distance()`

**Сигнатура по коду:** `_calculate_breakpoint_distance(self, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_breakpoint_distance`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает расстояние излома, отделяющее участки формулы распространения LOS.

**Принимает:** `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Расстояние breakpoint в метрах.

##### `_calculate_los_probability()`

**Сигнатура по коду:** `_calculate_los_probability(self, d_2D)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_probability`.

**Роль:** Внутренний метод.

**Как работает:** По расстоянию BS–UE и параметрам сценария вычисляет вероятность прямой видимости.

**Принимает:** `d_2D` — Горизонтальное расстояние BS–UE, м.

**Возвращает:** Вероятность LOS от 0 до 1.

##### `_calculate_los_path_loss_time_based()`

**Сигнатура по коду:** `_calculate_los_path_loss_time_based(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_time_based`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает потери LOS в варианте обновления, зависящем от времени и сохранённых характеристик канала.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss_time_based()`

**Сигнатура по коду:** `_calculate_nlos_path_loss_time_based(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_time_based`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает потери NLOS по временной ветви сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание NLOS в дБ.

##### `_calculate_los_path_loss_sc()`

**Сигнатура по коду:** `_calculate_los_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** Для режима SC рассчитывает затухание LOS и учитывает пространственно-коррелированное shadow fading.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss_sc()`

**Сигнатура по коду:** `_calculate_nlos_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** Для режима SC рассчитывает затухание NLOS и учитывает пространственно-коррелированное shadow fading.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** Затухание NLOS в дБ.

##### `_calculate_los_path_loss_base()`

**Сигнатура по коду:** `_calculate_los_path_loss_base(self, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_base`.

**Роль:** Внутренний метод.

**Как работает:** Считает базовый путь распространения LOS без случайного затенения.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Детерминированные потери LOS в дБ.

#### UMaModel_SC

Вариант Urban Macro с дополнительными методами пространственной согласованности.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, cond_update_period: float=0.0, o2i_model: str='low', spatial_consistency: bool=True, UE_height=1.5)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC.__init__`.

**Роль:** Инициализация.

**Как работает:** Задаёт коэффициенты и геометрические параметры данного сценария, затем инициализирует общую модель канала.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `o2i_model` — Тип модели потерь при проникновении в здание (str); по умолчанию `'low'`.
- `spatial_consistency` — Признак пространственно-согласованного режима (bool); по умолчанию `True`.
- `UE_height` — Высота UE над землёй, м; по умолчанию `1.5`.

**Возвращает:** `None`; инициализирует объект сценария.

##### `_calculate_breakpoint_distance()`

**Сигнатура по коду:** `_calculate_breakpoint_distance(self, UE_height: float, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_breakpoint_distance`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает расстояние излома, отделяющее участки формулы распространения LOS.

**Принимает:** `UE_height` — Высота UE над землёй, м (float); `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Расстояние breakpoint в метрах.

##### `_calculate_los_probability()`

**Сигнатура по коду:** `_calculate_los_probability(self, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_probability`.

**Роль:** Внутренний метод.

**Как работает:** По расстоянию BS–UE и параметрам сценария вычисляет вероятность прямой видимости.

**Принимает:** `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Вероятность LOS от 0 до 1.

##### `_calculate_los_path_loss_time_based()`

**Сигнатура по коду:** `_calculate_los_path_loss_time_based(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_time_based`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает потери LOS в варианте обновления, зависящем от времени и сохранённых характеристик канала.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss_time_based()`

**Сигнатура по коду:** `_calculate_nlos_path_loss_time_based(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_time_based`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает потери NLOS по временной ветви сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание NLOS в дБ.

##### `_calculate_los_path_loss_sc()`

**Сигнатура по коду:** `_calculate_los_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** Для режима SC рассчитывает затухание LOS и учитывает пространственно-коррелированное shadow fading.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss_sc()`

**Сигнатура по коду:** `_calculate_nlos_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** Для режима SC рассчитывает затухание NLOS и учитывает пространственно-коррелированное shadow fading.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** Затухание NLOS в дБ.

#### UMiModel_SC

Вариант Urban Micro с дополнительными методами пространственной согласованности.

##### `__init__()`

**Сигнатура по коду:** `__init__(self, bs: BaseStation, cond_update_period: float=0.0, o2i_model: str='low', spatial_consistency: bool=True, UE_height=1.5)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC.__init__`.

**Роль:** Инициализация.

**Как работает:** Задаёт коэффициенты и геометрические параметры данного сценария, затем инициализирует общую модель канала.

**Принимает:**

- `bs` — Объект базовой станции (BaseStation).
- `cond_update_period` — Период обновления состояния канала, мс (float); по умолчанию `0.0`.
- `o2i_model` — Тип модели потерь при проникновении в здание (str); по умолчанию `'low'`.
- `spatial_consistency` — Признак пространственно-согласованного режима (bool); по умолчанию `True`.
- `UE_height` — Высота UE над землёй, м; по умолчанию `1.5`.

**Возвращает:** `None`; инициализирует объект сценария.

##### `_calculate_breakpoint_distance()`

**Сигнатура по коду:** `_calculate_breakpoint_distance(self, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_breakpoint_distance`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает расстояние излома, отделяющее участки формулы распространения LOS.

**Принимает:** `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Расстояние breakpoint в метрах.

##### `_calculate_los_probability()`

**Сигнатура по коду:** `_calculate_los_probability(self, d_2D: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_probability`.

**Роль:** Внутренний метод.

**Как работает:** По расстоянию BS–UE и параметрам сценария вычисляет вероятность прямой видимости.

**Принимает:** `d_2D` — Горизонтальное расстояние BS–UE, м (float).

**Возвращает:** Вероятность LOS от 0 до 1.

##### `_calculate_los_path_loss_time_based()`

**Сигнатура по коду:** `_calculate_los_path_loss_time_based(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_time_based`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает потери LOS в варианте обновления, зависящем от времени и сохранённых характеристик канала.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss_time_based()`

**Сигнатура по коду:** `_calculate_nlos_path_loss_time_based(self, UE_ID: int, displacement: float, channel_condition: str, d_2D: float, d_3D: float, UE_height: float)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_time_based`.

**Роль:** Внутренний метод.

**Как работает:** Рассчитывает потери NLOS по временной ветви сценария.

**Принимает:**

- `UE_ID` — Идентификатор UE (int).
- `displacement` — Смещение UE между обновлениями, м (float).
- `channel_condition` — Состояние LOS/NLOS (str).
- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).

**Возвращает:** Затухание NLOS в дБ.

##### `_calculate_los_path_loss_sc()`

**Сигнатура по коду:** `_calculate_los_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** Для режима SC рассчитывает затухание LOS и учитывает пространственно-коррелированное shadow fading.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** Затухание LOS в дБ.

##### `_calculate_nlos_path_loss_sc()`

**Сигнатура по коду:** `_calculate_nlos_path_loss_sc(self, d_2D: float, d_3D: float, UE_height: float, current_position: np.ndarray)`.
**Источник:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_sc`.

**Роль:** Внутренний метод.

**Как работает:** Для режима SC рассчитывает затухание NLOS и учитывает пространственно-коррелированное shadow fading.

**Принимает:**

- `d_2D` — Горизонтальное расстояние BS–UE, м (float).
- `d_3D` — Трёхмерное расстояние BS–UE, м (float).
- `UE_height` — Высота UE над землёй, м (float).
- `current_position` — Координаты UE на плоскости (np.ndarray).

**Возвращает:** Затухание NLOS в дБ.


### Связь с другими частями симулятора

`BS_MODULE.py` создаёт экземпляр модели канала для базовой станции. `UE_MODULE.py` использует `calculate_SINR` для оценки качества связи; модель читает геометрию UE–BS и настройки базовой станции.

## 6. Один типичный сценарий

1. `UserEquipment.UPD_CH_QUALITY(channel_update_interval)` вызывается из `UECollection.UPDATE_ALL_USERS`.
2. UE рассчитывает расстояния, высоту при необходимости и displacement=`max(velocity,0.2)*channel_update_interval/1000`.
3. Вызывает модель своей serving_bs: `calculate_SINR(UE_ID,displacement,d_2D,d_2D_in,d_3D,UE_height,ue_class)`.
4. ChannelModel выполняет `_calculate_path_loss` с LOS/SF/O2I, затем рассчитывает SINR с постоянной помехой -95 dBm и запасом 2 дБ.
5. При enable_tdl добавляет массив gain_on_RB и возвращает ndarray; иначе возвращает скаляр. UE усредняет результат в SINR и вычисляет CQI.

SC-путь из стандартного calculate_SINR не вызывается. Методы SC и классы сохранены, но недоступность штатного маршрута и ошибки конструкторов подтверждены в audit. TDL берёт абсолютное время из GLOBALS.CURRENT_TIME, которое обновляет цикл SimulationManager.

## 7. Ограничения и открытые вопросы
- Статусы замечаний уточнены в audit.md: различаются воспроизведённые ошибки, подтверждённые конструкции кода и несогласованные архитектурные решения. Наличие метода не означает прохождение всех его ветвей.
- Сигнатуры и импорты сверены с кодом ветки dev. Симуляция запускалась автором; результаты дополнительных проверок приведены в разделе 4 и audit.md.
- File-режим, DTO и интерфейс RNG остаются предложениями в общей спецификации; исходники подтверждают текущую реализацию, но не согласование будущих изменений.

## 8. Связанные материалы
- [Диаграммы](docs/diagrams.md)
- [Ревизия](docs/audit.md)
- [План развития](docs/features.md)
- [Единая спецификация model/file](../integration_report.md)
