# Ревизия модуля CHANNEL_MODEL

> Ответственный: Кутенков Андрей | Проверено по коду: `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b` | Обновлено: 2026-10-17

Проверены `PyScheduler/CHANNEL_MODEL.py`, связанные вызовы SIMULATION_MANAGER/UE_MODULE и существующие тесты. «Подтверждено по коду» означает наблюдаемую конструкцию, а не автоматическое признание багом. «Подтверждённый баг» ниже имеет воспроизведение. Производительность не измерялась. Симуляция запускалась автором. Дополнительная ревизия включает анализ кода и целевые проверки ниже; исходный код при подготовке документации не изменялся.

Для блоков воспроизведения: из каталога PyScheduler запустить Python и выполнить общий импорт, затем блок нужной карточки:

```python
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from CHANNEL_MODEL import *
```

## 1. Реестр наблюдений

| ID | Категория | Объект | Статус |
|---|---|---|---|
| AUD-CHN-001 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_conditions` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-002 | подтверждённый баг | `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.create` | подтверждено, воспроизведено |
| AUD-CHN-003 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_available_conds` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-004 | вопрос о назначении вспомогательного API | `PyScheduler/CHANNEL_MODEL.py::ChannelState.reset` | открыт вопрос; существование и отсутствие штатного подключения проверены |
| AUD-CHN-005 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::ChannelModel.__init__` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-006 | вопрос о валидации | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._get_sigma_sf` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-007 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-008 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_channel_condition` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-009 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_shadow_fading` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-010 | подтверждённый баг | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss` | подтверждено, воспроизведено |
| AUD-CHN-011 | подтверждённый баг | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss` | подтверждено, воспроизведено |
| AUD-CHN-012 | ограничение модели | `PyScheduler/CHANNEL_MODEL.py::ChannelModel.calculate_SINR` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-013 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_correlated_shadow_fading_sc` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-014 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-015 | подтверждённый баг | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_spatially_consistent_los` | подтверждено, воспроизведено |
| AUD-CHN-016 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-017 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-018 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::TDLModel._jakes_sum_of_sinusoids` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-019 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::TDLModel._calculate_freq_response` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-020 | технический долг / ограничение | `PyScheduler/CHANNEL_MODEL.py::TDLModel.calculate_channel_gain` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-CHN-021 | подтверждённый баг | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC.__init__` | подтверждено, воспроизведено |
| AUD-CHN-022 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._get_sigma_sf` | подтверждено для штатного маршрута |
| AUD-CHN-023 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_breakpoint_distance` | подтверждено для штатного маршрута |
| AUD-CHN-024 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_probability` | подтверждено для штатного маршрута |
| AUD-CHN-025 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_time_based` | подтверждено для штатного маршрута |
| AUD-CHN-026 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_time_based` | подтверждено для штатного маршрута |
| AUD-CHN-027 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-028 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-029 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_base` | подтверждено для штатного маршрута |
| AUD-CHN-030 | подтверждённый баг | `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC.__init__` | подтверждено, воспроизведено |
| AUD-CHN-031 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_breakpoint_distance` | подтверждено для штатного маршрута |
| AUD-CHN-032 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_probability` | подтверждено для штатного маршрута |
| AUD-CHN-033 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_time_based` | подтверждено для штатного маршрута |
| AUD-CHN-034 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_time_based` | подтверждено для штатного маршрута |
| AUD-CHN-035 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-036 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-037 | подтверждённый баг | `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC.__init__` | подтверждено, воспроизведено |
| AUD-CHN-038 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_breakpoint_distance` | подтверждено для штатного маршрута |
| AUD-CHN-039 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_probability` | подтверждено для штатного маршрута |
| AUD-CHN-040 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_time_based` | подтверждено для штатного маршрута |
| AUD-CHN-041 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_time_based` | подтверждено для штатного маршрута |
| AUD-CHN-042 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_sc` | подтверждено для штатного маршрута |
| AUD-CHN-043 | ограничение штатной интеграции | `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_sc` | подтверждено для штатного маршрута |

## 2. Карточки наблюдений

### AUD-CHN-001 — ChannelInterface.get_conditions
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_conditions`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** В реестре фабрики есть RMa, UMa, UMi, RMa-SC, но отсутствуют UMa-SC и UMi-SC при наличии соответствующих классов.
- **Доказательство:** Метод найден в исходнике (строки 27–32); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_conditions`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `def get_conditions():     conditions = {'RMa': RMaModel, 'UMa': UMaModel, 'UMi': UMiModel, 'RMa-SC': RMaModel_SC}     return conditions`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-002 — ChannelInterface.create
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.create`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Для `RMa-SC` собирает стандартные TDL-аргументы, которых конструктор `RMaModel_SC` не принимает, поэтому создание этого сценария завершается `TypeError`.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "TypeError", "message": "RMaModel_SC.__init__() got an unexpected keyword argument 'freq_fad_nlos_model'"}`. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.create`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `ChannelInterface.validate_parameters(cond, cond_update_period, freq_fad_nlos_model, freq_fad_los_model, ds_profile, los_arrival_angle)     conds = self.get_conditions()     if cond not in conds:`.
- **Влияние:** Заявленный фабрикой RMa-SC нельзя создать штатным способом.
- **Как проверить:** Создать фабрикой RMa-SC, UMa-SC, UMi-SC; записать аргументы базового конструктора и тип tdl_profile. Флаг bool не должен попадать вместо имени профиля; spy фиксирует вызов SC-ветви SINR.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
ChannelInterface(NS()).create("RMa-SC")
```

### AUD-CHN-003 — ChannelInterface.get_available_conds
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_available_conds`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** В определении нет `self` и `@staticmethod`; вывести список можно через класс, но не через экземпляр фабрики.
- **Доказательство:** Метод найден в исходнике (строки 124–129); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelInterface.get_available_conds`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `conds = list(ChannelInterface.get_conditions().keys())     print('Доступные модели:', conds)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Вызвать метод через класс и экземпляр с одинаковыми аргументами; зафиксировать фактическую сигнатуру и наличие TypeError. Сначала уточнить, предусмотрен ли вызов через экземпляр.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-004 — ChannelState.reset
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelState.reset`.
- **Категория:** вопрос о назначении вспомогательного API.
- **Наблюдение:** Метод очистки реализован, но в обычном маршруте обновления состояния канала не вызывается; жизненный цикл SC пока с ним не связан.
- **Доказательство:** Метод найден в исходнике (строки 148–150); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Поиск вызовов в проекте не выявил подключения этой обёртки/очистки в SimulationManager; это не основание удалять публичный API. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelState.reset`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `self.__init__()`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** открыт вопрос; существование и отсутствие штатного подключения проверены.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-005 — ChannelModel.__init__
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel.__init__`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** `self.channel_state` назначается дважды подряд; сценарии также наследуют общие словари состояния для всех экземпляров канала.
- **Доказательство:** Метод найден в исходнике (строки 162–210); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel.__init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `self.bs = bs     self.cond_update_period = cond_update_period     self.spatial_consistency = spatial_consistency`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** С spy на ChannelState выполнить конструктор ChannelModel с spatial_consistency=True и корректной BS; ожидается два вызова ChannelState, из которых первый объект заменяется. Два экземпляра ChannelModel также имеют общий CHANNEL_COND_INFO. Разделить дублирующее создание и меж-BS изоляцию как разные исправления.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-006 — ChannelModel._get_sigma_sf
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._get_sigma_sf`.
- **Категория:** вопрос о валидации.
- **Наблюдение:** Для channel_condition вне LOS/NLOS метод не имеет явного возврата или исключения и возвращает None; штатные вызывающие ветви используют LOS/NLOS.
- **Доказательство:** Метод найден в исходнике (строки 212–234); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._get_sigma_sf`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if channel_condition == 'LOS':         return getattr(self, 'sigma_SF_LOS', None)     elif channel_condition == 'NLOS':`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** У экземпляра ChannelModel.__new__(ChannelModel) вызвать _get_sigma_sf("UNKNOWN"): возвращается None. Для валидных LOS/NLOS использовать полностью созданную модель и проверить положительное sigma. Решить, нужен ли явный ValueError для внешнего невалидного ввода.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-007 — ChannelModel._calculate_path_loss
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Состояние LOS/SF/O2I берётся из словарей класса с ключом только `UE_ID`; при нескольких BS данные одного UE способны пересекаться.
- **Доказательство:** Метод найден в исходнике (строки 236–301); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `channel_condition = self._calculate_channel_condition(UE_ID, self.cond_update_period, UE_height, d_2D)     sigma_sf = self._get_sigma_sf(channel_condition, d_2D, UE_height)     correlation_dist = self.correlation_dist_LOS if channel_condition == 'LOS' else self.correlation_dist_NLOS`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Рассчитать один ue_id для двух BS, поменять порядок BS и сбросить одну радиолинию. Сравнить LOS/SF/O2I каждой линии; изменения второй линии означают пересечение состояния.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-008 — ChannelModel._calculate_channel_condition
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_channel_condition`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Хранит состояние в общем словаре `CHANNEL_COND_INFO[UE_ID]` и смотрит на глобальное время; нужна изоляция по радиолинии BS–UE.
- **Доказательство:** Метод найден в исходнике (строки 303–345); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_channel_condition`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `channel_condition = None     not_found = False     update = False`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Рассчитать один ue_id для двух BS, поменять порядок BS и сбросить одну радиолинию. Сравнить LOS/SF/O2I каждой линии; изменения второй линии означают пересечение состояния.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-009 — ChannelModel._calculate_shadow_fading
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_shadow_fading`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Корреляция хранится в общем `SHADOW_FADING_INFO[UE_ID]`, поэтому результаты разных BS для одного UE не изолированы.
- **Доказательство:** Метод найден в исходнике (строки 347–392); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_shadow_fading`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `shadow_fading = None     not_found = False     new_condition = False`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Рассчитать один ue_id для двух BS, поменять порядок BS и сбросить одну радиолинию. Сравнить LOS/SF/O2I каждой линии; изменения второй линии означают пересечение состояния.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-010 — ChannelModel._calculate_o2i_building_loss
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Для неизвестного `model_type` переменные `PL_tw` и `sigma_p` остаются неинициализированными и вызывают `UnboundLocalError`.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "UnboundLocalError", "message": "cannot access local variable 'sigma_p' where it is not associated with a value"}`. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `o2i_loss = None     not_found = False     new_condition = False`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Вызвать O2I с каждым поддержанным model_type и неизвестной строкой; неизвестное значение должно давать явную ошибку валидации, без UnboundLocalError.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
ChannelModel.O2I_INFO.clear()
c=ChannelModel.__new__(ChannelModel)
c._calculate_o2i_building_loss(1,"LOS","bad",1,2)
```

### AUD-CHN-011 — ChannelModel._calculate_o2i_car_loss
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Повторно использует `O2I_INFO[UE_ID]` без признака типа проникновения; значение может быть записано другой O2I-ветвью.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"result": {"building": 19.825319569847395, "car": 19.825319569847395, "same": "True"}}`. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `o2i_loss = None     not_found = False     if UE_ID in ChannelModel.O2I_INFO:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Рассчитать один ue_id для двух BS, поменять порядок BS и сбросить одну радиолинию. Сравнить LOS/SF/O2I каждой линии; изменения второй линии означают пересечение состояния.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
ChannelModel.O2I_INFO.clear(); c=ChannelModel.__new__(ChannelModel)
with patch.object(np.random,"normal",return_value=3):
    print(c._calculate_o2i_building_loss(1,"LOS","low",10,2))
    print(c._calculate_o2i_car_loss(1))
```

### AUD-CHN-012 — ChannelModel.calculate_SINR
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel.calculate_SINR`.
- **Категория:** ограничение модели.
- **Наблюдение:** P_interference=-95 и interference_margin=2 заданы константами; комментарий TODO указывает отсутствие интеграции SC. Это ограничение текущего сценария, не доказательство ошибки всех SINR.
- **Доказательство:** Метод найден в исходнике (строки 482–534); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel.calculate_SINR`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `path_loss = self._calculate_path_loss(UE_ID, displacement, d_2D, d_2D_in, d_3D, UE_height, ue_class)     cable_loss = 2     interference_margin = 2`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-013 — ChannelModel._calculate_correlated_shadow_fading_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_correlated_shadow_fading_sc`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Для движения меньше `0.01 м` значение принудительно сохраняется; порог задан вручную, а не параметром корреляционной модели.
- **Доказательство:** Метод найден в исходнике (строки 536–573); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_correlated_shadow_fading_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `correlation_distance = self.sf_correlation_distance_los if is_los else self.sf_correlation_distance_nlos     if self.channel_state.last_position is None:         self.channel_state.last_shadow_fading = np.random.normal(0, sigma_sf)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Сравнить перемещения 0, 0.009, 0.01 и 0.011 м при фиксированном RNG; записать SF и проверить непрерывность/допустимый порог с владельцем модели.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-014 — ChannelModel._calculate_path_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Основной `calculate_SINR` обращается только к `_calculate_path_loss`; SC-версия не вызывается из стандартного потока.
- **Доказательство:** Метод найден в исходнике (строки 575–603); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_path_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `base_los_prob = self._calculate_los_probability(d_2D)     is_LOS = self._calculate_spatially_consistent_los(current_position, base_los_prob)     if is_LOS:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-015 — ChannelModel._calculate_spatially_consistent_los
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_spatially_consistent_los`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** В первой ветви устанавливает `last_los_state`, но ни разу не переключает `first_call` в `False`; коррелированная ветвь остаётся недоступной.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"result": {"states": [true, false], "first_call": true, "position_is_none": true}}`. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_spatially_consistent_los`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if self.channel_state.first_call or self.channel_state.last_position is None or self.channel_state.last_los_state is None:         self.channel_state.last_los_state = base_los_prob > 0.5         return self.channel_state.last_los_state`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Два последовательных вызова LOS при фиксированном UE и контролируемом RNG; после первого first_call должен быть False, второй должен попасть в коррелированную ветвь.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
c=ChannelModel.__new__(ChannelModel); c.channel_state=ChannelState()
print(c._calculate_spatially_consistent_los(np.array([1.,1.]),0.6))
print(c._calculate_spatially_consistent_los(np.array([2.,2.]),0.4))
print(c.channel_state.first_call, c.channel_state.last_position)
```

### AUD-CHN-016 — ChannelModel._calculate_o2i_building_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Работает только из `_calculate_path_loss_sc`, до которого стандартный `calculate_SINR` не доходит.
- **Доказательство:** Метод найден в исходнике (строки 645–669); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_building_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if self.channel_state.o2i_building_loss is None:         L_concrete = 5 + 4 * fc_GHz         if model_type == 'low':`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Поставить spy на _calculate_path_loss_sc и _calculate_o2i_building_loss_sc; стандартный calculate_SINR должен вызвать обычный _calculate_path_loss, а SC-spy останутся не вызваны. Прямую проверку SC проводить отдельно после исправления SC-конструктора; найти внутренний вызов из _calculate_path_loss_sc.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-017 — ChannelModel._calculate_o2i_car_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Работает только из `_calculate_path_loss_sc`, до которого стандартный `calculate_SINR` не доходит.
- **Доказательство:** Метод найден в исходнике (строки 671–676); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::ChannelModel._calculate_o2i_car_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if self.channel_state.o2i_car_loss is None:         self.channel_state.o2i_car_loss = np.random.normal(9, 5)     return self.channel_state.o2i_car_loss`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Поставить spy на _calculate_o2i_car_loss_sc; стандартный calculate_SINR с ue_class="car" использует обычную O2I-ветвь, SC-spy не вызывается. После исправления SC маршрута проверить отдельный вызов из _calculate_path_loss_sc.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-018 — TDLModel._jakes_sum_of_sinusoids
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._jakes_sum_of_sinusoids`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Использует глобальный `GLOBALS.CURRENT_TIME`, поэтому время нельзя задавать независимо для отдельного вызова.
- **Доказательство:** Метод найден в исходнике (строки 1450–1469); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::TDLModel._jakes_sum_of_sinusoids`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `t = GLOBALS.CURRENT_TIME * 0.001     arg = 2 * np.pi * f_doppler * t * np.cos(alpha) + phi     tap_coeff = np.exp(1j * arg).sum(axis=1) / np.sqrt(self.NUM_SINUSOIDS)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать фабрикой RMa-SC, UMa-SC, UMi-SC; записать аргументы базового конструктора и тип tdl_profile. Флаг bool не должен попадать вместо имени профиля; spy фиксирует вызов SC-ветви SINR.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-019 — TDLModel._calculate_freq_response
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::TDLModel._calculate_freq_response`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** ues_info кэширует фазы и path_phase_matrix при первом UE или смене LOS/NLOS. Метода удаления UE/инвалидирования при смене частоты/профиля в TDLModel нет; переиспользование UE_ID может сохранить старое состояние. Повторное вычисление всей матрицы каждый TTI не подтверждается.
- **Доказательство:** Метод найден в исходнике (строки 1471–1563); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::TDLModel._calculate_freq_response`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if UE_ID not in self.ues_info or self.ues_info[UE_ID]['cond'] != channel_cond:         if channel_cond == 'NLOS':             powers_lin = 10 ** (self.nlos_power_dB / 10.0)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить None, число состояний 2/3, форму NxN, неотрицательные скорости переходов и packet_rates. Строки матрицы задают интенсивности 1/с, сумма не обязана быть 1; нулевая сумма означает поглощающее состояние.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-020 — TDLModel.calculate_channel_gain
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::TDLModel.calculate_channel_gain`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Мощность нормализуется на её же среднее по RB: абсолютный масштаб частотного отклика отбрасывается ради относительных замираний.
- **Доказательство:** Метод найден в исходнике (строки 1565–1589); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/CHANNEL_MODEL.py::TDLModel.calculate_channel_gain`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `H_f = self._calculate_freq_response(UE_ID, channel_cond, channel_model, ue_class)     gain_lin = np.abs(H_f) ** 2     gain_lin_norm = gain_lin / np.mean(gain_lin)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать фабрикой RMa-SC, UMa-SC, UMi-SC; записать аргументы базового конструктора и тип tdl_profile. Флаг bool не должен попадать вместо имени профиля; spy фиксирует вызов SC-ветви SINR.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-021 — RMaModel_SC.__init__
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC.__init__`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** `RMa-SC` создаётся из фабрики с несовместимыми аргументами; даже при прямом вызове `spatial_consistency` передаётся третьим позиционным аргументом как имя TDL-профиля.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "ValueError", "message": "Недопустимая TDL модель для NLOS: 'True'. Разрешённые значения: ['TDL-A', 'TDL-B', 'TDL-C']"}`. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC.__init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `super().__init__(bs, cond_update_period, spatial_consistency)     self.W = W     self.h = h`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать фабрикой RMa-SC, UMa-SC, UMi-SC; записать аргументы базового конструктора и тип tdl_profile. Флаг bool не должен попадать вместо имени профиля; spy фиксирует вызов SC-ветви SINR.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
RMaModel_SC(NS())
```

### AUD-CHN-022 — RMaModel_SC._get_sigma_sf
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._get_sigma_sf`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1642–1664); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._get_sigma_sf`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if channel_condition == 'LOS':         d_BP = self._calculate_breakpoint_distance(UE_height)         if d_2D < d_BP:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-023 — RMaModel_SC._calculate_breakpoint_distance
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_breakpoint_distance`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1666–1678); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_breakpoint_distance`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = 2 * np.pi * self.default_height * UE_height * self.bs.frequency_Hz / 300000000.0     return d_BP`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-024 — RMaModel_SC._calculate_los_probability
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_probability`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1680–1696); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_probability`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if d_2D <= 10:         return 1.0     los_probability = np.exp(-(d_2D - 10) / 1000)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-025 — RMaModel_SC._calculate_los_path_loss_time_based
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_time_based`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1698–1746); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_time_based`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = self._calculate_breakpoint_distance(UE_height)     if 10 <= d_2D <= d_BP:         PL1 = 20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0) + min(0.03 * self.h ** 1.72, 10) * np.log10(d_3D) - min(0.044 * self.h ** 1.72, 14.77) + 0.002 * np.log10(self.h) * d_3D`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-026 — RMaModel_SC._calculate_nlos_path_loss_time_based
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_time_based`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1748–1792); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_time_based`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if 10 <= d_2D <= 5000:         PL_LOS = self._calculate_los_path_loss_time_based(UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height)         PL_NLOS = 161.04 - 7.1 * np.log10(self.W) + 7.5 * np.log10(self.h) - (24.37 - 3.7 * (self.h / self.default_height) ** 2) * np.log10(self.default_height) + (43.42 - 3.1 * np.log10(self.default_height)) * (np.log10(d_3D) - 3) + 20 * np.log10(self.bs.frequency_GHz) - (3.2 * np....`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-027 — RMaModel_SC._calculate_los_path_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1794–1831); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = self._calculate_breakpoint_distance(UE_height)     if 10 <= d_2D <= d_BP:         PL_base = 20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0) + min(0.03 * self.h ** 1.72, 10) * np.log10(d_3D) - min(0.044 * self.h ** 1.72, 14.77) + 0.002 * np.log10(self.h) * d_3D`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-028 — RMaModel_SC._calculate_nlos_path_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1833–1861); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_nlos_path_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if 10 <= d_2D <= 5000:         PL_LOS_base = self._calculate_los_path_loss_base(d_2D, d_3D, UE_height)         PL_NLOS_base = 161.04 - 7.1 * np.log10(self.W) + 7.5 * np.log10(self.h) - (24.37 - 3.7 * (self.h / self.default_height) ** 2) * np.log10(self.default_height) + (43.42 - 3.1 * np.log10(self.default_height)) * (np.log10(d_3D) - 3) + 20 * np.log10(self.bs.frequency_GHz) - (3.2 * np.log10(11.75 * UE_height) ** 2 - 4.97)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-029 — RMaModel_SC._calculate_los_path_loss_base
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_base`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `RMa-SC`: фабрика не создаёт этот сценарий из-за несовместимых аргументов, а `calculate_SINR` не переключается на SC-расчёт.
- **Доказательство:** Метод найден в исходнике (строки 1863–1883); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::RMaModel_SC._calculate_los_path_loss_base`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = self._calculate_breakpoint_distance(UE_height)     if 10 <= d_2D <= d_BP:         return 20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0) + min(0.03 * self.h ** 1.72, 10) * np.log10(d_3D) - min(0.044 * self.h ** 1.72, 14.77) + 0.002 * np.log10(self.h) * d_3D`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-030 — UMaModel_SC.__init__
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC.__init__`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Третий позиционный аргумент базового конструктора ошибочно заполняется флагом `spatial_consistency`, хотя это имя профиля TDL.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "ValueError", "message": "Недопустимая TDL модель для NLOS: 'True'. Разрешённые значения: ['TDL-A', 'TDL-B', 'TDL-C']"}`. Основание: `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC.__init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `super().__init__(bs, cond_update_period, spatial_consistency)     self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]     self.bs.height = 25.0`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать фабрикой RMa-SC, UMa-SC, UMi-SC; записать аргументы базового конструктора и тип tdl_profile. Флаг bool не должен попадать вместо имени профиля; spy фиксирует вызов SC-ветви SINR.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
UMaModel_SC(NS())
```

### AUD-CHN-031 — UMaModel_SC._calculate_breakpoint_distance
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_breakpoint_distance`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMa-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 1935–1976); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_breakpoint_distance`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if UE_height < 13:         C = 0     elif 13 <= UE_height <= 23:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-032 — UMaModel_SC._calculate_los_probability
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_probability`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMa-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 1978–2003); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_probability`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `UE_height = self.UE_height     if d_2D <= 18:         return 1.0`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-033 — UMaModel_SC._calculate_los_path_loss_time_based
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_time_based`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMa-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2005–2047); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_time_based`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = self._calculate_breakpoint_distance(UE_height, d_2D)     if 10 <= d_2D <= d_BP:         PL1 = 28 + 22 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-034 — UMaModel_SC._calculate_nlos_path_loss_time_based
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_time_based`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMa-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2049–2089); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_time_based`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if 10 <= d_2D <= 5000:         PL_LOS = self._calculate_los_path_loss_time_based(UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height)         PL_NLOS = 13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz) - 0.6 * (UE_height - 1.5)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-035 — UMaModel_SC._calculate_los_path_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMa-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2091–2116); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_los_path_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = self._calculate_breakpoint_distance(UE_height, d_2D)     if 10 <= d_2D <= d_BP:         PL_base = 28 + 22 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-036 — UMaModel_SC._calculate_nlos_path_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMa-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2118–2151); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMaModel_SC._calculate_nlos_path_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if 10 <= d_2D <= 5000:         d_BP = self._calculate_breakpoint_distance(UE_height, d_2D)         if 10 <= d_2D <= d_BP:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-037 — UMiModel_SC.__init__
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC.__init__`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Третий позиционный аргумент базового конструктора ошибочно заполняется флагом `spatial_consistency`, хотя это имя профиля TDL.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "ValueError", "message": "Недопустимая TDL модель для NLOS: 'True'. Разрешённые значения: ['TDL-A', 'TDL-B', 'TDL-C']"}`. Основание: `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC.__init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `super().__init__(bs, cond_update_period, spatial_consistency)     self.bs.tx_power = self.bs.MICROCELL_TX_POWER[self.bs.bandwidth]     self.bs.height = 10.0`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать фабрикой RMa-SC, UMa-SC, UMi-SC; записать аргументы базового конструктора и тип tdl_profile. Флаг bool не должен попадать вместо имени профиля; spy фиксирует вызов SC-ветви SINR.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
UMiModel_SC(NS())
```

### AUD-CHN-038 — UMiModel_SC._calculate_breakpoint_distance
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_breakpoint_distance`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMi-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2202–2219); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_breakpoint_distance`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `h_E = 1.0     bs_height_prime = self.bs.height - h_E     UE_height_prime = UE_height - h_E`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-039 — UMiModel_SC._calculate_los_probability
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_probability`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMi-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2221–2239); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_probability`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `UE_height = self.UE_height     if d_2D <= 18:         return 1.0`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-040 — UMiModel_SC._calculate_los_path_loss_time_based
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_time_based`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMi-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2241–2283); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_time_based`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = self._calculate_breakpoint_distance(UE_height)     if 10 <= d_2D <= d_BP:         PL1 = 32.4 + 21 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-041 — UMiModel_SC._calculate_nlos_path_loss_time_based
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_time_based`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMi-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2285–2325); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_time_based`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if 10 <= d_2D <= 5000:         PL_LOS = self._calculate_los_path_loss_time_based(UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height)         PL_NLOS = 35.3 * np.log10(d_3D) + 22.4 + 21.3 * np.log10(self.bs.frequency_GHz) - 0.3 * (UE_height - 1.5)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-042 — UMiModel_SC._calculate_los_path_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMi-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2327–2352); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_los_path_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `d_BP = self._calculate_breakpoint_distance(UE_height)     if 10 <= d_2D <= d_BP:         PL_base = 32.4 + 21 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-CHN-043 — UMiModel_SC._calculate_nlos_path_loss_sc
- **Объект:** `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_sc`.
- **Категория:** ограничение штатной интеграции.
- **Наблюдение:** Метод принадлежит `UMi-SC`: класс не зарегистрирован в фабрике, а стандартный `calculate_SINR` не использует SC-ветку.
- **Доказательство:** Метод найден в исходнике (строки 2354–2387); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. ChannelInterface.get_conditions содержит RMa, UMa, UMi, RMa-SC; calculate_SINR вызывает _calculate_path_loss. Внешние/тестовые ручные вызовы не объявляются невозможными. Основание: `PyScheduler/CHANNEL_MODEL.py::UMiModel_SC._calculate_nlos_path_loss_sc`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if 10 <= d_2D <= 5000:         d_BP = self._calculate_breakpoint_distance(UE_height)         if 10 <= d_2D <= d_BP:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить реестр фабрики и все вызовы SC-метода поиском по репозиторию; затем создать каждый SC-сценарий и выполнить два расчёта с перемещением UE. Отсутствие вызова только в одном сценарии не доказывает мёртвый код.
- **Статус:** подтверждено для штатного маршрута.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

## 3. Что пока не удалось понять
- Не согласованы будущие API RNG, DTO и file/model; в проекте отдельного RNG-модуля и реестра источников нет.
- Не измерены затраты CPU/памяти, ускорение не заявляется.
- Нужны решения владельцев по сбросу статистики, multibs и поддержке SC.
- Подтверждение отдельной ветви не означает прохождения всех тестов модуля.
