# Ревизия модуля TRAFFIC_MODEL

> Ответственный: Кутенков Андрей | Проверено по коду: `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b` | Обновлено: 2026-10-17

Проверены `PyScheduler/TRAFFIC_MODEL.py`, связанные вызовы SIMULATION_MANAGER/UE_MODULE и существующие тесты. «Подтверждено по коду» означает наблюдаемую конструкцию, а не автоматическое признание багом. «Подтверждённый баг» ниже имеет воспроизведение. Производительность не измерялась. Симуляция запускалась автором. Дополнительная ревизия включает анализ кода и целевые проверки ниже; исходный код при подготовке документации не изменялся.

Для блоков воспроизведения: из каталога PyScheduler запустить Python и выполнить общий импорт, затем блок нужной карточки:

```python
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from TRAFFIC_MODEL import *
```

## 1. Реестр наблюдений

| ID | Категория | Объект | Статус |
|---|---|---|---|
| AUD-TRF-001 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::Packet.__post_init__` | подтверждено, воспроизведено |
| AUD-TRF-002 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::Packet.from_dict` | подтверждено, воспроизведено |
| AUD-TRF-003 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::PoissonModel.generate_traffic` | подтверждено, воспроизведено |
| AUD-TRF-004 | технический долг / ограничение | `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.__init__` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-005 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.generate_traffic` | подтверждено, воспроизведено |
| AUD-TRF-006 | технический долг / ограничение | `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.__init__` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-007 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::MMPPModel._get_next_state` | подтверждено, воспроизведено |
| AUD-TRF-008 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.get_model_info` | подтверждено, воспроизведено |
| AUD-TRF-009 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_model` | подтверждено, воспроизведено |
| AUD-TRF-010 | технический долг / ограничение | `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.get_available_models` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-011 | вопрос о назначении вспомогательного API | `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_poisson_model` | открыт вопрос; существование и отсутствие штатного подключения проверены |
| AUD-TRF-012 | вопрос о назначении вспомогательного API | `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_onoff_model` | открыт вопрос; существование и отсутствие штатного подключения проверены |
| AUD-TRF-013 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_mmpp_model` | подтверждено, воспроизведено |
| AUD-TRF-014 | технический долг / ограничение | `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.__init__` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-015 | технический долг / ограничение | `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.set_model` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-016 | вопрос о политике статистики | `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.update` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-017 | технический долг / ограничение | `PyScheduler/TRAFFIC_MODEL.py::PacketManager.get_qos_statistics` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-018 | технический долг / ограничение | `PyScheduler/TRAFFIC_MODEL.py::PacketManager.reset_ue` | подтверждено по коду; необходимость изменения обсуждается |
| AUD-TRF-019 | подтверждённый баг | `PyScheduler/TRAFFIC_MODEL.py::Packet.split_packet` | подтверждено, воспроизведено |

## 2. Карточки наблюдений

### AUD-TRF-001 — Packet.__post_init__
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::Packet.__post_init__`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Всегда заменяет переданный `deadline` на `creation_time + ttl_ms`, даже когда срок уже задан.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"result": 150}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::Packet.__post_init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `self.deadline = self.creation_time + self.ttl_ms`.
- **Влияние:** Явный срок пакета теряется при создании; это влияет на конвертацию файловых пакетов.
- **Как проверить:** Создать Packet с creation_time=100, ttl_ms=50 и deadline=999; выполнить to_dict/from_dict. Сравнить deadline до и после: если контракт допускает явный deadline, ожидается 999; иначе документировать его производность.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
print(Packet(size=100,ue_id=1,creation_time=100,ttl_ms=50,deadline=999).deadline)
```

### AUD-TRF-002 — Packet.from_dict
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::Packet.from_dict`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Значение `deadline` из словаря не передаётся в конструктор и теряется при обратном преобразовании.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"result": 150}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::Packet.from_dict`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `def from_dict(data: Dict, ue_id: int) -> 'Packet':     return Packet(size=data['size'], ue_id=ue_id, creation_time=data['creation_time'], priority=data.get('priority', 0), qci=data.get('qci'), ttl_ms=data.get('ttl_ms', 1000), is_fragment=data.get('is_fragment', False), bearer_id=data.get('bearer_id'))`.
- **Влияние:** Срок не переживает сериализацию/обратное создание.
- **Как проверить:** Создать Packet с creation_time=100, ttl_ms=50 и deadline=999; выполнить to_dict/from_dict. Сравнить deadline до и после: если контракт допускает явный deadline, ожидается 999; иначе документировать его производность.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
print(Packet.from_dict({"size":100,"creation_time":100,"ttl_ms":50,"deadline":999},1).deadline)
```

### AUD-TRF-003 — PoissonModel.generate_traffic
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::PoissonModel.generate_traffic`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** При `packet_rate = 0` вычисляется `1000 / packet_rate`: возникает деление на ноль.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "ZeroDivisionError", "message": "float division by zero"}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::PoissonModel.generate_traffic`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `packets = []     generate_time = current_time - update_interval     end_generate = current_time`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Передать packet_rate=0, затем -1, затем 10; ноль должен давать пустой поток, отрицательное значение — явную ошибку валидации. Ограничить время теста; деление на ноль и зависание считаются провалом.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
PoissonModel(0).generate_traffic(1,100,10)
```

### AUD-TRF-004 — OnOffModel.__init__
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.__init__`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Отрицательные и нулевые длительности ON/OFF не отклоняются при создании, хотя от них зависит продвижение времени.
- **Доказательство:** Метод найден в исходнике (строки 349–367); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.__init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `super().__init__(min_packet_size, max_packet_size)     self.duration_on = duration_on     self.duration_off = duration_off`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать OnOffModel с duration_on/duration_off из {-1,0,1}; конструктор сейчас принимает эти значения. Для отрицательных длительностей вызвать генерацию и записать ValueError NumPy; для нулевых фаз использовать отдельный процесс с таймаутом 1 с. Требуемую политику отказа закрепить валидацией до старта.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-005 — OnOffModel.generate_traffic
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.generate_traffic`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** При активной ON-фазе и `packet_rate = 0` выполняется деление на ноль; нулевые длительности фаз могут не продвигать время.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "ZeroDivisionError", "message": "float division by zero"}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::OnOffModel.generate_traffic`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if ue_id not in self._device_states:         self._initialize_state(ue_id, current_time, update_interval)     state_data = self._device_states[ue_id]`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Передать packet_rate=0, затем -1, затем 10; ноль должен давать пустой поток, отрицательное значение — явную ошибку валидации. Ограничить время теста; деление на ноль и зависание считаются провалом.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
m=OnOffModel(1,1,0)
m._device_states[1]={"state":"ON","end_state_time":1000}
m.generate_traffic(1,100,10)
```

### AUD-TRF-006 — MMPPModel.__init__
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.__init__`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** packet_rates и transition_matrix сохраняются без проверки размерности/знака; _get_next_state трактует строки как интенсивности переходов и нормирует их суммой. Это не стохастическая матрица с обязательной суммой строки 1.
- **Доказательство:** Метод найден в исходнике (строки 472–491); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.__init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `super().__init__(min_packet_size, max_packet_size)     self.transition_matrix = transition_matrix     self.packet_rates = packet_rates`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить None, число состояний 2/3, форму NxN, неотрицательные скорости переходов и packet_rates. Строки матрицы задают интенсивности 1/с, сумма не обязана быть 1; нулевая сумма означает поглощающее состояние.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-007 — MMPPModel._get_next_state
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel._get_next_state`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Ожидает индексируемую матрицу переходов; `None`, допустимый в методе создания `create_mmpp_model`, приводит к ошибке.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "TypeError", "message": "'NoneType' object is not subscriptable"}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::MMPPModel._get_next_state`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `rates = self.transition_matrix[current_state]     total_rate = sum(rates)     if total_rate == 0:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить None, число состояний 2/3, форму NxN, неотрицательные скорости переходов и packet_rates. Строки матрицы задают интенсивности 1/с, сумма не обязана быть 1; нулевая сумма означает поглощающее состояние.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
TrafficModelFactory.create_mmpp_model([10,20])._get_next_state(0)
```

### AUD-TRF-008 — MMPPModel.get_model_info
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.get_model_info`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Вызывает `self.transition_matrix.tolist()`; если модель создана без матрицы через `create_mmpp_model`, возникает `AttributeError`.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "AttributeError", "message": "'NoneType' object has no attribute 'tolist'"}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::MMPPModel.get_model_info`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `info = super().get_model_info()     info.update({'packet_rates': self.packet_rates, 'num_states': len(self.packet_rates), 'transition_matrix': self.transition_matrix.tolist(), 'active_devices': len(self._device_states)})     return info`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить None, число состояний 2/3, форму NxN, неотрицательные скорости переходов и packet_rates. Строки матрицы задают интенсивности 1/с, сумма не обязана быть 1; нулевая сумма означает поглощающее состояние.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
TrafficModelFactory.create_mmpp_model([10,20]).get_model_info()
```

### AUD-TRF-009 — TrafficModelFactory.create_model
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_model`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Подставляет фиксированную матрицу 3×3 для MMPP без проверки длины `packet_rates`: при ином числе состояний возникает несоответствие размеров.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "ValueError", "message": "'a' and 'p' must have same size"}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_model`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `def create_model(model_type: str, **kwargs) -> ITrafficModel:     if model_type == 'Poisson':         return PoissonModel(**kwargs)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить None, число состояний 2/3, форму NxN, неотрицательные скорости переходов и packet_rates. Строки матрицы задают интенсивности 1/с, сумма не обязана быть 1; нулевая сумма означает поглощающее состояние.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
TrafficModelFactory.create_model("MMPP",packet_rates=[10,20])._get_next_state(0)
```

### AUD-TRF-010 — TrafficModelFactory.get_available_models
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.get_available_models`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Набор допустимых названий отдельно задан здесь и повторяется в `create_model`; при расширении фабрики списки потребуется синхронизировать.
- **Доказательство:** Метод найден в исходнике (строки 679–681); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.get_available_models`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `def get_available_models() -> List[str]:     return ['Poisson', 'OnOff', 'MMPP']`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Для каждого имени из get_available_models вызвать create_model с валидными параметрами: Poisson(packet_rate=10), OnOff(duration_on=1,duration_off=1,packet_rate=10), MMPP(packet_rates=[1,2,3]). Проверить, что ни одно имя не отвергается фабрикой, а неизвестное имя отвергается.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-011 — TrafficModelFactory.create_poisson_model
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_poisson_model`.
- **Категория:** вопрос о назначении вспомогательного API.
- **Наблюдение:** Это дополнительная обёртка над конструктором, тогда как основной путь создаёт Poisson через `create_model`.
- **Доказательство:** Метод найден в исходнике (строки 684–686); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Поиск вызовов в проекте не выявил подключения этой обёртки/очистки в SimulationManager; это не основание удалять публичный API. Основание: `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_poisson_model`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `def create_poisson_model(packet_rate: float, **kwargs) -> PoissonModel:     return PoissonModel(packet_rate=packet_rate, **kwargs)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать Poisson через create_poisson_model(10) и create_model("Poisson",packet_rate=10), сравнить тип и get_model_info. Затем выяснить у владельца, нужен ли публичный shortcut; отсутствие вызовов в SM не означает ошибку.
- **Статус:** открыт вопрос; существование и отсутствие штатного подключения проверены.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-012 — TrafficModelFactory.create_onoff_model
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_onoff_model`.
- **Категория:** вопрос о назначении вспомогательного API.
- **Наблюдение:** Это дополнительная обёртка над конструктором, тогда как основной путь создаёт OnOff через `create_model`.
- **Доказательство:** Метод найден в исходнике (строки 689–695); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Поиск вызовов в проекте не выявил подключения этой обёртки/очистки в SimulationManager; это не основание удалять публичный API. Основание: `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_onoff_model`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `def create_onoff_model(duration_on: float, duration_off: float, packet_rate: float, **kwargs) -> OnOffModel:     return OnOffModel(duration_on=duration_on, duration_off=duration_off, packet_rate=packet_rate, **kwargs)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать OnOff через create_onoff_model(1,2,10) и create_model("OnOff",duration_on=1,duration_off=2,packet_rate=10); сравнить тип и get_model_info. Решение о сохранении shortcut согласовать с владельцем API.
- **Статус:** открыт вопрос; существование и отсутствие штатного подключения проверены.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-013 — TrafficModelFactory.create_mmpp_model
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_mmpp_model`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** Значение `transition_matrix=None` передаётся напрямую в MMPP, хотя генератор переходов требует полноценную матрицу.
- **Доказательство:** Выполнен локальный сценарий ниже. Фактический результат: `{"exception": "TypeError", "message": "'NoneType' object is not subscriptable"}`. Основание: `PyScheduler/TRAFFIC_MODEL.py::TrafficModelFactory.create_mmpp_model`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `def create_mmpp_model(packet_rates: List[float], transition_matrix: Optional[np.ndarray]=None, **kwargs) -> MMPPModel:     return MMPPModel(packet_rates=packet_rates, transition_matrix=transition_matrix, **kwargs)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Проверить None, число состояний 2/3, форму NxN, неотрицательные скорости переходов и packet_rates. Строки матрицы задают интенсивности 1/с, сумма не обязана быть 1; нулевая сумма означает поглощающее состояние.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.


```python
TrafficModelFactory.create_mmpp_model([10,20])._get_next_state(0)
```

### AUD-TRF-014 — SimpleGenerator.__init__
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.__init__`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Сохраняется как альтернативный путь для legacy-буфера: один UE — одна модель без поддержки multi-bearer/QCI.
- **Доказательство:** Метод найден в исходнике (строки 744–751); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.__init__`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `self.models: Dict[int, ITrafficModel] = {}     self._total_packets_generated = 0     self._packets_per_ue: Dict[int, int] = {}`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать SimpleGenerator, назначить Poisson одному UE и с фиксированным np.random.seed сгенерировать 100 мс. Проверить счётчики SimpleGenerator; затем отдельно сравнить multi-bearer PacketManager. Legacy-режим сознательно ограничен одним генератором на UE.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-015 — SimpleGenerator.set_model
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.set_model`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Параметр `qci` принимается, но не применяется; вместо использования QoS выводится предупреждение. Это совместимый упрощённый режим.
- **Доказательство:** Метод найден в исходнике (строки 767–775); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::SimpleGenerator.set_model`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if qci is not None:         print('[TRAFFIC] WARNING: Got QCI parameter. SimpleGenerator does not support QoS traffic.')     model = TrafficModelFactory.create_model(model_type, **params)`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Вызвать set_model(1,"Poisson",qci=9,packet_rate=10), перехватить предупреждение, проверить отсутствие применения QCI. Для QoS использовать PacketManager и проверить qci каждого Packet. Это отличие режимов, а не найденный сбой.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-016 — TrafficStatistics.update
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.update`.
- **Категория:** вопрос о политике статистики.
- **Наблюдение:** Счётчики накопительные; PacketManager.reset_ue удаляет профиль/состояния bearer, но не очищает накопленную статистику UE. Требуется решить, должна ли история сохраняться; это не подтверждённый баг.
- **Доказательство:** Метод найден в исходнике (строки 1093–1132); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::TrafficStatistics.update`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if not packets:         return     if self.start_time is None:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Создать два Packet: UE1=100 байт и UE2=200 байт, вызвать update один раз; проверить total_packets=2,total_bytes=300. Затем reset_ue(1) у PacketManager не должен менять историю, пока политика её удаления не согласована. Проверить полный statistics.reset обнуляет все счётчики.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-017 — PacketManager.get_qos_statistics
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.get_qos_statistics`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** При заданном `qci` игнорирует `ue_id` и возвращает общую статистику всех UE с этим QCI.
- **Доказательство:** Метод найден в исходнике (строки 1355–1373); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::PacketManager.get_qos_statistics`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if qci is not None:         return self.statistics.get_qci_stats(qci)     else:`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Для двух UE с известными пакетами сверить суммы пакетов/байтов по UE, QCI, bearer и глобально; reset_ue одного UE не должен удалять профиль другого. Политику исторических счётчиков зафиксировать отдельно.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-018 — PacketManager.reset_ue
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::PacketManager.reset_ue`.
- **Категория:** технический долг / ограничение.
- **Наблюдение:** Очищает профиль UE, но не исторические счётчики `TrafficStatistics`. Название `reset_ue` выглядит как полный сброс.
- **Доказательство:** Метод найден в исходнике (строки 1375–1396); проверены тело и связанные штатные вызовы. Исполнение этой ветви отдельно не проверялось. Основание: `PyScheduler/TRAFFIC_MODEL.py::PacketManager.reset_ue`, версия `ae9db40ecfd27787b4cdeba79bcb6eafabd7d44b`.
- **Фрагмент тела по коду:** `if ue_id in self.ue_profiles:         profile = self.ue_profiles[ue_id]         profile.clear_all()`.
- **Влияние:** Затрагивает описанную ветвь; масштаб зависит от выбранной модели, входных данных и жизненного цикла.
- **Как проверить:** Измерить время и выделения памяти на 100 и 1000 UE за 100 TTI; отделить инициализацию профиля от обновления. Ускорение заявлять только после сравнения выходов и повторных замеров.
- **Статус:** подтверждено по коду; необходимость изменения обсуждается.
- **Связь:** issue/PR пока нет; исправление кода в эту работу не входит.

### AUD-TRF-019 — Packet.split_packet теряет QoS-deadline
- **Объект:** `PyScheduler/TRAFFIC_MODEL.py::Packet.split_packet`, `Packet.__post_init__`.
- **Категория:** подтверждённый баг.
- **Наблюдение:** фрагменту передаётся deadline исходного пакета, но конструктор заменяет его на creation_time+ttl_ms. UeTrafficProfile ранее мог установить иной срок из QCI.
- **Доказательство:** локальное воспроизведение: исходный deadline=400, deadline фрагмента=1100.
- **Влияние:** фрагмент может жить дольше срока, установленного QoS исходного пакета.
- **Как проверить:** выполнить пример ниже; после исправления ожидается 400 у обоих пакетов.
- **Статус:** подтверждено, воспроизведено.
- **Связь:** FEAT-TRF-001; issue/PR пока нет.

```python
p=Packet(size=100,ue_id=1,creation_time=100)
p.deadline=400
fragment=p.split_packet(20)
print(p.deadline,fragment.deadline)
```

## 3. Что пока не удалось понять
- Не согласованы будущие API RNG, DTO и file/model; в проекте отдельного RNG-модуля и реестра источников нет.
- Не измерены затраты CPU/памяти, ускорение не заявляется.
- Нужны решения владельцев по сбросу статистики, multibs и поддержке SC.
- Подтверждение отдельной ветви не означает прохождения всех тестов модуля.
