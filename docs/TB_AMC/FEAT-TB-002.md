# Спецификация по ТБ
>Ответственный: Влад
>Состояние: выполнено
>Написано по коду: `dev@b388218` | Обновлено: `2026-11-13`
## Правки к спецификации [FEAT-TB-001.md](FEAT-TB-001.md)
### Вынос в отдельный модуль
Транспортный блок был вынесен в отдельный модуль в соответствии с планом перехода на новую архитектуру проекта.
### Изменения кода 
Сама структура осталась без изменений однако изменилось поле `buffer` теперь это `list[Packet]`, а не `list[int]`.
```py
@dataclass
class TransportBlock:
    ue_id: int
    tbs: int
    mcs: int
    rb: int
    buffer: list[Packet] = field(default_factory=list)
    ndi: int = 1
    rv: int = 0
    harq_id: int = 0        
    def determination_of_size_TB(self):
        self.tbs = AMC.get_tbs_bits(self.mcs, self.rb)
        return self.tbs
```