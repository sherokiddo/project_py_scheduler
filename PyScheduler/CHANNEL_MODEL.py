"""
#------------------------------------------------------------------------------
# Модуль: CHANNEL_MODEL - Модели радиоканалов для различных сценариев развертывания
#------------------------------------------------------------------------------
# Описание:
#   Реализация стандартных моделей радиоканалов согласно спецификациям 3GPP TR 38.901.
#   Включает модели для сельской (RMa), городской макросотовой (UMa) и городской
#   микросотовой (UMi) местности с поддержкой различных частотных диапазонов.
#
# Версия: 2.0.0
# Дата последнего изменения: 2026-25-11
# Автор: Норицин Иван
# Версия Python Kernel: 3.12.9
#
# Изменения v2.0.0:
#   - Подключён RNG через интерфейс, фабрику и конструкторы моделей.
#   - Заменены 18 обращений к глобальному генератору NumPy.
#   - Добавлена передача UE_ID во внутренние стохастические расчёты.
#   - Исправлены передача аргументов и обновление состояния SC.
#   - Добавлена валидация параметров канала и входных данных расчёта.
#   - Обновлены описания интерфейсов и подготовлены тесты интеграции.
#
#------------------------------------------------------------------------------
"""


import GLOBALS
import numpy as np
from BS_MODULE import BaseStation
from typing import Optional
from RNG import RandomProvider

class ChannelInterface:
    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        """Принимает источники случайных чисел для канала и TDL."""
        if rng is None:
            raise ValueError("rng должен быть передан в ChannelInterface")

        self.bs = bs
        self.rng = rng
        self.tdl_rng = tdl_rng if tdl_rng is not None else rng

    @staticmethod
    def get_conditions():
        """
        Возвращает словарь доступных состояний канала.
        """
        conditions = {
            "RMa": RMaModel, 
            "UMa": UMaModel, 
            "UMi": UMiModel, 
            "RMa-SC": RMaModel_SC,
            "UMa-SC": UMaModel_SC, 
            "UMi-SC": UMiModel_SC,
            }
        return conditions

    @staticmethod
    def validate_parameters(
        model: str,
        cond_update_period: float,
        freq_fad_nlos_model: str,
        freq_fad_los_model: str,
        ds_profile: str,
        los_arrival_angle: float,
    ) -> None:
        """
        Валидация параметров модели канала.

        Args:
            model (str): Тип модели ('RMa', 'UMa', 'UMi', 'RMa-SC').
            cond_update_period (float): Период обновления (мс).
            freq_fad_nlos_model (str): TDL-профиль NLOS.
            freq_fad_los_model (str): TDL-профиль LOS.
            ds_profile (str): Профиль задержек.
            los_arrival_angle (float): Угол прихода LOS (радианы).

        Raises:
            ValueError: При невалидных параметрах.
            TypeError: При неправильном типе данных.
        """
        models = ChannelInterface.get_conditions()
        if not isinstance(model, str) or model not in models:
            raise ValueError(
                f"Неизвестная модель {model!r}. "
                f"Допустимые модели: {', '.join(models)}"
            )
        numeric_types = (int, float, np.integer, np.floating)

        for name, value in (
            ("cond_update_period", cond_update_period),
            ("los_arrival_angle", los_arrival_angle),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numeric_types
            ):
                raise TypeError(f"{name} должен быть числом")
            
            if not np.isfinite(value):
                raise ValueError(f"{name} должен быть конечным числом")

        if cond_update_period < 0:
            raise ValueError("cond_update_period не может быть отрицательным")

        if not 0 <= los_arrival_angle < 2 * np.pi:
            raise ValueError(
                f"los_arrival_angle должен быть в [0, 2π), получен {los_arrival_angle}"
            )
        if not isinstance(freq_fad_nlos_model, str) or freq_fad_nlos_model not in (
            "TDL-A", "TDL-B", "TDL-C"
        ):
            raise ValueError("Профиль NLOS должен быть TDL-A, TDL-B или TDL-C")
        if not isinstance(freq_fad_los_model, str) or freq_fad_los_model not in (
            "TDL-D", "TDL-E"
        ):
            raise ValueError("Профиль LOS должен быть TDL-D или TDL-E")
        if not isinstance(ds_profile, str) or ds_profile not in (
            "short", "normal", "long"
        ):
            raise ValueError("ds_profile должен быть short, normal или long")

    def create(
        self,
        cond: str,
        cond_update_period: float = 0.0,
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        o2i_model: str = "low",
        W: float = 20.0,
        h: float = 5.0,
    ):
        """
        Создание модели радиоканала с переданными источниками RNG.

        Args:
            cond (str): Название модели: "RMa", "UMa", "UMi",
                "RMa-SC", "UMa-SC" или "UMi-SC".
            cond_update_period (float): Период обновления состояния
                канала (мс). По умолчанию 0.0.
            freq_fad_nlos_model (str): TDL-профиль для NLOS.
                По умолчанию "TDL-A".
            freq_fad_los_model (str): TDL-профиль для LOS.
                По умолчанию "TDL-D".
            ds_profile (str): Профиль разброса задержек:
                "short", "normal" или "long". По умолчанию "normal".
            los_arrival_angle (float): Угол прихода LOS-компоненты
                (радианы). По умолчанию np.pi / 4.
            o2i_model (str): Модель проникновения в здание:
                "low" или "high". Используется в UMa и UMi,
                включая их SC-варианты.
            W (float): Средняя ширина улиц (м).
                Используется в RMa и RMa-SC. По умолчанию 20.0.
            h (float): Средняя высота зданий (м).
                Используется в RMa и RMa-SC. По умолчанию 5.0.

        Returns:
            ChannelModel: Созданная модель радиоканала.

        Raises:
            TypeError: Если числовой параметр имеет неправильный тип.
            ValueError: Если модель неизвестна или параметры
                не проходят проверку допустимых значений.
        """

        ChannelInterface.validate_parameters(
            cond,
            cond_update_period,
            freq_fad_nlos_model,
            freq_fad_los_model,
            ds_profile,
            los_arrival_angle,
        )

        conds = self.get_conditions()
        if cond not in conds:
            raise ValueError(f"Unknown condition '{cond}'. Valid: {', '.join(conds.keys())}")

        if cond in ("RMa", "RMa-SC"):
            numeric_types = (int, float, np.integer, np.floating)

            for name, value in (("W", W), ("h", h)):
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, numeric_types
                ):
                    raise TypeError(f"{name} должен быть числом")

                if not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"{name} должен быть конечным числом больше нуля"
                    )

        elif cond in ("UMa", "UMi", "UMa-SC", "UMi-SC"):
            if not isinstance(o2i_model, str) or o2i_model not in (
                "low", "high"
            ):
                raise ValueError(
                    "o2i_model должен быть 'low' или 'high'"
                )

        model_cls = conds[cond]

        # Базовые параметры
        kwargs = {
            "bs": self.bs,
            "rng": self.rng,
            "tdl_rng": self.tdl_rng,
            "cond_update_period": cond_update_period,
            "freq_fad_nlos_model": freq_fad_nlos_model,
            "freq_fad_los_model": freq_fad_los_model,
            "ds_profile": ds_profile,
            "los_arrival_angle": los_arrival_angle,
        }

        # Специфика моделей
        if cond in ("RMa", "RMa-SC"):
            kwargs.update({"W": W, "h": h})
        elif cond in ("UMa", "UMi", "UMa-SC", "UMi-SC"):
            kwargs["o2i_model"] = o2i_model

        return model_cls(**kwargs)

    def get_available_conds():
        """
        Выводит на экран список названий доступных моделей.
        """
        conds = list(ChannelInterface.get_conditions().keys())
        print("Доступные модели:", conds)


class ChannelState:
    """
    Класс для хранения состояния канала с поддержкой пространственной согласованности.

    Поддерживает пространственную и временную корреляцию shadow fading и LOS/NLOS состояний.
    """

    def __init__(self):
        """
        Инициализация состояния канала.

         Хранит предыдущие позиции отдельно для расчётов LOS/NLOS
        и теневого затенения, а также сохранённые потери проникновения.
        """
        self.last_position: np.ndarray = None
        self.last_sf_position: Optional[np.ndarray] = None
        self.last_los_position: Optional[np.ndarray] = None

        self.last_shadow_fading: float = 0.0
        self.last_los_state: bool = None
        self.first_call: bool = True

        self.o2i_building_loss: float = None
        self.o2i_car_loss: float = None


    def reset(self):
        """Сброс состояния канала."""
        self.__init__()


class ChannelModel:
    """
    Базовый класс для моделей радиоканалов.
    """


    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        cond_update_period: float = 0.0,
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        spatial_consistency=False,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        """
        Инициализация базового класса моделей радиоканала.

        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел для
                крупномасштабных расчётов канала: LOS/NLOS, SF и O2I.
            cond_update_period (float, optional): Период обновления состояния
                радиоканала (мс) По умолчанию 0.0.
            freq_fad_nlos_model (str, optional): Модель для получения
            частотно-селективных замираний при NLOS. По умолчанию "TDL-A".
            freq_fad_los_model (str, optional): Модель для получения
            частотно-селективных замираний при LOS. По умолчанию "TDL-D"
            ds_profile (str, optional): Профиль разброса задержек.
                По умолчанию "normal".
            los_arrival_angle (float, optional): Угол прихода LOS-компоненты.
                По умолчанию np.pi / 4.
            tdl_rng (Optional[RandomProvider]): Внешний источник случайных
                чисел для TDL. Если не передан, используется rng.
                Для независимости TDL от остальных расчётов следует
                передавать отдельный провайдер.
        """
        if rng is None:
            raise ValueError("rng должен быть передан в ChannelModel")
        
        self.bs = bs

        self.rng = rng
        self.tdl_rng = tdl_rng if tdl_rng is not None else rng

        self.cond_update_period = cond_update_period
        self.spatial_consistency = spatial_consistency

        self.SHADOW_FADING_INFO = {}
        self.CHANNEL_COND_INFO = {}
        self.O2I_INFO = {}

        self.channel_state = ChannelState() if spatial_consistency else None

        # Корреляционные расстояния
        self.correlation_dist_LOS = 37.0
        self.correlation_dist_NLOS = 50.0
        self.los_correlation_length = 50.0
        self.sf_correlation_distance_los = 37.0
        self.sf_correlation_distance_nlos = 50.0

        # O2I модель
        self.o2i_model = "low"  # или "high"
        self.freq_fading = TDLModel(
            bs=bs,
            rng=self.tdl_rng,
            nlos_model=freq_fad_nlos_model,
            los_model=freq_fad_los_model,
            delay_spread_profile=ds_profile,
            los_arrival_angle=los_arrival_angle,
        )

    def _get_sigma_sf(
        self, channel_condition: str, d_2D: float = None, UE_height: float = None
    ) -> float:
        """
        Возвращает стандартное отклонение теневого затенения (SF) для
        заданного состояния канала (LOS/NLOS).

        Args:
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float, optional): 2D расстояние между БС и UE (м).
                По умолчанию None.
            UE_height (float, optional): Высота пользовательского оборудования (м).
                По умолчанию None.

        Returns:
            float: Стандартное отклонение теневого затенения (дБ).

        """
        if channel_condition == "LOS":
            return getattr(self, "sigma_SF_LOS", None)

        elif channel_condition == "NLOS":
            return getattr(self, "sigma_SF_NLOS", None)

    def _calculate_path_loss(
        self,
        UE_ID: int,
        displacement: float,
        d_2D: float,
        d_2D_in: float,
        d_3D: float,
        UE_height: float,
        ue_class: str,
    ) -> float:
        """
        Расчёт значения затухания сигнала между базовой станцией (BS)
        и пользовательским оборудованием (UE) с учётом условий канала (LOS/NLOS),
        теневого затенения (shadow fading) и проникновения сигнала внутрь
        здания или автомобиля (O2I).

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_2D_in (float): Внутреннее расстояние в здании (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).
            ue_class (str): Класс UE.

        Returns:
            float: Затухание сигнала (дБ).

        """
        channel_condition = self._calculate_channel_condition(
            UE_ID, self.cond_update_period, UE_height, d_2D
        )

        sigma_sf = self._get_sigma_sf(channel_condition, d_2D, UE_height)

        correlation_dist = (
            self.correlation_dist_LOS if channel_condition == "LOS" else self.correlation_dist_NLOS
        )

        shadow_fading = self._calculate_shadow_fading(
            UE_ID, displacement, channel_condition, sigma_sf, correlation_dist
        )

        if channel_condition == "LOS":
            calculate_los = getattr(
                self,
                "_calculate_los_path_loss",
                None,
            )
            
            if calculate_los is None:
                calculate_los = self._calculate_los_path_loss_time_based

            PL = calculate_los(
                UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height
            )

        elif channel_condition == "NLOS":
            calculate_nlos = getattr(
                self,
                "_calculate_nlos_path_loss",
                None,
            )

            if calculate_nlos is None:
                calculate_nlos = self._calculate_nlos_path_loss_time_based

            PL = calculate_nlos(
                UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height
            )
        else:
            raise ValueError(
                f"Неизвестное состояние канала: {channel_condition!r}"
            )
        PL = PL + shadow_fading

        if ue_class == "indoor":
            o2i_penetration_loss = self._calculate_o2i_building_loss(
                UE_ID, channel_condition, self.o2i_model, d_2D_in, self.bs.frequency_GHz
            )
            PL = PL + o2i_penetration_loss

        elif ue_class == "car":
            o2i_penetration_loss = self._calculate_o2i_car_loss(UE_ID)
            PL = PL + o2i_penetration_loss

        return PL

    def _calculate_channel_condition(
        self, 
        UE_ID: int, 
        cond_update_period: float, 
        UE_height: float, 
        d_2D: float
    ) -> str:
        """
        Расчёт состояния радоканала (LOS/NLOS).

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            cond_update_period (float): Период обновления состояния радиоканала (мс).
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            str: Состояние радиоканала (LOS/NLOS).

        """
        channel_condition = None
        not_found = False
        update = False

        if UE_ID in self.CHANNEL_COND_INFO:
            ue_cond_info = self.CHANNEL_COND_INFO[UE_ID]
            channel_condition = ue_cond_info["cond"]

            if (
                cond_update_period != 0.0
                and GLOBALS.CURRENT_TIME - ue_cond_info["updated_time"] > cond_update_period
            ):
                update = True

        else:
            not_found = True

            self.CHANNEL_COND_INFO[UE_ID] = {"cond": None, "updated_time": None}
            ue_cond_info = self.CHANNEL_COND_INFO[UE_ID]

        if not_found or update:
            los_probability = self._calculate_los_probability(UE_height, d_2D)
            u = self.rng.uniform(0.0, 1.0, stream_offset=UE_ID)
            channel_condition = "LOS" if u <= los_probability else "NLOS"

            ue_cond_info["cond"] = channel_condition
            ue_cond_info["updated_time"] = GLOBALS.CURRENT_TIME

        return channel_condition

    def _calculate_shadow_fading(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        sigma_SF: float,
        correlation_dist: float,
    ) -> float:
        """
        Расчёт значения теневого затененеия (Shadow Fading).

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            sigma_SF (float): Стандартное отклонение теневого затенения (дБ)
            correlation_dist (float): Корреляционное расстояние (м).

        Returns:
            float: Теневое затененеие (дБ)

        """
        shadow_fading = None
        not_found = False
        new_condition = False

        if UE_ID in self.SHADOW_FADING_INFO:
            ue_sf_info = self.SHADOW_FADING_INFO[UE_ID]
            new_condition = ue_sf_info["cond"] != channel_condition
        else:
            not_found = True
            self.SHADOW_FADING_INFO[UE_ID] = {"SF_value": None, "cond": None}
            ue_sf_info = self.SHADOW_FADING_INFO[UE_ID]

        if not_found or new_condition:
            shadow_fading = self.rng.normal(
                mean=0.0,
                variance=sigma_SF**2,
                stream_offset=UE_ID,
            )
        else:
            R = np.exp(-1 * displacement / correlation_dist)

            z = self.rng.normal(
                mean=0.0,
                variance=sigma_SF**2,
                stream_offset=UE_ID,
            )

            shadow_fading = R * ue_sf_info["SF_value"] + np.sqrt(1 - R * R) * z
            

        ue_sf_info["SF_value"] = shadow_fading
        ue_sf_info["cond"] = channel_condition

        return shadow_fading

    def _calculate_o2i_building_loss(
        self, UE_ID: int, channel_condition: str, model_type: str, d_2D_in: float, fc_GHz: float
    ) -> float:
        """
        Расчёт значения потерь при проникновении сигнала внутрь здания.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            model_type (str): Тип модели проникновения в здание ('low' или 'high').
            d_2D_in (float): Внутреннее расстояние в здании (м).
            fc_GHz (float): Несущая частота сигнала (ГГц).

        Returns:
            float: Потери при проникновении сигнала внутрь здания (дБ).

        """
        o2i_loss = None
        not_found = False
        new_condition = False

        if UE_ID in self.O2I_INFO:
            ue_o2i_info = self.O2I_INFO[UE_ID]
            new_condition = ue_o2i_info["cond"] != channel_condition
        else:
            not_found = True
            self.O2I_INFO[UE_ID] = {"o2i_loss": None, "cond": None}
            ue_o2i_info = self.O2I_INFO[UE_ID]

        if not_found or new_condition:
            L_concrete = 5 + 4 * fc_GHz

            if model_type == "low":
                L_glass = 2 + 0.2 * fc_GHz
                PL_tw = 5 - 10 * np.log10(
                    0.3 * 10 ** (-L_glass / 10) + 0.7 * 10 ** (-L_concrete / 10)
                )
                sigma_p = 4.4

            elif model_type == "high":
                L_IRR_glass = 23 + 0.3 * fc_GHz
                PL_tw = 5 - 10 * np.log10(
                    0.7 * 10 ** (-L_IRR_glass / 10) + 0.3 * 10 ** (-L_concrete / 10)
                )
                sigma_p = 6.5

            PL_in = 0.5 * d_2D_in
            random_loss = self.rng.normal(
                mean=0.0,
                variance=sigma_p**2,
                stream_offset=UE_ID,
            )
            o2i_loss = PL_tw + PL_in + random_loss

        else:
            o2i_loss = ue_o2i_info["o2i_loss"]

        ue_o2i_info["o2i_loss"] = o2i_loss
        ue_o2i_info["cond"] = channel_condition

        return o2i_loss

    def _calculate_o2i_car_loss(self, UE_ID: int) -> float:
        """
        Расчёт значения потерь при проникновении сигнала внутрь автомобиля.

        Args:
            UE_ID (int): Уникальный идентификатор UE.

        Returns:
            float: Потери при проникновении сигнала внутрь автомобиля (дБ).

        """
        o2i_loss = None
        not_found = False

        if UE_ID in self.O2I_INFO:
            ue_o2i_info = self.O2I_INFO[UE_ID]
        else:
            not_found = True
            self.O2I_INFO[UE_ID] = {"o2i_loss": None, "cond": None}
            ue_o2i_info = self.O2I_INFO[UE_ID]

        if not_found:
            o2i_loss = self.rng.normal(
                mean=9.0,
                variance=25.0,
                stream_offset=UE_ID,
            )
        else:
            o2i_loss = ue_o2i_info["o2i_loss"]

        ue_o2i_info["o2i_loss"] = o2i_loss

        return o2i_loss

    def calculate_SINR(
        self,
        UE_ID: int,
        displacement: float,
        d_2D: float,
        d_2D_in: float,
        d_3D: float,
        UE_height: float,
        ue_class: str,
        current_position: Optional[np.ndarray] = None,
    ) -> float:
        """
        Расчет отношения сигнал-интерференция-шум (SINR).

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_2D_in (float): Внутреннее расстояние в здании (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).
            ue_class (str): Класс UE.
            current_position (Optional[np.ndarray]): Позиция UE
                [x, y] в метрах. Обязательна для SC-режима.

        Returns:
            float: Значение SINR (дБ).

        Raises:
            TypeError: Если идентификатор или числовые параметры
                имеют неверный тип.
            ValueError: Если значения, класс UE или позиция недопустимы.
        """

        if isinstance(UE_ID, (bool, np.bool_)) or not isinstance(
            UE_ID, (int, np.integer)
        ):
            raise TypeError("UE_ID должен быть целым числом")

        if UE_ID < 0:
            raise ValueError("UE_ID не может быть отрицательным")

        UE_ID = int(UE_ID)

        numeric_types = (int, float, np.integer, np.floating)

        for name, value in (
            ("displacement", displacement),
            ("d_2D", d_2D),
            ("d_2D_in", d_2D_in),
            ("d_3D", d_3D),
            ("UE_height", UE_height),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numeric_types
            ):
                raise TypeError(f"{name} должен быть числом")

            if not np.isfinite(value):
                raise ValueError(f"{name} должен быть конечным числом")

        for name, value in (
            ("displacement", displacement),
            ("d_2D", d_2D),
            ("d_2D_in", d_2D_in),
        ):
            if value < 0:
                raise ValueError(f"{name} не может быть отрицательным")

        if d_3D <= 0:
            raise ValueError("d_3D должен быть больше нуля")

        if UE_height <= 0:
            raise ValueError("UE_height должен быть больше нуля")

        if isinstance(self, (UMaModel, UMaModel_SC)) and UE_height > 23:
            raise ValueError("Для UMa высота UE не должна превышать 23 м")

        if not isinstance(ue_class, str) or ue_class not in (
            "outdoor", "indoor", "car"
        ):
            raise ValueError("ue_class должен быть outdoor, indoor или car")

        if self.spatial_consistency:
            if current_position is None:
                raise ValueError(
                    "Для SC-режима необходимо передать current_position"
                )
            
            try:
                current_position = np.asarray(
                    current_position,
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "current_position должна содержать две числовые координаты"
                ) from exc

            if (
                    current_position.shape != (2,)
                    or not np.all(np.isfinite(current_position))
                ):
                    raise ValueError(
                        "current_position должна содержать две "
                        "конечные координаты [x, y]"
                    )

            path_loss = self._calculate_path_loss_sc(
                d_2D=d_2D,
                d_2D_in=d_2D_in,
                d_3D=d_3D,
                UE_height=UE_height,
                ue_class=ue_class,
                current_position=current_position,
                UE_ID=UE_ID,
            )
        else:    
            path_loss = self._calculate_path_loss(
                UE_ID, 
                displacement, 
                d_2D, 
                d_2D_in, 
                d_3D, 
                UE_height, 
                ue_class
            )

        cable_loss = 2
        interference_margin = 2

        P_signal = (
            self.bs.tx_power + self.bs.antenna_gain - cable_loss - path_loss - interference_margin
        )

        P_interference = -95
        P_noise = -174 + 10 * np.log10(self.bs.bandwidth * 1e6)

        SINR = P_signal - 10 * np.log10(10 ** (P_interference / 10) + 10 ** (P_noise / 10))

        if self.bs.enable_tdl:
            if self.spatial_consistency:
                state = self._get_channel_state(UE_ID)
                channel_cond = (
                    "LOS" if state.last_los_state else "NLOS"
                )
            else:
                channel_cond = self.CHANNEL_COND_INFO[UE_ID]["cond"]

            channel_model = self.__class__.__name__

            if channel_model.endswith("_SC"):
                channel_model = channel_model[:-3]

            gain_on_RB = self.freq_fading.calculate_channel_gain(
                UE_ID,
                channel_cond,
                channel_model,
                ue_class,
            )

            return SINR + gain_on_RB

        return SINR

        
    def _get_channel_state(self, UE_ID: int) -> ChannelState:
        """
        Получение SC-состояния конкретного пользователя.

        Состояния принадлежат экземпляру модели канала.
        Для каждого UE создаётся отдельный объект ChannelState.
        Повторное обращение возвращает ранее созданный объект.

        Args:
            UE_ID (int): Уникальный идентификатор UE.

        Returns:
            ChannelState: Состояние данного UE в текущей модели канала.

        Raises:
            ValueError: Если пространственная согласованность отключена.
        """

        if not self.spatial_consistency:
            raise ValueError(
                "SC-состояние недоступно: spatial_consistency=False"
            )

        if not hasattr(self, "_channel_states"):
            self._channel_states = {}

        if UE_ID not in self._channel_states:
            self._channel_states[UE_ID] = ChannelState()

        return self._channel_states[UE_ID]

    def _calculate_correlated_shadow_fading_sc(
        self, 
        current_position: np.ndarray, 
        sigma_sf: float, 
        is_los: bool,
        UE_ID: int = 0,
    ) -> float:
        """
        Расчет коррелированного shadow fading для SC режима.

        Args:
            current_position: Текущая позиция UE [x, y] в метрах
            sigma_sf: Стандартное отклонение shadow fading в дБ
            is_los: Состояние LOS/NLOS канала
            UE_ID (int): Уникальный идентификатор UE.
                Значение 0 оставлено для совместимости старых вызовов.
                При расчёте необходимо передавать ID текущего UE.

        Returns:
            Коррелированное значение shadow fading в дБ
        """

        state = self._get_channel_state(UE_ID)

        correlation_distance = (
            self.sf_correlation_distance_los 
            if is_los 
            else self.sf_correlation_distance_nlos
        )

        if state.last_sf_position is None:
            state.last_shadow_fading = self.rng.normal(
                mean=0.0,
                variance=sigma_sf**2,
                stream_offset=UE_ID
            )
            state.last_sf_position = current_position.copy()
            return state.last_shadow_fading

        delta_d = np.linalg.norm(current_position - state.last_sf_position)

        if delta_d < 0.01:  # Обновлять только при движении > 0.01м, костыль
            return state.last_shadow_fading

        R = np.exp(-delta_d / correlation_distance)

        innovation = self.rng.normal(
            mean=0.0,
            variance=sigma_sf**2,
            stream_offset=UE_ID,
        )

        new_sf = R * state.last_shadow_fading + np.sqrt(1 - R**2) * innovation

        state.last_shadow_fading = new_sf
        state.last_sf_position = current_position.copy()

        return new_sf

    def _calculate_path_loss_sc(
        self,
        d_2D: float,
        d_2D_in: float,
        d_3D: float,
        UE_height: float,
        ue_class: str,
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        Расчёт LOS-потерь для UMi в SC-режиме.

        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Текущая позиция UE
                [x, y] в метрах.
            UE_ID (int): Идентификатор UE для выбора подпотока RNG.
                По умолчанию 0 для совместимости старых вызовов.

        Returns:
            float: LOS-потери с учётом теневого затенения (дБ).
                Вне допустимого диапазона расстояний — 10000.0.
        """

        base_los_prob = self._calculate_los_probability(
            UE_height, 
            d_2D,
            )
        
        is_LOS = self._calculate_spatially_consistent_los(
            current_position,
            base_los_prob,
            UE_ID=UE_ID,
            )

        if is_LOS:
            PL = self._calculate_los_path_loss_sc(
                d_2D,
                d_3D,
                UE_height, 
                current_position,
                UE_ID=UE_ID,
                )
        else:
            PL = self._calculate_nlos_path_loss_sc(
                d_2D, 
                d_3D, 
                UE_height, 
                current_position,
                UE_ID=UE_ID,
                )

        # O2I потери для SC режима
        if ue_class == "indoor":
            o2i_loss = self._calculate_o2i_building_loss_sc(
                getattr(self, "o2i_model", "low"),
                d_2D_in,
                self.bs.frequency_GHz,
                UE_ID=UE_ID
            )
            PL = PL + o2i_loss
        elif ue_class == "car":
            o2i_loss = self._calculate_o2i_car_loss_sc(UE_ID=UE_ID)
            PL = PL + o2i_loss

        return PL

    def _calculate_spatially_consistent_los(
        self, 
        current_position: np.ndarray, 
        base_los_prob: float,
        UE_ID: int = 0,
    ) -> bool:
        """
        Расчет пространственно-согласованного состояния LOS/NLOS.

        Args:
            current_position: Текущая позиция UE [x, y] в метрах
            base_los_prob: Базовая вероятность LOS
            UE_ID (int): Уникальный идентификатор UE.
                Значение 0 оставлено для совместимости старых вызовов.
                При расчёте необходимо передавать ID текущего UE.

        Returns:
            Состояние канала: True для LOS, False для NLOS
        """

        state = self._get_channel_state(UE_ID)

        if (
            state.first_call
            or state.last_los_position is None
            or state.last_los_state is None
        ):
            # Первый вызов - инициализация на основе базовой вероятности

            state.last_los_state = base_los_prob > 0.5
            state.last_los_position = current_position.copy()
            state.first_call = False
            
            return state.last_los_state

        delta_d = np.linalg.norm(current_position - state.last_los_position)

        if delta_d < 0.01:  # non-3GPP костыль, но допустимый
            return state.last_los_state

        state_correlation = np.exp(-delta_d / self.los_correlation_length)

        if self.rng.uniform(
            0.0, 1.0, stream_offset=UE_ID
        ) < state_correlation:
            current_los_state = state.last_los_state
        else:
            current_los_state = self.rng.uniform(
                0.0, 1.0, stream_offset=UE_ID
            ) <= base_los_prob

        state.last_los_state = current_los_state
        state.last_los_position = current_position.copy()

        return current_los_state

    def _calculate_o2i_building_loss_sc(
        self, 
        model_type: str, 
        d_2D_in: float, 
        fc_GHz: float,
        UE_ID: int = 0,
    ) -> float:
        """
        Расчёт O2I потерь для зданий (SC режим).
        
        Случайная составляющая формируется через переданный RNG
        с подпотоком stream_offset=UE_ID. Полученные потери
        сохраняются в состоянии канала.

        Args:
            model_type (str): Тип модели проникновения:
                "low" или "high".
            d_2D_in (float): Расстояние внутри здания (м).
            fc_GHz (float): Несущая частота сигнала (ГГц).
            UE_ID (int): Уникальный идентификатор UE.
                Значение 0 оставлено для совместимости старых вызовов.
                При расчёте необходимо передавать ID текущего UE.

        Returns:
            float: Потери проникновения сигнала в здание (дБ).
        """
        state = self._get_channel_state(UE_ID)
        
        if state.o2i_building_loss is None:
            L_concrete = 5 + 4 * fc_GHz

            if model_type == "low":
                L_glass = 2 + 0.2 * fc_GHz
                PL_tw = 5 - 10 * np.log10(
                    0.3 * 10 ** (-L_glass / 10) + 0.7 * 10 ** (-L_concrete / 10)
                )
                sigma_p = 4.4
            elif model_type == "high":
                L_IRR_glass = 23 + 0.3 * fc_GHz
                PL_tw = 5 - 10 * np.log10(
                    0.7 * 10 ** (-L_IRR_glass / 10) + 0.3 * 10 ** (-L_concrete / 10)
                )
                sigma_p = 6.5

            PL_in = 0.5 * d_2D_in

            random_loss = self.rng.normal(
                mean=0.0,
                variance=sigma_p**2,
                stream_offset=UE_ID,
            )
            state.o2i_building_loss = PL_tw + PL_in + random_loss

        return state.o2i_building_loss

    def _calculate_o2i_car_loss_sc(self, UE_ID: int = 0) -> float:
        """
        Расчёт O2I потерь для автомобилей (SC режим).

        Потери имеют нормальное распределение со средним 9 дБ
        и стандартным отклонением 5 дБ. Полученное значение
        сохраняется в состоянии канала.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
                Значение 0 оставлено для совместимости старых вызовов.
                При расчёте необходимо передавать ID текущего UE.

        Returns:
            float: Потери проникновения сигнала в автомобиль (дБ).
        """

        state = self._get_channel_state(UE_ID)
        
        if state.o2i_car_loss is None:
            state.o2i_car_loss = self.rng.normal(
                mean=9.0,
                variance=25.0,
                stream_offset=UE_ID,
            )

        return state.o2i_car_loss

    def _calculate_los_path_loss_sc(
        self,
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        SC LOS path loss. Должен быть переопределен в дочерних классах.

        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Текущая позиция UE
                [x, y] в метрах.
            UE_ID (int): Уникальный идентификатор UE.
                Значение 0 оставлено для совместимости старых вызовов.
                При расчёте необходимо передавать ID текущего UE.

        Returns:
            float: Потери распространения в условиях LOS (дБ).

        Raises:
            NotImplementedError: Если дочерний класс не реализует метод.
        """
        raise NotImplementedError

    def _calculate_nlos_path_loss_sc(
        self, 
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        SC NLOS path loss. Должен быть переопределен в дочерних классах.

        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Позиция UE [x, y] в метрах.
            UE_ID (int): ID пользователя для выбора подпотока RNG.
                По умолчанию 0 для совместимости старых вызовов.

        Returns:
            float: Потери распространения в условиях NLOS (дБ).

        Raises:
            NotImplementedError: Если дочерний класс не реализует метод.
        """
        raise NotImplementedError


class RMaModel(ChannelModel):
    """
    Модель радиоканала для сельской местности (Rural Macro - RMa).
    Реализует расчеты для сценариев макросотового покрытия в сельской местности.
    """

    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        W: float = 20.0,
        h: float = 5.0,
        cond_update_period: float = 0.0,
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        """
        Инициализация модели RMa.

        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел
                для крупномасштабных расчётов канала.
            W (float, optional): Средняя ширина улиц (м). По умолчанию 20.0.
            h (float, optional): Средняя высота зданий в метрах. По умолчанию 5.0.
            cond_update_period (float, optional): Период обновления состояния
                радиоканала (мс). По умолчанию 0.0.
            freq_fad_nlos_model (str, optional): Модель для получения
            частотно-селективных замираний при NLOS. По умолчанию "TDL-A".
            freq_fad_los_model (str, optional): Модель для получения
            частотно-селективных замираний при LOS. По умолчанию "TDL-D"
            ds_profile (str, optional): Профиль разброса задержек.
                По умолчанию "normal".
            los_arrival_angle (float, optional): Угол прихода LOS-компоненты.
                По умолчанию np.pi / 4.
            tdl_rng (Optional[RandomProvider]): Отдельный источник
                случайных чисел для TDL. Если не передан, используется rng.

        Raises:
            ValueError: Если rng равен None или параметры TDL
                не проходят проверку допустимых значений.

        """
        if rng is None:
            raise ValueError("rng не может быть None")
        
        super().__init__(
            bs=bs,
            rng=rng,
            cond_update_period=cond_update_period,
            freq_fad_nlos_model=freq_fad_nlos_model,
            freq_fad_los_model=freq_fad_los_model,
            ds_profile=ds_profile,
            los_arrival_angle=los_arrival_angle,
            spatial_consistency=False,
            tdl_rng=tdl_rng,
        )

        self.W = W
        self.h = h

        self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]
        self.default_height = 35.0

        self.sigma_SF_LOS_1 = 4.0
        self.sigma_SF_LOS_2 = 6.0
        self.sigma_SF_NLOS = 8.0

        self.correlation_dist_LOS = 37
        self.correlation_dist_NLOS = 120

        self.o2i_model = "low"

    def _get_sigma_sf(self, channel_condition: str, d_2D: float, UE_height: float) -> float:
        """
        Возвращает стандартное отклонение теневого затенения (SF) для
        заданного состояния канала (LOS/NLOS).

        Args:
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Стандартное отклонение теневого затенения (дБ).

        """
        if channel_condition == "LOS":
            d_BP = self._calculate_breakpoint_distance(UE_height)
            if d_2D < d_BP:
                return self.sigma_SF_LOS_1
            else:
                return self.sigma_SF_LOS_2

        elif channel_condition == "NLOS":
            return self.sigma_SF_NLOS

    def _calculate_breakpoint_distance(self, UE_height: float) -> float:
        """
        Расчет дистанции излома (breakpoint distance).

        Args:
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Дистанция излома (м).

        """
        d_BP = (2 * np.pi * self.default_height * UE_height * self.bs.frequency_Hz) / 3.0e8
        return d_BP

    def _calculate_los_probability(self, UE_height: float, d_2D: float) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между UE и базовой станцией.

        Args:
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            float: Вероятность LOS в диапазоне [0, 1]

        """
        if d_2D <= 10:
            return 1.0

        los_probability = np.exp(-(d_2D - 10) / 1000)
        return los_probability

    def _calculate_los_path_loss(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для LOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для LOS случая (дБ).

        """
        d_BP = self._calculate_breakpoint_distance(UE_height)

        if 10 <= d_2D <= d_BP:
            PL1 = (
                20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_3D)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_3D
            )

            return PL1

        elif d_BP < d_2D <= 10000:
            PL1_at_dBP = (
                20 * np.log10(40 * np.pi * d_BP * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_BP)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_BP
            )
            PL2 = PL1_at_dBP + 40 * np.log10(d_3D / d_BP)

            return PL2

        else:
            return 10000.0

    def _calculate_nlos_path_loss(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для NLOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для NLOS случая (дБ).

        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss(
                UE_ID, 
                displacement, 
                channel_condition, 
                d_2D, 
                d_3D, 
                UE_height
            )
            PL_NLOS = (
                161.04
                - 7.1 * np.log10(self.W)
                + 7.5 * np.log10(self.h)
                - (24.37 - 3.7 * (self.h / self.default_height) ** 2)
                * np.log10(self.default_height)
                + (43.42 - 3.1 * np.log10(self.default_height)) * (np.log10(d_3D) - 3)
                + 20 * np.log10(self.bs.frequency_GHz)
                - (3.2 * (np.log10(11.75 * UE_height)) ** 2 - 4.97)
            )

            PL = max(PL_LOS, PL_NLOS)

            return PL

        else:
            return 10000.0


class UMaModel(ChannelModel):
    """
    Модель радиоканала для городской макросотовой местности (Urban Macro - UMa).
    Реализует расчеты для сценариев макросотового покрытия в городских условиях.
    """

    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        cond_update_period: float = 0.0,
        o2i_model: str = "low",
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        """
        Инициализация модели UMa.

        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел
                для крупномасштабных расчётов канала.
            cond_update_period (float, optional): Период обновления состояния
                радиоканала (мс). По умолчанию 0.0.
            o2i_model (str, optional): Модель проникновения в здание ('low' или 'high').
                По умолчанию "low".
            freq_fad_nlos_model (str, optional): Модель для получения
            частотно-селективных замираний при NLOS. По умолчанию "TDL-A".
            freq_fad_los_model (str, optional): Модель для получения
            частотно-селективных замираний при LOS. По умолчанию "TDL-D"
            ds_profile (str, optional): Профиль разброса задержек.
                По умолчанию "normal".
            los_arrival_angle (float, optional): Угол прихода LOS-компоненты.
                По умолчанию np.pi / 4.
             tdl_rng (Optional[RandomProvider]): Отдельный источник
                случайных чисел для TDL. Если не передан, используется rng.

        """
        super().__init__(
            bs=bs,
            rng=rng,
            cond_update_period=cond_update_period,
            freq_fad_nlos_model=freq_fad_nlos_model,
            freq_fad_los_model=freq_fad_los_model,
            ds_profile=ds_profile,
            los_arrival_angle=los_arrival_angle,
            spatial_consistency=False,
            tdl_rng=tdl_rng,
        )

        self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]
        self.default_height = 25.0

        self.sigma_SF_LOS = 4.0
        self.sigma_SF_NLOS = 6.0

        self.correlation_dist_LOS = 37
        self.correlation_dist_NLOS = 50

        self.o2i_model = o2i_model

    def _calculate_breakpoint_distance(
        self, 
        UE_ID: int,
        UE_height: float, 
        d_2D: float
    ) -> float:
        """
        Расчет дистанции излома (breakpoint distance).

        Args:
            UE_ID (int): ID пользователя для выбора подпотока RNG.
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            float: Дистанция излома (м).

        """
        if UE_height < 13:
            C = 0

        elif 13 <= UE_height <= 23:
            if d_2D <= 18:
                g = 0
            else:
                g = (5 / 4) * (d_2D / 100) ** 3 * np.exp(-d_2D / 150)

            C = ((UE_height - 13) / 10) ** 1.5 * g

        h_E_probability = 1 / (1 + C)

        if self.rng.uniform(
            0.0, 1.0, stream_offset=UE_ID
        )  <= h_E_probability:
            h_E = 1

        else:
            h_max = int(np.floor(UE_height - 1.5))
            h_Es = list(range(12, h_max + 1, 3))

            if not h_Es:
                h_Es = [12]

            idx = self.rng.get_integer(
                0,
                len(h_Es) - 1,
                stream_offset=UE_ID,
            )
            h_E = h_Es[idx]

        bs_height_prime = self.default_height - h_E
        UE_height_prime = UE_height - h_E

        d_BP = (4 * bs_height_prime * UE_height_prime * self.bs.frequency_Hz) / 3.0e8
        return d_BP

    def _calculate_los_probability(self, UE_height: float, d_2D: float) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между UE и базовой станцией.

        Args:
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            float: Вероятность LOS в диапазоне [0, 1]

        """
        if d_2D <= 18:
            return 1.0

        if UE_height <= 13:
            C_prime = 0
        elif 13 <= UE_height <= 23:
            C_prime = ((UE_height - 13) / 10) ** 1.5

        los_probability = ((18 / d_2D) + np.exp(-d_2D / 63) * (1 - (18 / d_2D))) * (
            1 + C_prime * (5 / 4) * (d_2D / 100) ** 3 * np.exp(-d_2D / 150)
        )

        return los_probability

    def _calculate_los_path_loss(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для LOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для LOS случая (дБ).

        """
        d_BP = self._calculate_breakpoint_distance(
            UE_ID,
            UE_height,
            d_2D,
            )

        if 10 <= d_2D <= d_BP:
            PL1 = 28 + 22 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)

            return PL1

        elif d_BP < d_2D <= 5000:
            PL2 = (
                28
                + 40 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 9 * np.log10(d_BP**2 + (self.default_height - UE_height) ** 2)
            )

            return PL2

        else:
            return 10000.0

    def _calculate_nlos_path_loss(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для NLOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для NLOS случая (дБ).

        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss(
                UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height
            )
            PL_NLOS = (
                13.54
                + 39.08 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 0.6 * (UE_height - 1.5)
            )

            PL = max(PL_LOS, PL_NLOS)

            return PL

        else:
            return 10000.0


class UMiModel(ChannelModel):
    """
    Модель радиоканала для городской микросотовой местности (Urban Micro - UMi).
    Реализует расчеты для сценариев микросотового покрытия в городских условиях.
    """

    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        cond_update_period: float = 0.0,
        o2i_model: str = "low",
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        """
        Инициализация модели UMi.

        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел
                для крупномасштабных расчётов канала.
            cond_update_period (float, optional): Период обновления состояния
                радиоканала (мс). По умолчанию 0.0.
            o2i_model (str, optional): Модель проникновения в здание ('low' или 'high').
                По умолчанию "low".
            freq_fad_nlos_model (str, optional): Модель для получения
            частотно-селективных замираний при NLOS. По умолчанию "TDL-A".
            freq_fad_los_model (str, optional): Модель для получения
            частотно-селективных замираний при LOS. По умолчанию "TDL-D"
            ds_profile (str, optional): Профиль разброса задержек.
                По умолчанию "normal".
            los_arrival_angle (float, optional): Угол прихода LOS-компоненты.
                По умолчанию np.pi / 4.
            tdl_rng (Optional[RandomProvider]): Отдельный источник
                случайных чисел для TDL. Если не передан, используется rng.
        Raises:
            ValueError: Если rng равен None или параметры TDL
                не проходят проверку допустимых значений.

        """
        if rng is None:
            raise ValueError("rng не может быть None")
        
        super().__init__(
            bs=bs,
            rng=rng,
            cond_update_period=cond_update_period,
            freq_fad_nlos_model=freq_fad_nlos_model,
            freq_fad_los_model=freq_fad_los_model,
            ds_profile=ds_profile,
            los_arrival_angle=los_arrival_angle,
            spatial_consistency=False,
            tdl_rng=tdl_rng,
        )

        self.bs.tx_power = self.bs.MICROCELL_TX_POWER[self.bs.bandwidth]
        self.default_height = 10.0

        self.sigma_SF_LOS = 4.0
        self.sigma_SF_NLOS = 7.82

        self.correlation_dist_LOS = 10
        self.correlation_dist_NLOS = 13

        self.o2i_model = o2i_model

    def _calculate_breakpoint_distance(
        self, 
        UE_height: float
    ) -> float:
        """
        Расчет дистанции излома (breakpoint distance).

        Args:
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Дистанция излома (м).

        """
        h_E = 1.0

        bs_height_prime = self.default_height - h_E
        UE_height_prime = UE_height - h_E

        d_BP = (4 * bs_height_prime * UE_height_prime * self.bs.frequency_Hz) / 3.0e8
        return d_BP

    def _calculate_los_probability(self, UE_height: float, d_2D: float) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между UE и базовой станцией.

        Args:
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            float: Вероятность LOS в диапазоне [0, 1]

        """
        if d_2D <= 18:
            return 1.0

        los_probability = (18 / d_2D) + np.exp(-d_2D / 36) * (1 - (18 / d_2D))

        return los_probability

    def _calculate_los_path_loss(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для LOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для LOS случая (дБ).

        """
        d_BP = self._calculate_breakpoint_distance(UE_height)

        if 10 <= d_2D <= d_BP:
            PL1 = 32.4 + 21 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)

            return PL1

        elif d_BP < d_2D <= 5000:
            PL2 = (
                32.4
                + 40 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 9.5 * np.log10(d_BP**2 + (self.default_height - UE_height) ** 2)
            )

            return PL2

        else:
            return 10000.0

    def _calculate_nlos_path_loss(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для NLOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для NLOS случая (дБ).

        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss(
                UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height
            )
            PL_NLOS = (
                35.3 * np.log10(d_3D)
                + 22.4
                + 21.3 * np.log10(self.bs.frequency_GHz)
                - 0.3 * (UE_height - 1.5)
            )

            PL = max(PL_LOS, PL_NLOS)

            return PL

        else:
            return 10000.0


class TDLModel:
    """
    Мелкомасштабная модель многолучевого канала на основе Tapped Delay Line
    по 3GPP TR 38.901. Выполняет моделирование частотно-селективных замираний
    и расчёт значения услиления/ослабления сигнала на различных частотах.
    """

    NUM_SINUSOIDS = 32

    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        nlos_model: str = "TDL-A",
        los_model: str = "TDL-D",
        delay_spread_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
    ):
        """
        Инициализация модели TDL.

        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел
                для углов прихода и начальных фаз лучей.
                Для независимости от крупномасштабных расчётов канала
                следует передавать отдельный провайдер.
            nlos_model (str, optional): TDL модель для NLOS (TDL-A/B/C).
                По умолчанию "TDL-A".
            los_model (str, optional): TDL модель для LOS (TDL-D/E).
                По умолчанию "TDL-D".
            delay_spread_profile (str, optional): Профиль разброса задержек.
                По умолчанию "normal".
            los_arrival_angle (float, optional): Угол прихода LOS-компоненты.
                По умолчанию np.pi / 4.
         Raises:
            ValueError: Если rng равен None или параметры модели
                не проходят проверку допустимых значений.

        """
        if rng is None:
            raise ValueError("rng должен быть передан в TDLModel")
        
        self._validate_parameters(nlos_model, los_model, delay_spread_profile, los_arrival_angle)

        self.params_table = GLOBALS.TDL_PARAMS_TABLE
        self.delay_spread_table = GLOBALS.TDL_DELAY_SPREAD_TABLE

        self.bs = bs
        self.rng = rng
        
        self.delay_spread_profile = delay_spread_profile
        self.los_arrival_angle = los_arrival_angle

        self.nlos_norm_delay = self._get_params_value(nlos_model, "normalized_delay")
        self.los_norm_delay = self._get_params_value(los_model, "normalized_delay")
        self.nlos_power_dB = self._get_params_value(nlos_model, "power_dB")
        self.los_power_dB = self._get_params_value(los_model, "power_dB")
        self.K_factor = self._get_params_value(los_model, "K_factor")

        self.lam = GLOBALS.SPEED_OF_LIGHT / self.bs.frequency_Hz
        self.rb_freqs = self._get_rb_frequencies()

        self.ues_info = {}

    def _validate_parameters(
        self, nlos_model: str, los_model: str, delay_spread_profile: str, los_arrival_angle: float
    ) -> None:
        """
        Валидация входных параметров.

        Args:
            nlos_model (str): TDL модель для NLOS (TDL-A/B/C).
            los_model (str): TDL модель для LOS (TDL-D/E).
            delay_spread_profile (str): Профиль разброса задержек.
            los_arrival_angle (float): Угол прихода LOS-компоненты.

        Raises:
            ValueError: Если передана недопустимая TDL модель, профиль разброса
            задержек или угол прихода LOS-компоненты вне диапазона.
            TypeError: Если угол прихода LOS-компоненты не является числом.

        """
        valid_nlos_models = ["TDL-A", "TDL-B", "TDL-C"]
        if nlos_model not in valid_nlos_models:
            raise ValueError(
                f"Недопустимая TDL модель для NLOS: '{nlos_model}'. "
                f"Разрешённые значения: {valid_nlos_models}"
            )

        valid_los_models = ["TDL-D", "TDL-E"]
        if los_model not in valid_los_models:
            raise ValueError(
                f"Недопустимая TDL модель для LOS: '{los_model}'. "
                f"Разрешённые значения: {valid_los_models}"
            )

        valid_ds_profiles = ["short", "normal", "long"]
        if delay_spread_profile not in valid_ds_profiles:
            raise ValueError(
                f"Недопустимый профиль задержек: '{delay_spread_profile}'. "
                f"Разрешённые значения: {valid_ds_profiles}"
            )

        if not isinstance(los_arrival_angle, (int, float)):
            raise TypeError("Значение los_arrival_angle должно быть числом (радианы)")

        if not (0 <= los_arrival_angle < 2 * np.pi):
            raise ValueError(
                f"Недопустимое los_arrival_angle: {los_arrival_angle}. "
                f"Значение должно быть в диапазоне [0, 2π)"
            )

    def _get_params_value(self, model: str, key: str):
        """
        Получение значения параметра TDL-модели из таблицы параметров.

        Args:
            model (str): Название TDL модели (TDL-A/B/C/D/E).
            key (str): Название параметра (normalized_delay, power_dB, K_factor).

        Returns:
            Any: Значение параметра из таблицы.

        """
        return self.params_table[model].get(key)

    def _get_delay_spread(self, channel_model: str, ue_class: str) -> float:
        """
        Получение значения разброса задержек на основе крупномасштабной модели
        канала и класса пользователя.

        Args:
            channel_model (str): Название крупномасштабной модели канала.
            ue_class (str): Класс UE.

        Returns:
            float: Значение разброса задержек (нс).

        """
        if ue_class in {"car", "indoor"} and channel_model != "RMaModel":
            table_key = "UMi_UMa_O2I"
        else:
            table_key = channel_model

        profile = (
            "normal"
            if self.delay_spread_profile == "short" and table_key == "UMi_UMa_O2I"
            else self.delay_spread_profile
        )

        return self.delay_spread_table[table_key].get(profile)

    def _get_rb_frequencies(self) -> np.ndarray:
        """
        Вычисление частотных смещений для ресурсных блоков (RB) относительно
        несущей.

        Returns:
            np.ndarray: Массив частотных смещений ресурсных блоков (Гц).

        """
        rb_idx = np.arange(-self.bs.rb_per_slot // 2, self.bs.rb_per_slot // 2)
        rb_freqs = rb_idx * GLOBALS.RB_BANDWIDTH

        return rb_freqs

    def _jakes_sum_of_sinusoids(
        self, f_doppler: float, alpha: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        """
        Генерация коэффициентов замирания канала методом суммы синусоид Джейкса.

        Args:
            f_doppler (float): Доплеровская частота (Гц).
            alpha (np.ndarray): Углы прихода синусоид (рад).
            phi (np.ndarray): Начальные фазы синусоид (рад).

        Returns:
            np.ndarray: Комплексные коэффициенты замирания для каждого луча.

        """
        t = GLOBALS.CURRENT_TIME * 1e-3
        arg = 2 * np.pi * f_doppler * t * np.cos(alpha) + phi
        tap_coeff = np.exp(1j * arg).sum(axis=1) / np.sqrt(self.NUM_SINUSOIDS)

        return tap_coeff

    def _calculate_freq_response(
        self,
        UE_ID: int, 
        channel_cond: str, 
        channel_model: str, 
        ue_class: str
    ) -> np.ndarray:
        """
        Расчёт частотной характеристики канала для заданного UE на всех RB.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            channel_cond (str): Состояние радиоканала (LOS/NLOS).
            channel_model (str): Название крупномасштабной модели канала.
            ue_class (str): Класс UE.

        Returns:
            np.ndarray: Частотная характеристика канала.

        """
        if UE_ID not in self.ues_info or self.ues_info[UE_ID]["cond"] != channel_cond:
            if channel_cond == "NLOS":
                powers_lin = 10 ** (self.nlos_power_dB / 10.0)
                normalized_delay = self.nlos_norm_delay

            elif channel_cond == "LOS":
                powers_lin = 10 ** (self.los_power_dB / 10.0)
                normalized_delay = self.los_norm_delay

            powers_lin = powers_lin / np.sum(powers_lin)

            delay_spread = self._get_delay_spread(channel_model, ue_class)
            delays_ns = normalized_delay * delay_spread

            sqrt_path_powers = np.sqrt(powers_lin)

            if UE_ID not in self.ues_info:
                max_L = max(len(self.nlos_norm_delay), len(self.los_norm_delay))

                n = np.arange(1, self.NUM_SINUSOIDS + 1)
                alpha_base = 2 * np.pi * n / self.NUM_SINUSOIDS
                delta_theta = np.pi / self.NUM_SINUSOIDS
                theta = np.array(
                    [
                        [
                            self.rng.uniform(
                                -delta_theta,
                                delta_theta,
                                stream_offset=UE_ID,
                            )
                            for _ in range(self.NUM_SINUSOIDS)
                        ]
                        for _ in range(max_L)
                    ],
                    dtype=np.float64,
                )
                alpha = alpha_base + theta
                phi = np.array(
                    [
                        [
                            self.rng.uniform(
                                -np.pi,
                                np.pi,
                                stream_offset=UE_ID,
                            )
                            for _ in range(self.NUM_SINUSOIDS)
                        ]
                        for _ in range(max_L)
                    ],
                    dtype=np.float64,
                )

                self.ues_info[UE_ID] = {"alpha": alpha, "phi": phi}

            delays_s = delays_ns * 1e-9
            path_phase_matrix = np.exp(
                -1j * 2 * np.pi * (self.rb_freqs[None, :] * delays_s[:, None])
            )
            self.ues_info[UE_ID].update(
                {
                    "cond": channel_cond,
                    "sqrt_path_powers": sqrt_path_powers,
                    "path_phase_matrix": path_phase_matrix,
                }
            )

        sqrt_path_powers = self.ues_info[UE_ID]["sqrt_path_powers"]
        alpha = self.ues_info[UE_ID]["alpha"]
        phi = self.ues_info[UE_ID]["phi"]
        path_phase_matrix = self.ues_info[UE_ID]["path_phase_matrix"]

        ue = self.bs.registered_ues.get(UE_ID)
        f_doppler = ue.velocity / self.lam

        L = len(path_phase_matrix)
        tap_coeff = self._jakes_sum_of_sinusoids(f_doppler, alpha[:L, :], phi[:L, :])

        if channel_cond == "LOS":
            K_factor_lin = 10 ** (self.K_factor / 10)

            los_weight = np.sqrt(K_factor_lin / (K_factor_lin + 1.0))
            nlos_weight = np.sqrt(1.0 / (K_factor_lin + 1.0))

            if "phi_los" not in self.ues_info[UE_ID]:
                phi_los = self.rng.uniform(
                    -np.pi,
                    np.pi,
                    stream_offset=UE_ID,
                )
                self.ues_info[UE_ID]["phi_los"] = phi_los
            else:
                phi_los = self.ues_info[UE_ID]["phi_los"]

            t = GLOBALS.CURRENT_TIME * 1e-3
            los_phase = 2.0 * np.pi * f_doppler * t * np.cos(self.los_arrival_angle) + phi_los

            los_comp = np.exp(1j * los_phase)
            tap_coeff[0] = los_weight * los_comp + nlos_weight * tap_coeff[0]

        scaled_tap_coeff = sqrt_path_powers * tap_coeff

        H_f = (scaled_tap_coeff[:, None] * path_phase_matrix).sum(axis=0)

        return H_f

    def calculate_channel_gain(
        self, 
        UE_ID: int, 
        channel_cond: str, 
        channel_model: str, 
        ue_class: str
    ) -> np.ndarray:
        """
        Расчет коэффициента усиления/ослабления канала в дБ для каждого
        ресурсного блока.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            channel_cond (str): Состояние радиоканала (LOS/NLOS).
            channel_model (str): Название крупномасштабной модели канала.
            ue_class (str): Класс UE.

        Returns:
            np.ndarray: Коэффициент усиления канала в дБ для каждого ресурсного
            блока.

        """

        if not isinstance(channel_cond, str) or channel_cond not in (
            "LOS", "NLOS"
        ):
            raise ValueError("channel_cond должен быть 'LOS' или 'NLOS'")
        
        H_f = self._calculate_freq_response(UE_ID, channel_cond, channel_model, ue_class)
        #@sherokiddo: я осмелился добавить сюда нормализацию значений
        #аналогично той, что есть в MATLAB
        gain_lin = np.abs(H_f) ** 2
        gain_lin_norm = gain_lin / np.mean(gain_lin)
        gain_dB = 10 * np.log10(gain_lin_norm)
        return gain_dB


class RMaModel_SC(ChannelModel):
    """
    Модель радиоканала для сельской местности (Rural Macro - RMa).
    Реализует расчеты для сценариев макросотового покрытия в сельской местности.
    """

    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        W: float = 20.0,
        h: float = 5.0,
        cond_update_period: float = 0.0,
        spatial_consistency: bool = True,
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        
        """
        Инициализация модели RMa с пространственной согласованностью.

        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел
                для крупномасштабных расчётов канала.
            W (float): Средняя ширина улиц (м). По умолчанию 20.0.
            h (float): Средняя высота зданий (м). По умолчанию 5.0.
            cond_update_period (float): Период обновления состояния
                канала (мс). По умолчанию 0.0.
            spatial_consistency (bool): Включение пространственной
                согласованности канала. По умолчанию True.
            freq_fad_nlos_model (str): TDL-профиль для NLOS.
                По умолчанию "TDL-A".
            freq_fad_los_model (str): TDL-профиль для LOS.
                По умолчанию "TDL-D".
            ds_profile (str): Профиль разброса задержек:
                "short", "normal" или "long". По умолчанию "normal".
            los_arrival_angle (float): Угол прихода LOS-компоненты
                (радианы). По умолчанию np.pi / 4.
            tdl_rng (Optional[RandomProvider]): Отдельный источник
                случайных чисел для TDL. Если не передан, используется rng.

        Raises:
            ValueError: Если rng равен None или параметры TDL
                не проходят проверку допустимых значений.
        """
        super().__init__(
            bs, 
            rng,
            cond_update_period,
            freq_fad_nlos_model,
            freq_fad_los_model,
            ds_profile,
            los_arrival_angle, 
            spatial_consistency,
            tdl_rng,
            )

        self.W = W
        self.h = h

        self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]
        self.default_height = 35.0

        self.sigma_SF_LOS_1 = 4.0
        self.sigma_SF_LOS_2 = 6.0
        self.sigma_SF_NLOS = 8.0

        if spatial_consistency:
            self.sf_correlation_distance_los = 37.0
            self.sf_correlation_distance_nlos = 120.0
            self.los_correlation_length = 60.0
        else:
            self.correlation_dist_LOS = 37
            self.correlation_dist_NLOS = 120

        self.o2i_model = "low"

    def _get_sigma_sf(self, channel_condition: str, d_2D: float, UE_height: float) -> float:
        """
        Возвращает стандартное отклонение теневого затенения (SF) для
        заданного состояния канала (LOS/NLOS).

        Args:
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Стандартное отклонение теневого затенения (дБ).

        """
        if channel_condition == "LOS":
            d_BP = self._calculate_breakpoint_distance(UE_height)
            if d_2D < d_BP:
                return self.sigma_SF_LOS_1
            else:
                return self.sigma_SF_LOS_2

        elif channel_condition == "NLOS":
            return self.sigma_SF_NLOS

    def _calculate_breakpoint_distance(self, UE_height: float) -> float:
        """
        Расчет дистанции излома (breakpoint distance).

        Args:
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Дистанция излома (м).

        """
        d_BP = (2 * np.pi * self.default_height * UE_height * self.bs.frequency_Hz) / 3.0e8
        return d_BP

    def _calculate_los_probability(
        self,
        UE_height: float, 
        d_2D: float,
    ) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между UE и базовой станцией.

        Args:
            UE_height (float): Высота пользовательского устройства (м).
            d_2D (float): Горизонтальное расстояние между БС и UE (м).

        Returns:
            float: Вероятность LOS в диапазоне [0, 1]

        """
        if d_2D <= 10:
            return 1.0

        los_probability = np.exp(-(d_2D - 10) / 1000)
        return los_probability

    def _calculate_los_path_loss_time_based(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для LOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для LOS случая (дБ).

        """
        d_BP = self._calculate_breakpoint_distance(UE_height)

        if 10 <= d_2D <= d_BP:
            PL1 = (
                20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_3D)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_3D
            )

            return PL1

        elif d_BP < d_2D <= 10000:
            PL1_at_dBP = (
                20 * np.log10(40 * np.pi * d_BP * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_BP)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_BP
            )
            PL2 = PL1_at_dBP + 40 * np.log10(d_3D / d_BP)

            return PL2

        else:
            return 10000.0

    def _calculate_nlos_path_loss_time_based(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для NLOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для NLOS случая (дБ).

        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss_time_based(
                UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height
            )
            PL_NLOS = (
                161.04
                - 7.1 * np.log10(self.W)
                + 7.5 * np.log10(self.h)
                - (24.37 - 3.7 * (self.h / self.default_height) ** 2)
                * np.log10(self.default_height)
                + (43.42 - 3.1 * np.log10(self.default_height)) * (np.log10(d_3D) - 3)
                + 20 * np.log10(self.bs.frequency_GHz)
                - (3.2 * (np.log10(11.75 * UE_height)) ** 2 - 4.97)
            )

            PL = max(PL_LOS, PL_NLOS)

            return PL

        else:
            return 10000.0

    def _calculate_los_path_loss_sc(
        self,
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        Расчет затухания сигнала для LOS условий (SC).
        
        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Позиция UE [x, y] в метрах.
            UE_ID (int): ID пользователя для выбора подпотока RNG.
                По умолчанию 0 для совместимости старых вызовов.

        Returns:
            float: LOS-потери с учётом теневого затенения (дБ).
                Вне допустимого диапазона расстояний — 10000.0.
        """
        d_BP = self._calculate_breakpoint_distance(UE_height)

        if 10 <= d_2D <= d_BP:
            PL_base = (
                20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_3D)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_3D
            )

            # Корреляционный shadow fading
            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position, 
                self.sigma_SF_LOS_1, 
                is_los=True,
                UE_ID=UE_ID,
            )

            return PL_base + shadow_fading

        elif d_BP < d_2D <= 10000:
            PL1_at_dBP = (
                20 * np.log10(40 * np.pi * d_BP * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_BP)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_BP
            )
            PL_base = PL1_at_dBP + 40 * np.log10(d_3D / d_BP)

            # Корреляционный shadow fading
            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position, 
                self.sigma_SF_LOS_2, 
                is_los=True,
                UE_ID=UE_ID,
            )

            return PL_base + shadow_fading
        else:
            return 10000.0

    def _calculate_nlos_path_loss_sc(
        self, 
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        Расчет затухания сигнала для NLOS условий (SC).
        
        Выбирается максимум базовых LOS- и NLOS-потерь,
        затем добавляется коррелированное теневое затенение.

        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Позиция UE [x, y] в метрах.
            UE_ID (int): ID пользователя для выбора подпотока RNG.
                По умолчанию 0 для совместимости старых вызовов.

        Returns:
            float: NLOS-потери с учётом теневого затенения (дБ).
                Вне допустимого диапазона расстояний — 10000.0.
        """
        if 10 <= d_2D <= 5000:
            # Базовый расчет без shadow fading
            PL_LOS_base = self._calculate_los_path_loss_base(d_2D, d_3D, UE_height)

            PL_NLOS_base = (
                161.04
                - 7.1 * np.log10(self.W)
                + 7.5 * np.log10(self.h)
                - (24.37 - 3.7 * (self.h / self.default_height) ** 2)
                * np.log10(self.default_height)
                + (43.42 - 3.1 * np.log10(self.default_height)) * (np.log10(d_3D) - 3)
                + 20 * np.log10(self.bs.frequency_GHz)
                - (3.2 * (np.log10(11.75 * UE_height)) ** 2 - 4.97)
            )

            PL_base = max(PL_LOS_base, PL_NLOS_base)

            # Корреляционный shadow fading
            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position, 
                self.sigma_SF_NLOS, 
                is_los=False,
                UE_ID=UE_ID,
            )

            return PL_base + shadow_fading
        else:
            return 10000.0

    def _calculate_los_path_loss_base(self, d_2D: float, d_3D: float, UE_height: float) -> float:
        """Базовый расчет LOS path loss без shadow fading."""
        d_BP = self._calculate_breakpoint_distance(UE_height)

        if 10 <= d_2D <= d_BP:
            return (
                20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_3D)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_3D
            )
        elif d_BP < d_2D <= 10000:
            PL1_at_dBP = (
                20 * np.log10(40 * np.pi * d_BP * self.bs.frequency_GHz / 3.0)
                + min(0.03 * self.h**1.72, 10) * np.log10(d_BP)
                - min(0.044 * self.h**1.72, 14.77)
                + 0.002 * np.log10(self.h) * d_BP
            )
            return PL1_at_dBP + 40 * np.log10(d_3D / d_BP)
        else:
            return 10000.0


class UMaModel_SC(ChannelModel):
    """
    Модель радиоканала для городской макросотовой местности (Urban Macro - UMa).
    Реализует расчеты для сценариев макросотового покрытия в городских условиях.
    """

    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        cond_update_period: float = 0.0,
        o2i_model: str = "low",
        spatial_consistency: bool = True,
        UE_height: float = 1.5,
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        """
        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел
                для крупномасштабных расчётов канала.
            cond_update_period (float): Период обновления состояния
                канала (мс). По умолчанию 0.0.
            o2i_model (str): Модель проникновения сигнала в здание:
                "low" или "high". По умолчанию "low".
            spatial_consistency (bool): Включение пространственной
                согласованности канала. По умолчанию True.
            UE_height (float): Высота пользовательского устройства
                (м). По умолчанию 1.5.
            freq_fad_nlos_model (str): TDL-профиль для NLOS.
                По умолчанию "TDL-A".
            freq_fad_los_model (str): TDL-профиль для LOS.
                По умолчанию "TDL-D".
            ds_profile (str): Профиль разброса задержек:
                "short", "normal" или "long". По умолчанию "normal".
            los_arrival_angle (float): Угол прихода LOS-компоненты
                (радианы). По умолчанию np.pi / 4.
            tdl_rng (Optional[RandomProvider]): Отдельный источник
                случайных чисел для TDL. Если не передан, используется rng.

        Raises:
            ValueError: Если rng равен None или параметры TDL
                не проходят проверку допустимых значений.
        """


        super().__init__(
            bs,
            rng,
            cond_update_period,
            freq_fad_nlos_model,
            freq_fad_los_model,
            ds_profile,
            los_arrival_angle,
            spatial_consistency,
            tdl_rng,
            )

        self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]
        self.bs.height = 25.0
        self.UE_height = UE_height

        self.sigma_SF_LOS = 4.0
        self.sigma_SF_NLOS = 6.0

        # Параметры корреляции
        if spatial_consistency:
            self.sf_correlation_distance_los = 37.0
            self.sf_correlation_distance_nlos = 50.0
            self.los_correlation_length = 120.0
        else:
            self.correlation_dist_LOS = 37
            self.correlation_dist_NLOS = 50

        self.o2i_model = o2i_model

    def _calculate_breakpoint_distance(
        self, 
        UE_ID: int,
        UE_height: float, 
        d_2D: float
    ) -> float:
        """
        Расчет дистанции излома (breakpoint distance).

        Args:
            UE_ID (int): ID пользователя для выбора подпотока RNG.
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            float: Дистанция излома (м).

        """
        if UE_height < 13:
            C = 0

        elif 13 <= UE_height <= 23:
            if d_2D <= 18:
                g = 0
            else:
                g = (5 / 4) * (d_2D / 100) ** 3 * np.exp(-d_2D / 150)

            C = ((UE_height - 13) / 10) ** 1.5 * g

        h_E_probability = 1 / (1 + C)

        if self.rng.uniform(
            0.0, 1.0, stream_offset=UE_ID
        ) <= h_E_probability:
            h_E = 1

        else:
            h_max = int(np.floor(UE_height - 1.5))
            h_Es = list(range(12, h_max + 1, 3))

            if not h_Es:
                h_Es = [12]

            idx = self.rng.get_integer(
                0,
                len(h_Es) - 1,
                stream_offset=UE_ID,
            )
            h_E = h_Es[idx]

        bs_height_prime = self.bs.height - h_E
        UE_height_prime = UE_height - h_E

        d_BP = (4 * bs_height_prime * UE_height_prime * self.bs.frequency_Hz) / 3.0e8
        return d_BP

    def _calculate_los_probability(
        self, 
        UE_height: float,
        d_2D: float,
    ) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между UE и базовой станцией.

        Args:
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            float: Вероятность LOS в диапазоне [0, 1]

        Raises:
        ValueError: Если высота не является конечным числом
            или превышает поддерживаемую формулой границу 23 м.
        """
        if not np.isfinite(UE_height) or UE_height > 23:
            raise ValueError(
                "Для UMa_SC высота UE должна быть конечной "
                "и не превышать 23 м"
            )
        
        if d_2D <= 18:
            return 1.0

        if UE_height <= 13:
            C_prime = 0
        elif 13 <= UE_height <= 23:
            C_prime = ((UE_height - 13) / 10) ** 1.5

        los_probability = ((18 / d_2D) + np.exp(-d_2D / 63) * (1 - (18 / d_2D))) * (
            1 + C_prime * (5 / 4) * (d_2D / 100) ** 3 * np.exp(-d_2D / 150)
        )

        return los_probability

    def _calculate_los_path_loss_time_based(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для LOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для LOS случая (дБ).

        """
        d_BP = self._calculate_breakpoint_distance(
            UE_ID,
            UE_height, 
            d_2D)

        if 10 <= d_2D <= d_BP:
            PL1 = 28 + 22 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)

            return PL1

        elif d_BP < d_2D <= 5000:
            PL2 = (
                28
                + 40 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 9 * np.log10(d_BP**2 + (self.bs.height - UE_height) ** 2)
            )

            return PL2

        else:
            return 10000.0

    def _calculate_nlos_path_loss_time_based(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для NLOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для NLOS случая (дБ).

        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss_time_based(
                UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height
            )
            PL_NLOS = (
                13.54
                + 39.08 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 0.6 * (UE_height - 1.5)
            )

            PL = max(PL_LOS, PL_NLOS)

            return PL

        else:
            return 10000.0

    def _calculate_los_path_loss_sc(
        self, 
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        Расчёт LOS-потерь для UMa в SC-режиме.

        Передаёт UE_ID в расчёт дистанции излома
        и коррелированного теневого затенения.

        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Позиция UE [x, y] в метрах.
            UE_ID (int): ID пользователя для выбора подпотока RNG.
                По умолчанию 0 для совместимости старых вызовов.

        Returns:
            float: LOS-потери с учётом теневого затенения (дБ).
                Вне допустимого диапазона расстояний — 10000.0.
        """
        d_BP = self._calculate_breakpoint_distance(
            UE_ID,
            UE_height, 
            d_2D,
            )

        if 10 <= d_2D <= d_BP:
            PL_base = 28 + 22 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)
            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position,
                self.sigma_SF_LOS, 
                is_los=True,
                UE_ID=UE_ID,
            )
            return PL_base + shadow_fading

        elif d_BP < d_2D <= 5000:
            PL_base = (
                28
                + 40 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 9 * np.log10(d_BP**2 + (self.bs.height - UE_height) ** 2)
            )
            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position,
                self.sigma_SF_LOS, 
                is_los=True,
                UE_ID=UE_ID,
            )
            return PL_base + shadow_fading
        else:
            return 10000.0

    def _calculate_nlos_path_loss_sc(
        self, 
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        Расчёт NLOS-потерь для UMa в SC-режиме.

        Выбирается максимум базовых LOS- и NLOS-потерь,
        затем добавляется коррелированное теневое затенение.

        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Позиция UE [x, y] в метрах.
            UE_ID (int): ID пользователя для выбора подпотока RNG.
                По умолчанию 0 для совместимости старых вызовов.

        Returns:
            float: NLOS-потери с учётом теневого затенения (дБ).
                Вне допустимого диапазона расстояний — 10000.0.
        """
        if 10 <= d_2D <= 5000:
            # Базовый LOS расчет без shadow fading
            d_BP = self._calculate_breakpoint_distance(
                UE_ID,
                UE_height,
                d_2D,
                )

            if 10 <= d_2D <= d_BP:
                PL_LOS_base = 28 + 22 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)
            else:
                PL_LOS_base = (
                    28
                    + 40 * np.log10(d_3D)
                    + 20 * np.log10(self.bs.frequency_GHz)
                    - 9 * np.log10(d_BP**2 + (self.bs.height - UE_height) ** 2)
                )

            PL_NLOS_base = (
                13.54
                + 39.08 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 0.6 * (UE_height - 1.5)
            )

            PL_base = max(PL_LOS_base, PL_NLOS_base)

            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position,
                self.sigma_SF_NLOS, 
                is_los=False,
                UE_ID=UE_ID,
            )

            return PL_base + shadow_fading
        else:
            return 10000.0


class UMiModel_SC(ChannelModel):
    """
    Модель радиоканала для городской микросотовой местности (Urban Micro - UMi).
    Реализует расчеты для сценариев микросотового покрытия в городских условиях.
    """

    def __init__(
        self,
        bs: BaseStation,
        rng: RandomProvider,
        cond_update_period: float = 0.0,
        o2i_model: str = "low",
        spatial_consistency: bool = True,
        UE_height: float = 1.5,
        freq_fad_nlos_model: str = "TDL-A",
        freq_fad_los_model: str = "TDL-D",
        ds_profile: str = "normal",
        los_arrival_angle: float = np.pi / 4,
        tdl_rng: Optional[RandomProvider] = None,
    ):
        """
        Инициализация модели UMi с пространственной согласованностью.

        Args:
            bs (BaseStation): Объект базовой станции.
            rng (RandomProvider): Внешний источник случайных чисел
                для крупномасштабных расчётов канала.
            cond_update_period (float): Период обновления состояния
                канала (мс). По умолчанию 0.0.
            o2i_model (str): Модель проникновения сигнала в здание:
                "low" или "high". По умолчанию "low".
            spatial_consistency (bool): Включение пространственной
                согласованности канала. По умолчанию True.
            UE_height (float): Высота пользовательского устройства
                (м). По умолчанию 1.5.
            freq_fad_nlos_model (str): TDL-профиль для NLOS.
                По умолчанию "TDL-A".
            freq_fad_los_model (str): TDL-профиль для LOS.
                По умолчанию "TDL-D".
            ds_profile (str): Профиль разброса задержек:
                "short", "normal" или "long". По умолчанию "normal".
            los_arrival_angle (float): Угол прихода LOS-компоненты
                (радианы). По умолчанию np.pi / 4.
            tdl_rng (Optional[RandomProvider]): Отдельный источник
                случайных чисел для TDL. Если не передан, используется rng.

        Raises:
            ValueError: Если rng равен None или параметры TDL
                не проходят проверку допустимых значений.
        """

        
        super().__init__(
            bs,
            rng, 
            cond_update_period, 
            freq_fad_nlos_model,
            freq_fad_los_model,
            ds_profile,
            los_arrival_angle,
            spatial_consistency,
            tdl_rng,
            )

        self.bs.tx_power = self.bs.MICROCELL_TX_POWER[self.bs.bandwidth]
        self.bs.height = 10.0
        self.UE_height = UE_height

        self.sigma_SF_LOS = 4.0
        self.sigma_SF_NLOS = 7.82

        if spatial_consistency:
            self.sf_correlation_distance_los = 10.0
            self.sf_correlation_distance_nlos = 13.0
            self.los_correlation_length = 50.0
        else:
            self.correlation_dist_LOS = 10
            self.correlation_dist_NLOS = 13

        self.o2i_model = o2i_model

    def _calculate_breakpoint_distance(self, UE_height: float) -> float:
        """
        Расчет дистанции излома (breakpoint distance).

        Args:
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Дистанция излома (м).

        """
        h_E = 1.0

        bs_height_prime = self.bs.height - h_E
        UE_height_prime = UE_height - h_E

        d_BP = (4 * bs_height_prime * UE_height_prime * self.bs.frequency_Hz) / 3.0e8
        return d_BP

    def _calculate_los_probability(
        self, 
        UE_height: float,
        d_2D: float,
    ) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между UE и базовой станцией.

        Args:
            UE_height (float): Высота пользовательского оборудования (м).
            d_2D (float): 2D расстояние между БС и UE (м).

        Returns:
            float: Вероятность LOS в диапазоне [0, 1]

        """
        if d_2D <= 18:
            return 1.0

        los_probability = (18 / d_2D) + np.exp(-d_2D / 36) * (1 - (18 / d_2D))

        return los_probability

    def _calculate_los_path_loss_time_based(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для LOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для LOS случая (дБ).

        """
        d_BP = self._calculate_breakpoint_distance(
            UE_height,
            )

        if 10 <= d_2D <= d_BP:
            PL1 = 32.4 + 21 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)

            return PL1

        elif d_BP < d_2D <= 5000:
            PL2 = (
                32.4
                + 40 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 9.5 * np.log10(d_BP**2 + (self.bs.height - UE_height) ** 2)
            )

            return PL2

        else:
            return 10000.0

    def _calculate_nlos_path_loss_time_based(
        self,
        UE_ID: int,
        displacement: float,
        channel_condition: str,
        d_2D: float,
        d_3D: float,
        UE_height: float,
    ) -> float:
        """
        Расчет затухания сигнала для NLOS условий.

        Args:
            UE_ID (int): Уникальный идентификатор UE.
            displacement (float): Смещение UE относительно предыдущего положения (м).
            channel_condition (str): Состояние радиоканала (LOS/NLOS).
            d_2D (float): 2D расстояние между БС и UE (м).
            d_3D (float): 3D расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского оборудования (м).

        Returns:
            float: Затухание сигнала для NLOS случая (дБ).

        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss_time_based(
                UE_ID, displacement, channel_condition, d_2D, d_3D, UE_height
            )
            PL_NLOS = (
                35.3 * np.log10(d_3D)
                + 22.4
                + 21.3 * np.log10(self.bs.frequency_GHz)
                - 0.3 * (UE_height - 1.5)
            )

            PL = max(PL_LOS, PL_NLOS)

            return PL

        else:
            return 10000.0

    def _calculate_los_path_loss_sc(
        self, 
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """
        Интерфейс расчёта LOS-потерь в SC-режиме.

        Метод должен быть переопределён в дочерней модели канала.

        Args:
            d_2D (float): Горизонтальное расстояние между БС и UE (м).
            d_3D (float): Пространственное расстояние между БС и UE (м).
            UE_height (float): Высота пользовательского устройства (м).
            current_position (np.ndarray): Текущая позиция UE
                [x, y] в метрах.
            UE_ID (int): Уникальный идентификатор UE.
                Значение 0 оставлено для совместимости старых вызовов.
                При расчёте необходимо передавать ID текущего UE.

        Returns:
            float: Потери распространения в условиях LOS (дБ).

        """
        d_BP = self._calculate_breakpoint_distance(UE_height)

        if 10 <= d_2D <= d_BP:
            PL_base = 32.4 + 21 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)
            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position,
                self.sigma_SF_LOS, 
                is_los=True,
                UE_ID=UE_ID,
            )
            return PL_base + shadow_fading

        elif d_BP < d_2D <= 5000:
            PL_base = (
                32.4
                + 40 * np.log10(d_3D)
                + 20 * np.log10(self.bs.frequency_GHz)
                - 9.5 * np.log10(d_BP**2 + (self.bs.height - UE_height) ** 2)
            )
            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position,
                self.sigma_SF_LOS,
                is_los=True, 
                UE_ID=UE_ID,
            )
            return PL_base + shadow_fading
        else:
            return 10000.0

    def _calculate_nlos_path_loss_sc(
        self, 
        d_2D: float, 
        d_3D: float, 
        UE_height: float, 
        current_position: np.ndarray,
        UE_ID: int = 0,
    ) -> float:
        """SC NLOS path loss для UMi."""
        if 10 <= d_2D <= 5000:
            # Базовый LOS расчет без shadow fading
            d_BP = self._calculate_breakpoint_distance(UE_height)

            if 10 <= d_2D <= d_BP:
                PL_LOS_base = 32.4 + 21 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)
            else:
                PL_LOS_base = (
                    32.4
                    + 40 * np.log10(d_3D)
                    + 20 * np.log10(self.bs.frequency_GHz)
                    - 9.5 * np.log10(d_BP**2 + (self.bs.height - UE_height) ** 2)
                )

            PL_NLOS_base = (
                35.3 * np.log10(d_3D)
                + 22.4
                + 21.3 * np.log10(self.bs.frequency_GHz)
                - 0.3 * (UE_height - 1.5)
            )

            PL_base = max(PL_LOS_base, PL_NLOS_base)

            shadow_fading = self._calculate_correlated_shadow_fading_sc(
                current_position, 
                self.sigma_SF_NLOS, 
                is_los=False,
                UE_ID=UE_ID,
            )

            return PL_base + shadow_fading
        else:
            return 10000.0
