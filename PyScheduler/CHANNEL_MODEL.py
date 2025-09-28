"""
#------------------------------------------------------------------------------
# Модуль: CHANNEL_MODEL - Модели радиоканалов для различных сценариев развертывания
#------------------------------------------------------------------------------
# Описание:
#   Реализация стандартных моделей радиоканалов согласно спецификациям 3GPP TR 38.901.
#   Включает модели для сельской (RMa), городской макросотовой (UMa) и городской 
#   микросотовой (UMi) местности с поддержкой различных частотных диапазонов.
#
# Версия: 1.0.0
# Дата последнего изменения: 2025-04-01
# Автор: Норицин Иван
# Версия Python Kernel: 3.12.9
#------------------------------------------------------------------------------
"""
import numpy as np
import GLOBALS
from BS_MODULE import BaseStation

class ChannelModel:
    """
    """
    
    SHADOW_FADING_INFO = {}
    CHANNEL_COND_INFO = {}
    O2I_INFO = {}
    
    def __init__(self, bs: BaseStation, cond_update_period: float = 0.0):
        """
        """
        self.bs = bs
        self.cond_update_period = cond_update_period
        
    def _get_sigma_sf(self, channel_condition: str, d_2D: float = None, 
                      UE_height: float = None) -> float:
        """
        """
        if channel_condition == "LOS":
            return getattr(self, "sigma_SF_LOS", None)
        
        elif channel_condition == "NLOS":
            return getattr(self, "sigma_SF_NLOS", None)
        
    def _calculate_path_loss(self, UE_ID: int, displacement: float, d_2D: float, 
                            d_2D_in: float, d_3D: float, UE_height: float, 
                            ue_class: str) -> float:
        """
        Расчет затухания сигнала с учетом LOS/NLOS условий.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_2D_in: Внутреннее расстояние в здании в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
            ue_class: Класс UE
    
        Returns:
            PL: Затухание сигнала в дБ
        """
        channel_condition = self._calculate_channel_condition(UE_ID,
                                                              self.cond_update_period, 
                                                              UE_height, 
                                                              d_2D)
        
        sigma_sf = self._get_sigma_sf(channel_condition, d_2D, UE_height)
        
        correlation_dist = (self.correlation_dist_LOS if channel_condition == "LOS"
                    else self.correlation_dist_NLOS)
        
        shadow_fading = self._calculate_shadow_fading(UE_ID, 
                                                     displacement, 
                                                     channel_condition, 
                                                     sigma_sf, 
                                                     correlation_dist)
        
        if channel_condition == "LOS":
            PL = self._calculate_los_path_loss(UE_ID, displacement, 
                                               channel_condition, d_2D, d_3D, 
                                               UE_height)
            PL = PL + shadow_fading
        
        elif channel_condition == "NLOS":
            PL = self._calculate_nlos_path_loss(UE_ID, displacement, 
                                           channel_condition, d_2D, d_3D, 
                                           UE_height)
            PL = PL + shadow_fading
            
        if ue_class == "indoor":
            o2i_penetration_loss = self._calculate_o2i_building_loss(UE_ID, 
                                                                    channel_condition,
                                                                    self.o2i_model, 
                                                                    d_2D_in, 
                                                                    self.bs.frequency_GHz)
            PL = PL + o2i_penetration_loss
            
        elif ue_class == "car":
            o2i_penetration_loss = self._calculate_o2i_car_loss(UE_ID)
            PL = PL + o2i_penetration_loss
            
        return PL
    
    def _calculate_channel_condition(self, UE_ID: int, cond_update_period: float, 
                                    UE_height: float, d_2D: float) -> str:
        """
        """
        channel_condition = None
        not_found = False
        update = False
        
        if UE_ID in ChannelModel.CHANNEL_COND_INFO:
            ue_cond_info = ChannelModel.CHANNEL_COND_INFO[UE_ID]
            channel_condition = ue_cond_info["cond"]
            
            if (cond_update_period != 0.0 and GLOBALS.CURRENT_TIME - 
                ue_cond_info["updated_time"] > cond_update_period):
                update = True  
                
        else:
            not_found = True
            ChannelModel.CHANNEL_COND_INFO[UE_ID] = {"cond": None,
                                                     "updated_time": None}
            ue_cond_info = ChannelModel.CHANNEL_COND_INFO[UE_ID]
            
        if not_found or update:
            los_probability = self._calculate_los_probability(UE_height, d_2D)
            channel_condition = ("LOS" if np.random.random() <= los_probability 
                                 else "NLOS")
            
            ue_cond_info["cond"] = channel_condition
            ue_cond_info["updated_time"] = GLOBALS.CURRENT_TIME
            
        return channel_condition
    
    def _calculate_shadow_fading(self, UE_ID: int, displacement: float, 
                                channel_condition: str, sigma_SF: float,
                                correlation_dist: float) -> float:
        """
        """
        shadow_fading = None
        not_found = False
        new_condition = False
        
        if UE_ID in ChannelModel.SHADOW_FADING_INFO:
            ue_sf_info = ChannelModel.SHADOW_FADING_INFO[UE_ID]
            new_condition = ue_sf_info["cond"] != channel_condition
        else:
            not_found = True
            ChannelModel.SHADOW_FADING_INFO[UE_ID] = {"SF_value": None,
                                                      "cond": None}
            ue_sf_info = ChannelModel.SHADOW_FADING_INFO[UE_ID]
            
        if not_found or new_condition:
            shadow_fading = np.random.normal(0, sigma_SF)
        else:
            R = np.exp(-1 * displacement / correlation_dist)
            shadow_fading = (R * ue_sf_info["SF_value"] + np.sqrt(1 - R * R) * 
                             np.random.normal(0, sigma_SF))
            
        ue_sf_info["SF_value"] = shadow_fading
        ue_sf_info["cond"] = channel_condition
        
        return shadow_fading
    
    def _calculate_o2i_building_loss(self, UE_ID: int, channel_condition: str, 
                                    model_type: str, d_2D_in: float, 
                                    fc_GHz: float) -> float:
        """
        """
        o2i_loss = None
        not_found = False
        new_condition = False
        
        if UE_ID in ChannelModel.O2I_INFO:
            ue_o2i_info = ChannelModel.O2I_INFO[UE_ID]
            new_condition = ue_o2i_info["cond"] != channel_condition
        else:
            not_found = True
            ChannelModel.O2I_INFO[UE_ID] = {"o2i_loss": None,
                                            "cond": None}
            ue_o2i_info = ChannelModel.O2I_INFO[UE_ID]
            
        if not_found or new_condition:
            L_concrete = 5 + 4 * fc_GHz
            
            if model_type == "low":
                L_glass = 2 + 0.2 * fc_GHz
                PL_tw = 5 - 10 * np.log10(0.3 * 10**(-L_glass / 10) + 0.7 * 10**(-L_concrete / 10))
                sigma_p = 4.4
                
            elif model_type == "high":
                L_IRR_glass = 23 + 0.3 * fc_GHz
                PL_tw = 5 - 10 * np.log10(0.7 * 10**(-L_IRR_glass / 10) + 0.3 * 10**(-L_concrete/10))
                sigma_p = 6.5
                
            PL_in = 0.5 * d_2D_in
            random_loss = np.random.normal(0, sigma_p)
            o2i_loss = PL_tw + PL_in + random_loss
            
        else:
            o2i_loss = ue_o2i_info["o2i_loss"]
            
        ue_o2i_info["o2i_loss"] = o2i_loss
        ue_o2i_info["cond"] = channel_condition
        
        return o2i_loss
    
    def _calculate_o2i_car_loss(self, UE_ID: int):
        """
        """
        o2i_loss = None
        not_found = False
        
        if UE_ID in ChannelModel.O2I_INFO:
            ue_o2i_info = ChannelModel.O2I_INFO[UE_ID]
        else:
            not_found = True
            ChannelModel.O2I_INFO[UE_ID] = {"o2i_loss": None,
                                            "cond": None}
            ue_o2i_info = ChannelModel.O2I_INFO[UE_ID]
            
        if not_found:
            o2i_loss = np.random.normal(9, 5)
        else:
            o2i_loss = ue_o2i_info["o2i_loss"]
            
        ue_o2i_info["o2i_loss"] = o2i_loss
        
        return o2i_loss
        
    def calculate_SINR(self, UE_ID: int, displacement: float, d_2D: float, 
                       d_2D_in: float, d_3D: float, UE_height: float, 
                       ue_class: str) -> float:
        """
        Расчет отношения сигнал-интерференция-шум (SINR).
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_2D_in: Внутреннее расстояние в здании в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
            ue_class: Класс UE
    
        Returns:
            Значение SINR в дБ
        """
        path_loss = self._calculate_path_loss(UE_ID, displacement, d_2D, d_2D_in, 
                                              d_3D, UE_height, ue_class)
        cable_loss = 2
        interference_margin = 2
        
        P_signal = self.bs.tx_power + self.bs.antenna_gain - cable_loss - path_loss - interference_margin
            
        P_interference = -95
        P_noise = -174 + 10 * np.log10(self.bs.bandwidth * 1e6)
        
        SINR = P_signal - 10 * np.log10(10 ** (P_interference / 10) + 10 ** (P_noise / 10))
        
        return SINR


class RMaModel(ChannelModel):
    """
    Модель радиоканала для сельской местности (Rural Macro - RMa).
    Реализует расчеты для сценариев макросотового покрытия в сельской местности.
    """
    def __init__(self, bs: BaseStation, W: float = 20.0, h: float = 5.0, 
                 cond_update_period: float = 0.0):
        """
        Инициализация модели RMa.

        Args:
            bs: Объект базовой станции
            W: Средняя ширина улиц в метрах (по умолчанию 20.0)
            h: Средняя высота зданий в метрах (по умолчанию 5.0)
        """
        super().__init__(bs, cond_update_period)
        
        self.W = W
        self.h = h
        
        self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]
        self.bs.height = 35.0
        
        self.sigma_SF_LOS_1 = 4.0
        self.sigma_SF_LOS_2 = 6.0
        self.sigma_SF_NLOS = 8.0
        
        self.correlation_dist_LOS = 37
        self.correlation_dist_NLOS = 120
        
        self.o2i_model = "low"
        
    def _get_sigma_sf(self, channel_condition: str, d_2D: float = None, 
                      UE_height: float = None) -> float:
        """
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
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            d_BP: Дистанция излома в метрах
        """
        d_BP = (2 * np.pi * self.bs.height * UE_height * self.bs.frequency_Hz) / 3.0e8
        return d_BP
    
    def _calculate_los_probability(self, UE_height: float, d_2D: float) -> float:
        """
        Расчет вероятности прямой видимости (LOS).
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
    
        Returns:
            los_probability: Вероятность LOS в диапазоне [0, 1]
        """
        if d_2D <= 10:
            return 1.0
        
        los_probability = np.exp(-(d_2D - 10) / 1000)
        return los_probability
            
    def _calculate_los_path_loss(self, UE_ID: int, displacement: float, 
                                 channel_condition: str, d_2D: float, 
                                 d_3D: float, UE_height: float) -> float:
        """
        Расчет затухания сигнала для LOS условий.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            PL1: Затухание сигнала в дБ для LOS случая 1
            PL2: Затухание сигнала в дБ для LOS случая 2
        """
        d_BP = self._calculate_breakpoint_distance(UE_height)
        
        if 10 <= d_2D <= d_BP:
            PL1 = (20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0) + 
                   min(0.03 * self.h**1.72, 10) * np.log10(d_3D) - 
                   min(0.044 * self.h**1.72, 14.77) + 0.002 * np.log10(self.h) * d_3D)
            
            return PL1
    
        elif d_BP < d_2D <= 10000:
            PL1_at_dBP = (20 * np.log10(40 * np.pi * d_BP * self.bs.frequency_GHz / 3.0) + 
                   min(0.03 * self.h**1.72, 10) * np.log10(d_BP) - 
                   min(0.044 * self.h**1.72, 14.77) + 0.002 * np.log10(self.h) * d_BP)
            PL2 = PL1_at_dBP + 40 * np.log10(d_3D / d_BP)
            
            return PL2
        
        else:
            return 10000.0
        
    def _calculate_nlos_path_loss(self, UE_ID: int, displacement: float, 
                                 channel_condition: str, d_2D: float, 
                                 d_3D: float, UE_height: float) -> float:
        """
        Расчет затухания сигнала для NLOS условий.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            PL: Затухание сигнала в дБ для NLOS случая
        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss(UE_ID, displacement, channel_condition, 
                                                   d_2D, d_3D, UE_height)
            PL_NLOS = (161.04 - 7.1 * np.log10(self.W) + 
                         7.5 * np.log10(self.h) -
                         (24.37 - 3.7 * (self.h / self.bs.height)**2) * np.log10(self.bs.height) +
                         (43.42 - 3.1 * np.log10(self.bs.height)) * (np.log10(d_3D) - 3) +
                         20 * np.log10(self.bs.frequency_GHz) -
                         (3.2 * (np.log10(11.75 * UE_height))**2 - 4.97))
            
            PL = max(PL_LOS, PL_NLOS)
            
            return PL
            
        else:
            return 10000.0
        
    
class UMaModel(ChannelModel):
    """
    Модель радиоканала для городской макросотовой местности (Urban Macro - UMa).
    Реализует расчеты для сценариев макросотового покрытия в городских условиях.
    """
    def __init__(self, bs: BaseStation, cond_update_period: float = 0.0, 
                 o2i_model: str = "low"):
        """Инициализация модели UMa.
        
        Args:
            bs: Объект базовой станции с параметрами антенны, мощности и частоты.
            o2i_model: Модель проникновения в здание ('low' или 'high'). 
        """
        super().__init__(bs, cond_update_period)
        
        self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]
        self.bs.height = 25.0
        
        self.sigma_SF_LOS = 4.0
        self.sigma_SF_NLOS = 6.0
        
        self.correlation_dist_LOS = 37
        self.correlation_dist_NLOS = 50
        
        self.o2i_model = o2i_model
        
    def _calculate_breakpoint_distance(self, UE_height: float, d_2D: float) -> float:
        """
        Расчет дистанции излома (breakpoint distance).
    
        Args:
            UE_height: Высота пользовательского оборудования в метрах
            d_2D: 2D расстояние между БС и UE в метрах
    
        Returns:
            d_BP: Дистанция излома в метрах
        """
        if UE_height < 13:
            C = 0
            
        elif 13 <= UE_height <= 23:
            if d_2D <= 18:
                g = 0
            else:
                g = (5 / 4) * (d_2D / 100)**3 * np.exp(-d_2D / 150)
                
            C = ((UE_height - 13) / 10)**1.5 * g
            
        h_E_probability = 1 / (1 + C)
        
        if np.random.random() <= h_E_probability:
            h_E = 1
            
        else:
            h_max = int(np.floor(UE_height - 1.5))
            h_Es = list(range(12, h_max + 1, 3))
            
            if not h_Es:
                h_Es = [12]
                
            h_E = np.random.choice(h_Es)
        
        bs_height_prime = self.bs.height - h_E
        UE_height_prime = UE_height - h_E
        
        d_BP = (4 * bs_height_prime * UE_height_prime * self.bs.frequency_Hz) / 3.0e8
        return d_BP    
    
    def _calculate_los_probability(self, UE_height: float, d_2D: float) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между БС и UE.
    
        Args:
            UE_height: Высота пользовательского оборудования в метрах
            d_2D: 2D расстояние между БС и UE в метрах
    
        Returns:
            los_probability: Вероятность LOS в диапазоне [0, 1]
        """
        if d_2D <= 18:
            return 1.0
        
        if UE_height <= 13:
            C_prime = 0
        elif 13 <= UE_height <= 23:
            C_prime = ((UE_height - 13) / 10)**1.5
        
        los_probability = (((18 / d_2D) + np.exp(-d_2D / 63) * (1 - (18 / d_2D))) * 
                         (1 + C_prime * (5 / 4) * (d_2D / 100)**3 * 
                          np.exp(-d_2D / 150)))
        
        return los_probability 
    
    def _calculate_los_path_loss(self, UE_ID: int, displacement: float, 
                                 channel_condition: str, d_2D: float, 
                                 d_3D: float, UE_height: float) -> float:
        """
        Расчет затухания сигнала для LOS условий.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            PL1: Затухание сигнала в дБ для LOS случая 1
            PL2: Затухание сигнала в дБ для LOS случая 2
        """
        d_BP = self._calculate_breakpoint_distance(UE_height, d_2D)
        
        if 10 <= d_2D <= d_BP:
            PL1 = 28 + 22 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)
            
            return PL1
    
        elif d_BP < d_2D <= 5000:
            PL2 = (28 + 40 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz) -
                   9 * np.log10(d_BP**2 + (self.bs.height - UE_height)**2))
            
            return PL2
        
        else:
            return 10000.0
        
    def _calculate_nlos_path_loss(self, UE_ID: int, displacement: float, 
                                 channel_condition: str, d_2D: float, 
                                 d_3D: float, UE_height: float) -> float:
        """
        Расчет затухания сигнала для NLOS условий.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            PL: Затухание сигнала в дБ для NLOS случая
        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss(UE_ID, displacement, channel_condition,
                                                   d_2D, d_3D, UE_height)
            PL_NLOS = (13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz) -
                       0.6 * (UE_height - 1.5))
            
            PL = max(PL_LOS, PL_NLOS)
            
            return PL
            
        else:
            return 10000.0
        
    
class UMiModel(ChannelModel):
    """
    Модель радиоканала для городской микросотовой местности (Urban Micro - UMi).
    Реализует расчеты для сценариев микросотового покрытия в городских условиях.
    """
    def __init__(self, bs: BaseStation, cond_update_period: float = 0.0, 
                 o2i_model: str = "low"):
        """
        Инициализация модели UMi.
        
        Args:
            bs: Объект базовой станции с параметрами антенны, мощности и частоты.
            o2i_model: Модель проникновения в здание ('low' или 'high'). 
        """
        super().__init__(bs, cond_update_period)
        
        self.bs.tx_power = self.bs.MICROCELL_TX_POWER[self.bs.bandwidth]
        self.bs.height = 10.0
        
        self.sigma_SF_LOS = 4.0
        self.sigma_SF_NLOS = 7.82
        
        self.correlation_dist_LOS = 10
        self.correlation_dist_NLOS = 13
        
        self.o2i_model = o2i_model
        
    def _calculate_breakpoint_distance(self, UE_height: float) -> float:
        """
        Расчет дистанции излома (breakpoint distance).
    
        Args:
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            d_BP: Дистанция излома в метрах
        """
        h_E = 1.0
        
        bs_height_prime = self.bs.height - h_E
        UE_height_prime = UE_height - h_E
        
        d_BP = (4 * bs_height_prime * UE_height_prime * self.bs.frequency_Hz) / 3.0e8
        return d_BP
    
    def _calculate_los_probability(self, UE_height: float, d_2D: float) -> float:
        """
        Расчет вероятности прямой видимости (LOS) между БС и UE.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
    
        Returns:
            los_probability: Вероятность LOS в диапазоне [0, 1]
        """
        if d_2D <= 18:
            return 1.0
        
        los_probability = (18 / d_2D) + np.exp(-d_2D / 36) * (1 - (18 / d_2D))
        
        return los_probability
    
    def _calculate_los_path_loss(self, UE_ID: int, displacement: float, 
                                 channel_condition: str, d_2D: float, 
                                 d_3D: float, UE_height: float) -> float:
        """
        Расчет затухания сигнала для LOS условий.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            PL1: Затухание сигнала в дБ для LOS случая 1
            PL2: Затухание сигнала в дБ для LOS случая 2
        """
        d_BP = self._calculate_breakpoint_distance(UE_height)
        
        if 10 <= d_2D <= d_BP:
            PL1 = 32.4 + 21 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz)
            
            return PL1
    
        elif d_BP < d_2D <= 5000:
            PL2 = (32.4 + 40 * np.log10(d_3D) + 20 * np.log10(self.bs.frequency_GHz) -
                   9.5 * np.log10(d_BP**2 + (self.bs.height - UE_height)**2))
            
            return PL2
        
        else:
            return 10000.0
        
    def _calculate_nlos_path_loss(self, UE_ID: int, displacement: float, 
                                 channel_condition: str, d_2D: float, 
                                 d_3D: float, UE_height: float) -> float:
        """
        Расчет затухания сигнала для NLOS условий.
    
        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах
    
        Returns:
            PL: Затухание сигнала в дБ для NLOS случая
        """
        if 10 <= d_2D <= 5000:
            PL_LOS = self._calculate_los_path_loss(UE_ID, displacement, channel_condition,
                                                   d_2D, d_3D, UE_height)
            PL_NLOS = (35.3 * np.log10(d_3D) + 22.4 + 21.3 * np.log10(self.bs.frequency_GHz) -
                       0.3 * (UE_height - 1.5))
            
            PL = max(PL_LOS, PL_NLOS)
            
            return PL
            
        else:
            return 10000.0   