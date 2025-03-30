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
# Дата последнего изменения: 2025-03-30
# Автор: Норицин Иван
# Версия Python Kernel: 3.12.9
#------------------------------------------------------------------------------
"""
import numpy as np
from BS_MODULE import BaseStation

class RMaModel:
    """
    Модель радиоканала для сельской местности (Rural Macro - RMa).
    Реализует расчеты вероятности прямой видимости (LOS), затухания радиосигнала и
    отношения сигнал/шум (SINR) согласно модели радоканала RMa по спецификации 3GPP.
    """
    def __init__(self, bs: BaseStation, W: float = 20.0, h: float = 5.0):
        """
        Инициализация модели RMa.

        Args:
            bs: Объект базовой станции
            W: Средняя ширина улиц в метрах (по умолчанию 20.0)
            h: Средняя высота зданий в метрах (по умолчанию 5.0)
        """
        self.bs = bs
        self.W = W
        self.h = h
        
        self.bs.tx_power = self.bs.MACROCELL_TX_POWER[self.bs.bandwidth]
        self.bs.height = 35.0
        
        self.sigma_SF_LOS_1 = 4.0
        self.sigma_SF_LOS_2 = 6.0
        self.sigma_SF_NLOS = 8.0
        
    def calculate_breakpoint_distance(self, UE_height: float):
        """
        Расчет дистанции излома (breakpoint distance).

        Args:
            UE_height: Высота пользовательского оборудования в метрах

        Returns:
            d_BP: Дистанция излома в метрах
        """
        d_BP = (2 * np.pi * self.bs.height * UE_height * self.bs.frequency_Hz) / 3.0e8
        return d_BP
    
    def calculate_los_probability(self, d_2D: float):
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
         
    def calculate_path_loss(self, d_2D: float, d_3D: float, UE_height: float):
        """
        Расчет затухания сигнала с учетом LOS/NLOS условий.

        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах

        Returns:
            PL: Затухание сигнала в дБ
        """
        los_probability = self.calculate_los_probability(d_2D)
        is_LOS = np.random.random() <= los_probability
        
        if is_LOS:
            PL = self._calculate_los_path_loss(d_2D, d_3D, UE_height)
        else:
            PL = self._calculate_nlos_path_loss(d_2D, d_3D, UE_height)
            
        return PL
            
    def _calculate_los_path_loss(self, d_2D: float, d_3D: float, UE_height: float):
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
        d_BP = self.calculate_breakpoint_distance(UE_height)
        
        if 10 <= d_2D <= d_BP:
            PL1 = (20 * np.log10(40 * np.pi * d_3D * self.bs.frequency_GHz / 3.0) + 
                   min(0.03 * self.h**1.72, 10) * np.log10(d_3D) - 
                   min(0.044 * self.h**1.72, 14.77) + 0.002 * np.log10(self.h) * d_3D)
            
            shadow_fading = np.random.normal(0, self.sigma_SF_LOS_1)
            PL1 = PL1 + shadow_fading
            return PL1

        elif d_BP < d_2D <= 10000:
            PL1_at_dBP = (20 * np.log10(40 * np.pi * d_BP * self.bs.frequency_GHz / 3.0) + 
                   min(0.03 * self.h**1.72, 10) * np.log10(d_BP) - 
                   min(0.044 * self.h**1.72, 14.77) + 0.002 * np.log10(self.h) * d_BP)
            PL2 = PL1_at_dBP + 40 * np.log10(d_3D / d_BP)
            
            shadow_fading = np.random.normal(0, self.sigma_SF_LOS_2)
            PL2 = PL2 + shadow_fading
            return PL2
        else:
            return 10000.0
        
    def _calculate_nlos_path_loss(self, d_2D: float, d_3D: float, UE_height: float):
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
            PL_LOS = self._calculate_los_path_loss(d_2D, d_3D, UE_height)
            PL_NLOS = (161.04 - 7.1 * np.log10(self.W) + 
                         7.5 * np.log10(self.h) -
                         (24.37 - 3.7 * (self.h / self.bs.height)**2) * np.log10(self.bs.height) +
                         (43.42 - 3.1 * np.log10(self.bs.height)) * (np.log10(d_3D) - 3) +
                         20 * np.log10(self.bs.frequency_GHz) -
                         (3.2 * (np.log10(11.75 * UE_height))**2 - 4.97))
            
            PL = max(PL_LOS, PL_NLOS)
            
            shadow_fading = np.random.normal(0, self.sigma_SF_NLOS)
            PL = PL + shadow_fading
            return PL
            
        else:
            return 10000.0
        
    def calculate_SINR(self, d_2D: float, d_3D: float, UE_height: float):
        """
        Расчет отношения сигнал-интерференция-шум (SINR).

        Args:
            d_2D: 2D расстояние между БС и UE в метрах
            d_3D: 3D расстояние между БС и UE в метрах
            UE_height: Высота пользовательского оборудования в метрах

        Returns:
            Значение SINR в дБ
        """
        path_loss = self.calculate_path_loss(d_2D, d_3D, UE_height)
        cable_loss = 2
        interference_margin = 2
        
        P_signal = self.bs.tx_power + self.bs.antenna_gain - cable_loss - path_loss - interference_margin
            
        P_interference = -95
        P_noise = -174 * 10 * np.log10(self.bs.frequency_Hz)
        
        SINR = P_signal - 10 * np.log10(10 ** (P_interference / 10) + 10 ** (P_noise / 10))
        
        return SINR