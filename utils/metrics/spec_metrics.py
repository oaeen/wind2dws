import os
import sys

import numpy as np
from loguru import logger
from scipy import integrate
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

sys.path.append(os.getcwd())

from utils.metrics.spectra_info import generate_dirs, generate_freq, get_cdip_buyo_freq


class Spec_Metrics:
    """
    How are significant wave height, dominant period, average period,
    and wave steepness calculated?
    https://www.ndbc.noaa.gov/faq/wavecalc.shtml
    """

    def __init__(self, freq_num=36, direction_num=24, source_type=None):
        self.x_freq = generate_freq(new_n=freq_num, spec_source_type=source_type)
        self.theta = generate_dirs(n=direction_num, spec_source_type=source_type)
        self.direction = direction_num

    def get_significant_wave_height(self, m0):
        """
        :param m0: the variance of the wave displacement time series acquired during the wave acquisition period.
        :return: Hs: significant wave height, 有效波高
        :formula: Hs=4\sqrt{m0}
        """
        # 有效波高
        Hs = 4 * (m0**0.5)
        return Hs

    def get_mean_wave_period_minus1(self, m0, m_minus1):
        """
        :param m0: the variance of the wave displacement time series acquired during the wave acquisition period.
        :param m_minus1: the negative first moment of the wave displacement time series acquired during the wave acquisition period.
        :return: 平均波周期
        """
        T_m_minus1 = m_minus1 / m0
        return T_m_minus1

    def get_mean_wave_period1(self, m0, m1):
        """
        :param m0: the variance of the wave displacement time series acquired during the wave acquisition period.
        :param m1: the first moment of the wave displacement time series acquired during the wave acquisition period.
        :return: 平均波周期
        """
        T_m1 = m0 / m1
        return T_m1

    def get_mean_wave_period2(self, m0, m2):
        """
        :param m0: the variance of the wave displacement time series acquired during the wave acquisition period.
        :param m2: the second moment of the wave displacement time series acquired during the wave acquisition period.
        :return: 平均波周期
        """
        T_m2 = (m0 / m2) ** 0.5
        return T_m2

    def get_mean_wave_direction(self, y_spec):
        """
        :param y_spec: 海浪谱
        :return: 平均波向
        """
        SF = self.get_sin_func_integrate(y_spec)
        CF = self.get_cos_func_integrate(y_spec)
        theta = np.arctan2(SF, CF) * 180 / np.pi
        theta = theta + 360 if theta < 0 else theta
        return theta

    def get_m_minus1(self, y_spec_dir_integrate):
        """
        计算海浪谱负一阶矩
        """
        m_minus1 = integrate.trapz(y_spec_dir_integrate / self.x_freq, x=self.x_freq)
        return m_minus1

    def get_m0(self, y_spec_dir_integrate):
        """
        计算海浪谱零阶矩（积分）
        """
        m0 = integrate.trapz(y_spec_dir_integrate, x=self.x_freq)
        return m0

    def get_m1(self, y_spec_dir_integrate):
        """
        计算海浪谱一阶矩
        """
        m1 = integrate.trapz(y_spec_dir_integrate * self.x_freq, x=self.x_freq)
        return m1

    def get_m2(self, y_spec_dir_integrate):
        """
        计算海浪谱二阶矩
        """
        m2 = integrate.trapz(y_spec_dir_integrate * (self.x_freq**2), x=self.x_freq)
        return m2

    def get_sin_func_integrate(self, y_spec):
        """
        SF is the integral of sin(θ) F (f, θ) over f and θ
        """
        SF_list = []
        for dir_idx in range(self.direction):
            SF_dir = integrate.trapz(
                np.sin(self.theta[dir_idx]) * y_spec[:, dir_idx], self.x_freq
            )
            SF_list.append(SF_dir)
        SF = integrate.trapz(SF_list, self.theta)
        return SF

    def get_cos_func_integrate(self, y_spec):
        """
        CF is the integral of cos(θ) F (f, θ) over f and θ
        """
        CF_list = []
        for dir_idx in range(self.direction):
            CF_dir = integrate.trapz(
                np.cos(self.theta[dir_idx]) * y_spec[:, dir_idx], self.x_freq
            )
            CF_list.append(CF_dir)
        CF = integrate.trapz(CF_list, self.theta)
        return CF

    def get_swh_from_spec1d(self, spec_1d_list):
        """
        从CDIP的1D谱中计算浪高
        """
        cdip_freq = get_cdip_buyo_freq()
        swh_from_spec1d = []
        for i in tqdm(range(len(spec_1d_list)), desc="calculate swh from spec1d"):
            integral_value = integrate.trapz(spec_1d_list[i], cdip_freq)
            Hs = 4 * np.sqrt(integral_value)
            swh_from_spec1d.append(Hs)
        swh_from_spec1d = np.array(swh_from_spec1d)
        return swh_from_spec1d

    def integral_predict_spec_parameters(self, spec_list):
        """
        从海浪谱中计算有效波高、平均波向、平均波周期等积分参数
        """
        logger.info("Start to integral predict spec parameters...")
        swh_list, mwd_list = [], []
        mwp_minus1_list, mwp1_list, mwp2_list = [], [], []
        for idx in tqdm(range(len(spec_list)), desc=f"integral spec parameters"):
            y_spec = spec_list[idx]

            # 在方向上积分
            each_dir_degree = (360 / self.direction) * np.pi / 180
            y_spec_dir_integrate = y_spec.sum(axis=1) * each_dir_degree
            # n阶矩
            m_minus1 = self.get_m_minus1(y_spec_dir_integrate)
            m0 = self.get_m0(y_spec_dir_integrate)
            m1 = self.get_m1(y_spec_dir_integrate)
            m2 = self.get_m2(y_spec_dir_integrate)

            swh = self.get_significant_wave_height(m0)
            mwd = self.get_mean_wave_direction(y_spec)
            mwp_minus1 = self.get_mean_wave_period_minus1(m0, m_minus1)
            mwp1 = self.get_mean_wave_period1(m0, m1)
            mwp2 = self.get_mean_wave_period2(m0, m2)

            swh_list.append(swh)
            mwd_list.append(mwd)
            mwp_minus1_list.append(mwp_minus1)
            mwp1_list.append(mwp1)
            mwp2_list.append(mwp2)

        return (
            np.array(swh_list),
            np.array(mwd_list),
            np.array(mwp_minus1_list),
            np.array(mwp1_list),
            np.array(mwp2_list),
        )

    def evaluate_predict_spec_loss(self, y_true, y_predict, data_type="null"):
        """
        对预测的海浪谱进行评估
        """
        y_true = y_true.flatten()
        y_predict = y_predict.flatten()

        rmse = round(np.sqrt(mean_squared_error(y_true, y_predict)), 5)
        bias = round(np.mean(y_predict - y_true), 5)
        corrcoef = round(np.corrcoef(y_true, y_predict)[1, 0], 5)

        # cited: A Global Ocean Wave (GOW) calibrated reanalysis from 1948 onwards
        # residual_scatter_index = rmse / np.mean(y_true)

        logger.info(f"{data_type}_RMSE: {rmse}")
        logger.info(f"{data_type}_bias: {bias}")
        logger.info(f"{data_type}_corrcoef: {corrcoef}")
        # logger.info(f"{data_type}_residual_scatter_index: {residual_scatter_index}")

        return rmse, bias, corrcoef
