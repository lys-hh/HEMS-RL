"""
此文件用于模拟电动汽车与环境交互，集成洗衣机和空调设备
"""
from datetime import datetime, timedelta
from scipy.stats import uniform
import matplotlib.dates as mdates
import torch
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MaxNLocator
from interface import DataInterface
import csv  # 添加CSV模块
import os   # 添加OS模块
import pandas as pd

class HomeEnergyManagementEnv:
    def __init__(self, ev_capacity=24, ess_capacity=24, charge_efficiency=0.95, discharge_efficiency=0.95):
        self.ev_capacity = ev_capacity  # 电动汽车电池容量
        self.ess_capacity = ess_capacity  # 储能电池容量
        self.charge_efficiency = charge_efficiency  # 充电效率
        self.discharge_efficiency = discharge_efficiency  # 放电效率
        self.ev_min_charge = 12  # 设置 EV 离家时的最低电量需求

        # 初始化惩罚系数，实现动态调整
        self.energy_weight = 0.1  # 电网成本权重
        self.user_satisfaction_weight0 = 0.5  # 用户不满意度权重
        self.user_satisfaction_weight1 = 0.5  #
        self.user_satisfaction_weight2 = 0.2
        self.violation_weight = 0.05
        self.temp_weight = 0.1
        self.ess_weight = 0.1  # 原1
        self.ev_weight = 0.1  # 原1

        self.total_cost = 0

        self.episode_costs = []  # 存储每个episode的总成本
        self.current_step_cost = 0  # 存储当前时间步的成本

        self.state_space = {
            'home_load': (0, 10),  # 家庭用电量范围
            'pv_generation': (0, 5),  # 光伏发电量范围
            'ess_state': (0, ess_capacity),  # 储能电池容量x
            'ev_battery_state': (0, ev_capacity),  # 电动汽车电池容量
            'time_index': (0, 48),  # 一天中的时间，以0.5为步长
            'electricity_price': (0, 2),  # 电价范围，示例值
            'temperature': (0, 40),
            'wash_machine_state': (0, 1),  # 洗衣机状态，0表示关闭、1表示运行
            # 'start_time': (-1, 48),
            # 'price_trend': (-1, 1),  # 电价趋势（过去3小时均值与当前对比）
            'Air_conditioner_power': (0, 5),  # 空调功率范围值
            'Air_conditioner_power2': (0, 5),
            'ewh_temp': (40, 70),  # 热水器水温
            'ewh_power': (0, 2)   # 热水器功率
        }
        self.action_space = {
            'ev_power': (-6.6, -3.3, 0, 3.3, 6.6),  # 电动汽车充放电功率范围

            'battery_power': (-4.4, -2.2, 0, 2.2, 4.4),  # 储能电池充电功率范围

            'wash_machine_schedule': (0, 1, 2, 3, 4, 5, 6),  # 洗衣机调度动作，0表示不运行，1表示运行
            'Air_conditioner_set_temp': (16, 18, 20, 22, 24, 26, 28, 30),  # 空调设定温度
            'Air_conditioner_set_temp2': (16, 18, 20, 22, 24, 26, 28, 30),

            'ewh_set_temp': (40, 45, 50, 55, 60, 65, 70)  # 离散温度设定动作
        }
        self.current_time = '2011-07-03'
        self.current_time_index = 0
        self.data_interface = DataInterface('data/daily_pivot_cons_2011-2012.csv',
                                            'data/daily_pivot_prod_2011-2012.csv')
        self.current_ev_power = 0
        self.current_battery_power = 0

        self.ev_battery_record = []  # 记录 EV 电量
        self.ess_state_record = []  # 记录 ESS 电量
        self.home_load_record = []  # 记录电网负荷
        self.pv_generation_record = []  # 记录光伏发电量
        self.electricity_price_record = []  # 记录电价
        self.ev_at_home_record = []  # 记录 EV 是否在家
        self.wash_machine_record = []  # 记录洗衣机状态
        self.air_conditioner_power_record = []  # 记录空调功率
        self.ess_actions = []  # 记录储能电池充电功率
        self.wash_machine_actions = []  # 记录洗衣机动作
        self.air_conditioner_actions = []  # 记录空调动作

        # 新增ESS充放电来源记录
        self.ess_charge_pv = []  # 光伏充电量
        self.ess_charge_grid = []  # 电网充电量
        self.ess_discharge_ev = []  # 用于EV的放电量
        self.ess_discharge_house = []  # 用于家庭负荷的放电量

        # 新增记录变量
        self.records = {
            'timestamps': [],
            'ev_soc': [],
            'ess_soc': [],
            'grid_power': [],
            'energy_cost': [],
            'daily_costs': [],
            'indoor_temp': [],
            'indoor_temp2': [],
            'outdoor_temp': [],
            'current_daily_cost': 0,
            'total_load':[],
            'user_dissatisfaction': []  # 用户不满意度记录
        }

        # 新增奖励记录容器
        self.reward_components = {
            'total': [],
            'energy_cost': [],
            'violation_penalty': [],
            'ess_reward': [],
            'ev_reward': [],
            'user_penalty': [],
            'temp_reward': []
            # 'sell': []
        }

        # ================== 新增热水器参数 ==================
        self.ewh_params = {
            # 'ξ_ewh': 0.993,  # 水温衰减系数
            # 'R_prime': 4.18,  # 热传导参数 (kJ/°C)
            'h': 10,  # 对流换热系数
            'temp_cold': 18,  # 冷水温度 (°C)
            'surface_area': 2,  # 水箱表面积(m^2)
            'volume': 100,  # 水箱容量 (L)
            'mass': 100,  # 水的质量（kg）
            'temp_min': 30.0,  # 最低可接受水温 (°C)
            'temp_max': 100.0,  # 最高安全水温 (°C)
            'temp_init': 40,  # 初始水温 (°C)
            'user_flow_mean': 1.5,  # 平均用水流量 (L/h)
            'user_flow_std': 0.3  # 用水流量标准差
        }

        # 新增记录变量
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []

        # 洗衣机相关参数
        self.wash_machine_power = 1.5  # 洗衣机固定功率（kW）
        self.wash_machine_duration = 1  # 洗衣机运行时长（小时）
        self.current_period_start = None  # 当前周期开始时间
        self.has_run_in_current_period = False  # 当前周期是否已运行
        self.time_deviation = 0  # 时间偏移量（分钟）
        self.wash_machine_state = 0  # 初始状态为关闭
        self.last_action_time = None  # 记录启动时间
        self.remaining_runtime = 0  # 剩余运行时间（30分钟步数）
        self.wash_machine_preferred_time = (6, 20)  # 用户偏好的洗衣机运行时间范围

        # 空调相关参数
        self.indoor_temp = 25  # 初始室内温度
        self.indoor_temp2 = 25  # 第二台空调
        self.outdoor_temp = 25  # 初始室外温度
        self.temp_change_rate = 0.5  # 室内温度变化率
        self.user_temp_preference = 22  # 用户偏好温度
        self.user_temp_preference2 = 18  # 第二台空调

    def get_state_vector(self, state_dict):
        """将状态字典转换为有序列表"""
        ordered_keys = sorted(state_dict.keys())  # 按字母顺序排序
        return [state_dict[k] for k in ordered_keys]

    def get_action_mask(self, state):
        """返回基于当前电量的动态动作掩码"""
        masks = {
            'battery_power': [True] * len(self.action_space['battery_power']),
            'ev_power': [True] * len(self.action_space['ev_power'])
        }

        delta_t = 0.5

        # 1. ESS电池动作掩码 - 动态计算
        # 考虑放电效率（假设95%）
        max_discharge = (state['ess_state'] / delta_t) * 0.95
        max_charge = ((self.ess_capacity - state['ess_state']) / delta_t) / 0.95
        
        for idx, action_value in enumerate(self.action_space['battery_power']):
            # 放电动作：只能选择小于等于当前可放电量的动作
            if action_value < 0 and abs(action_value) > max_discharge:
                masks['battery_power'][idx] = False

            # 充电动作：只能选择小于等于当前可充电空间的充电功率
            if action_value > 0 and action_value > max_charge:
                masks['battery_power'][idx] = False

        # 2. EV电池动作掩码 - 同样动态计算
        max_ev_discharge = (state['ev_battery_state'] / delta_t) * 0.95
        max_ev_charge = ((self.ev_capacity - state['ev_battery_state']) / delta_t) / 0.95
        
        for idx, action_value in enumerate(self.action_space['ev_power']):
            # 放电限制
            if action_value < 0 and abs(action_value) > max_ev_discharge:
                masks['ev_power'][idx] = False

            # 充电限制
            if action_value > 0 and action_value > max_ev_charge:
                masks['ev_power'][idx] = False

        # 3. EV不在家时掩码
        if not self.is_ev_at_home():
            for idx, action_value in enumerate(self.action_space['ev_power']):
                if action_value != 0:  # 只能选择0功率动作
                    masks['ev_power'][idx] = False

        return masks

    def reset(self):
        self.total_cost = 0
        # 使用实际输入数据初始化状态
        self.current_time = '2011-07-03'
        self.current_time_index = 0
        self.current_ev_power = 0
        self.current_battery_power = 0
        # wash_feature = self._get_wash_machine_features(self.current_time_index)
        self.state = {
            'home_load': self.data_interface.get_home_load(self.current_time, self.current_time_index),
            'pv_generation': self.data_interface.get_pv_generation(self.current_time, self.current_time_index),
            'ess_state': 2.4,
            'ev_battery_state': 12,
            'time_index': self.current_time_index,
            'electricity_price': self.data_interface.get_electricity_price(self.current_time, self.current_time_index),
            'temperature': 20,
            'wash_machine_state': 0,
            # 'start_time': wash_feature['start_time'],
            # 'price_trend': wash_feature['price_trend'],
            'Air_conditioner_power': 0,
            'Air_conditioner_power2': 0,
            'ewh_temp': self.ewh_params['temp_init'],
            'ewh_power': 0
        }
        self.ev_battery_record = []   # 重置记录
        self.ess_state_record = []
        self.home_load_record = []
        self.pv_generation_record = []
        self.electricity_price_record = []
        self.ev_at_home_record = []
        self.wash_machine_record = []
        self.air_conditioner_power_record = []
        self.air_conditioner_power_record2 = []
        self.ess_actions = []  # 重置 ESS 动作记录

        # 重置记录
        self.records = {
            'timestamps': [],
            'ev_soc': [],
            'ess_soc': [],
            'grid_power': [],
            'energy_cost': [],
            'daily_costs': [],
            'indoor_temp': [],
            'indoor_temp2': [],
            'outdoor_temp': [],
            'current_daily_cost': 0,
            'total_load': [],
            'user_dissatisfaction': []
        }

        # 重置奖励记录容器
        self.reward_components = {
            'total': [],
            'energy_cost': [],
            'violation_penalty': [],
            'ess_reward': [],
            'ev_reward': [],
            'user_penalty': [],
            'temp_reward': []
            # 'sell': []
        }

        # 重置洗衣机相关变量
        self.current_period_start = None  # 当前周期开始时间
        self.has_run_in_current_period = False  # 当前周期是否已运行
        self.time_deviation = 0  # 时间偏移量（分钟）
        self.wash_machine_state = 0  # 初始状态为关闭
        self.last_action_time = None  # 记录启动时间
        self.remaining_runtime = 0  # 剩余运行时间（30分钟步数）

        # 重置空调相关变量
        self.indoor_temp = 25
        self.indoor_temp2 = 20
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)

        # 重置记录
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []

        return self.state

    def step(self, state, action):

        # === 动作物理裁剪，保证环境物理合理性 ===
        # --- 储能电池功率裁剪 ---
        ess_soc = state['ess_state']
        if action['battery_power'] < 0:  # 放电
            max_discharge = min(abs(action['battery_power']), ess_soc / 0.5 * self.discharge_efficiency)
            action['battery_power'] = -max_discharge
        elif action['battery_power'] > 0:  # 充电
            max_charge = min(action['battery_power'], (self.ess_capacity - ess_soc) / 0.5 / self.charge_efficiency)
            action['battery_power'] = max_charge

        # --- 电动汽车功率裁剪 ---
        ev_soc = state['ev_battery_state']
        if action['ev_power'] < 0:  # 放电
            max_discharge = min(abs(action['ev_power']), ev_soc / 0.5 * self.discharge_efficiency)
            action['ev_power'] = -max_discharge
        elif action['ev_power'] > 0:  # 充电
            max_charge = min(action['ev_power'], (self.ev_capacity - ev_soc) / 0.5 / self.charge_efficiency)
            action['ev_power'] = max_charge


        self.current_ev_power=action['ev_power']  # 存储当前动作
        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + \
                     timedelta(minutes=30 * self.current_time_index)

        # 更新室外温度
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)

        # 更新空调状态
        new_air_conditioner_power, self.indoor_temp = self.update_air_conditioner(action['Air_conditioner_set_temp'], self.indoor_temp)
        self.state['Air_conditioner_power'] = new_air_conditioner_power

        new_air_conditioner_power2, self.indoor_temp2 = self.update_air_conditioner(action['Air_conditioner_set_temp2'], self.indoor_temp2)
        self.state['Air_conditioner_power2'] = new_air_conditioner_power2

        # 更新洗衣机状态
        new_wash_machine_state = self.update_wash_machine2(action['wash_machine_schedule'])
        self.state['wash_machine_state'] = new_wash_machine_state

        # wash_feature = self._get_wash_machine_features(self.current_time_index)

        # 更新热水器功率
        power = self._fuzzy_heating_control(
            action['ewh_set_temp'],
            state['ewh_temp']
        )

        #  更新热水器状态
        new_ewh_temp, user_flow = self.update_water_heater(
            power,
            state['ewh_temp']
        )

        # 更新电动汽车和储能电池状态
        self.state = {
            'home_load': self.data_interface.get_home_load(self.current_time, self.current_time_index),
            'pv_generation': self.data_interface.get_pv_generation(self.current_time, self.current_time_index),
            'ess_state': self.update_ess(action['battery_power'], state['pv_generation']),
            'ev_battery_state': self.update_ev_battery(action['ev_power']),
            'time_index': self.current_time_index,
            'electricity_price': self.data_interface.get_electricity_price(self.current_time, self.current_time_index),
            'temperature': self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index),
            'wash_machine_state': new_wash_machine_state,
            # 'start_time': wash_feature['start_time'],
            # 'price_trend': wash_feature['price_trend'],
            'Air_conditioner_power': new_air_conditioner_power,
            'Air_conditioner_power2': new_air_conditioner_power2,
            'ewh_temp': new_ewh_temp,
            'ewh_power': power
        }

        reward = self.calculate_reward(state, action)
        done = self.is_terminal_state()

        self.ev_battery_record.append(self.state['ev_battery_state'])
        self.ess_state_record.append(self.state['ess_state'])
        self.home_load_record.append(self.state['home_load'])
        self.pv_generation_record.append(self.state['pv_generation'])
        self.electricity_price_record.append(self.state['electricity_price'])
        self.ev_at_home_record.append(self.is_ev_at_home())
        self.wash_machine_record.append(self.state['wash_machine_state'])
        self.air_conditioner_power_record.append(self.state['Air_conditioner_power'])
        self.air_conditioner_power_record2.append(self.state['Air_conditioner_power2'])

        # 记录 ESS 动作
        self.ess_actions.append(action['battery_power'])

        # 记录温度数据
        self.records['indoor_temp'].append(self.indoor_temp)
        self.records['indoor_temp2'].append(self.indoor_temp2)
        self.records['outdoor_temp'].append(self.outdoor_temp)

        # 更新记录
        self.records['timestamps'].append(current_dt)
        self.records['ev_soc'].append(self.state['ev_battery_state'])
        self.records['ess_soc'].append(self.state['ess_state'])

        self.records['total_load'].append(self.total_load_compute())

        # 记录热水器数据
        self.ewh_temp_record.append(new_ewh_temp)
        self.ewh_power_record.append(state['ewh_power'])
        self.user_flow_record.append(user_flow)

        self.current_time_index += 1  # 增加半小时

        if self.current_time_index >= 48:  # 如果当前时间索引达到48，则增加一天
            self.current_time_index = 0
            # 将当前日期字符串转换为datetime对象
            current_date = datetime.strptime(self.current_time, '%Y-%m-%d')
            # 增加一天
            current_date += timedelta(days=1)
            # 将datetime对象转换回字符串格式
            self.current_time = current_date.strftime('%Y-%m-%d')

            # 强制重置洗衣机状态（避免跨天残留）
            self.wash_machine_used_today = False

        return self.state, reward, done

    def update_ev_battery(self, ev_charge_discharge):
        # 如果 EV 刚刚到家，更新电量状态
        if (not self.data_interface.is_ev_at_home(self.current_time, self.current_time_index - 1)
                and self.is_ev_at_home()):
            # 使用上一次出行前的电量状态来模拟到家后的电量
            ev_state_init = self.update_ev_state_after_trip(self.state['ev_battery_state'])
            if ev_charge_discharge > 0:
                new_soc = ev_state_init + (ev_charge_discharge * self.charge_efficiency) * 0.5
            else:
                new_soc = ev_state_init + (ev_charge_discharge / self.charge_efficiency) * 0.5
        else:
            if ev_charge_discharge > 0:
                new_soc = self.state['ev_battery_state'] + (ev_charge_discharge * self.charge_efficiency) * 0.5
            else:
                new_soc = self.state['ev_battery_state'] + (ev_charge_discharge / self.charge_efficiency) * 0.5

        # 强制SOC边界
        # min_soc = self.ev_min_charge * 0.8  # 保持安全裕量
        return np.clip(new_soc, 0, self.ev_capacity)

    def total_load_compute(self):
        ev_power = self.current_ev_power
        ess_power = self.current_battery_power
        pv_load = self.state['pv_generation']
        home_load = self.state['home_load']
        air_cond_power = self.state['Air_conditioner_power']
        air_cond_power2 = self.state['Air_conditioner_power2']
        wash_machine_power = self.state['wash_machine_state'] * self.wash_machine_power
        ewh_power = self.state['ewh_power']
        total_load = (home_load + air_cond_power + air_cond_power2 + wash_machine_power + ewh_power + ev_power
                      + ess_power - pv_load)
        return total_load

    def update_ess(self, ess_charge_discharge, pv_generation):
        # assert charge_power * discharge_power == 0, "ESS cannot charge and discharge simultaneously"
        # diff = pv_generation - self.total_load_compute()

        if ess_charge_discharge > 0:  # 充电动作   =  pv + grid
            new_ess = self.state['ess_state'] + ess_charge_discharge * self.charge_efficiency * 0.5
        else:   # 放电动作 = home + grid
            new_ess = self.state['ess_state'] + ess_charge_discharge / self.charge_efficiency * 0.5

        return np.clip(new_ess, 0, self.ess_capacity)

    def _get_period_start(self, dt):
        if dt.hour >= 0:
            return datetime(dt.year, dt.month, dt.day, 0, 0)
        else:
            prev_day = dt - timedelta(days=1)
            return datetime(prev_day.year, prev_day.month, prev_day.day, 0, 0)

    def _get_period_end(self, dt):
        return datetime(dt.year, dt.month, dt.day, 23, 0)

    def update_time_deviation(self, scheduled_time):
        pref_start = self.current_period_start.replace(hour=self.wash_machine_preferred_time[0], minute=0)
        pref_end = self.current_period_start.replace(hour=self.wash_machine_preferred_time[1],
                                                     minute=0) - timedelta(hours=self.wash_machine_duration)
        # 计算时间偏移量（小时）
        if scheduled_time < pref_start:
            # 在偏好时间之前
            self.time_deviation = (pref_start - scheduled_time).total_seconds() / 3600
        elif scheduled_time > pref_end:
            # 在偏好时间之后
            self.time_deviation = (scheduled_time - pref_end).total_seconds() / 3600
        else:
            # 在偏好时间段内
            self.time_deviation = 0

    def update_wash_machine2(self, schedule_action):
        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + \
                     timedelta(minutes=30 * self.current_time_index)
        # 转换为Python内置整数
        schedule_action = int(schedule_action)  # 添加这一行

        # 处理运行中状态
        if self.remaining_runtime > 0:
            self.wash_machine_state = 1
            self.remaining_runtime -= 1
            return self.wash_machine_state

        # 周期检测（每天6点重置）
        current_period_start = self._get_period_start(current_dt)
        current_period_end = self._get_period_end(current_dt)  # 获取当前周期结束时间
        if current_period_start != self.current_period_start:
            self.current_period_start = current_period_start
            self.has_run_in_current_period = False
            self.wash_machine_state = 0

        # 强制启动逻辑：如果距离周期结束不足1小时且尚未运行
        time_to_end = (current_period_end - current_dt).total_seconds() / 3600
        if time_to_end <= 1 and not self.has_run_in_current_period:
            schedule_action = 1  # 强制立即启动

        if schedule_action > 1 and not self.has_run_in_current_period:
            scheduled_time = current_dt + timedelta(hours=schedule_action - 1)
            self.wash_machine_state = 0
            self.update_time_deviation(scheduled_time)
        elif schedule_action == 1 and not self.has_run_in_current_period:
            scheduled_time = current_dt  # 当前时间启动
            self.wash_machine_state = 1
            self.has_run_in_current_period = True
            self.remaining_runtime = 1  # 1小时=2个时间步
            self.update_time_deviation(scheduled_time)
        else:
            self.wash_machine_state = 0
            self.time_deviation = 0

        return self.wash_machine_state

    def update_air_conditioner(self, set_temp, indoor_temp):
        """更新空调功率"""
        # 计算设定温度与当前室内温度的差异
        temp_diff = set_temp - indoor_temp
        # 模糊控制规则：根据温差决定功率
        if temp_diff > 0:  # 制热模式
            # 定义制热模式下的模糊控制规则
            rules = [
                {'range': (0, 0.5), 'power': 0},
                {'range': (0.5, 1), 'power': 0.5},  # 0.5可实现精准控制
                {'range': (1, 2), 'power': 1.0},
                {'range': (2, 3), 'power': 1.5},
                {'range': (3, 4), 'power': 2.0},
                {'range': (4, np.inf), 'power': 3.0}
            ]
        else:
            # 定义制冷模式下的模糊控制规则
            rules = [
                {'range': (-0.5, 0), 'power': 0},
                {'range': (-1, -0.5), 'power': 0.5},  # 0.5可实现精准控制
                {'range': (-2, -1), 'power': 1.0},
                {'range': (-3, -2), 'power': 1.5},
                {'range': (-4, -3), 'power': 2.0},
                {'range': (-np.inf, -4), 'power': 3.0}
            ]

        # 根据温差找到对应的功率
        power = 0  # 默认功率
        for rule in rules:
            if rule['range'][0] <= temp_diff < rule['range'][1]:
                power = rule['power']
                break

        # 动态计算温度变化率
        max_power = 3.0  # 空调最大功率
        efficiency = power / max_power if max_power > 0 else 0  # 计算当前功率下的效率
        temp_change = self.temp_change_rate * efficiency * temp_diff  # 根据功率和温差计算温度变化

        # 模拟用户行为的随机扰动
        user_behavior = np.random.normal(0, 0.2)
        temp_change += user_behavior  # 将随机扰动加入温度变化中

        # 当空调关闭时（power == 0），室内温度逐渐向室外温度靠拢
        if power == 0:
            # 室内温度向室外温度靠拢的速率可以调整
            temp_change += (self.outdoor_temp - indoor_temp) * 0.4 * self.temp_change_rate

        # 更新室内温度
        indoor_temp += temp_change

        # 确保室内温度在合理范围内
        indoor_temp = np.clip(indoor_temp, 10, 40)

        # 确保空调功率在合理范围内
        power = np.clip(power, 0, 3.0)

        return power, indoor_temp

    def update_water_heater(self, power, current_temp):
        """更新热水器状态（物理模型）"""
        params = self.ewh_params
        delta_t = 0.5  # 半小时时间间隔

        # # 生成随机用水量 (L/h)
        # flow_rate = np.random.uniform(1, 2) if np.random.rand() < 0.3 else 0  # 假设30%的概率有用水行为

        # 定义用水高峰时间段
        peak_morning = 6 <= self.current_time_index / 2 <= 9  # 早晨高峰 6-9点
        peak_evening = 18 <= self.current_time_index / 2 <= 22  # 傍晚高峰 18-22点

        # 根据时间段设置不同的用水概率和用水量范围
        if peak_morning or peak_evening:
            # 高峰时段用水概率和用水量
            if peak_morning:
                # 早晨高峰用水量更大，概率更高
                flow_rate_prob = 0.6  # 60% 概率有用水行为
                flow_rate_min, flow_rate_max = 3, 5  # 3-5 L/h
            else:
                # 傍晚高峰用水量稍小，概率稍低
                flow_rate_prob = 0.5  # 50% 概率有用水行为
                flow_rate_min, flow_rate_max = 2, 4  # 2-4 L/h

            # 高峰时段有更高的用水概率
            if np.random.rand() < flow_rate_prob:
                flow_rate = np.random.uniform(flow_rate_min, flow_rate_max)
            else:
                flow_rate = 0
        else:
            # 平峰时段用水概率和用水量
            flow_rate_prob = 0.2  # 20% 概率有用水行为
            if np.random.rand() < flow_rate_prob:
                # 平峰时段用水量较小
                flow_rate = np.random.uniform(0.5, 1.5)  # 0.5-1.5 L/h
            else:
                flow_rate = 0

        # 特殊处理：在平峰时段偶尔会有稍大的用水量
        if not (peak_morning or peak_evening) and np.random.rand() < 0.1:
            flow_rate = np.random.uniform(1.5, 2)  # 1.5-2 L/h
        # === 用水量逻辑结束 ===

        # 环境参数
        env_temp = self.state['temperature']
        temp_cold = max(18, env_temp - 2)  # 最低18°C冷水
        # temp_cold = 18 # 最低18°C冷水

        new_temp = current_temp  # 初始化新温度

        # 情况1: 注入冷水时的温度变化
        if flow_rate > 0:
            # 计算用水量对应的体积变化（假设水箱总体积不变）
            used_water_volume = flow_rate * delta_t   # 假设流量单位为 L/h，转换为 L
            # 注入相同体积的冷水
            injected_cold_volume = used_water_volume

            # 混合效应计算新温度
            remaining_hot_volume = params['volume'] - used_water_volume
            new_temp = (current_temp * remaining_hot_volume + temp_cold * injected_cold_volume) / params['volume']

        # 情况2: 加热时的升温
        if power > 0:
            hour = self.state['time_index'] // 2
            efficiency = 0.9 if (6 <= hour <= 9 or 18 <= hour <= 22) else 0.8
            # 将功率从 kW 转换为 W
            power_in_watts = power * 1000 * efficiency

            # 计算加热带来的温度上升
            heat_input = power_in_watts * 3600 * delta_t  # 能量输入（J）
            temp_gain = heat_input / (params['mass'] * 4180)  # 温度上升（°C）
            new_temp += temp_gain

        # 情况3: 无加热时的自然冷却
        else:
            # 计算自然冷却带来的温度下降   冷却系数 k
            cooling_coefficient = params['h'] * params['surface_area'] / (params['mass'] * 4180)
            new_temp = env_temp + (new_temp - env_temp) * np.exp(-cooling_coefficient * delta_t * 3600)

        # 更新参数
        params['temp_min'] = env_temp
        new_temp = np.clip(new_temp, params['temp_min'], params['temp_max'])

        return new_temp, flow_rate

    def _fuzzy_heating_control(self, set_tem, current_tem):
        # 模糊控制逻辑：根据目标温度与当前温度的差值决定加热功率
        # temp_diff: 目标温度 - 当前温度
        temp_diff = set_tem - current_tem
        hour = self.state['time_index'] // 2  # 获取当前小时

        # 动态调整控制规则（高峰期更激进）
        if 6 <= hour <= 9 or 18 <= hour <= 22:  # 高峰期
            rules = [
                {'range': (-np.inf, -3), 'power': 0.0},
                {'range': (-3, 1), 'power': 0.0},
                {'range': (1, 4), 'power': 0.4 + 0.1 * max(0, temp_diff-1)},  # 动态比例
                {'range': (4, 6), 'power': 1.0},
                {'range': (6, np.inf), 'power': 1.2}  # # 允许短时超功率
            ]
        else:  # 非高峰期
            rules = [
                {'range': (-np.inf, -5), 'power': 0.0},
                {'range': (-5, 2), 'power': 0.0},
                {'range': (2, 5), 'power': 0},  # 温和加热
                {'range': (5, 8), 'power': 0.2},  # 原0
                {'range': (8, np.inf), 'power': 0.5}  # 原0.5
            ]

        # rules = [
        #     {'range': (-np.inf, -3), 'power': 0.0},
        #     {'range': (-3, 1), 'power': 0.0},
        #     {'range': (1, 4), 'power': 0.4 + 0.1 * max(0, temp_diff - 1)},  # 动态比例
        #     {'range': (4, 6), 'power': 1.0},
        #     {'range': (6, np.inf), 'power': 1.2}  # # 允许短时超功率
        # ]

        for rule in rules:
            if rule['range'][0] <= temp_diff < rule['range'][1]:
                return min(max(rule['power'], 0.0), 1.2)  # 功率钳位
        return 0.0  # 默认不加热

    def calculate_reward(self, state, action):
        # 定义奖励

        # 1. 约束惩罚
        violation = 0
        # violation += max(0.1*self.ess_capacity-self.state['ess_state'],
        #                 self.state['ess_state']-0.9*self.ess_capacity, 0) ** 2 * 20
        #
        # violation += max(0.1 * self.ev_capacity - self.state['ev_battery_state'],
        #                 self.state['ev_battery_state'] - 0.9 * self.ev_capacity, 0) ** 2 * 20

        # # EV约束（最低电量）
        if self.data_interface.is_ev_departing_soon(self.current_time, self.current_time_index):
            ev_violation = max(0, self.ev_min_charge - self.state['ev_battery_state'])
            # violation += np.clip(ev_violation ** 2, 0, 500)  # 添加数值裁剪
            violation += ev_violation ** 2

        # 2. 电网成本计算（半小时）
        ev_charge = max(action['ev_power'], 0)
        ev_discharge = max(-action['ev_power'], 0)
        battery_charge = max(action['battery_power'], 0)
        battery_discharge = max(-action['battery_power'], 0)

        # 计算总消耗和总发电（kW）
        total_consumption = (
                state['home_load']
                + ev_charge
                + battery_charge
                + state['Air_conditioner_power']
                + state['Air_conditioner_power2']
                + state['wash_machine_state'] * self.wash_machine_power
                + state['ewh_power']
        )

        total_generation = (
                state['pv_generation']
                + ev_discharge
                + battery_discharge
        )
        # 计算净需求（kW）
        net_demand = total_consumption - total_generation  # 这个需求就是与电网的交互

        # 转换为能量（kWh）并拆分购售电
        purchase_kwh = max(net_demand, 0) * 0.5  # 这里计算的是理想成本而非实际成本
        sell_kwh = max(-net_demand, 0) * 0.5

        # 计算能源成本（考虑售电价格折扣）
        energy_cost = (
                purchase_kwh * state['electricity_price']
                - sell_kwh * state['electricity_price'] * 0.9  # 假设售电价格是购电的0.9
        )

        self.total_cost += energy_cost
        self.records['energy_cost'].append(self.total_cost)

        self.current_step_cost = energy_cost

        # 3. ESS与ev充放电引导奖励

        # ESS奖励：鼓励低充高放
        ess_reward = 0
        price = self.state['electricity_price']
        soc = self.state['ess_state'] / self.ess_capacity
        if price < 0.5:
            # ess_reward += -action['battery_power'] * (price - 0.5) * (1-soc) * 10
            ess_reward += -action['battery_power'] * (price - 0.5) * 20
        elif price == 0.5:
            ess_reward += -action['battery_power'] * (soc-0.8) * 20
        else:
            # ess_reward += -action['battery_power'] * (price - 0.5) * soc * 10
            ess_reward += -action['battery_power'] * (price - 0.5) * 20
        # ess_reward = np.tanh(ess_reward / 10) * 3  # 使用tanh压缩幅度

        # Ev奖励：鼓励低充高放
        ev_reward = 0
        # 获取当前SOC比例
        soc_ev = state['ev_battery_state'] / self.ev_capacity
        if price < 0.5:
            # ev_reward += -action['ev_power'] * (price - 0.5) * (1-soc_ev) * 10
            ev_reward += -action['ev_power'] * (price - 0.5) * 20
        elif price == 0.5:
            ev_reward += -action['ev_power'] * (soc_ev-0.8) * 20
        else:
            # ev_reward += -action['ev_power'] * (price - 0.5) * soc_ev * 10
            ev_reward += -action['ev_power'] * (price - 0.5) * 20
        # ev_reward = np.tanh(ev_reward / 10) * 3  # 使用tanh压缩幅度

        # 4. 用户不满意度惩罚
        user_dissatisfaction_penalty = 0
        user_dissatisfaction_penalty = (self.user_satisfaction_weight0 * self.calculate_user_dissatisfaction0() +
                                        self.user_satisfaction_weight1 * self.calculate_user_dissatisfaction1() +
                                    self.user_satisfaction_weight2 * self.calculate_user_dissatisfaction2(state, action))
        # user_dissatisfaction_penalty = np.clip(user_dissatisfaction_penalty, -20, 20)

        temp_reward = self.calculate_temp_reward(state['ewh_temp'])
        # temp_reward = np.clip(temp_reward, -5, 5)  # 限制温度奖励范围

        # 5. 组合奖励
        reward = (
                - self.energy_weight * energy_cost
                - self.violation_weight * violation    # EV及ESS的上下界约束
                + self.ess_weight * ess_reward
                + self.ev_weight * ev_reward
                - user_dissatisfaction_penalty   # 用户不满意度惩罚
                + self.temp_weight * temp_reward
        )



        # # 新增探索奖励（防止早熟）
        # if np.random.rand() < 0.1:  # 10%概率添加噪声
        #     reward += np.random.normal(0, 5)

        # 在计算奖励后添加记录
        reward_breakdown = {
            'total': reward,
            'energy_cost': - self.energy_weight * energy_cost,
            'violation_penalty': - self.violation_weight * violation,
            'ess_reward': self.ess_weight * ess_reward,
            'ev_reward': self.ev_weight * ev_reward,
            'user_penalty': - user_dissatisfaction_penalty,
            'temp_reward': self.temp_weight * temp_reward,
        }

        for key in self.reward_components:
            self.reward_components[key].append(reward_breakdown[key])

        #     # 添加数值稳定性检查
        # if not np.isfinite(reward):
        #     reward = -10  # 对异常奖励值进行兜底处理

        return reward

    def calculate_user_dissatisfaction0(self):
        """用户不满意度计算（空调）"""
        dissatisfaction = 0

        # 检查是否在舒适区间内
        comfort_range = 2  # 舒适区间范围（例如±2℃）
        temp_diff = self.indoor_temp2 - self.user_temp_preference2

        if abs(temp_diff) > comfort_range:
            # 温度不满意度（非线性计算）
            dissatisfaction += min((abs(temp_diff) - comfort_range) ** 1.5 * 0.5, 500)  # 指数增长但设置上限

        return dissatisfaction

    def calculate_user_dissatisfaction1(self):
        """用户不满意度计算（空调）"""
        dissatisfaction = 0

        # 检查是否在舒适区间内
        comfort_range = 2  # 舒适区间范围（例如±2℃）
        temp_diff = self.indoor_temp - self.user_temp_preference

        if abs(temp_diff) > comfort_range:
            # 温度不满意度（非线性计算）
            dissatisfaction += min((abs(temp_diff) - comfort_range) ** 1.5 * 0.5, 500)  # 指数增长但设置上限

        # # 空调频繁启停惩罚
        # if len(self.air_conditioner_power_record) > 2:
        #     recent_changes = sum(
        #         abs(a - b) for a, b in zip(
        #             self.air_conditioner_power_record[-3:],
        #             self.air_conditioner_power_record[-4:-1]
        #         )
        #     )
        #     dissatisfaction += recent_changes * 0.2

        return dissatisfaction

    def calculate_user_dissatisfaction2(self, state, action):
        """改进的洗衣机相关惩罚计算"""
        if action['wash_machine_schedule'] >= 1:
            price = self.data_interface.get_electricity_price(self.current_time,
                                                          self.current_time_index+2*(action['wash_machine_schedule']-1))
            price_penalty = (price-0.5) * 10  #
        else:
            price = state['electricity_price']
            price_penalty = 0

        # 时间偏移惩罚（非线性）
        time_penalty = 5 * self.time_deviation   #

        return time_penalty + price_penalty

    def calculate_temp_reward(self, current_temp):
        # 强化分时控制参数
        hour = self.state['time_index'] // 2
        if 6 <= hour <= 9 or 18 <= hour <= 22:  # 用水高峰时段
            target, low, high = 55, 53, 57
            strict_factor = 5.0  # 强化高峰时段惩罚
        else:
            target, low, high = 45, 43, 47
            strict_factor = 1.0

        # 动态偏差计算
        temp_diff = current_temp - target
        abs_diff = abs(temp_diff)

        # # 阶梯式奖励机制
        if low <= current_temp <= high:
            # 核心区奖励 (S型曲线)
            reward = 5.0 / (1 + np.exp(abs_diff * 0.8))  # 中心最高，边缘平滑下降
        else:
            # 危险区惩罚（指数增长）
            deviation = max(low - current_temp, current_temp - high)
            reward = -strict_factor * deviation   # 严惩越界
        return reward

    def is_terminal_state(self):
        done = False
        if self.current_time > '2011-07-09':
            done = True
        return done

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        torch.manual_seed(seed)

    def is_ev_at_home(self):
        return self.data_interface.is_ev_at_home(self.current_time, self.current_time_index)

    def update_ev_state_after_trip(self, current_ev_soc):
        #  生成日行驶里程（对数正态分布）
        avg_daily_km = 50  # 平均日行驶里程（单位：公里）
        km_std = 0.2  # 日行驶里程的标准差
        daily_km = np.random.lognormal(mean=np.log(avg_daily_km), sigma=km_std)

        # 假设每公里耗电量为0.2 kWh
        energy_consumption_per_km = 0.2  # kWh/km

        # 计算到家时的电量状态
        soc_home = current_ev_soc - daily_km * energy_consumption_per_km

        # 确保电量状态在合理范围内（0-100%）
        soc_home = np.clip(soc_home, 0, 100)

        return soc_home

    def _find_contiguous_segments(self, bool_list):
        """检测连续为True的时间段"""
        segments = []
        start_idx = None
        for i, value in enumerate(bool_list):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:  # 处理最后一段
            segments.append((start_idx, len(bool_list) - 1))
        return segments

    def reward_shape(self, progress):
        """
        根据训练进度动态调整奖励函数的系数
        :param progress: 训练进度，范围从0到1
        :return: 各个奖励项的权重
        """

        # self.energy_weight = 5.0 * (1 - 0.8*progress)  # 线性衰减
        # self.temp_weight = 1 / (1 + np.exp(-10*(progress-0.3)))  # S型增长
        self.violation_weight = 3-2*progress  # 4月29号尝试下这个方法，无效后尝试保存模型快照
        # self.user_satisfaction_weight1 = 0.3 + 0.7*progress
        # self.user_satisfaction_weight2 = 0.1 + 0.4*progress
        # self.ess_weight = 3.0 + 2*progress
        # self.ev_weight = 1

    def save_cost_data(self):
        """保存成本数据到CSV文件"""
        # 创建结果目录
        results_dir = "model/cost_results"
        os.makedirs(results_dir, exist_ok=True)

        # 创建唯一的文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(results_dir, f"cost_data_{timestamp}.csv")

        # 写入数据到CSV文件
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入标题行
            writer.writerow(["Timestamp", "Energy Cost"])

            # 写入每个时间步的成本数据
            for i, (ts, cost) in enumerate(zip(self.records['timestamps'], self.records['energy_cost'])):
                writer.writerow([ts.strftime('%Y-%m-%d %H:%M:%S'), cost])

        print(f"成本数据已保存到: {csv_filename}")

    def save_episode_costs(self):
        """保存每个episode的总成本到CSV文件"""
        # 创建结果目录
        results_dir = "model/episode_cost_results"
        os.makedirs(results_dir, exist_ok=True)

        # 创建唯一的文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"episode_costs_{timestamp}.csv")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入标题行
            writer.writerow(["Episode", "Total Cost"])

            # 写入每个episode的成本数据
            for episode, cost in enumerate(self.episode_costs):
                writer.writerow([episode + 1, cost])

        print(f"Episode成本数据已保存到: {filename}")

    def visualize(self):
        # ===== 第一个画布：EV的SOC变化图与电价图 =====
        plt.figure(figsize=(20, 5))

        ax1 = plt.subplot(1, 1, 1)
        ev_soc = np.array(self.ev_battery_record, dtype=np.float32)
        ev_soc[~np.array(self.ev_at_home_record)] = np.nan  # 离家时段设为NaN

        mpl_dates = mdates.date2num(self.records['timestamps'])

        # 绘制EV SOC曲线
        ax1.plot(mpl_dates, ev_soc, color='blue', label='EV SOC')
        ax1.set_ylabel('EV SOC (kWh)')
        ax1.set_title('EV State of Charge and Electricity Price')
        ax1.legend(loc='upper left')

        # 绘制电价曲线（右轴）
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax1_price.set_ylabel('Price ($/kWh)')
        ax1_price.legend(loc='upper right')

        # 分段绘制在家时间段的填充区域
        home_segments = self._find_contiguous_segments(self.ev_at_home_record)
        for start, end in home_segments:
            segment_dates = mdates.date2num(self.records['timestamps'][start:end + 1])
            ax1.fill_between(segment_dates, 0, 1, color='green', alpha=0.3, transform=ax1.get_xaxis_transform())

        # 设置时间轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p3.png')
        plt.close()

        # ===== 第二个画布：ESS充放电功率、光伏发电与电价变化图 =====
        plt.figure(figsize=(20, 10))

        # 确保两个子图的横坐标范围一致
        min_date = mpl_dates[0]
        max_date = mpl_dates[-1]

        # ESS充放电功率图
        ax2_1 = plt.subplot(2, 1, 1)
        if len(self.ess_actions) != len(mpl_dates):
            print(
                f"Warning: Length mismatch between ess_actions ({len(self.ess_actions)}) and timestamps ({len(mpl_dates)}).")
            if len(self.ess_actions) < len(mpl_dates):
                self.ess_actions.append(0)
            else:
                self.ess_actions = self.ess_actions[:len(mpl_dates)]
        ess_actions = self.ess_actions

        ess_charge_power = [max(power, 0) for power in ess_actions]
        ess_discharge_power = [min(power, 0) for power in ess_actions]

        # 绘制ESS充放电柱状图，调整柱体宽度
        ax2_1.bar(mpl_dates, ess_charge_power, width=0.015, color='#05B9E2',
                  label='ESS Charging Power')  # 充电颜色更深，柱体宽度调整
        ax2_1.bar(mpl_dates, ess_discharge_power, width=0.015, color='#FFBE7A', label='ESS Discharging Power')

        # 在0刻度线绘制横线
        ax2_1.axhline(0, color='black', linewidth=0.8, linestyle='--')

        # 设置左轴范围
        ax2_1.set_ylim(-5, 5)

        ax2_1.set_ylabel('Power (kW)')
        ax2_1.set_title('ESS Charging/Discharging Power, PV Generation and Electricity Price')
        ax2_1.legend(loc='upper left')

        # 设置时间轴格式和范围
        ax2_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax2_1.set_xlim(min_date, max_date)

        # 光伏发电功率图（右轴）
        ax2_1_pv = ax2_1.twinx()
        # 绘制带点的曲线
        ax2_1_pv.plot(mpl_dates, self.pv_generation_record, color='green', marker='o', linestyle='-',
                      label='PV Generation')
        ax2_1_pv.set_ylabel('PV Generation (kW)')
        ax2_1_pv.legend(loc='upper right')

        # 确保光伏发电曲线在0刻度线上方
        ax2_1_pv.set_ylim(-1, 1)  # 右轴范围从-2到2，光伏发电显示在0上方

        # 电价图（右轴右侧）
        ax2_1_price = ax2_1.twinx()
        ax2_1_price.spines['right'].set_position(('outward', 60))  # 将电价轴向外移动
        ax2_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax2_1_price.set_ylabel('Price ($/kWh)')
        ax2_1_price.legend(loc='lower right')

        # ESS SOC变化图
        ax2_2 = plt.subplot(2, 1, 2)
        time_interval = 0.5 / 24  # 30分钟间隔
        bar_width = 0.8 * time_interval

        bars = ax2_2.bar(
            x=mpl_dates,
            height=self.ess_state_record,
            width=bar_width,
            color='#23BAC5',
            edgecolor='none',
            align='edge',
            label='ESS SOC'
        )

        ax2_2.set_xlim(min_date, max_date)

        # 设置时间轴格式
        ax2_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        ax2_2.set_ylabel('ESS SOC (kWh)', color='#4EC0E9')
        ax2_2.tick_params(axis='y', labelcolor='#4EC0E9')
        ax2_2.set_title('ESS State of Charge ')

        ax2_2.legend([bars], ['ESS SOC'], loc='upper left')

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p4.png')
        plt.close()

        # ===== 第三个画布：空调功率变化图和温度变化图 =====
        plt.figure(figsize=(20, 10))

        # 空调功率与电价图
        ax3_1 = plt.subplot(2, 1, 1)
        # 使用step函数绘制空调功率变化
        ax3_1.step(mpl_dates, self.air_conditioner_power_record, color='#B2DBB9', label='AC Power', where='post')
        ax3_1.set_ylabel('AC Power (kW)')
        ax3_1.set_title('AC Power and Electricity Price')
        ax3_1.legend(loc='upper left')

        ax3_1_price = ax3_1.twinx()
        ax3_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax3_1_price.set_ylabel('Price ($/kWh)')
        ax3_1_price.legend(loc='upper right')

        # 设置时间轴格式
        ax3_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # 温度变化图
        ax3_2 = plt.subplot(2, 1, 2)
        ax3_2.plot(mpl_dates, self.records['indoor_temp'], color='limegreen', label='Indoor Temperature')
        ax3_2.plot(mpl_dates, self.records['outdoor_temp'], color='deepskyblue', label='Outdoor Temperature')

        # 添加适宜温度范围的横线和填充区域
        ax3_2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax3_2.axhline(24, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax3_2.fill_between(mpl_dates, 20, 24, color='lightblue', alpha=0.3)

        ax3_2.set_ylabel('Temperature (°C)')
        ax3_2.set_title('Indoor and Outdoor Temperature Trends')
        ax3_2.legend(loc='upper left')

        # 设置时间轴格式
        ax3_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p5.png')
        plt.close()

        # ===== 第四个画布：第二台空调功率变化图和温度变化图 =====
        plt.figure(figsize=(20, 10))

        # 空调功率与电价图
        ax4_1 = plt.subplot(2, 1, 1)
        # 使用step函数绘制空调功率变化
        ax4_1.step(mpl_dates, self.air_conditioner_power_record2, color='#B2DBB9', label='AC Power', where='post')
        ax4_1.set_ylabel('AC Power (kW)')
        ax4_1.set_title('AC Power and Electricity Price')
        ax4_1.legend(loc='upper left')

        ax4_1_price = ax4_1.twinx()
        ax4_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax4_1_price.set_ylabel('Price ($/kWh)')
        ax4_1_price.legend(loc='upper right')

        # 设置时间轴格式
        ax4_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # 温度变化图
        ax4_2 = plt.subplot(2, 1, 2)
        ax4_2.plot(mpl_dates, self.records['indoor_temp2'], color='limegreen', label='Indoor Temperature')
        ax4_2.plot(mpl_dates, self.records['outdoor_temp'], color='deepskyblue', label='Outdoor Temperature')

        # 添加适宜温度范围的横线和填充区域
        ax4_2.axhline(16, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax4_2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax4_2.fill_between(mpl_dates, 16, 20, color='lightblue', alpha=0.3)

        ax4_2.set_ylabel('Temperature (°C)')
        ax4_2.set_title('Indoor and Outdoor Temperature Trends')
        ax4_2.legend(loc='upper left')

        # 设置时间轴格式
        ax4_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p6.png')
        plt.close()

        # ===== 第五个画布：洗衣机状态图 =====
        plt.figure(figsize=(20, 5))

        ax5 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])  # 转换时间戳为matplotlib格式

        # 绘制洗衣机状态，将柱状图向左对齐时间戳的起点
        time_interval = 0.5 / 24  # 30分钟转换为天数
        bar_width = time_interval  # 设置柱状图宽度与时间间隔一致

        # 调整柱状图的位置，使其左侧边与时间戳对齐
        bar_left_edges = mpl_dates

        ax5.bar(bar_left_edges, self.wash_machine_record, width=bar_width, color='#F0A19A',
                label='Washing Machine State', align='edge')
        ax5.set_ylabel('Washing Machine State')
        ax5.set_title('Washing Machine State and Electricity Price')
        ax5.legend(loc='upper left')

        ax5_price = ax5.twinx()
        # 使用step函数绘制阶梯状电价曲线
        ax5_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax5_price.set_ylabel('Price ($/kWh)')
        ax5_price.legend(loc='upper right')

        # 设置时间轴格式
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax5.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # 获取记录的日期范围
        start_date = self.records['timestamps'][0]
        end_date = self.records['timestamps'][-1]

        # 遍历每一天
        current_date = start_date
        while current_date <= end_date:
            # 计算当天的偏好时段
            preferred_start = current_date.replace(hour=self.wash_machine_preferred_time[0], minute=0)
            preferred_end = current_date.replace(hour=self.wash_machine_preferred_time[1], minute=0)

            # 添加垂直的阴影区域和虚线
            ax5.axvspan(preferred_start, preferred_end, facecolor='#5A9BD5', alpha=0.1)  # 淡蓝色填充
            ax5.axvline(preferred_start, color='#5A9BD5', linestyle='--', linewidth=1)  # 虚线
            ax5.axvline(preferred_end, color='#5A9BD5', linestyle='--', linewidth=1)

            # 进入下一天
            current_date += timedelta(days=1)

        # 设置时间轴范围为记录的时间戳范围
        ax5.set_xlim(mpl_dates[0], mpl_dates[-1])

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p7.png')
        plt.close()


        # ===== 第六个画布：热水器状态 =====
        plt.figure(figsize=(20, 10))
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # 创建一个 2 行 1 列的网格布局
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # 双轴显示功率和用水量（上面的子图）
        ax6_2 = plt.subplot(gs[0])
        ax6_2.bar(mpl_dates, self.ewh_power_record,
                  width=0.015, color='#1F77B4', label='Power')
        ax6_2.set_ylabel('Power (kW)', color='#1F77B4')
        ax6_2.tick_params(axis='y', labelcolor='#1F77B4')

        ax6_2_flow = ax6_2.twinx()
        ax6_2_flow.plot(mpl_dates, self.user_flow_record,
                        color='#2CA02C', marker='o', label='Water Flow')
        ax6_2_flow.set_ylabel('Flow Rate (L/h)', color='#2CA02C')
        ax6_2_flow.tick_params(axis='y', labelcolor='#2CA02C')

        # 水温曲线（下面的子图）
        ax6_1 = plt.subplot(gs[1])
        ax6_1.plot(mpl_dates, self.ewh_temp_record,
                   color='#FF7F0E', label='Water Temperature')
        ax6_1.axhline(40, color='grey', linestyle='--', label='Target Temp')
        ax6_1.set_ylabel('Temperature (°C)')
        ax6_1.set_title('Water Heater Status')

        # 添加两个适宜温度范围的填充区域
        # 高峰时段 53-57℃ (6-9点和18-22点)
        peak_low = 50
        peak_high = 60
        non_peak_low = 40
        non_peak_high = 50

        # 创建一个统一的温度范围用于填充
        all_low = [peak_low if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_low for ts in
                   self.records['timestamps']]
        all_high = [peak_high if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_high for ts in
                    self.records['timestamps']]

        # 绘制统一的填充区域
        ax6_1.fill_between(mpl_dates, all_low, all_high,
                           color='lightgreen', alpha=0.3, label='Comfort Zone')

        # 绘制高峰时段的特殊标记
        peak_mask = [(6 <= ts.hour <= 9) or (18 <= ts.hour <= 22) for ts in self.records['timestamps']]
        ax6_1.fill_between(mpl_dates, peak_low, peak_high,
                           where=peak_mask, color='lightcoral', alpha=0.3, label='Peak Comfort Zone')

        # 统一时间轴格式
        for ax in [ax6_1, ax6_2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # 添加图例
        ax6_1.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p8.png')
        plt.close()

        # ===== 第七个画布：家庭总负载变化图 =====
        plt.figure(figsize=(20, 5))
        ax7 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])  # 转换时间戳为matplotlib格式

        # 绘制家庭总负载曲线
        ax7.plot(mpl_dates, self.records['total_load'], color='purple', label='Total Load')
        ax7.set_ylabel('Load (kW)')
        ax7.set_title('Household Total Load Over Time')
        ax7.legend(loc='upper left')

        # 设置时间轴格式
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax7.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # 获取当前图的y轴范围
        ymin, ymax = ax7.get_ylim()

        # 设置网格
        ax7.grid(alpha=0.3)

        # 填充红色区域（上半部分）
        ax7.fill_between(mpl_dates, self.records['total_load'], 0, where=(np.array(self.records['total_load']) > 0),
                         color='red', alpha=0.3)

        # 填充绿色区域（下半部分）
        ax7.fill_between(mpl_dates, self.records['total_load'], 0, where=(np.array(self.records['total_load']) < 0),
                         color='green', alpha=0.3)

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p9.png')
        plt.close()

        # ===== 第8个画布：成本随时间变化的图 =====
        plt.figure(figsize=(20, 5))

        ax8 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # 绘制成本曲线
        ax8.plot(mpl_dates, self.records['energy_cost'], color='purple', label='Energy Cost')
        ax8.set_ylabel('Cost ($)')
        ax8.set_title('Energy Cost Over Time')
        ax8.legend(loc='upper left')

        # 设置时间轴格式
        ax8.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax8.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        plt.savefig('figures/environment_plots/p10.png')
        plt.close()

        # 新增：保存成本数据到文件
        # self.save_cost_data()

    def plot_reward_components(self):
        plt.figure(figsize=(20, 8))
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # 创建颜色映射
        colors = {
            'energy_cost': '#FF6B6B',
            'violation_penalty': '#4ECDC4',
            'ess_reward': '#D95319',
            'ev_reward': '#96CEB4',
            'user_penalty': '#FFEEAD',
            'temp_reward': '#D4A5A5'
            # 'sell': '#925EBO'
        }

        # 绘制堆叠区域图
        components = ['energy_cost', 'violation_penalty', 'ess_reward',
                      'ev_reward', 'user_penalty', 'temp_reward']

        # 累积值用于堆叠
        cumulative = np.zeros(len(mpl_dates))

        for comp in components:
            values = np.array(self.reward_components[comp])
            plt.fill_between(mpl_dates, cumulative, cumulative + values,
                             label=comp.replace('_', ' ').title(),
                             color=colors[comp], alpha=0.8)
            cumulative += values

        # 绘制总奖励线
        plt.plot(mpl_dates, self.reward_components['total'],
                 color='#2C3E50', linewidth=2, label='Total Reward')

        # 格式设置
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.xticks(rotation=45)

        plt.ylabel('Reward Value')
        plt.title('Reward Component Breakdown')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig('figures/environment_plots/p2.png')
        plt.close()
        # plt.show()

    def save_simulation_data(self, filename=None):
        """保存模拟数据到CSV文件，用于后续绘图分析"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.csv"
        
        # 确保数据目录存在
        os.makedirs('simulation_data', exist_ok=True)
        filepath = os.path.join('simulation_data', filename)
        
        # 准备数据字典
        data_dict = {
            'timestamp': self.records['timestamps'],
            'ev_soc': self.ev_battery_record,
            'ess_soc': self.ess_state_record,
            'home_load': self.home_load_record,
            'pv_generation': self.pv_generation_record,
            'electricity_price': self.electricity_price_record,
            'ev_at_home': self.ev_at_home_record,
            'wash_machine_state': self.wash_machine_record,
            'air_conditioner_power': self.air_conditioner_power_record,
            'ess_actions': self.ess_actions,
            'wash_machine_actions': self.wash_machine_actions,
            'air_conditioner_actions': self.air_conditioner_actions,
            'ess_charge_pv': self.ess_charge_pv,
            'ess_charge_grid': self.ess_charge_grid,
            'ess_discharge_ev': self.ess_discharge_ev,
            'ess_discharge_house': self.ess_discharge_house,
            'indoor_temp': self.records['indoor_temp'],
            'indoor_temp2': self.records['indoor_temp2'],
            'outdoor_temp': self.records['outdoor_temp'],
            'total_load': self.records['total_load'],
            'energy_cost': self.records['energy_cost'],
            'user_dissatisfaction': self.records['user_dissatisfaction'],
            'ewh_temp': self.ewh_temp_record,
            'ewh_power': self.ewh_power_record,
            'user_flow': self.user_flow_record,
            'air_conditioner_power2': self.air_conditioner_power_record2,
            'daily_costs': self.records['daily_costs']
        }
        
        # 处理长度不匹配的问题
        max_length = max(len(v) for v in data_dict.values() if isinstance(v, list))
        
        # 确保所有列表长度一致
        for key, value in data_dict.items():
            if isinstance(value, list):
                if len(value) < max_length:
                    # 用最后一个值填充
                    data_dict[key] = value + [value[-1]] * (max_length - len(value))
                elif len(value) > max_length:
                    # 截断到最大长度
                    data_dict[key] = value[:max_length]
        
        # 创建DataFrame
        df = pd.DataFrame(data_dict)
        
        # 保存到CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"模拟数据已保存到: {filepath}")
        
        # 同时保存奖励组件数据
        reward_filename = filename.replace('.csv', '_rewards.csv')
        reward_filepath = os.path.join('simulation_data', reward_filename)
        
        reward_data = {}
        for key, values in self.reward_components.items():
            if isinstance(values, list):
                if len(values) < max_length:
                    reward_data[key] = values + [values[-1]] * (max_length - len(values))
                else:
                    reward_data[key] = values[:max_length]
        
        reward_df = pd.DataFrame(reward_data)
        reward_df.to_csv(reward_filepath, index=False, encoding='utf-8-sig')
        print(f"奖励组件数据已保存到: {reward_filepath}")
        
        return filepath

