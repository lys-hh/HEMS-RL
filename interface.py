"""
此文件用于写一些处理数据的接口函数
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
class DataInterface:
    def __init__(self, cons_file, prod_file):
        # 定义文件路径
        self.cons_file = cons_file
        self.prod_file = prod_file

        # 读取数据
        self.cons_data = pd.read_csv(self.cons_file, parse_dates=['date'], index_col='date')
        self.prod_data = pd.read_csv(self.prod_file, parse_dates=['date'], index_col='date')

        # 存储每天的 EV 到家时间和离开家时间
        self.ev_schedule = {}
        # 设置随机种子
        self.np_random = np.random.RandomState(0)
    def get_home_load(self, current_date, current_time_index):
        # 获取家庭当前的用电量
        return self.cons_data.loc[current_date, self.cons_data.columns[current_time_index]]

    def get_pv_generation(self, current_date, current_time_index):
        # 获取光伏系统的当前发电量
        return self.prod_data.loc[current_date, self.cons_data.columns[current_time_index]]

    def get_electricity_price(self, current_date, current_time_index):
        # 将当前日期字符串转换为datetime对象
        if current_time_index >= 48:
            current_time_index = current_time_index-48

        # 定义一天中的不同时间段及其对应的电价
        time_based_prices = {
            '低谷': 0.2,  # 假设低谷电价为0.4元/千瓦时
            '平段': 0.5,  # 假设平段电价为0.6元/千瓦时
            '高峰': 0.8,  # 假设高峰电价为0.9元/千瓦时
        }

        # 定义一天中的时间段（以小时为单位）
        # 假设低谷时段为00:00-06:00，平段为06:00-18:00，高峰为18:00-24:00
        if 0 <= current_time_index < 20 or 44 <= current_time_index < 48:
            price = time_based_prices['低谷']
        elif 8 <= current_time_index < 34:
            price = time_based_prices['平段']
        else:
            price = time_based_prices['高峰']

        # 返回电价
        return price

    def get_date_time(self, current_date, current_time_index):
        return current_date


    def is_ev_at_home(self, current_date, current_time_index):
        """
        判断当前时间电动汽车是否在家。
        每天只生成一次到家时间和离开家时间，第二天重新生成。
        """
        # 检查当前日期是否已经有生成的到家时间和离开家时间
        if current_date not in self.ev_schedule:
            self.generate_daily_ev_schedule(current_date)

        # 如果是周六或周日，直接返回 True 表示 EV 全天在家
        current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
        if current_weekday >= 5:  # 5 表示周六，6 表示周日
            return True

        t1, t2 = self.ev_schedule[current_date]
        current_hour = current_time_index
        return not t2 <= current_hour < t1

    def generate_daily_ev_schedule(self, current_date):
        """
        为当前日期生成 EV 的到家时间和离开家时间。
        """
        # 如果是周六或周日，直接设置 EV 全天在家
        current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
        if current_weekday >= 5:  # 5 表示周六，6 表示周日
            self.ev_schedule[current_date] = (0, 24 * 2)  # 全天在家
            return

        # 调整到家和离开家时间的分布参数
        t1_mean = 20  # 增加到家时间的均值（更晚回家）
        t1_std = 1
        t1_range = (18, 22)  # 扩大到家时间的范围

        t2_mean = 8  # 减少离开家时间的均值（更早出门）
        t2_std = 1
        t2_range = (7, 9)  # 缩小离开家时间的范围，确保 t2 < t1

        while True:
            t1 = self.np_random.normal(t1_mean, t1_std)
            t2 = self.np_random.normal(t2_mean, t2_std)
            t1 = max(min(t1, t1_range[1]), t1_range[0])
            t2 = max(min(t2, t2_range[1]), t2_range[0])
            if t2 < t1:
                break  # 确保离开家时间早于到家时间

        # 将生成的时间存储为以 0.5 为单位（代表半小时）
        self.ev_schedule[current_date] = (t1 * 2, t2 * 2)

    def is_ev_departing_soon(self,current_date, current_time_idx):
        """判断EV是否即将离家（未来2小时内）"""
        return self.get_hours_until_departure(current_date, current_time_idx) <= 2

    def get_hours_until_departure(self, current_date, current_time_idx):
        """获取距离下次离家的剩余时间（小时）"""
        # current_date = datetime.strptime(current_date, '%Y-%m-%d')
        if current_date not in self.ev_schedule:
            self.generate_daily_ev_schedule(current_date)

        departure_time = self.ev_schedule[current_date][1] / 2  # 转换为小时
        current_hour = current_time_idx / 2
        return departure_time - current_hour if current_hour < departure_time else (
                24 + departure_time - current_hour)

    def get_outdoor_temp(self, current_time, current_time_index):
        """
        室外温度模拟
        """
        current_datetime = datetime.strptime(current_time, '%Y-%m-%d')
        day_of_year = current_datetime.timetuple().tm_yday

        # ===== 改进参数设置 =====
        # 全年基础温度模型
        base_temp = 24 + 10 * np.sin(2 * np.pi * (day_of_year - 200) / 365)  # 7月峰值调整

        # 日温度波动模型（增大昼夜温差）
        hour = current_time_index // 2
        minute = 30 * (current_time_index % 2)
        time_of_day = hour + minute / 60
        daily_temp_variation = 8 * np.sin(2 * np.pi * (time_of_day - 14.5) / 24)

        # 夏季高温
        summer_boost = 0
        if 172 <= day_of_year <= 265:  # 6/21~9/22期间
            summer_boost = 4 * np.sin(np.pi * (day_of_year - 172) / 93)  # 夏季额外升温

        # 天气不确定性
        random_noise = np.random.normal(0, 2.5)

        # ===== 最终温度计算 =====
        outdoor_temp = base_temp + daily_temp_variation + summer_boost + random_noise

        # 温度边界保护
        outdoor_temp = np.clip(outdoor_temp, -5, 45)


        return round(outdoor_temp, 1)