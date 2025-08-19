"""
从CSV数据生成图片
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta

class CSVPlotter:
    def __init__(self, csv_file_path):
        """初始化绘图器"""
        self.csv_file_path = csv_file_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """加载CSV数据"""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_file_path}")
        
        self.data = pd.read_csv(self.csv_file_path)
        print(f"已加载数据: {self.csv_file_path}")
        print(f"数据形状: {self.data.shape}")
        print(f"数据列: {list(self.data.columns)}")
        
        # 转换时间戳
        if 'timestamp' in self.data.columns:
            try:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                print("时间戳转换成功")
            except Exception as e:
                print(f"时间戳转换失败: {e}")
                # 如果转换失败，尝试其他可能的列名
                possible_timestamp_cols = ['time', 'date', 'datetime', 'Time', 'Date', 'DateTime']
                for col in possible_timestamp_cols:
                    if col in self.data.columns:
                        try:
                            self.data['timestamp'] = pd.to_datetime(self.data[col])
                            print(f"使用列 '{col}' 作为时间戳")
                            break
                        except:
                            continue
                else:
                    print("警告: 无法找到有效的时间戳列")
                    # 创建一个简单的时间索引
                    self.data['timestamp'] = pd.date_range('2011-07-03', periods=len(self.data), freq='30min')
        else:
            print("警告: CSV文件中没有'timestamp'列")
            # 创建一个简单的时间索引
            self.data['timestamp'] = pd.date_range('2011-07-03', periods=len(self.data), freq='30min')
    
    def plot_ev_soc_and_price(self, save_path='figures/environment_plots/ev_soc_and_price.png'):
        """绘制EV SOC和电价图"""
        plt.figure(figsize=(20, 5))
        
        # 使用已经转换的时间戳
        timestamps = self.data['timestamp']
        ev_soc = self.data['ev_soc'].values
        ev_at_home = self.data['ev_at_home'].values
        electricity_price = self.data['electricity_price'].values
        
        # 离家时段设为NaN
        ev_soc[~ev_at_home.astype(bool)] = np.nan
        
        ax1 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(timestamps)
        
        # 绘制EV SOC曲线
        ax1.plot(mpl_dates, ev_soc, color='blue', label='EV SOC', linewidth=2)
        ax1.set_ylabel('EV SOC (kWh)', fontsize=12)
        ax1.set_title('EV State of Charge and Electricity Price', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        
        # 绘制电价曲线（右轴）- 使用step函数
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', label='Electricity Price', 
                       where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=12)
        ax1_price.legend(loc='upper right', fontsize=10)
        
        # 分段绘制在家时间段的填充区域
        home_segments = self._find_contiguous_segments(ev_at_home.astype(bool))
        for start, end in home_segments:
            segment_dates = mdates.date2num(timestamps.iloc[start:end + 1])
            ax1.fill_between(segment_dates, 0, 1, color='green', alpha=0.3, 
                           transform=ax1.get_xaxis_transform())
        
        # 设置时间轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_ess_power_and_pv(self, save_path='figures/environment_plots/ess_power_and_pv.png'):
        """绘制ESS功率和光伏发电图"""
        plt.figure(figsize=(20, 10))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # 确保两个子图的横坐标范围一致
        min_date = mpl_dates[0]
        max_date = mpl_dates[-1]
        
        # 检查是否存在ess_actions列，如果不存在则通过ess_soc计算
        if 'ess_actions' not in self.data.columns:
            print("警告: CSV文件中缺少'ess_actions'列，通过ess_soc差值计算ESS功率")
            
            # 通过ess_soc差值计算ess_actions
            ess_soc = self.data['ess_soc'].values
            ess_actions = []
            
            for i in range(len(ess_soc)):
                if i == 0:
                    # 第一个时间步，假设初始功率为0
                    ess_actions.append(0.0)
                else:
                    # 计算SOC变化率，转换为功率
                    # 假设时间间隔为0.5小时，效率为0.95
                    time_interval = 0.5  # 小时
                    efficiency = 0.95
                    
                    soc_change = ess_soc[i] - ess_soc[i-1]
                    power = soc_change / (time_interval * efficiency)
                    ess_actions.append(power)
            
            ess_actions = np.array(ess_actions)
        else:
            ess_actions = self.data['ess_actions'].values
        
        # ESS充放电功率
        ess_charge_power = [max(power, 0) for power in ess_actions]
        ess_discharge_power = [min(power, 0) for power in ess_actions]
        
        # 光伏发电
        pv_generation = self.data['pv_generation'].values
        electricity_price = self.data['electricity_price'].values
        
        # 第一个子图：ESS功率和光伏发电
        ax1 = plt.subplot(2, 1, 1)
        
        # 绘制ESS充放电柱状图
        ax1.bar(mpl_dates, ess_charge_power, width=0.015, color='#05B9E2',
                label='ESS Charging Power', alpha=0.8)
        ax1.bar(mpl_dates, ess_discharge_power, width=0.015, color='#FFBE7A', 
                label='ESS Discharging Power', alpha=0.8)
        
        # 在0刻度线绘制横线
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax1.set_ylim(-5, 5)
        ax1.set_ylabel('Power (kW)', fontsize=12)
        ax1.set_title('ESS Charging/Discharging Power, PV Generation and Electricity Price', 
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        
        # 设置时间轴格式和范围
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax1.set_xlim(min_date, max_date)
        
        # 光伏发电功率图（右轴）
        ax1_pv = ax1.twinx()
        ax1_pv.plot(mpl_dates, pv_generation, color='green', marker='o', 
                    linestyle='-', label='PV Generation', markersize=3)
        ax1_pv.set_ylabel('PV Generation (kW)', fontsize=12)
        ax1_pv.legend(loc='upper right', fontsize=10)
        ax1_pv.set_ylim(-1, 1)
        
        # 电价图（右轴右侧）
        ax1_price = ax1.twinx()
        ax1_price.spines['right'].set_position(('outward', 60))
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=12)
        ax1_price.legend(loc='lower right', fontsize=10)
        
        # 第二个子图：ESS SOC变化
        ax2 = plt.subplot(2, 1, 2)
        ess_soc = self.data['ess_soc'].values
        
        time_interval = 0.5 / 24
        bar_width = 0.8 * time_interval
        
        bars = ax2.bar(x=mpl_dates, height=ess_soc, width=bar_width,
                       color='#23BAC5', edgecolor='none', align='edge', label='ESS SOC')
        
        ax2.set_xlim(min_date, max_date)
        ax2.set_ylabel('ESS SOC (kWh)', color='#4EC0E9', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#4EC0E9')
        ax2.set_title('ESS State of Charge', fontsize=14, fontweight='bold')
        ax2.legend([bars], ['ESS SOC'], loc='upper left', fontsize=10)
        
        # 设置时间轴格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_combined_ac_temp(self, save_path='figures/environment_plots/combined_ac_temp.png'):
        """绘制两个空调的室内温度对比图（室外温度一致）"""
        plt.figure(figsize=(20, 10))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # 温度数据
        indoor_temp = self.data['indoor_temp'].values
        indoor_temp2 = self.data['indoor_temp2'].values if 'indoor_temp2' in self.data.columns else None
        outdoor_temp = self.data['outdoor_temp'].values
        
        # 空调功率数据
        ac_power = self.data['air_conditioner_power'].values
        ac_power2 = self.data['air_conditioner_power2'].values if 'air_conditioner_power2' in self.data.columns else None
        electricity_price = self.data['electricity_price'].values
        
        # 第一个子图：空调功率对比（上方）
        ax1 = plt.subplot(2, 1, 1)
        
        # 绘制第一台空调功率
        ax1.step(mpl_dates, ac_power, color='limegreen', label='AC1 Power', 
                 where='post', linewidth=2)
        
        # 绘制第二台空调功率（如果存在）
        if ac_power2 is not None:
            ax1.step(mpl_dates, ac_power2, color='orange', label='AC2 Power', 
                     where='post', linewidth=2)
        
        ax1.set_ylabel('AC Power (kW)', fontsize=12)
        ax1.set_title('Air Conditioner Power and Electricity Price', fontsize=14, fontweight='bold')
        
        # 电价曲线（右轴）
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=12)
        
        # 合并两个轴的图例，都放在左上角
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_price.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        # 设置时间轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        # 第二个子图：温度对比（下方）
        ax2 = plt.subplot(2, 1, 2)
        
        # 绘制室外温度（两个空调共用）
        ax2.plot(mpl_dates, outdoor_temp, color='deepskyblue', label='Outdoor Temperature', linewidth=2)
        
        # 绘制第一台空调的室内温度
        ax2.plot(mpl_dates, indoor_temp, color='limegreen', label='AC1 Indoor Temperature', linewidth=2)
        
        # 绘制第二台空调的室内温度（如果存在）
        if indoor_temp2 is not None:
            ax2.plot(mpl_dates, indoor_temp2, color='orange', label='AC2 Indoor Temperature', linewidth=2)
        
        # 添加第一台空调的舒适温度范围（20-24℃）
        ax2.axhline(20, color='limegreen', linestyle='--', alpha=0.7, label='AC1 Comfort Zone Lower')
        ax2.axhline(24, color='limegreen', linestyle='--', alpha=0.7, label='AC1 Comfort Zone Upper')
        ax2.fill_between(mpl_dates, 20, 24, color='lightgreen', alpha=0.3, label='AC1 Comfort Zone')
        
        # 添加第二台空调的舒适温度范围（16-20℃）
        if indoor_temp2 is not None:
            ax2.axhline(16, color='orange', linestyle='--', alpha=0.7, label='AC2 Comfort Zone Lower')
            ax2.axhline(20, color='orange', linestyle='--', alpha=0.7, label='AC2 Comfort Zone Upper')
            ax2.fill_between(mpl_dates, 16, 20, color='lightcoral', alpha=0.3, label='AC2 Comfort Zone')
        
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_title('Indoor and Outdoor Temperature Comparison', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        
        # 设置时间轴格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    def plot_ac_power_and_temp(self, save_path='figures/environment_plots/ac_power_and_temp.png'):
        """绘制空调功率和温度图"""
        # 调用新的合并函数
        self.plot_combined_ac_temp(save_path.replace('ac_power_and_temp.png', 'combined_ac_temp.png'))
        
        # 保留原有的单独绘制功能
        plt.figure(figsize=(20, 10))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # 空调功率与电价
        ac_power = self.data['air_conditioner_power'].values
        electricity_price = self.data['electricity_price'].values
        
        ax1 = plt.subplot(2, 1, 1)
        # 使用step函数绘制空调功率变化
        ax1.step(mpl_dates, ac_power, color='#B2DBB9', label='AC Power', 
                 where='post', linewidth=2)
        ax1.set_ylabel('AC Power (kW)', fontsize=12)
        ax1.set_title('AC Power and Electricity Price', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=12)
        ax1_price.legend(loc='upper right', fontsize=10)
        
        # 设置时间轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        # 温度变化图
        ax2 = plt.subplot(2, 1, 2)
        indoor_temp = self.data['indoor_temp'].values
        outdoor_temp = self.data['outdoor_temp'].values
        
        ax2.plot(mpl_dates, indoor_temp, color='limegreen', label='Indoor Temperature', linewidth=2)
        ax2.plot(mpl_dates, outdoor_temp, color='deepskyblue', label='Outdoor Temperature', linewidth=2)
        
        # 添加适宜温度范围的横线和填充区域
        ax2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax2.axhline(24, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax2.fill_between(mpl_dates, 20, 24, color='lightblue', alpha=0.3)
        
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_title('Indoor and Outdoor Temperature Trends', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        
        # 设置时间轴格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_ac_power_and_temp2(self, save_path='figures/environment_plots/ac_power_and_temp2.png'):
        """绘制第二台空调功率和温度图"""
        plt.figure(figsize=(20, 10))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # 检查是否存在第二台空调的数据
        if 'air_conditioner_power2' not in self.data.columns:
            print("警告: CSV文件中缺少'air_conditioner_power2'列，跳过第二台空调图绘制")
            plt.close()
            return
        
        # 第二台空调功率与电价
        ac_power2 = self.data['air_conditioner_power2'].values
        electricity_price = self.data['electricity_price'].values
        
        ax1 = plt.subplot(2, 1, 1)
        # 使用step函数绘制空调功率变化
        ax1.step(mpl_dates, ac_power2, color='#B2DBB9', label='AC Power', 
                 where='post', linewidth=2)
        ax1.set_ylabel('AC Power (kW)', fontsize=12)
        ax1.set_title('AC Power and Electricity Price', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=12)
        ax1_price.legend(loc='upper right', fontsize=10)
        
        # 设置时间轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        # 温度变化图
        ax2 = plt.subplot(2, 1, 2)
        indoor_temp2 = self.data['indoor_temp2'].values
        outdoor_temp = self.data['outdoor_temp'].values
        
        ax2.plot(mpl_dates, indoor_temp2, color='limegreen', label='Indoor Temperature', linewidth=2)
        ax2.plot(mpl_dates, outdoor_temp, color='deepskyblue', label='Outdoor Temperature', linewidth=2)
        
        # 添加适宜温度范围的横线和填充区域（第二台空调的舒适区域是16-20℃）
        ax2.axhline(16, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax2.fill_between(mpl_dates, 16, 20, color='lightblue', alpha=0.3)
        
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_title('Indoor and Outdoor Temperature Trends', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        
        # 设置时间轴格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_wash_machine_state(self, save_path='figures/environment_plots/wash_machine_state.png'):
        """绘制洗衣机状态图"""
        plt.figure(figsize=(20, 5))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        wash_machine_state = self.data['wash_machine_state'].values
        electricity_price = self.data['electricity_price'].values
        
        ax = plt.subplot(1, 1, 1)
        time_interval = 0.5 / 24
        bar_width = time_interval
        
        # 使用align='edge'左对齐，与environment.py保持一致
        ax.bar(mpl_dates, wash_machine_state, width=bar_width, color='#F0A19A',
               label='Washing Machine State', align='edge', alpha=0.8)
        ax.set_ylabel('Washing Machine State', fontsize=12)
        ax.set_title('Washing Machine State and Electricity Price', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # 使用step函数绘制阶梯状电价曲线
        ax_price = ax.twinx()
        ax_price.step(mpl_dates, electricity_price, color='red', 
                     label='Electricity Price', where='post', linewidth=2)
        ax_price.set_ylabel('Price ($/kWh)', fontsize=12)
        ax_price.legend(loc='upper right', fontsize=10)
        
        # 设置时间轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        # 添加洗衣机偏好时段阴影（假设偏好时段为8-12点）
        # 注意：这里需要根据实际的偏好时段设置
        wash_machine_preferred_time = [6, 20]  # 假设偏好时段为8-12点
        
        # 获取记录的日期范围
        start_date = timestamps.iloc[0]
        end_date = timestamps.iloc[-1]
        
        # 遍历每一天
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_rounded = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        while current_date <= end_date_rounded:
            # 计算当天的偏好时段
            preferred_start = current_date.replace(hour=wash_machine_preferred_time[0], minute=0)
            preferred_end = current_date.replace(hour=wash_machine_preferred_time[1], minute=0)
            
            # 添加垂直的阴影区域和虚线
            ax.axvspan(preferred_start, preferred_end, facecolor='#5A9BD5', alpha=0.1)  # 淡蓝色填充
            ax.axvline(preferred_start, color='#5A9BD5', linestyle='--', linewidth=1)  # 虚线
            ax.axvline(preferred_end, color='#5A9BD5', linestyle='--', linewidth=1)
            
            # 进入下一天
            current_date += timedelta(days=1)
        
        # 设置时间轴范围为记录的时间戳范围
        ax.set_xlim(mpl_dates[0], mpl_dates[-1])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_water_heater_status(self, save_path='figures/environment_plots/water_heater_status.png'):
        """绘制热水器状态图"""
        plt.figure(figsize=(20, 10))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        ewh_power = self.data['ewh_power'].values
        ewh_temp = self.data['ewh_temp'].values
        user_flow = self.data['user_flow'].values
        
        # 创建2行1列的网格布局
        from matplotlib import gridspec
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        
        # 双轴显示功率和用水量（上面的子图）
        ax1 = plt.subplot(gs[0])
        ax1.bar(mpl_dates, ewh_power, width=0.015, color='#1F77B4', label='Power', alpha=0.8)
        ax1.set_ylabel('Power (kW)', color='#1F77B4', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#1F77B4')
        
        ax1_flow = ax1.twinx()
        ax1_flow.plot(mpl_dates, user_flow, color='#2CA02C', marker='o', 
                      label='Water Flow', markersize=3)
        ax1_flow.set_ylabel('Flow Rate (L/h)', color='#2CA02C', fontsize=12)
        ax1_flow.tick_params(axis='y', labelcolor='#2CA02C')
        
        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_flow.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        
        # 水温曲线（下面的子图）
        ax2 = plt.subplot(gs[1])
        ax2.plot(mpl_dates, ewh_temp, color='#FF7F0E', label='Water Temperature', linewidth=2)
        # ax2.axhline(40, color='grey', linestyle='--', label='Target Temp')
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_title('Water Heater Status', fontsize=14, fontweight='bold')
        
        # 添加适宜温度范围的填充区域
        peak_low = 50
        peak_high = 60
        non_peak_low = 40
        non_peak_high = 50
        
        # 创建统一的温度范围用于填充
        all_low = [peak_low if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_low 
                   for ts in timestamps]
        all_high = [peak_high if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_high 
                    for ts in timestamps]
        
        # 绘制统一的填充区域
        ax2.fill_between(mpl_dates, all_low, all_high, color='lightgreen', alpha=0.3, 
                        label='Comfort Zone')
        
        # 绘制高峰时段的特殊标记
        peak_mask = [(6 <= ts.hour <= 9) or (18 <= ts.hour <= 22) for ts in timestamps]
        ax2.fill_between(mpl_dates, peak_low, peak_high, where=peak_mask, 
                        color='lightcoral', alpha=0.3, label='Peak Comfort Zone')
        
        # 统一时间轴格式
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        # 添加图例
        ax2.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_total_load(self, save_path='figures/environment_plots/total_load.png'):
        """绘制家庭总负载图"""
        plt.figure(figsize=(20, 5))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        total_load = self.data['total_load'].values
        
        ax = plt.subplot(1, 1, 1)
        ax.plot(mpl_dates, total_load, color='purple', label='Total Load', linewidth=2)
        ax.set_ylabel('Load (kW)', fontsize=12)
        ax.set_title('Household Total Load Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # 设置时间轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        # 设置网格
        ax.grid(alpha=0.3)
        
        # 填充红色区域（上半部分）
        ax.fill_between(mpl_dates, total_load, 0, where=(total_load > 0),
                       color='red', alpha=0.3)
        # 填充绿色区域（下半部分）
        ax.fill_between(mpl_dates, total_load, 0, where=(total_load < 0),
                       color='green', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_energy_cost(self, save_path='figures/environment_plots/energy_cost.png'):
        """绘制能源成本图"""
        plt.figure(figsize=(20, 5))
        
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        energy_cost = self.data['energy_cost'].values
        
        ax = plt.subplot(1, 1, 1)
        ax.plot(mpl_dates, energy_cost, color='purple', label='Energy Cost', linewidth=2)
        ax.set_ylabel('Cost ($)', fontsize=12)
        ax.set_title('Energy Cost Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # 设置时间轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def plot_reward_components(self, save_path='figures/environment_plots/reward_components.png'):
        """绘制奖励组件图"""
        # 尝试加载奖励数据
        reward_file = self.csv_file_path.replace('.csv', '_rewards.csv')
        if not os.path.exists(reward_file):
            print(f"奖励数据文件不存在: {reward_file}")
            return
        
        reward_data = pd.read_csv(reward_file)
        timestamps = self.data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        plt.figure(figsize=(20, 8))
        
        # 创建颜色映射
        colors = {
            'energy_cost': '#FF6B6B',
            'violation_penalty': '#4ECDC4',
            'ess_reward': '#D95319',
            'ev_reward': '#96CEB4',
            'user_penalty': '#FFEEAD',
            'temp_reward': '#D4A5A5'
        }
        
        # 绘制堆叠区域图
        components = ['energy_cost', 'violation_penalty', 'ess_reward',
                     'ev_reward', 'user_penalty', 'temp_reward']
        
        # 累积值用于堆叠
        cumulative = np.zeros(len(mpl_dates))
        
        for comp in components:
            if comp in reward_data.columns:
                values = np.array(reward_data[comp])
                plt.fill_between(mpl_dates, cumulative, cumulative + values,
                               label=comp.replace('_', ' ').title(),
                               color=colors[comp], alpha=0.8)
                cumulative += values
        
        # 绘制总奖励线
        if 'total' in reward_data.columns:
            plt.plot(mpl_dates, reward_data['total'],
                    color='#2C3E50', linewidth=2, label='Total Reward')
        
        # 格式设置
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.xticks(rotation=45)
        
        plt.ylabel('Reward Value', fontsize=12)
        plt.title('Reward Component Breakdown', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")
    
    def _find_contiguous_segments(self, bool_list):
        """找到连续的True段"""
        segments = []
        start = None
        
        for i, value in enumerate(bool_list):
            if value and start is None:
                start = i
            elif not value and start is not None:
                segments.append((start, i - 1))
                start = None
        
        if start is not None:
            segments.append((start, len(bool_list) - 1))
        
        return segments
    
    def generate_all_plots(self, output_dir='save'):
        """生成所有图片"""
        print("开始生成所有图片...")
        
        # 设置matplotlib参数
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
        # 生成所有图片
        self.plot_ev_soc_and_price(f'{output_dir}/ev_soc_and_price.png')
        self.plot_ess_power_and_pv(f'{output_dir}/ess_power_and_pv.png')
        self.plot_ac_power_and_temp(f'{output_dir}/ac_power_and_temp.png')
        self.plot_ac_power_and_temp2(f'{output_dir}/ac_power_and_temp2.png')
        # 生成合并的空调温度图
        self.plot_combined_ac_temp(f'{output_dir}/combined_ac_temp.png')
        self.plot_wash_machine_state(f'{output_dir}/wash_machine_state.png')
        self.plot_water_heater_status(f'{output_dir}/water_heater_status.png')
        self.plot_total_load(f'{output_dir}/total_load.png')
        self.plot_energy_cost(f'{output_dir}/energy_cost.png')
        self.plot_reward_components(f'{output_dir}/reward_components.png')
        
        print("所有图片生成完成！")

def main():
    """主函数"""
    # 直接在代码中指定文件路径
    csv_file_path = "simulation_data/simulation_data_episode_5000.csv"
    output_dir = "figures/algorithm_comparison"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"文件不存在: {csv_file_path}")
        print("请检查文件路径是否正确")
        return
    
    # 检查是否是rewards文件
    if '_rewards.csv' in csv_file_path:
        print("警告: 检测到rewards文件，请指定主数据文件！")
        print("主数据文件通常不包含'_rewards'后缀")
        print("示例: simulation_data/simulation_data_episode_5000.csv")
        return
    
    try:
        # 创建绘图器
        plotter = CSVPlotter(csv_file_path)
        
        # 生成所有图片
        plotter.generate_all_plots(output_dir)
        
        print(f"\n所有图片已保存到 {output_dir} 文件夹")
        print(f"数据来源: {csv_file_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 