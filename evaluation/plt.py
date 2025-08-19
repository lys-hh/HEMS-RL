import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加项目根目录到Python路径（使用相对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model import rl_utils


def plot_returns(return_list, window_size=9):
    """
    绘制原始回报曲线和移动平均回报曲线。

    参数:
        return_list (list): 每个episode的回报列表。
        num_episodes (int): 总的episode数量。
        window_size (int): 移动平均窗口大小。
    """
    episodes_list = list(range(len(return_list)))

    # 计算移动平均回报
    mv_return = rl_utils.moving_average(return_list, window_size)

    # 绘制原始回报曲线和移动平均回报曲线
    plt.figure(figsize=(12, 6))

    # 绘制原始回报曲线，淡紫色
    plt.plot(episodes_list, return_list, label='Original Returns', color='thistle', alpha=0.7)

    # 绘制移动平均回报曲线，稍微深一点的颜色
    plt.plot(episodes_list, mv_return, label=f'Moving Average Returns (Window={window_size})', color='purple')

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Original and Moving Average Returns Over Episodes')
    plt.legend(loc='lower right')  # 将图例放在右上角
    plt.show()

def plot_charge_states(ev_records, ess_records, home_load_records, pv_generation_records, electricity_price_records,
                       ev_at_home_records, selected_episodes, env):
    """
    绘制指定 episode 的 EV 和 ESS 电量变化，以及电网负荷、光伏发电量、电价和 EV 是否在家的状态。

    参数:
        ev_records (list): 每个 episode 的 EV 电量记录。
        ess_records (list): 每个 episode 的 ESS 电量记录。
        home_load_records (list): 每个 episode 的电网负荷记录。
        pv_generation_records (list): 每个 episode 的光伏发电量记录。
        electricity_price_records (list): 每个 episode 的电价记录。
        ev_at_home_records (list): 每个 episode 的 EV 是否在家记录。
        selected_episodes (list): 需要绘制的 episode 编号列表。
        env (HomeEnergyManagementEnv): 环境实例，用于获取 EV 和 ESS 的容量。
    """
    for episode in selected_episodes:
        if episode >= len(ev_records) or episode >= len(ess_records):
            raise ValueError(f"Episode {episode} is out of range.")

    # 绘制指定 episode 的 EV 电量变化
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        ev_charge = ev_records[episode]
        ev_at_home = ev_at_home_records[episode]

        # 创建时间索引
        time_steps = list(range(len(ev_charge)))

        # 绘制 EV 电量
        ev_charge_values = np.array([c if c is not None else np.nan for c in ev_charge])
        plt.plot(time_steps, ev_charge_values, label=f'EV State (Episode {episode + 1})', drawstyle='steps', alpha=0.7)

        # 标记 EV 不在家的时段
        for i in range(len(ev_at_home)):
            if not ev_at_home[i]:
                plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    plt.title('EV Battery State for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('EV State')
    plt.axhline(y=0, color='r', linestyle='--', label='Min State')
    plt.axhline(y=env.ev_capacity, color='g', linestyle='--', label='Max State')
    plt.legend(loc='upper right')  # 将图例放在右上角
    plt.show()

    # 绘制指定 episode 的 ESS 电量变化
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(ess_records[episode], label=f'ESS State (Episode {episode + 1})')
    plt.title('ESS Battery State for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('ESS State')
    plt.axhline(y=0, color='r', linestyle='--', label='Min State')
    plt.axhline(y=env.ess_capacity, color='g', linestyle='--', label='Max State')
    plt.legend(loc='upper right')  # 将图例放在右上角
    plt.show()

    # 绘制指定 episode 的电网负荷变化
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(home_load_records[episode], label=f'Home Load (Episode {episode + 1})')
    plt.title('Home Load for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('Home Load')
    plt.legend(loc='upper right')  # 将图例放在右上角
    plt.show()

    # 绘制指定 episode 的光伏发电量变化
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(pv_generation_records[episode], label=f'PV Generation (Episode {episode + 1})')
    plt.title('PV Generation for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('PV Generation')
    plt.legend(loc='upper right')  # 将图例放在右上角
    plt.show()

    # 绘制指定 episode 的电价变化
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(electricity_price_records[episode], label=f'Electricity Price (Episode {episode + 1})')
    plt.title('Electricity Price for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('Electricity Price')
    plt.legend(loc='upper right')  # 将图例放在右上角
    plt.show()

    # 绘制指定 episode 的 EV 是否在家状态
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(ev_at_home_records[episode], label=f'EV at Home (Episode {episode + 1})', drawstyle='steps-pre')
    plt.title('EV at Home Status for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('EV at Home (1: At Home, 0: Not at Home)')
    plt.legend(loc='upper right')  # 将图例放在右上角
    plt.show()


# def plot_ev_charge_discharge(ev_charge_records, ev_discharge_records, ev_at_home_records, selected_episodes):
#     """
#     绘制指定 episode 的 EV 充放电功率图。
#
#     参数:
#         ev_charge_records (list): 每个 episode 的 EV 充电功率记录。
#         ev_discharge_records (list): 每个 episode 的 EV 放电功率记录。
#         ev_at_home_records (list): 每个 episode 的 EV 是否在家记录。
#         selected_episodes (list): 需要绘制的 episode 编号列表。
#     """
#     plt.figure(figsize=(12, 6))
#     for episode in selected_episodes:
#         charge = ev_charge_records[episode]
#         discharge = ev_discharge_records[episode]
#         at_home = ev_at_home_records[episode]
#
#         # 创建时间索引
#         time_steps = list(range(len(charge)))
#
#         # 绘制充电功率
#         charge_values = np.array([c if c is not None else np.nan for c in charge])
#         plt.plot(time_steps, charge_values, label=f'Episode {episode + 1} Charge', drawstyle='steps', alpha=0.7)
#
#         # 绘制放电功率
#         discharge_values = np.array([d if d is not None else np.nan for d in discharge])
#         plt.plot(time_steps, discharge_values, label=f'Episode {episode + 1} Discharge', drawstyle='steps', alpha=0.7)
#
#     plt.title('EV Charge and Discharge Power for Selected Episodes')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Power (kW)')
#     plt.legend(loc='upper right')  # 将图例放在右上角
#     plt.show()

def plot_episode_returns(ep_returns):
    """绘制 episode 回报曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(ep_returns, label='Episode Returns', color='purple')

    # 计算移动平均值（窗口大小为 10）
    window_size = 10
    if len(ep_returns) >= window_size:
        moving_avg = np.convolve(ep_returns, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size - 1, len(ep_returns)), moving_avg, label=f'Rolling Mean (Window={window_size})', color='cyan', linewidth=2)

    plt.title('Episode Returns Over Training')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.show()