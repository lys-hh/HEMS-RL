import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root directory to Python path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from model import rl_utils


def plot_returns(return_list, window_size=9):
    """
    Plot original returns curve and moving average returns curve.

    Args:
        return_list (list): List of returns for each episode.
        window_size (int): Moving average window size.
    """
    episodes_list = list(range(len(return_list)))

    # Calculate moving average returns
    mv_return = rl_utils.moving_average(return_list, window_size)

    # Plot original returns curve and moving average returns curve
    plt.figure(figsize=(12, 6))

    # Plot original returns curve, light purple
    plt.plot(episodes_list, return_list, label='Original Returns', color='thistle', alpha=0.7)

    # Plot moving average returns curve, slightly darker color
    plt.plot(episodes_list, mv_return, label=f'Moving Average Returns (Window={window_size})', color='purple')

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Original and Moving Average Returns Over Episodes')
    plt.legend(loc='lower right')
    plt.show()

def plot_charge_states(ev_records, ess_records, home_load_records, pv_generation_records, electricity_price_records,
                       ev_at_home_records, selected_episodes, env):
    """
    Plot EV and ESS charge state changes, grid load, PV generation, electricity price, and EV at home status for specified episodes.

    Args:
        ev_records (list): EV charge records for each episode.
        ess_records (list): ESS charge records for each episode.
        home_load_records (list): Grid load records for each episode.
        pv_generation_records (list): PV generation records for each episode.
        electricity_price_records (list): Electricity price records for each episode.
        ev_at_home_records (list): EV at home records for each episode.
        selected_episodes (list): List of episode numbers to plot.
        env (HomeEnergyManagementEnv): Environment instance, used to get EV and ESS capacity.
    """
    for episode in selected_episodes:
        if episode >= len(ev_records) or episode >= len(ess_records):
            raise ValueError(f"Episode {episode} is out of range.")

    # Plot EV charge state changes for specified episodes
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        ev_charge = ev_records[episode]
        ev_at_home = ev_at_home_records[episode]

        # Create time index
        time_steps = list(range(len(ev_charge)))

        # Plot EV charge
        ev_charge_values = np.array([c if c is not None else np.nan for c in ev_charge])
        plt.plot(time_steps, ev_charge_values, label=f'EV State (Episode {episode + 1})', drawstyle='steps', alpha=0.7)

        # Mark periods when EV is not at home
        for i in range(len(ev_at_home)):
            if not ev_at_home[i]:
                plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    plt.title('EV Battery State for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('EV State')
    plt.axhline(y=0, color='r', linestyle='--', label='Min State')
    plt.axhline(y=env.ev_capacity, color='g', linestyle='--', label='Max State')
    plt.legend(loc='upper right')
    plt.show()

    # Plot ESS charge state changes for specified episodes
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(ess_records[episode], label=f'ESS State (Episode {episode + 1})')
    plt.title('ESS Battery State for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('ESS State')
    plt.axhline(y=0, color='r', linestyle='--', label='Min State')
    plt.axhline(y=env.ess_capacity, color='g', linestyle='--', label='Max State')
    plt.legend(loc='upper right')
    plt.show()

    # Plot grid load changes for specified episodes
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(home_load_records[episode], label=f'Home Load (Episode {episode + 1})')
    plt.title('Home Load for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('Home Load')
    plt.legend(loc='upper right')
    plt.show()

    # Plot PV generation changes for specified episodes
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(pv_generation_records[episode], label=f'PV Generation (Episode {episode + 1})')
    plt.title('PV Generation for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('PV Generation')
    plt.legend(loc='upper right')
    plt.show()

    # Plot electricity price changes for specified episodes
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(electricity_price_records[episode], label=f'Electricity Price (Episode {episode + 1})')
    plt.title('Electricity Price for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('Electricity Price')
    plt.legend(loc='upper right')
    plt.show()

    # Plot EV at home status for specified episodes
    plt.figure(figsize=(12, 6))
    for episode in selected_episodes:
        plt.plot(ev_at_home_records[episode], label=f'EV at Home (Episode {episode + 1})', drawstyle='steps-pre')
    plt.title('EV at Home Status for Selected Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('EV at Home (1: At Home, 0: Not at Home)')
    plt.legend(loc='upper right')
    plt.show()


# def plot_ev_charge_discharge(ev_charge_records, ev_discharge_records, ev_at_home_records, selected_episodes):
#     """
#     Plot EV charge and discharge power for specified episodes.
#
#     Args:
#         ev_charge_records (list): EV charging power records for each episode.
#         ev_discharge_records (list): EV discharging power records for each episode.
#         ev_at_home_records (list): EV at home records for each episode.
#         selected_episodes (list): List of episode numbers to plot.
#     """
#     plt.figure(figsize=(12, 6))
#     for episode in selected_episodes:
#         charge = ev_charge_records[episode]
#         discharge = ev_discharge_records[episode]
#         at_home = ev_at_home_records[episode]
#
#         # Create time index
#         time_steps = list(range(len(charge)))
#
#         # Plot charging power
#         charge_values = np.array([c if c is not None else np.nan for c in charge])
#         plt.plot(time_steps, charge_values, label=f'Episode {episode + 1} Charge', drawstyle='steps', alpha=0.7)
#
#         # Plot discharging power
#         discharge_values = np.array([d if d is not None else np.nan for d in discharge])
#         plt.plot(time_steps, discharge_values, label=f'Episode {episode + 1} Discharge', drawstyle='steps', alpha=0.7)
#
#     plt.title('EV Charge and Discharge Power for Selected Episodes')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Power (kW)')
#     plt.legend(loc='upper right')
#     plt.show()

def plot_episode_returns(ep_returns):
    """Plot episode returns curve"""
    plt.figure(figsize=(10, 5))
    plt.plot(ep_returns, label='Episode Returns', color='purple')

    # Calculate moving average (window size 10)
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