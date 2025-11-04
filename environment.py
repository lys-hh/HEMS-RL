"""
This file is used to simulate the interaction between electric vehicles and the environment, integrating washing machines and air conditioning equipment
"""
from datetime import datetime, timedelta
from scipy.stats import uniform
import matplotlib.dates as mdates
import torch
import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MaxNLocator
from interface import DataInterface
import csv  # Add CSV module
import os   # Add OS module
import pandas as pd

class HomeEnergyManagementEnv:
    def __init__(self, ev_capacity=24, ess_capacity=24, charge_efficiency=0.95, discharge_efficiency=0.95):
        self.ev_capacity = ev_capacity  # EV battery capacity
        self.ess_capacity = ess_capacity  # ESS battery capacity
        self.charge_efficiency = charge_efficiency  # Charge efficiency
        self.discharge_efficiency = discharge_efficiency  # Discharge efficiency
        self.ev_min_charge = 12  # Set minimum charge requirement when EV leaves home

        # Initialize penalty coefficients for dynamic adjustment
        self.energy_weight = 0.1  # Grid cost weight
        self.user_satisfaction_weight0 = 0.5  # User dissatisfaction weight
        self.user_satisfaction_weight1 = 0.5  #
        self.user_satisfaction_weight2 = 0.2
        self.violation_weight = 0.05
        self.temp_weight = 0.1
        self.ess_weight = 0.1  # Originally 1
        self.ev_weight = 0.1  # Originally 1

        self.total_cost = 0

        self.episode_costs = []  # Store total cost for each episode
        self.current_step_cost = 0  # Store cost for current time step

        self.state_space = {
            'home_load': (0, 10),  # Household electricity consumption range
            'pv_generation': (0, 5),  # PV generation range
            'ess_state': (0, ess_capacity),  # ESS battery capacity
            'ev_battery_state': (0, ev_capacity),  # EV battery capacity
            'time_index': (0, 48),  # Time of day, with 0.5 as step size
            'electricity_price': (0, 2),  # Electricity price range, example values
            'temperature': (0, 40),
            'wash_machine_state': (0, 1),  # Washing machine state, 0 means off, 1 means running
            # 'start_time': (-1, 48),
            # 'price_trend': (-1, 1),  # Electricity price trend (past 3-hour average vs current)
            'Air_conditioner_power': (0, 5),  # AC power range
            'Air_conditioner_power2': (0, 5),
            'ewh_temp': (40, 70),  # Water heater water temperature
            'ewh_power': (0, 2)   # Water heater power
        }
        self.action_space = {
            'ev_power': (-6.6, -3.3, 0, 3.3, 6.6),  # EV charge/discharge power range

            'battery_power': (-4.4, -2.2, 0, 2.2, 4.4),  # ESS charge/discharge power range

            'wash_machine_schedule': (0, 1, 2, 3, 4, 5, 6),  # Washing machine scheduling actions, 0 means not running, 1 means running
            'Air_conditioner_set_temp': (16, 18, 20, 22, 24, 26, 28, 30),  # AC set temperature
            'Air_conditioner_set_temp2': (16, 18, 20, 22, 24, 26, 28, 30),

            'ewh_set_temp': (40, 45, 50, 55, 60, 65, 70)  # Discrete temperature setting actions
        }
        self.current_time = '2011-07-03'
        self.current_time_index = 0
        self.data_interface = DataInterface('data/daily_pivot_cons_2011-2012.csv',
                                            'data/daily_pivot_prod_2011-2012.csv')
        self.current_ev_power = 0
        self.current_battery_power = 0

        self.ev_battery_record = []  # Record EV charge level
        self.ess_state_record = []  # Record ESS charge level
        self.home_load_record = []  # Record grid load
        self.pv_generation_record = []  # Record PV generation
        self.electricity_price_record = []  # Record electricity price
        self.ev_at_home_record = []  # Record whether EV is at home
        self.wash_machine_record = []  # Record washing machine state
        self.air_conditioner_power_record = []  # Record AC power
        self.ess_actions = []  # Record ESS charge/discharge power
        self.wash_machine_actions = []  # Record washing machine actions
        self.air_conditioner_actions = []  # Record AC actions

        # New ESS charge/discharge source records
        self.ess_charge_pv = []  # PV charging amount
        self.ess_charge_grid = []  # Grid charging amount
        self.ess_discharge_ev = []  # Discharge amount for EV
        self.ess_discharge_house = []  # Discharge amount for household load

        # New record variables
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
            'user_dissatisfaction': []  # User dissatisfaction record
        }

        # New reward record containers
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

        # ================== New Water Heater Parameters ==================
        self.ewh_params = {
            # 'ξ_ewh': 0.993,  # Water temperature decay coefficient
            # 'R_prime': 4.18,  # Heat transfer parameter (kJ/°C)
            'h': 10,  # Convective heat transfer coefficient
            'temp_cold': 18,  # Cold water temperature (°C)
            'surface_area': 2,  # Tank surface area (m^2)
            'volume': 100,  # Tank capacity (L)
            'mass': 100,  # Water mass (kg)
            'temp_min': 30.0,  # Minimum acceptable water temperature (°C)
            'temp_max': 100.0,  # Maximum safe water temperature (°C)
            'temp_init': 40,  # Initial water temperature (°C)
            'user_flow_mean': 1.5,  # Average water flow rate (L/h)
            'user_flow_std': 0.3  # Water flow rate standard deviation
        }

        # New record variables
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []

        # Washing machine related parameters
        self.wash_machine_power = 1.5  # Washing machine fixed power (kW)
        self.wash_machine_duration = 1  # Washing machine runtime (hours)
        self.current_period_start = None  # Current period start time
        self.has_run_in_current_period = False  # Whether has run in current period
        self.time_deviation = 0  # Time offset (minutes)
        self.wash_machine_state = 0  # Initial state is off
        self.last_action_time = None  # Record start time
        self.remaining_runtime = 0  # Remaining runtime (30-minute steps)
        self.wash_machine_preferred_time = (6, 20)  # User preferred washing machine runtime range

        # AC related parameters
        self.indoor_temp = 25  # Initial indoor temperature
        self.indoor_temp2 = 25  # Second AC
        self.outdoor_temp = 25  # Initial outdoor temperature
        self.temp_change_rate = 0.5  # Indoor temperature change rate
        self.user_temp_preference = 22  # User preferred temperature
        self.user_temp_preference2 = 18  # Second AC

    def get_state_vector(self, state_dict):
        """Convert state dictionary to ordered list"""
        ordered_keys = sorted(state_dict.keys())  # Sort alphabetically
        return [state_dict[k] for k in ordered_keys]

    def get_action_mask(self, state):
        """Return dynamic action mask based on current charge level"""
        masks = {
            'battery_power': [True] * len(self.action_space['battery_power']),
            'ev_power': [True] * len(self.action_space['ev_power'])
        }

        delta_t = 0.5

        # 1. ESS battery action mask - dynamic calculation
        # Consider discharge efficiency (assume 95%)
        max_discharge = (state['ess_state'] / delta_t) * 0.95
        max_charge = ((self.ess_capacity - state['ess_state']) / delta_t) / 0.95
        
        for idx, action_value in enumerate(self.action_space['battery_power']):
            # Discharge action: can only select actions less than or equal to current dischargeable amount
            if action_value < 0 and abs(action_value) > max_discharge:
                masks['battery_power'][idx] = False

            # Charge action: can only select charge power less than or equal to current chargeable space
            if action_value > 0 and action_value > max_charge:
                masks['battery_power'][idx] = False

        # 2. EV battery action mask - also dynamic calculation
        max_ev_discharge = (state['ev_battery_state'] / delta_t) * 0.95
        max_ev_charge = ((self.ev_capacity - state['ev_battery_state']) / delta_t) / 0.95
        
        for idx, action_value in enumerate(self.action_space['ev_power']):
            # Discharge limit
            if action_value < 0 and abs(action_value) > max_ev_discharge:
                masks['ev_power'][idx] = False

            # Charge limit
            if action_value > 0 and action_value > max_ev_charge:
                masks['ev_power'][idx] = False

        # 3. EV not at home mask
        if not self.is_ev_at_home():
            for idx, action_value in enumerate(self.action_space['ev_power']):
                if action_value != 0:  # Can only select 0 power action
                    masks['ev_power'][idx] = False

        return masks

    def reset(self):
        self.total_cost = 0
        # Initialize state using actual input data
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
        self.ev_battery_record = []   # Reset records
        self.ess_state_record = []
        self.home_load_record = []
        self.pv_generation_record = []
        self.electricity_price_record = []
        self.ev_at_home_record = []
        self.wash_machine_record = []
        self.air_conditioner_power_record = []
        self.air_conditioner_power_record2 = []
        self.ess_actions = []  # Reset ESS action records

        # Reset records
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

        # Reset reward record containers
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

        # Reset washing machine related variables
        self.current_period_start = None  # Current period start time
        self.has_run_in_current_period = False  # Whether has run in current period
        self.time_deviation = 0  # Time offset (minutes)
        self.wash_machine_state = 0  # Initial state is off
        self.last_action_time = None  # Record start time
        self.remaining_runtime = 0  # Remaining runtime (30-minute steps)

        # Reset AC related variables
        self.indoor_temp = 25
        self.indoor_temp2 = 20
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)

        # Reset records
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []

        return self.state

    def step(self, state, action):

        # === Action physical clipping to ensure environment physical rationality ===
        # --- ESS battery power clipping ---
        ess_soc = state['ess_state']
        if action['battery_power'] < 0:  # Discharge
            max_discharge = min(abs(action['battery_power']), ess_soc / 0.5 * self.discharge_efficiency)
            action['battery_power'] = -max_discharge
        elif action['battery_power'] > 0:  # Charge
            max_charge = min(action['battery_power'], (self.ess_capacity - ess_soc) / 0.5 / self.charge_efficiency)
            action['battery_power'] = max_charge

        # --- EV power clipping ---
        ev_soc = state['ev_battery_state']
        if action['ev_power'] < 0:  # Discharge
            max_discharge = min(abs(action['ev_power']), ev_soc / 0.5 * self.discharge_efficiency)
            action['ev_power'] = -max_discharge
        elif action['ev_power'] > 0:  # Charge
            max_charge = min(action['ev_power'], (self.ev_capacity - ev_soc) / 0.5 / self.charge_efficiency)
            action['ev_power'] = max_charge


        self.current_ev_power=action['ev_power']  # Store current action
        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + \
                     timedelta(minutes=30 * self.current_time_index)

        # Update outdoor temperature
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)

        # Update AC state
        new_air_conditioner_power, self.indoor_temp = self.update_air_conditioner(action['Air_conditioner_set_temp'], self.indoor_temp)
        self.state['Air_conditioner_power'] = new_air_conditioner_power

        new_air_conditioner_power2, self.indoor_temp2 = self.update_air_conditioner(action['Air_conditioner_set_temp2'], self.indoor_temp2)
        self.state['Air_conditioner_power2'] = new_air_conditioner_power2

        # Update washing machine state
        new_wash_machine_state = self.update_wash_machine2(action['wash_machine_schedule'])
        self.state['wash_machine_state'] = new_wash_machine_state

        # wash_feature = self._get_wash_machine_features(self.current_time_index)

        # Update water heater power
        power = self._fuzzy_heating_control(
            action['ewh_set_temp'],
            state['ewh_temp']
        )

        # Update water heater state
        new_ewh_temp, user_flow = self.update_water_heater(
            power,
            state['ewh_temp']
        )

        # Update EV and ESS battery state
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

        # Record ESS action
        self.ess_actions.append(action['battery_power'])

        # Record temperature data
        self.records['indoor_temp'].append(self.indoor_temp)
        self.records['indoor_temp2'].append(self.indoor_temp2)
        self.records['outdoor_temp'].append(self.outdoor_temp)

        # Update records
        self.records['timestamps'].append(current_dt)
        self.records['ev_soc'].append(self.state['ev_battery_state'])
        self.records['ess_soc'].append(self.state['ess_state'])

        self.records['total_load'].append(self.total_load_compute())

        # Record water heater data
        self.ewh_temp_record.append(new_ewh_temp)
        self.ewh_power_record.append(state['ewh_power'])
        self.user_flow_record.append(user_flow)

        self.current_time_index += 1  # Add half hour

        if self.current_time_index >= 48:  # If current time index reaches 48, add one day
            self.current_time_index = 0
            # Convert current date string to datetime object
            current_date = datetime.strptime(self.current_time, '%Y-%m-%d')
            # Add one day
            current_date += timedelta(days=1)
            # Convert datetime object back to string format
            self.current_time = current_date.strftime('%Y-%m-%d')

            # Force reset washing machine state (avoid cross-day residue)
            self.wash_machine_used_today = False

        return self.state, reward, done

    def update_ev_battery(self, ev_charge_discharge):
        # If EV just arrived home, update charge state
        if (not self.data_interface.is_ev_at_home(self.current_time, self.current_time_index - 1)
                and self.is_ev_at_home()):
            # Use charge state before last trip to simulate charge state after arriving home
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

        # Force SOC boundary
        # min_soc = self.ev_min_charge * 0.8  # Maintain safety margin
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

        if ess_charge_discharge > 0:  # Charge action = pv + grid
            new_ess = self.state['ess_state'] + ess_charge_discharge * self.charge_efficiency * 0.5
        else:   # Discharge action = home + grid
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
        # Calculate time offset (hours)
        if scheduled_time < pref_start:
            # Before preferred time
            self.time_deviation = (pref_start - scheduled_time).total_seconds() / 3600
        elif scheduled_time > pref_end:
            # After preferred time
            self.time_deviation = (scheduled_time - pref_end).total_seconds() / 3600
        else:
            # Within preferred time period
            self.time_deviation = 0

    def update_wash_machine2(self, schedule_action):
        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + \
                     timedelta(minutes=30 * self.current_time_index)
        # Convert to Python built-in integer
        schedule_action = int(schedule_action)  # Add this line

        # Handle running state
        if self.remaining_runtime > 0:
            self.wash_machine_state = 1
            self.remaining_runtime -= 1
            return self.wash_machine_state

        # Period detection (reset at 6 AM daily)
        current_period_start = self._get_period_start(current_dt)
        current_period_end = self._get_period_end(current_dt)  # Get current period end time
        if current_period_start != self.current_period_start:
            self.current_period_start = current_period_start
            self.has_run_in_current_period = False
            self.wash_machine_state = 0

        # Force start logic: if less than 1 hour until period end and hasn't run yet
        time_to_end = (current_period_end - current_dt).total_seconds() / 3600
        if time_to_end <= 1 and not self.has_run_in_current_period:
            schedule_action = 1  # Force immediate start

        if schedule_action > 1 and not self.has_run_in_current_period:
            scheduled_time = current_dt + timedelta(hours=schedule_action - 1)
            self.wash_machine_state = 0
            self.update_time_deviation(scheduled_time)
        elif schedule_action == 1 and not self.has_run_in_current_period:
            scheduled_time = current_dt  # Start at current time
            self.wash_machine_state = 1
            self.has_run_in_current_period = True
            self.remaining_runtime = 1  # 1 hour = 2 time steps
            self.update_time_deviation(scheduled_time)
        else:
            self.wash_machine_state = 0
            self.time_deviation = 0

        return self.wash_machine_state

    def update_air_conditioner(self, set_temp, indoor_temp):
        """Update AC power"""
        # Calculate difference between set temperature and current indoor temperature
        temp_diff = set_temp - indoor_temp
        # Fuzzy control rules: determine power based on temperature difference
        if temp_diff > 0:  # Heating mode
            # Define fuzzy control rules for heating mode
            rules = [
                {'range': (0, 0.5), 'power': 0},
                {'range': (0.5, 1), 'power': 0.5},  # 0.5 enables precise control
                {'range': (1, 2), 'power': 1.0},
                {'range': (2, 3), 'power': 1.5},
                {'range': (3, 4), 'power': 2.0},
                {'range': (4, np.inf), 'power': 3.0}
            ]
        else:
            # Define fuzzy control rules for cooling mode
            rules = [
                {'range': (-0.5, 0), 'power': 0},
                {'range': (-1, -0.5), 'power': 0.5},  # 0.5 enables precise control
                {'range': (-2, -1), 'power': 1.0},
                {'range': (-3, -2), 'power': 1.5},
                {'range': (-4, -3), 'power': 2.0},
                {'range': (-np.inf, -4), 'power': 3.0}
            ]

        # Find corresponding power based on temperature difference
        power = 0  # Default power
        for rule in rules:
            if rule['range'][0] <= temp_diff < rule['range'][1]:
                power = rule['power']
                break

        # Dynamically calculate temperature change rate
        max_power = 3.0  # AC maximum power
        efficiency = power / max_power if max_power > 0 else 0  # Calculate efficiency at current power
        temp_change = self.temp_change_rate * efficiency * temp_diff  # Calculate temperature change based on power and temperature difference

        # Simulate random disturbance from user behavior
        user_behavior = np.random.normal(0, 0.2)
        temp_change += user_behavior  # Add random disturbance to temperature change

        # When AC is off (power == 0), indoor temperature gradually approaches outdoor temperature
        if power == 0:
            # Rate at which indoor temperature approaches outdoor temperature can be adjusted
            temp_change += (self.outdoor_temp - indoor_temp) * 0.4 * self.temp_change_rate

        # Update indoor temperature
        indoor_temp += temp_change

        # Ensure indoor temperature is within reasonable range
        indoor_temp = np.clip(indoor_temp, 10, 40)

        # Ensure AC power is within reasonable range
        power = np.clip(power, 0, 3.0)

        return power, indoor_temp

    def update_water_heater(self, power, current_temp):
        """Update water heater state (physical model)"""
        params = self.ewh_params
        delta_t = 0.5  # Half-hour time interval

        # # Generate random water usage (L/h)
        # flow_rate = np.random.uniform(1, 2) if np.random.rand() < 0.3 else 0  # Assume 30% probability of water usage

        # Define peak water usage time periods
        peak_morning = 6 <= self.current_time_index / 2 <= 9  # Morning peak 6-9 AM
        peak_evening = 18 <= self.current_time_index / 2 <= 22  # Evening peak 6-10 PM

        # Set different water usage probabilities and ranges based on time period
        if peak_morning or peak_evening:
            # Peak period water usage probability and amount
            if peak_morning:
                # Morning peak has larger water usage and higher probability
                flow_rate_prob = 0.6  # 60% probability of water usage
                flow_rate_min, flow_rate_max = 3, 5  # 3-5 L/h
            else:
                # Evening peak has slightly smaller water usage and slightly lower probability
                flow_rate_prob = 0.5  # 50% probability of water usage
                flow_rate_min, flow_rate_max = 2, 4  # 2-4 L/h

            # Peak periods have higher water usage probability
            if np.random.rand() < flow_rate_prob:
                flow_rate = np.random.uniform(flow_rate_min, flow_rate_max)
            else:
                flow_rate = 0
        else:
            # Off-peak period water usage probability and amount
            flow_rate_prob = 0.2  # 20% probability of water usage
            if np.random.rand() < flow_rate_prob:
                # Off-peak periods have smaller water usage
                flow_rate = np.random.uniform(0.5, 1.5)  # 0.5-1.5 L/h
            else:
                flow_rate = 0

        # Special handling: occasionally have slightly larger water usage during off-peak periods
        if not (peak_morning or peak_evening) and np.random.rand() < 0.1:
            flow_rate = np.random.uniform(1.5, 2)  # 1.5-2 L/h
        # === Water usage logic end ===

        # Environment parameters
        env_temp = self.state['temperature']
        temp_cold = max(18, env_temp - 2)  # Minimum 18°C cold water
        # temp_cold = 18 # Minimum 18°C cold water

        new_temp = current_temp  # Initialize new temperature

        # Case 1: Temperature change when injecting cold water
        if flow_rate > 0:
            # Calculate volume change corresponding to water usage (assuming total tank volume unchanged)
            used_water_volume = flow_rate * delta_t   # Assume flow rate unit is L/h, convert to L
            # Inject same volume of cold water
            injected_cold_volume = used_water_volume

            # Calculate new temperature using mixing effect
            remaining_hot_volume = params['volume'] - used_water_volume
            new_temp = (current_temp * remaining_hot_volume + temp_cold * injected_cold_volume) / params['volume']

        # Case 2: Temperature rise when heating
        if power > 0:
            hour = self.state['time_index'] // 2
            efficiency = 0.9 if (6 <= hour <= 9 or 18 <= hour <= 22) else 0.8
            # Convert power from kW to W
            power_in_watts = power * 1000 * efficiency

            # Calculate temperature rise from heating
            heat_input = power_in_watts * 3600 * delta_t  # Energy input (J)
            temp_gain = heat_input / (params['mass'] * 4180)  # Temperature rise (°C)
            new_temp += temp_gain

        # Case 3: Natural cooling when not heating
        else:
            # Calculate temperature drop from natural cooling    Cooling coefficient k
            cooling_coefficient = params['h'] * params['surface_area'] / (params['mass'] * 4180)
            new_temp = env_temp + (new_temp - env_temp) * np.exp(-cooling_coefficient * delta_t * 3600)

        # Update parameters
        params['temp_min'] = env_temp
        new_temp = np.clip(new_temp, params['temp_min'], params['temp_max'])

        return new_temp, flow_rate

    def _fuzzy_heating_control(self, set_tem, current_tem):
        # Fuzzy control logic: determine heating power based on difference between target temperature and current temperature
        # temp_diff: target temperature - current temperature
        temp_diff = set_tem - current_tem
        hour = self.state['time_index'] // 2  # Get current hour

        # Dynamically adjust control rules (more aggressive during peak periods)
        if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak period
            rules = [
                {'range': (-np.inf, -3), 'power': 0.0},
                {'range': (-3, 1), 'power': 0.0},
                {'range': (1, 4), 'power': 0.4 + 0.1 * max(0, temp_diff-1)},  # Dynamic proportion
                {'range': (4, 6), 'power': 1.0},
                {'range': (6, np.inf), 'power': 1.2}  # Allow short-term over-power
            ]
        else:  # Off-peak period
            rules = [
                {'range': (-np.inf, -5), 'power': 0.0},
                {'range': (-5, 2), 'power': 0.0},
                {'range': (2, 5), 'power': 0},  # Gentle heating
                {'range': (5, 8), 'power': 0.2},  # Originally 0
                {'range': (8, np.inf), 'power': 0.5}  # Originally 0.5
            ]

        # rules = [
        #     {'range': (-np.inf, -3), 'power': 0.0},
        #     {'range': (-3, 1), 'power': 0.0},
        #     {'range': (1, 4), 'power': 0.4 + 0.1 * max(0, temp_diff - 1)},  # Dynamic proportion
        #     {'range': (4, 6), 'power': 1.0},
        #     {'range': (6, np.inf), 'power': 1.2}  # Allow short-term over-power
        # ]

        for rule in rules:
            if rule['range'][0] <= temp_diff < rule['range'][1]:
                return min(max(rule['power'], 0.0), 1.2)  # Power clamping
        return 0.0  # Default no heating

    def calculate_reward(self, state, action):
        # Define reward

        # 1. Constraint penalty
        violation = 0
        # violation += max(0.1*self.ess_capacity-self.state['ess_state'],
        #                 self.state['ess_state']-0.9*self.ess_capacity, 0) ** 2 * 20
        #
        # violation += max(0.1 * self.ev_capacity - self.state['ev_battery_state'],
        #                 self.state['ev_battery_state'] - 0.9 * self.ev_capacity, 0) ** 2 * 20

        # # EV constraint (minimum charge)
        if self.data_interface.is_ev_departing_soon(self.current_time, self.current_time_index):
            ev_violation = max(0, self.ev_min_charge - self.state['ev_battery_state'])
            # violation += np.clip(ev_violation ** 2, 0, 500)  # Add numerical clipping
            violation += ev_violation ** 2

        # 2. Grid cost calculation (half hour)
        ev_charge = max(action['ev_power'], 0)
        ev_discharge = max(-action['ev_power'], 0)
        battery_charge = max(action['battery_power'], 0)
        battery_discharge = max(-action['battery_power'], 0)

        # Calculate total consumption and total generation (kW)
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
        # Calculate net demand (kW)
        net_demand = total_consumption - total_generation  # This demand is the interaction with the grid

        # Convert to energy (kWh) and split purchase/sale
        purchase_kwh = max(net_demand, 0) * 0.5  # This calculates ideal cost, not actual cost
        sell_kwh = max(-net_demand, 0) * 0.5

        # Calculate energy cost (considering electricity sale price discount)
        energy_cost = (
                purchase_kwh * state['electricity_price']
                - sell_kwh * state['electricity_price'] * 0.9  # Assume sale price is 0.9 of purchase price
        )

        self.total_cost += energy_cost
        self.records['energy_cost'].append(self.total_cost)

        self.current_step_cost = energy_cost

        # 3. ESS and EV charge/discharge guidance reward

        # ESS reward: encourage low price charging and high price discharging
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
        # ess_reward = np.tanh(ess_reward / 10) * 3  # Use tanh to compress amplitude

        # EV reward: encourage low price charging and high price discharging
        ev_reward = 0
        # Get current SOC ratio
        soc_ev = state['ev_battery_state'] / self.ev_capacity
        if price < 0.5:
            # ev_reward += -action['ev_power'] * (price - 0.5) * (1-soc_ev) * 10
            ev_reward += -action['ev_power'] * (price - 0.5) * 20
        elif price == 0.5:
            ev_reward += -action['ev_power'] * (soc_ev-0.8) * 20
        else:
            # ev_reward += -action['ev_power'] * (price - 0.5) * soc_ev * 10
            ev_reward += -action['ev_power'] * (price - 0.5) * 20
        # ev_reward = np.tanh(ev_reward / 10) * 3  # Use tanh to compress amplitude

        # 4. User dissatisfaction penalty
        user_dissatisfaction_penalty = 0
        user_dissatisfaction_penalty = (self.user_satisfaction_weight0 * self.calculate_user_dissatisfaction0() +
                                        self.user_satisfaction_weight1 * self.calculate_user_dissatisfaction1() +
                                    self.user_satisfaction_weight2 * self.calculate_user_dissatisfaction2(state, action))
        # user_dissatisfaction_penalty = np.clip(user_dissatisfaction_penalty, -20, 20)

        temp_reward = self.calculate_temp_reward(state['ewh_temp'])
        # temp_reward = np.clip(temp_reward, -5, 5)  # Limit temperature reward range

        # 5. Combined reward
        reward = (
                - self.energy_weight * energy_cost
                - self.violation_weight * violation    # EV and ESS upper and lower bound constraints
                + self.ess_weight * ess_reward
                + self.ev_weight * ev_reward
                - user_dissatisfaction_penalty   # User dissatisfaction penalty
                + self.temp_weight * temp_reward
        )



        # # New exploration reward (prevent premature convergence)
        # if np.random.rand() < 0.1:  # 10% probability add noise
        #     reward += np.random.normal(0, 5)

        # Add records after calculating reward
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

        #     # Add numerical stability check
        # if not np.isfinite(reward):
        #     reward = -10  # Fallback handling for abnormal reward values

        return reward

    def calculate_user_dissatisfaction0(self):
        """User dissatisfaction calculation (AC)"""
        dissatisfaction = 0

        # Check if within comfort range
        comfort_range = 2  # Comfort range (e.g., ±2°C)
        temp_diff = self.indoor_temp2 - self.user_temp_preference2

        if abs(temp_diff) > comfort_range:
            # Temperature dissatisfaction (non-linear calculation)
            dissatisfaction += min((abs(temp_diff) - comfort_range) ** 1.5 * 0.5, 500)  # Exponential growth but set upper limit

        return dissatisfaction

    def calculate_user_dissatisfaction1(self):
        """User dissatisfaction calculation (AC)"""
        dissatisfaction = 0

        # Check if within comfort range
        comfort_range = 2  # Comfort range (e.g., ±2°C)
        temp_diff = self.indoor_temp - self.user_temp_preference

        if abs(temp_diff) > comfort_range:
            # Temperature dissatisfaction (non-linear calculation)
            dissatisfaction += min((abs(temp_diff) - comfort_range) ** 1.5 * 0.5, 500)  # Exponential growth but set upper limit

        # # AC frequent on/off penalty
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
        """Improved washing machine related penalty calculation"""
        if action['wash_machine_schedule'] >= 1:
            price = self.data_interface.get_electricity_price(self.current_time,
                                                          self.current_time_index+2*(action['wash_machine_schedule']-1))
            price_penalty = (price-0.5) * 10  #
        else:
            price = state['electricity_price']
            price_penalty = 0

        # Time offset penalty (non-linear)
        time_penalty = 5 * self.time_deviation   #

        return time_penalty + price_penalty

    def calculate_temp_reward(self, current_temp):
        # Strengthen time-of-use control parameters
        hour = self.state['time_index'] // 2
        if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak water usage hours
            target, low, high = 55, 53, 57
            strict_factor = 5.0  # Strengthen peak period penalty
        else:
            target, low, high = 45, 43, 47
            strict_factor = 1.0

        # Dynamic deviation calculation
        temp_diff = current_temp - target
        abs_diff = abs(temp_diff)

        # # Step-wise reward mechanism
        if low <= current_temp <= high:
            # Core region reward (S-shaped curve)
            reward = 5.0 / (1 + np.exp(abs_diff * 0.8))  # Highest at center, smooth decline at edges
        else:
            # Danger zone penalty (exponential growth)
            deviation = max(low - current_temp, current_temp - high)
            reward = -strict_factor * deviation   # Strictly penalize boundary violations
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
        # Generate daily mileage (log-normal distribution)
        avg_daily_km = 50  # Average daily mileage (unit: km)
        km_std = 0.2  # Standard deviation of daily mileage
        daily_km = np.random.lognormal(mean=np.log(avg_daily_km), sigma=km_std)

        # Assume energy consumption per km is 0.2 kWh
        energy_consumption_per_km = 0.2  # kWh/km

        # Calculate charge state when arriving home
        soc_home = current_ev_soc - daily_km * energy_consumption_per_km

        # Ensure charge state is within reasonable range (0-100%)
        soc_home = np.clip(soc_home, 0, 100)

        return soc_home

    def _find_contiguous_segments(self, bool_list):
        """Detect contiguous time periods where value is True"""
        segments = []
        start_idx = None
        for i, value in enumerate(bool_list):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:  # Handle last segment
            segments.append((start_idx, len(bool_list) - 1))
        return segments

    def reward_shape(self, progress):
        """
        Dynamically adjust reward function coefficients based on training progress
        :param progress: Training progress, range from 0 to 1
        :return: Weights for each reward component
        """

        # self.energy_weight = 5.0 * (1 - 0.8*progress)  # Linear decay
        # self.temp_weight = 1 / (1 + np.exp(-10*(progress-0.3)))  # S-shaped growth
        self.violation_weight = 3-2*progress  # April 29 attempt this method, try saving model snapshots if ineffective
        # self.user_satisfaction_weight1 = 0.3 + 0.7*progress
        # self.user_satisfaction_weight2 = 0.1 + 0.4*progress
        # self.ess_weight = 3.0 + 2*progress
        # self.ev_weight = 1

    def save_cost_data(self):
        """Save cost data to CSV file"""
        # Create results directory
        results_dir = "model/cost_results"
        os.makedirs(results_dir, exist_ok=True)

        # Create unique filename (include timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(results_dir, f"cost_data_{timestamp}.csv")

        # Write data to CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(["Timestamp", "Energy Cost"])

            # Write cost data for each time step
            for i, (ts, cost) in enumerate(zip(self.records['timestamps'], self.records['energy_cost'])):
                writer.writerow([ts.strftime('%Y-%m-%d %H:%M:%S'), cost])

        print(f"Cost data saved to: {csv_filename}")

    def save_episode_costs(self):
        """Save total cost for each episode to CSV file"""
        # Create results directory
        results_dir = "model/episode_cost_results"
        os.makedirs(results_dir, exist_ok=True)

        # Create unique filename (include timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"episode_costs_{timestamp}.csv")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(["Episode", "Total Cost"])

            # Write cost data for each episode
            for episode, cost in enumerate(self.episode_costs):
                writer.writerow([episode + 1, cost])

        print(f"Episode cost data saved to: {filename}")

    def visualize(self):
        # ===== First canvas: EV SOC change plot and electricity price plot =====
        plt.figure(figsize=(20, 5))

        ax1 = plt.subplot(1, 1, 1)
        ev_soc = np.array(self.ev_battery_record, dtype=np.float32)
        ev_soc[~np.array(self.ev_at_home_record)] = np.nan  # Set away periods to NaN

        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Plot EV SOC curve
        ax1.plot(mpl_dates, ev_soc, color='blue', label='EV SOC')
        ax1.set_ylabel('EV SOC (kWh)')
        ax1.set_title('EV State of Charge and Electricity Price')
        ax1.legend(loc='upper left')

        # Plot electricity price curve (right axis)
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax1_price.set_ylabel('Price ($/kWh)')
        ax1_price.legend(loc='upper right')

        # Segmentally plot filled areas for at-home time periods
        home_segments = self._find_contiguous_segments(self.ev_at_home_record)
        for start, end in home_segments:
            segment_dates = mdates.date2num(self.records['timestamps'][start:end + 1])
            ax1.fill_between(segment_dates, 0, 1, color='green', alpha=0.3, transform=ax1.get_xaxis_transform())

        # Set time axis format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p3.png')
        plt.close()

        # ===== Second canvas: ESS charge/discharge power, PV generation and electricity price change plot =====
        plt.figure(figsize=(20, 10))

        # Ensure horizontal coordinate ranges are consistent for both subplots
        min_date = mpl_dates[0]
        max_date = mpl_dates[-1]

        # ESS charge/discharge power plot
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

        # Plot ESS charge/discharge bar chart, adjust bar width
        ax2_1.bar(mpl_dates, ess_charge_power, width=0.015, color='#05B9E2',
                  label='ESS Charging Power')  # Charging color darker, bar width adjusted
        ax2_1.bar(mpl_dates, ess_discharge_power, width=0.015, color='#FFBE7A', label='ESS Discharging Power')

        # Draw horizontal line at 0 scale
        ax2_1.axhline(0, color='black', linewidth=0.8, linestyle='--')

        # Set left axis range
        ax2_1.set_ylim(-5, 5)

        ax2_1.set_ylabel('Power (kW)')
        ax2_1.set_title('ESS Charging/Discharging Power, PV Generation and Electricity Price')
        ax2_1.legend(loc='upper left')

        # Set time axis format and range
        ax2_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax2_1.set_xlim(min_date, max_date)

        # PV generation power plot (right axis)
        ax2_1_pv = ax2_1.twinx()
        # Plot curve with points
        ax2_1_pv.plot(mpl_dates, self.pv_generation_record, color='green', marker='o', linestyle='-',
                      label='PV Generation')
        ax2_1_pv.set_ylabel('PV Generation (kW)')
        ax2_1_pv.legend(loc='upper right')

        # Ensure PV generation curve is above 0 scale
        ax2_1_pv.set_ylim(-1, 1)  # Right axis range from -2 to 2, PV generation displayed above 0

        # Electricity price plot (right axis right side)
        ax2_1_price = ax2_1.twinx()
        ax2_1_price.spines['right'].set_position(('outward', 60))  # Move price axis outward
        ax2_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax2_1_price.set_ylabel('Price ($/kWh)')
        ax2_1_price.legend(loc='lower right')

        # ESS SOC change plot
        ax2_2 = plt.subplot(2, 1, 2)
        time_interval = 0.5 / 24  # 30-minute interval
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

        # Set time axis format
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

        # ===== Third canvas: AC power change plot and temperature change plot =====
        plt.figure(figsize=(20, 10))

        # AC power and electricity price plot
        ax3_1 = plt.subplot(2, 1, 1)
        # Use step function to plot AC power changes
        ax3_1.step(mpl_dates, self.air_conditioner_power_record, color='#B2DBB9', label='AC Power', where='post')
        ax3_1.set_ylabel('AC Power (kW)')
        ax3_1.set_title('AC Power and Electricity Price')
        ax3_1.legend(loc='upper left')

        ax3_1_price = ax3_1.twinx()
        ax3_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax3_1_price.set_ylabel('Price ($/kWh)')
        ax3_1_price.legend(loc='upper right')

        # Set time axis format
        ax3_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Temperature change plot
        ax3_2 = plt.subplot(2, 1, 2)
        ax3_2.plot(mpl_dates, self.records['indoor_temp'], color='limegreen', label='Indoor Temperature')
        ax3_2.plot(mpl_dates, self.records['outdoor_temp'], color='deepskyblue', label='Outdoor Temperature')

        # Add horizontal lines and filled areas for comfortable temperature range
        ax3_2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax3_2.axhline(24, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax3_2.fill_between(mpl_dates, 20, 24, color='lightblue', alpha=0.3)

        ax3_2.set_ylabel('Temperature (°C)')
        ax3_2.set_title('Indoor and Outdoor Temperature Trends')
        ax3_2.legend(loc='upper left')

        # Set time axis format
        ax3_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p5.png')
        plt.close()

        # ===== Fourth canvas: Second AC power change plot and temperature change plot =====
        plt.figure(figsize=(20, 10))

        # AC power and electricity price plot
        ax4_1 = plt.subplot(2, 1, 1)
        # Use step function to plot AC power changes
        ax4_1.step(mpl_dates, self.air_conditioner_power_record2, color='#B2DBB9', label='AC Power', where='post')
        ax4_1.set_ylabel('AC Power (kW)')
        ax4_1.set_title('AC Power and Electricity Price')
        ax4_1.legend(loc='upper left')

        ax4_1_price = ax4_1.twinx()
        ax4_1_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax4_1_price.set_ylabel('Price ($/kWh)')
        ax4_1_price.legend(loc='upper right')

        # Set time axis format
        ax4_1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4_1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Temperature change plot
        ax4_2 = plt.subplot(2, 1, 2)
        ax4_2.plot(mpl_dates, self.records['indoor_temp2'], color='limegreen', label='Indoor Temperature')
        ax4_2.plot(mpl_dates, self.records['outdoor_temp'], color='deepskyblue', label='Outdoor Temperature')

        # Add horizontal lines and filled areas for comfortable temperature range
        ax4_2.axhline(16, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax4_2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax4_2.fill_between(mpl_dates, 16, 20, color='lightblue', alpha=0.3)

        ax4_2.set_ylabel('Temperature (°C)')
        ax4_2.set_title('Indoor and Outdoor Temperature Trends')
        ax4_2.legend(loc='upper left')

        # Set time axis format
        ax4_2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4_2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p6.png')
        plt.close()

        # ===== Fifth canvas: Washing machine state plot =====
        plt.figure(figsize=(20, 5))

        ax5 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])  # Convert timestamps to matplotlib format

        # Plot washing machine state, align bar chart left to timestamp start point
        time_interval = 0.5 / 24  # 30 minutes converted to days
        bar_width = time_interval  # Set bar width consistent with time interval

        # Adjust bar chart position so left edge aligns with timestamp
        bar_left_edges = mpl_dates

        ax5.bar(bar_left_edges, self.wash_machine_record, width=bar_width, color='#F0A19A',
                label='Washing Machine State', align='edge')
        ax5.set_ylabel('Washing Machine State')
        ax5.set_title('Washing Machine State and Electricity Price')
        ax5.legend(loc='upper left')

        ax5_price = ax5.twinx()
        # Use step function to plot step-like electricity price curve
        ax5_price.step(mpl_dates, self.electricity_price_record, color='red', label='Electricity Price', where='post')
        ax5_price.set_ylabel('Price ($/kWh)')
        ax5_price.legend(loc='upper right')

        # Set time axis format
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax5.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Get recorded date range
        start_date = self.records['timestamps'][0]
        end_date = self.records['timestamps'][-1]

        # Iterate through each day
        current_date = start_date
        while current_date <= end_date:
            # Calculate preferred time period for the day
            preferred_start = current_date.replace(hour=self.wash_machine_preferred_time[0], minute=0)
            preferred_end = current_date.replace(hour=self.wash_machine_preferred_time[1], minute=0)

            # Add vertical shaded regions and dashed lines
            ax5.axvspan(preferred_start, preferred_end, facecolor='#5A9BD5', alpha=0.1)  # Light blue fill
            ax5.axvline(preferred_start, color='#5A9BD5', linestyle='--', linewidth=1)  # Dashed line
            ax5.axvline(preferred_end, color='#5A9BD5', linestyle='--', linewidth=1)

            # Move to next day
            current_date += timedelta(days=1)

        # Set time axis range to recorded timestamp range
        ax5.set_xlim(mpl_dates[0], mpl_dates[-1])

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p7.png')
        plt.close()


        # ===== Sixth canvas: Water heater status =====
        plt.figure(figsize=(20, 10))
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Create a 2 row 1 column grid layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # Dual axis display power and water usage (top subplot)
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

        # Water temperature curve (bottom subplot)
        ax6_1 = plt.subplot(gs[1])
        ax6_1.plot(mpl_dates, self.ewh_temp_record,
                   color='#FF7F0E', label='Water Temperature')
        ax6_1.axhline(40, color='grey', linestyle='--', label='Target Temp')
        ax6_1.set_ylabel('Temperature (°C)')
        ax6_1.set_title('Water Heater Status')

        # Add two comfortable temperature range filled areas
        # Peak period 53-57°C (6-9 AM and 6-10 PM)
        peak_low = 50
        peak_high = 60
        non_peak_low = 40
        non_peak_high = 50

        # Create a unified temperature range for filling
        all_low = [peak_low if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_low for ts in
                   self.records['timestamps']]
        all_high = [peak_high if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_high for ts in
                    self.records['timestamps']]

        # Plot unified filled area
        ax6_1.fill_between(mpl_dates, all_low, all_high,
                           color='lightgreen', alpha=0.3, label='Comfort Zone')

        # Plot special markers for peak periods
        peak_mask = [(6 <= ts.hour <= 9) or (18 <= ts.hour <= 22) for ts in self.records['timestamps']]
        ax6_1.fill_between(mpl_dates, peak_low, peak_high,
                           where=peak_mask, color='lightcoral', alpha=0.3, label='Peak Comfort Zone')

        # Unify time axis format
        for ax in [ax6_1, ax6_2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Add legend
        ax6_1.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p8.png')
        plt.close()

        # ===== Seventh canvas: Household total load change plot =====
        plt.figure(figsize=(20, 5))
        ax7 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])  # Convert timestamps to matplotlib format

        # Plot household total load curve
        ax7.plot(mpl_dates, self.records['total_load'], color='purple', label='Total Load')
        ax7.set_ylabel('Load (kW)')
        ax7.set_title('Household Total Load Over Time')
        ax7.legend(loc='upper left')

        # Set time axis format
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax7.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # Get current plot y-axis range
        ymin, ymax = ax7.get_ylim()

        # Set grid
        ax7.grid(alpha=0.3)

        # Fill red area (upper part)
        ax7.fill_between(mpl_dates, self.records['total_load'], 0, where=(np.array(self.records['total_load']) > 0),
                         color='red', alpha=0.3)

        # Fill green area (lower part)
        ax7.fill_between(mpl_dates, self.records['total_load'], 0, where=(np.array(self.records['total_load']) < 0),
                         color='green', alpha=0.3)

        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/environment_plots/p9.png')
        plt.close()

        # ===== Eighth canvas: Cost over time plot =====
        plt.figure(figsize=(20, 5))

        ax8 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Plot cost curve
        ax8.plot(mpl_dates, self.records['energy_cost'], color='purple', label='Energy Cost')
        ax8.set_ylabel('Cost ($)')
        ax8.set_title('Energy Cost Over Time')
        ax8.legend(loc='upper left')

        # Set time axis format
        ax8.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax8.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        plt.tight_layout()
        plt.savefig('figures/environment_plots/p10.png')
        plt.close()

        # New: save cost data to file
        # self.save_cost_data()

    def plot_reward_components(self):
        plt.figure(figsize=(20, 8))
        mpl_dates = mdates.date2num(self.records['timestamps'])

        # Create color mapping
        colors = {
            'energy_cost': '#FF6B6B',
            'violation_penalty': '#4ECDC4',
            'ess_reward': '#D95319',
            'ev_reward': '#96CEB4',
            'user_penalty': '#FFEEAD',
            'temp_reward': '#D4A5A5'
            # 'sell': '#925EBO'
        }

        # Plot stacked area chart
        components = ['energy_cost', 'violation_penalty', 'ess_reward',
                      'ev_reward', 'user_penalty', 'temp_reward']

        # Cumulative values for stacking
        cumulative = np.zeros(len(mpl_dates))

        for comp in components:
            values = np.array(self.reward_components[comp])
            plt.fill_between(mpl_dates, cumulative, cumulative + values,
                             label=comp.replace('_', ' ').title(),
                             color=colors[comp], alpha=0.8)
            cumulative += values

        # Plot total reward line
        plt.plot(mpl_dates, self.reward_components['total'],
                 color='#2C3E50', linewidth=2, label='Total Reward')

        # Format settings
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
        """Save simulation data to CSV file for subsequent plotting and analysis"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_data_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs('simulation_data', exist_ok=True)
        filepath = os.path.join('simulation_data', filename)
        
        # Prepare data dictionary
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
        
        # Handle length mismatch issues
        max_length = max(len(v) for v in data_dict.values() if isinstance(v, list))
        
        # Ensure all lists have consistent length
        for key, value in data_dict.items():
            if isinstance(value, list):
                if len(value) < max_length:
                    # Fill with last value
                    data_dict[key] = value + [value[-1]] * (max_length - len(value))
                elif len(value) > max_length:
                    # Truncate to maximum length
                    data_dict[key] = value[:max_length]
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Save to CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Simulation data saved to: {filepath}")
        
        # Also save reward component data
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
        print(f"Reward component data saved to: {reward_filepath}")
        
        return filepath

