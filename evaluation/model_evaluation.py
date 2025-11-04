"""
Model Evaluation Module
For evaluating trained PPO models in home energy management tasks
"""
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import sys

# Add project root directory to Python path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from environment import HomeEnergyManagementEnv
sys.path.append(os.path.join(project_root, 'model'))
from PPO_3rd import HomeEnergyPPO, RunningStats, SharedFeatureExtractor, ActionBranch, ValueNet
from matplotlib.patches import FancyBboxPatch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelEvaluator:
    def __init__(self, model_path, env=None):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained model file
            env: Environment instance, create new one if None
        """
        self.model_path = model_path
        self.env = env if env is not None else HomeEnergyManagementEnv()
        
        # Load model
        self.agent, self.running_stats, self.state_keys = self._load_model()
        
        # Evaluation result storage
        self.evaluation_results = {
            'episode_returns': [],
            'total_costs': [],
            'avg_hourly_costs': [],
            'energy_costs': [],
            'user_satisfaction': [],
            'temperature_comfort': [],
            'ac1_temp_comfort': [],
            'ac2_temp_comfort': [],
            'ewh_temp_comfort': [],
            'peak_valley_arbitrage': [],
            'ess_soc_mean': [],
            'ess_soc_std': [],
            'ev_soc_mean': [],
            'ev_soc_std': [],
            'ess_safety_margin': [],
            'ev_safety_margin': [],
            'ess_violation_mean': [],
            'ev_violation_mean': [],
            'constraint_violations': [],
            'ess_violation_rate': [],
            'ev_violation_rate': [],
            'total_violation_rate': [],
            'device_usage': {
                'ev_charging': [],
                'ess_charging': [],
                'air_conditioner': [],
                'wash_machine': [],
                'water_heater': []
            },
            'wash_avg_deviation': [],
            'wash_in_pref_ratio': [],
            'wash_avg_price': [],
            'arbitrage_details': {
                'valley_charging': [],
                'low_mid_charging': [],
                'mid_high_discharging': [],
                'peak_discharging': [],
                'mid_arbitrage': [],
                'price_levels': [],
                'charging_power': [],
                'discharging_power': [],
                'ev_valley_charging': [],
                'ess_valley_charging': [],
                'ev_mid_arbitrage': [],
                'ess_mid_arbitrage': [],
                'ev_peak_discharging': [],
                'ess_peak_discharging': []
            }
        }
    
    def _load_model(self):
        """Load trained model"""
        print(f"Loading model: {self.model_path}")
        
        # Load model file
        checkpoint = torch.load(self.model_path, map_location=device)
        
        # Rebuild network structure
        training_config = checkpoint['training_config']
        agent = HomeEnergyPPO(
            env=self.env,
            state_dim=training_config['state_dim'],
            hidden_dim=training_config['hidden_dim'],
            action_space_config=training_config['action_space_config'],
            gamma=training_config['gamma'],
            lmbda=training_config['lmbda'],
            eps=training_config['eps'],
            epochs=training_config['epochs'],
            ent_coef=training_config['ent_coef'],
            max_grad_norm=training_config['max_grad_norm'],
            device=device
        )
        
        # Load model weights
        agent.shared_backbone.load_state_dict(checkpoint['shared_backbone_state_dict'])
        agent.actor_branches.load_state_dict(checkpoint['actor_branches_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Rebuild running_stats
        running_stats = RunningStats(shape=training_config['state_dim'])
        running_stats.mean = checkpoint['running_stats_mean']
        running_stats.std = checkpoint['running_stats_std']
        running_stats.count = checkpoint['running_stats_count']
        
        # Get state keys
        state_keys = checkpoint['state_keys']
        
        print("Model loaded successfully!")
        return agent, running_stats, state_keys
    
    def evaluate_episode(self, episode_id, render=False):
        """
        Evaluate a single episode
        
        Args:
            episode_id: Episode number
            render: Whether to display detailed information
            
        Returns:
            episode_results: Dictionary containing episode evaluation results
        """
        state = self.env.reset()
        episode_return = 0
        episode_cost = 0
        step_count = 0
        # Automatically infer step length (hours)
        # If 48 steps per day, then 0.5 hours per step; otherwise use 24/steps_per_day
        steps_per_day = 48
        if hasattr(self.env, 'steps_per_day'):
            steps_per_day = getattr(self.env, 'steps_per_day')
        elif hasattr(self.env, 'step_per_day'):
            steps_per_day = getattr(self.env, 'step_per_day')
        elif hasattr(self.env, 'current_time_index') and hasattr(self.env, 'reset'):
            # Try counting after reset
            _ = self.env.reset()
            for _ in range(200):
                if hasattr(self.env, 'current_time_index') and self.env.current_time_index == 0:
                    break
                self.env.current_time_index += 1
            steps_per_day = self.env.current_time_index if self.env.current_time_index > 0 else 48
        step_hour = 24 / steps_per_day if steps_per_day > 0 else 0.5
        
        # Record device usage
        device_usage = {
            'ev_charging': [],
            'ess_charging': [],
            'air_conditioner': [],
            'wash_machine': [],
            'water_heater': []
        }
        
        # Record constraint violations
        constraint_violations = []
        
        # New: per-step metric collection
        temperature_comforts = []
        ac1_temp_comforts = []
        ac2_temp_comforts = []
        ewh_temp_comforts = []
        peak_valley_arbitrages = []
        ess_socs = []
        ev_socs = []
        ess_safety_margins = []
        ev_safety_margins = []
        ess_violation_severities = []
        ev_violation_severities = []
        ess_violation_count = 0
        ev_violation_count = 0
        wash_machine_start_times = []  # Record time_index for each start
        wash_machine_start_hours = []  # Record hour for each start
        wash_machine_start_prices = [] # Record electricity price at each start
        preferred_start, preferred_end = self.env.wash_machine_preferred_time
        started_this_day = False
        last_day = None
        
        # New: detailed arbitrage data collection
        arbitrage_details = {
            'valley_charging': [],      # Valley charging
            'low_mid_charging': [],     # Low-mid price charging
            'mid_high_discharging': [], # Mid-high price discharging
            'peak_discharging': [],     # Peak discharging
            'mid_arbitrage': [],        # Mid-price arbitrage
            'price_levels': [],         # Price levels
            'charging_power': [],       # Charging power
            'discharging_power': [],    # Discharging power
            'ev_valley_charging': [],   # EV valley charging contribution
            'ess_valley_charging': [],  # ESS valley charging contribution
            'ev_mid_arbitrage': [],     # EV mid arbitrage contribution
            'ess_mid_arbitrage': [],    # ESS mid arbitrage contribution
            'ev_peak_discharging': [],  # EV peak discharging contribution
            'ess_peak_discharging': []  # ESS peak discharging contribution
        }
        # New: SOC and power trajectory records
        soc_trace = {'ev_soc': [], 'ess_soc': []}
        power_trace = {'ev_power': [], 'ess_power': []}
        while True:
            # Prepare state tensor
            state_values = [state[k] for k in self.state_keys]
            state_tensor = torch.FloatTensor(state_values).unsqueeze(0).to(device)
            normalized_state = self.running_stats.normalize(state_tensor).clamp(-5, 5)
            
            # Get action mask
            action_mask = self.env.get_action_mask(state)
            
            # Agent selects action (no exploration in evaluation mode)
            with torch.no_grad():
                actions, _, _, value = self.agent.take_action(normalized_state, action_mask=action_mask)
            
            # Environment executes action
            next_state, reward, done = self.env.step(state, actions)
            
            # Record device usage (count as used if charging/discharging)
            ev_power = actions.get('ev_power', 0)
            ess_power = actions.get('battery_power', 0)
            device_usage['ev_charging'].append(1 if abs(ev_power) > 1e-4 else 0)
            device_usage['ess_charging'].append(1 if abs(ess_power) > 1e-4 else 0)
            device_usage['air_conditioner'].append(actions.get('Air_conditioner_set_temp', 22))
            device_usage['wash_machine'].append(actions.get('wash_machine_schedule', 0))
            device_usage['water_heater'].append(actions.get('ewh_set_temp', 50))
            
            # Record SOC and power
            soc_trace['ev_soc'].append(state['ev_battery_state'] / self.env.ev_capacity)
            soc_trace['ess_soc'].append(state['ess_state'] / self.env.ess_capacity)
            power_trace['ev_power'].append(actions.get('ev_power', 0))
            power_trace['ess_power'].append(actions.get('battery_power', 0))
            
            # Calculate SOC statistics
            ess_soc = state['ess_state'] / self.env.ess_capacity
            ev_soc = state['ev_battery_state'] / self.env.ev_capacity
            ess_socs.append(ess_soc)
            ev_socs.append(ev_soc)
            # Constraint boundaries
            soc_lower = 0.05
            soc_upper = 0.95
            # Constraint severity and safety margin
            def calc_violation_metrics(soc, lower, upper):
                if lower <= soc <= upper:
                    return 0.0, min(soc - lower, upper - soc) / (upper - lower)
                if soc < lower:
                    v = lower - soc  # Modified: use absolute difference
                    return v, -v
                else:
                    v = soc - upper  # Modified: use absolute difference
                    return v, -v
            ess_v, ess_margin = calc_violation_metrics(ess_soc, soc_lower, soc_upper)
            ev_v, ev_margin = calc_violation_metrics(ev_soc, soc_lower, soc_upper)
            ess_violation_severities.append(ess_v)
            ev_violation_severities.append(ev_v)
            ess_safety_margins.append(ess_margin)
            ev_safety_margins.append(ev_margin)
            # Peak-valley arbitrage
            price = state['electricity_price']
            ev_power = actions.get('ev_power', 0)
            battery_power = actions.get('battery_power', 0)
            def calc_peak_valley_arbitrage(electricity_price, ev_pwr, bat_pwr, collect_details=False, details_dict=None):
                arbitrage_score = 0.0
                ev_contribution = 0.0
                ess_contribution = 0.0
                max_arbitrage_power = 11.0
                
                if collect_details and details_dict is not None:
                    details_dict['price_levels'].append(electricity_price)
                    details_dict['charging_power'].append(max(0, ev_pwr) + max(0, bat_pwr))
                    details_dict['discharging_power'].append(abs(min(0, ev_pwr)) + abs(min(0, bat_pwr)))
                
                # Low price region
                if electricity_price <= 0.2:
                    ev_charging = max(0, ev_pwr)
                    ess_charging = max(0, bat_pwr)
                    ev_contribution = ev_charging * 0.95
                    ess_contribution = ess_charging * 0.95
                    arbitrage_score = ev_contribution + ess_contribution
                    
                    if collect_details and details_dict is not None:
                        details_dict['valley_charging'].append(arbitrage_score)
                        details_dict['mid_arbitrage'].append(0)
                        details_dict['peak_discharging'].append(0)
                        details_dict['ev_valley_charging'].append(ev_contribution)
                        details_dict['ess_valley_charging'].append(ess_contribution)
                        details_dict['ev_mid_arbitrage'].append(0)
                        details_dict['ess_mid_arbitrage'].append(0)
                        details_dict['ev_peak_discharging'].append(0)
                        details_dict['ess_peak_discharging'].append(0)
                        
                # High price region
                elif electricity_price >= 0.8:
                    ev_discharging = abs(min(0, ev_pwr))
                    ess_discharging = abs(min(0, bat_pwr))
                    ev_contribution = ev_discharging * 0.95
                    ess_contribution = ess_discharging * 0.95
                    arbitrage_score = ev_contribution + ess_contribution
                    
                    if collect_details and details_dict is not None:
                        details_dict['peak_discharging'].append(arbitrage_score)
                        details_dict['valley_charging'].append(0)
                        details_dict['mid_arbitrage'].append(0)
                        details_dict['ev_peak_discharging'].append(ev_contribution)
                        details_dict['ess_peak_discharging'].append(ess_contribution)
                        details_dict['ev_valley_charging'].append(0)
                        details_dict['ess_valley_charging'].append(0)
                        details_dict['ev_mid_arbitrage'].append(0)
                        details_dict['ess_mid_arbitrage'].append(0)
                        
                # Mid price region
                else:
                    ev_charging = max(0, ev_pwr)
                    ess_charging = max(0, bat_pwr)
                    ev_discharging = abs(min(0, ev_pwr))
                    ess_discharging = abs(min(0, bat_pwr))
                    ev_contribution = (ev_charging + ev_discharging) * 0.2
                    ess_contribution = (ess_charging + ess_discharging) * 0.2
                    arbitrage_score = ev_contribution + ess_contribution
                    
                    if collect_details and details_dict is not None:
                        details_dict['mid_arbitrage'].append(arbitrage_score)
                        details_dict['valley_charging'].append(0)
                        details_dict['peak_discharging'].append(0)
                        details_dict['ev_mid_arbitrage'].append(ev_contribution)
                        details_dict['ess_mid_arbitrage'].append(ess_contribution)
                        details_dict['ev_valley_charging'].append(0)
                        details_dict['ess_valley_charging'].append(0)
                        details_dict['ev_peak_discharging'].append(0)
                        details_dict['ess_peak_discharging'].append(0)
                
                normalized_score = min(1.0, arbitrage_score / max_arbitrage_power)
                return normalized_score
            peak_valley_arbitrages.append(calc_peak_valley_arbitrage(price, ev_power, battery_power, 
                                                                   collect_details=True, details_dict=arbitrage_details))
            # Temperature comfort
            indoor_temp1 = self.env.indoor_temp
            indoor_temp2 = self.env.indoor_temp2
            user_pref1 = self.env.user_temp_preference
            user_pref2 = self.env.user_temp_preference2
            temp_diff1 = abs(indoor_temp1 - user_pref1)
            temp_diff2 = abs(indoor_temp2 - user_pref2)
            ac1_comfort = max(0, 1 - max(0, temp_diff1 - 2) / 8)
            ac2_comfort = max(0, 1 - max(0, temp_diff2 - 2) / 8)
            ac1_temp_comforts.append(ac1_comfort)
            ac2_temp_comforts.append(ac2_comfort)
            # Water heater comfort
            ewh_temp = self.env.state['ewh_temp']
            hour = int(self.env.state['time_index'] // 2)
            if 6 <= hour <= 9 or 18 <= hour <= 22:
                low_temp, high_temp = 50, 60
            else:
                low_temp, high_temp = 40, 50
            if low_temp <= ewh_temp <= high_temp:
                ewh_comfort = 1.0
            else:
                deviation = max(low_temp - ewh_temp, ewh_temp - high_temp)
                ewh_comfort = max(0, 1 - deviation / 10)
            ewh_temp_comforts.append(ewh_comfort)
            # Overall temperature comfort
            overall_comfort = (ac1_comfort + ac2_comfort + ewh_comfort) / 3
            temperature_comforts.append(overall_comfort)
            # User satisfaction directly uses weighted overall temperature comfort
            user_satisfaction = overall_comfort * 0.7 + 0.3
            
            # Check constraint violations (unified logic to avoid duplicate calculations)
            ess_violation = 0
            ev_violation = 0
            
            # Use unified boundary check
            if state['ess_state'] < soc_lower * self.env.ess_capacity or state['ess_state'] > soc_upper * self.env.ess_capacity:
                ess_violation = 1
                ess_violation_count += 1
                
            if state['ev_battery_state'] < soc_lower * self.env.ev_capacity or state['ev_battery_state'] > soc_upper * self.env.ev_capacity:
                ev_violation = 1
                ev_violation_count += 1
                
            constraint_violations.append(ess_violation + ev_violation)
            
            # Check if it's a new day
            current_day = int(state['time_index'] // 48) if 'time_index' in state else None
            if current_day != last_day:
                started_this_day = False
                last_day = current_day
            # Check if wash machine is started
            if actions.get('wash_machine_schedule', 0) == 1 and not started_this_day:
                hour = (state['time_index'] % 48) * 0.5  # 0.5 hour step length
                wash_machine_start_times.append(state['time_index'])
                wash_machine_start_hours.append(hour)
                wash_machine_start_prices.append(state['electricity_price'])
                started_this_day = True
            
            episode_return += reward
            episode_cost += self.env.current_step_cost
            state = next_state
            step_count += 1  # New: increment counter each step
            
            if render:
                print(f"Step {step_count}: Actions={actions}, Reward={reward:.3f}, Cost={self.env.current_step_cost:.3f}")
            
            if done:
                break
        
        # Wash machine behavior metrics statistics
        wash_machine_deviation = []
        wash_machine_in_preference = []
        for hour in wash_machine_start_hours:
            if preferred_start <= hour <= preferred_end:
                wash_machine_deviation.append(0)
                wash_machine_in_preference.append(1)
            else:
                # Calculate deviation in hours
                if hour < preferred_start:
                    deviation = preferred_start - hour
                else:
                    deviation = hour - preferred_end
                wash_machine_deviation.append(deviation)
                wash_machine_in_preference.append(0)
        avg_wash_deviation = np.mean(wash_machine_deviation) if wash_machine_deviation else 0
        wash_in_pref_ratio = np.mean(wash_machine_in_preference) if wash_machine_in_preference else 0
        avg_wash_price = np.mean(wash_machine_start_prices) if wash_machine_start_prices else 0
        # Calculate violation rate
        ess_violation_rate = ess_violation_count / step_count if step_count > 0 else 0
        ev_violation_rate = ev_violation_count / step_count if step_count > 0 else 0
        total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2
        # Calculate average hourly cost
        avg_hourly_cost = episode_cost / (step_count * step_hour) if step_count > 0 and step_hour > 0 else 0
        # Calculate mean/variance etc.
        episode_results = {
            'episode_id': episode_id,
            'episode_return': episode_return,
            'total_cost': episode_cost,
            'avg_hourly_cost': avg_hourly_cost,
            'step_count': step_count,
            'user_satisfaction': np.mean(user_satisfaction),
            'temperature_comfort': np.mean(temperature_comforts),
            'ac1_temp_comfort': np.mean(ac1_temp_comforts),
            'ac2_temp_comfort': np.mean(ac2_temp_comforts),
            'ewh_temp_comfort': np.mean(ewh_temp_comforts),
            'peak_valley_arbitrage': np.mean(peak_valley_arbitrages),
            'ess_soc_mean': np.mean(ess_socs),
            'ess_soc_std': np.std(ess_socs),
            'ev_soc_mean': np.mean(ev_socs),
            'ev_soc_std': np.std(ev_socs),
            'ess_safety_margin': np.mean(ess_safety_margins),
            'ev_safety_margin': np.mean(ev_safety_margins),
            'ess_violation_mean': np.mean(ess_violation_severities),
            'ev_violation_mean': np.mean(ev_violation_severities),
            'ess_violation_rate': ess_violation_rate,
            'ev_violation_rate': ev_violation_rate,
            'total_violation_rate': total_violation_rate,
            'device_usage': device_usage,
            'wash_avg_deviation': avg_wash_deviation,
            'wash_in_pref_ratio': wash_in_pref_ratio,
            'wash_avg_price': avg_wash_price,
            'arbitrage_details': arbitrage_details
        }
        episode_results['soc_trace'] = soc_trace
        episode_results['power_trace'] = power_trace
        
        # Calculate usage rate for each device in this episode
        device_usage_rate = {k: (sum(v)/len(v) if len(v)>0 else 0) for k,v in device_usage.items()}
        episode_results['device_usage_rate'] = device_usage_rate
        
        return episode_results
    
    def evaluate_model(self, num_episodes=10, render_episodes=None):
        """
        Evaluate model performance
        
        Args:
            num_episodes: Number of episodes to evaluate
            render_episodes: List of episodes to display in detail
        """
        print(f"Starting model evaluation with {num_episodes} episodes...")
        
        if render_episodes is None:
            render_episodes = []
        
        all_device_usage_rates = {k: [] for k in ['ev_charging', 'ess_charging', 'air_conditioner', 'wash_machine', 'water_heater']}
        for episode in range(num_episodes):
            render = episode in render_episodes
            if render:
                print(f"\n=== Episode {episode + 1} (Detailed Mode) ===")
            
            episode_results = self.evaluate_episode(episode + 1, render=render)
            
            # Store results
            self.evaluation_results['episode_returns'].append(episode_results['episode_return'])
            self.evaluation_results['total_costs'].append(episode_results['total_cost'])
            self.evaluation_results['avg_hourly_costs'].append(episode_results['avg_hourly_cost'])
            self.evaluation_results['user_satisfaction'].append(episode_results['user_satisfaction'])
            self.evaluation_results['temperature_comfort'].append(episode_results['temperature_comfort'])
            self.evaluation_results['ac1_temp_comfort'].append(episode_results['ac1_temp_comfort'])
            self.evaluation_results['ac2_temp_comfort'].append(episode_results['ac2_temp_comfort'])
            self.evaluation_results['ewh_temp_comfort'].append(episode_results['ewh_temp_comfort'])
            self.evaluation_results['peak_valley_arbitrage'].append(episode_results['peak_valley_arbitrage'])
            self.evaluation_results['ess_soc_mean'].append(episode_results['ess_soc_mean'])
            self.evaluation_results['ess_soc_std'].append(episode_results['ess_soc_std'])
            self.evaluation_results['ev_soc_mean'].append(episode_results['ev_soc_mean'])
            self.evaluation_results['ev_soc_std'].append(episode_results['ev_soc_std'])
            self.evaluation_results['ess_safety_margin'].append(episode_results['ess_safety_margin'])
            self.evaluation_results['ev_safety_margin'].append(episode_results['ev_safety_margin'])
            self.evaluation_results['ess_violation_mean'].append(episode_results['ess_violation_mean'])
            self.evaluation_results['ev_violation_mean'].append(episode_results['ev_violation_mean'])
            self.evaluation_results['ess_violation_rate'].append(episode_results['ess_violation_rate'])
            self.evaluation_results['ev_violation_rate'].append(episode_results['ev_violation_rate'])
            self.evaluation_results['total_violation_rate'].append(episode_results['total_violation_rate'])
            
            # Merge device usage data
            for device, usage in episode_results['device_usage'].items():
                self.evaluation_results['device_usage'][device].extend(usage)
            
            self.evaluation_results['wash_avg_deviation'].append(episode_results['wash_avg_deviation'])
            self.evaluation_results['wash_in_pref_ratio'].append(episode_results['wash_in_pref_ratio'])
            self.evaluation_results['wash_avg_price'].append(episode_results['wash_avg_price'])
            
            # Save arbitrage detailed data
            self.evaluation_results['arbitrage_details']['valley_charging'].append(np.mean(episode_results['arbitrage_details']['valley_charging']))
            self.evaluation_results['arbitrage_details']['low_mid_charging'].append(np.mean(episode_results['arbitrage_details']['low_mid_charging']))
            self.evaluation_results['arbitrage_details']['mid_high_discharging'].append(np.mean(episode_results['arbitrage_details']['mid_high_discharging']))
            self.evaluation_results['arbitrage_details']['peak_discharging'].append(np.mean(episode_results['arbitrage_details']['peak_discharging']))
            self.evaluation_results['arbitrage_details']['mid_arbitrage'].append(np.mean(episode_results['arbitrage_details']['mid_arbitrage']))
            self.evaluation_results['arbitrage_details']['price_levels'].append(np.mean(episode_results['arbitrage_details']['price_levels']))
            self.evaluation_results['arbitrage_details']['charging_power'].append(np.mean(episode_results['arbitrage_details']['charging_power']))
            self.evaluation_results['arbitrage_details']['discharging_power'].append(np.mean(episode_results['arbitrage_details']['discharging_power']))
            
            # Save EV and ESS contribution data
            self.evaluation_results['arbitrage_details']['ev_valley_charging'].append(np.mean(episode_results['arbitrage_details']['ev_valley_charging']))
            self.evaluation_results['arbitrage_details']['ess_valley_charging'].append(np.mean(episode_results['arbitrage_details']['ess_valley_charging']))
            self.evaluation_results['arbitrage_details']['ev_mid_arbitrage'].append(np.mean(episode_results['arbitrage_details']['ev_mid_arbitrage']))
            self.evaluation_results['arbitrage_details']['ess_mid_arbitrage'].append(np.mean(episode_results['arbitrage_details']['ess_mid_arbitrage']))
            self.evaluation_results['arbitrage_details']['ev_peak_discharging'].append(np.mean(episode_results['arbitrage_details']['ev_peak_discharging']))
            self.evaluation_results['arbitrage_details']['ess_peak_discharging'].append(np.mean(episode_results['arbitrage_details']['ess_peak_discharging']))
            
            for device in all_device_usage_rates:
                all_device_usage_rates[device].append(episode_results['device_usage_rate'][device])
            
            if not render:
                print(f"Episode {episode + 1}: Return={episode_results['episode_return']:.2f}, "
                      f"Cost={episode_results['total_cost']:.2f}, "
                      f"Satisfaction={episode_results['user_satisfaction']:.3f}")
            
            # Only visualize environment for first episode
            if episode == 0:
                self.env.visualize()
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Save evaluation results
        self._save_evaluation_results()
        
        # Generate evaluation report
        self._generate_evaluation_report()

        # Write device average usage rates for subsequent plotting
        self.evaluation_results['device_usage_means'] = {device: np.mean(all_device_usage_rates[device]) for device in all_device_usage_rates}

    
    def _calculate_statistics(self):
        """Calculate evaluation statistics"""
        self.statistics = {
            'mean_return': np.mean(self.evaluation_results['episode_returns']),
            'std_return': np.std(self.evaluation_results['episode_returns']),
            'mean_cost': np.mean(self.evaluation_results['avg_hourly_costs']),
            'std_cost': np.std(self.evaluation_results['avg_hourly_costs']),
            'mean_satisfaction': np.mean(self.evaluation_results['user_satisfaction']),
            'mean_temperature_comfort': np.mean(self.evaluation_results['temperature_comfort']),
            'mean_ac1_temp_comfort': np.mean(self.evaluation_results['ac1_temp_comfort']),
            'mean_ac2_temp_comfort': np.mean(self.evaluation_results['ac2_temp_comfort']),
            'mean_ewh_temp_comfort': np.mean(self.evaluation_results['ewh_temp_comfort']),
            'mean_peak_valley_arbitrage': np.mean(self.evaluation_results['peak_valley_arbitrage']),
            'mean_ess_soc': np.mean(self.evaluation_results['ess_soc_mean']),
            'std_ess_soc': np.mean(self.evaluation_results['ess_soc_std']),
            'mean_ev_soc': np.mean(self.evaluation_results['ev_soc_mean']),
            'std_ev_soc': np.mean(self.evaluation_results['ev_soc_std']),
            'mean_ess_safety_margin': np.mean(self.evaluation_results['ess_safety_margin']),
            'mean_ev_safety_margin': np.mean(self.evaluation_results['ev_safety_margin']),
            'mean_ess_violation': np.mean(self.evaluation_results['ess_violation_mean']),
            'mean_ev_violation': np.mean(self.evaluation_results['ev_violation_mean']),
            'mean_ess_violation_rate': np.mean(self.evaluation_results['ess_violation_rate']),
            'mean_ev_violation_rate': np.mean(self.evaluation_results['ev_violation_rate']),
            'mean_total_violation_rate': np.mean(self.evaluation_results['total_violation_rate']),
            'device_usage_stats': {},
            'mean_wash_deviation': np.mean(self.evaluation_results['wash_avg_deviation']),
            'mean_wash_in_pref_ratio': np.mean(self.evaluation_results['wash_in_pref_ratio']),
            'mean_wash_avg_price': np.mean(self.evaluation_results['wash_avg_price'])
        }
        
        # Calculate device usage statistics
        for device, usage in self.evaluation_results['device_usage'].items():
            if usage:
                self.statistics['device_usage_stats'][device] = {
                    'mean': np.mean(usage),
                    'std': np.std(usage),
                    'min': np.min(usage),
                    'max': np.max(usage)
                }
    
    def _save_evaluation_results(self):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create evaluation results directory
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(eval_dir, f"evaluation_results_{timestamp}.csv")
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Return', 'Total_Cost', 'Avg_Hourly_Cost', 'User_Satisfaction', 'Temperature_Comfort',
                'AC1_Temp_Comfort', 'AC2_Temp_Comfort', 'EWH_Temp_Comfort', 'Peak_Valley_Arbitrage',
                'ESS_SOC_Mean', 'ESS_SOC_Std', 'EV_SOC_Mean', 'EV_SOC_Std',
                'ESS_Safety_Margin', 'EV_Safety_Margin', 'ESS_Violation_Mean', 'EV_Violation_Mean',
                'ESS_Violation_Rate', 'EV_Violation_Rate', 'Total_Violation_Rate',
                'Wash_Avg_Deviation', 'Wash_In_Pref_Ratio', 'Wash_Avg_Price'
            ])
            for i in range(len(self.evaluation_results['episode_returns'])):
                writer.writerow([
                    i + 1,
                    self.evaluation_results['episode_returns'][i],
                    self.evaluation_results['total_costs'][i],
                    self.evaluation_results['avg_hourly_costs'][i],
                    self.evaluation_results['user_satisfaction'][i],
                    self.evaluation_results['temperature_comfort'][i],
                    self.evaluation_results['ac1_temp_comfort'][i],
                    self.evaluation_results['ac2_temp_comfort'][i],
                    self.evaluation_results['ewh_temp_comfort'][i],
                    self.evaluation_results['peak_valley_arbitrage'][i],
                    self.evaluation_results['ess_soc_mean'][i],
                    self.evaluation_results['ess_soc_std'][i],
                    self.evaluation_results['ev_soc_mean'][i],
                    self.evaluation_results['ev_soc_std'][i],
                    self.evaluation_results['ess_safety_margin'][i],
                    self.evaluation_results['ev_safety_margin'][i],
                    self.evaluation_results['ess_violation_mean'][i],
                    self.evaluation_results['ev_violation_mean'][i],
                    self.evaluation_results['ess_violation_rate'][i],
                    self.evaluation_results['ev_violation_rate'][i],
                    self.evaluation_results['total_violation_rate'][i],
                    self.evaluation_results['wash_avg_deviation'][i],
                    self.evaluation_results['wash_in_pref_ratio'][i],
                    self.evaluation_results['wash_avg_price'][i]
                ])
        
        # Save statistical summary
        summary_file = os.path.join(eval_dir, f"evaluation_summary_{timestamp}.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Mean_Return', self.statistics['mean_return']])
            writer.writerow(['Std_Return', self.statistics['std_return']])
            writer.writerow(['Mean_Avg_Hourly_Cost', self.statistics['mean_cost']])
            writer.writerow(['Std_Avg_Hourly_Cost', self.statistics['std_cost']])
            writer.writerow(['Mean_Satisfaction', self.statistics['mean_satisfaction']])
            writer.writerow(['Mean_Temperature_Comfort', self.statistics['mean_temperature_comfort']])
            writer.writerow(['Mean_ESS_Violation_Rate', self.statistics['mean_ess_violation_rate']])
            writer.writerow(['Mean_EV_Violation_Rate', self.statistics['mean_ev_violation_rate']])
            writer.writerow(['Mean_Total_Violation_Rate', self.statistics['mean_total_violation_rate']])
            writer.writerow(['Mean_Wash_Deviation', self.statistics['mean_wash_deviation']])
            writer.writerow(['Mean_Wash_In_Pref_Ratio', self.statistics['mean_wash_in_pref_ratio']])
            writer.writerow(['Mean_Wash_Avg_Price', self.statistics['mean_wash_avg_price']])
        
        print(f"Evaluation results saved to: {eval_dir}")
    
    def _generate_evaluation_report(self):
        """Generate evaluation report"""
        print("\n" + "="*50)
        print("Model Evaluation Report")
        print("="*50)
        print(f"Mean Return: {self.statistics['mean_return']:.2f} ± {self.statistics['std_return']:.2f}")
        print(f"Mean Cost: {self.statistics['mean_cost']:.2f} ± {self.statistics['std_cost']:.2f}")
        print(f"Mean User Satisfaction: {self.statistics['mean_satisfaction']:.3f}")
        print(f"Mean Constraint Violation Rate: {self.statistics['mean_total_violation_rate']:.3f}")
        print(f"Mean Temperature Comfort: {self.statistics['mean_temperature_comfort']:.3f}")
        print(f"AC1 Comfort: {self.statistics['mean_ac1_temp_comfort']:.3f}, AC2 Comfort: {self.statistics['mean_ac2_temp_comfort']:.3f}, EWH Comfort: {self.statistics['mean_ewh_temp_comfort']:.3f}")
        print(f"Mean Peak-Valley Arbitrage Score: {self.statistics['mean_peak_valley_arbitrage']:.3f}")
        print(f"ESS_SOC Mean: {self.statistics['mean_ess_soc']:.3f} ± {self.statistics['std_ess_soc']:.3f}, EV_SOC Mean: {self.statistics['mean_ev_soc']:.3f} ± {self.statistics['std_ev_soc']:.3f}")
        print(f"ESS Safety Margin Mean: {self.statistics['mean_ess_safety_margin']:.3f}, EV Safety Margin Mean: {self.statistics['mean_ev_safety_margin']:.3f}")
        print(f"ESS Violation Mean: {self.statistics['mean_ess_violation']:.3f}, EV Violation Mean: {self.statistics['mean_ev_violation']:.3f}")
        print(f"ESS Violation Rate Mean: {self.statistics['mean_ess_violation_rate']:.3f}, EV Violation Rate Mean: {self.statistics['mean_ev_violation_rate']:.3f}, Total Violation Rate Mean: {self.statistics['mean_total_violation_rate']:.3f}")
        print(f"Wash Machine Avg Start Time Deviation (hours): {self.statistics['mean_wash_deviation']:.2f}")
        print(f"Wash Machine Preference Time Ratio: {self.statistics['mean_wash_in_pref_ratio']:.2%}")
        print(f"Wash Machine Avg Price: {self.statistics['mean_wash_avg_price']:.3f}")
        
        print("\nDevice Usage Statistics:")
        for device, stats in self.statistics['device_usage_stats'].items():
            print(f"  {device}: {stats['mean']:.2f} ± {stats['std']:.2f} "
                  f"(Range: {stats['min']:.2f} - {stats['max']:.2f})")
        
        print("="*50)
    
    def plot_arbitrage_analysis(self):
        """Plot dedicated arbitrage analysis charts - reorganized layout"""
        plt.style.use('seaborn-darkgrid')
        
        # Create dedicated arbitrage analysis chart - 3x3 layout
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        font_title = {'fontsize': 16, 'fontweight': 'bold'}
        font_label = {'fontsize': 14}
        font_tick = {'labelsize': 12}
        
        episodes = range(1, len(self.evaluation_results['peak_valley_arbitrage']) + 1)
        
        # ===== Row 1: Distribution analysis (bar charts) =====
        # 1. Arbitrage score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.evaluation_results['peak_valley_arbitrage'], bins=10, alpha=0.7, color='#E17C05', edgecolor='black')
        ax1.axvline(self.statistics['mean_peak_valley_arbitrage'], color='red', linestyle='--', 
                    label=f'Mean: {self.statistics["mean_peak_valley_arbitrage"]:.3f}')
        # ax1.set_title('Distribution of Arbitrage Scores', fontsize=16)
        ax1.text(-0.2, 1.1, 'a', transform=ax1.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax1.set_xlabel('Arbitrage Score', **font_label)
        ax1.set_ylabel('Count', **font_label)
        legend1 = ax1.legend(loc='upper right', fontsize=11,
                             bbox_to_anchor=(1.0, 1.0),
                             fancybox=True, shadow=True,
                             frameon=True)
        legend1.get_frame().set_facecolor('white')
        legend1.get_frame().set_edgecolor('gray')
        legend1.get_frame().set_linewidth(1.5)
        legend1.get_frame().set_alpha(0.5)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Return distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.evaluation_results['episode_returns'], bins=10, alpha=0.7, color='#4C72B0', edgecolor='black')
        ax2.axvline(self.statistics['mean_return'], color='red', linestyle='--', 
                    label=f'Mean: {self.statistics["mean_return"]:.2f}')
        # ax2.set_title('Distribution of Episode Returns', fontsize=16)
        ax2.text(-0.2, 1.1, 'b', transform=ax2.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax2.set_xlabel('Return', **font_label)
        ax2.set_ylabel('Count', **font_label)
        legend2 = ax2.legend(loc='upper right', fontsize=11,
                             bbox_to_anchor=(1.0, 1.0),
                             fancybox=True, shadow=True,
                             frameon=True)
        legend2.get_frame().set_facecolor('white')
        legend2.get_frame().set_edgecolor('gray')
        legend2.get_frame().set_linewidth(1.5)
        legend2.get_frame().set_alpha(0.5)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Cost distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.evaluation_results['avg_hourly_costs'], bins=10, alpha=0.7, color='#55A868', edgecolor='black')
        ax3.axvline(self.statistics['mean_cost'], color='red', linestyle='--', 
                    label=f'Mean: {self.statistics["mean_cost"]:.2f}')
        # ax3.set_title('Distribution of Average Hourly Cost', fontsize=16)
        ax3.text(-0.2, 1.1, 'c', transform=ax3.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax3.set_xlabel('Avg Hourly Cost', **font_label)
        ax3.set_ylabel('Count', **font_label)
        legend3 = ax3.legend(loc='upper right', fontsize=11,
                             bbox_to_anchor=(1.0, 1.0),
                             fancybox=True, shadow=True,
                             frameon=True)
        legend3.get_frame().set_facecolor('white')
        legend3.get_frame().set_edgecolor('gray')
        legend3.get_frame().set_linewidth(1.5)
        legend3.get_frame().set_alpha(0.5)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # ===== Row 2: Time series analysis (line charts) =====
        # 4. Arbitrage score time series
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(episodes, self.evaluation_results['peak_valley_arbitrage'], 
                color='#E17C05', marker='o', linewidth=2, markersize=6)
        ax4.axhline(self.statistics['mean_peak_valley_arbitrage'], color='red', linestyle='--', 
                    label=f'Mean: {self.statistics["mean_peak_valley_arbitrage"]:.3f}')
        ax4.fill_between(episodes, self.evaluation_results['peak_valley_arbitrage'], 
                        alpha=0.3, color='#E17C05')
        # ax4.set_title('Arbitrage Score per Episode', fontsize=16)
        ax4.text(-0.2, 1.1, 'd', transform=ax4.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax4.set_xlabel('Episode', **font_label)
        ax4.set_ylabel('Arbitrage Score', **font_label)
        # Increase y-axis upper limit to avoid legend overlap
        y_max = max(self.evaluation_results['peak_valley_arbitrage']) if self.evaluation_results['peak_valley_arbitrage'] else 1.0
        ax4.set_ylim(0, y_max * 1.5)
        legend4 = ax4.legend(loc='upper right', fontsize=11,
                             bbox_to_anchor=(1.0, 1.0),
                             fancybox=True, shadow=True,
                             frameon=True)
        legend4.get_frame().set_facecolor('white')
        legend4.get_frame().set_edgecolor('gray')
        legend4.get_frame().set_linewidth(1.5)
        legend4.get_frame().set_alpha(0.5)
        ax4.grid(True, linestyle='--', alpha=0.5)
        
        # 5. Power analysis
        ax5 = fig.add_subplot(gs[1, 1])
        charging_powers = self.evaluation_results['arbitrage_details']['charging_power']
        discharging_powers = self.evaluation_results['arbitrage_details']['discharging_power']
        
        ax5.plot(episodes, charging_powers, label='Charging Power', color='#4C72B0', marker='o')
        ax5.plot(episodes, discharging_powers, label='Discharging Power', color='#E17C05', marker='s')
        # ax5.set_title('Power Usage Analysis', fontsize=16)
        ax5.text(-0.2, 1.1, 'e', transform=ax5.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax5.set_xlabel('Episode', **font_label)
        ax5.set_ylabel('Power (kW)', **font_label)
        # Increase y-axis upper limit to avoid legend overlap
        y_max = max(max(charging_powers), max(discharging_powers)) if charging_powers and discharging_powers else 10
        ax5.set_ylim(0, y_max * 1.5)
        legend5 = ax5.legend(loc='upper right', fontsize=11,
                             bbox_to_anchor=(1.0, 1.0),
                             fancybox=True, shadow=True,
                             frameon=True)
        legend5.get_frame().set_facecolor('white')
        legend5.get_frame().set_edgecolor('gray')
        legend5.get_frame().set_linewidth(1.5)
        legend5.get_frame().set_alpha(0.5)
        ax5.grid(True, linestyle='--', alpha=0.5)
        
        # 6. Return time series
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(episodes, self.evaluation_results['episode_returns'], 
                color='#4C72B0', marker='o', linewidth=2, markersize=6)
        ax6.axhline(self.statistics['mean_return'], color='red', linestyle='--', 
                    label=f'Mean: {self.statistics["mean_return"]:.2f}')
        ax6.fill_between(episodes, self.evaluation_results['episode_returns'], 
                        alpha=0.3, color='#4C72B0')
        # ax6.set_title('Episode Returns per Episode', fontsize=16)
        ax6.text(-0.2, 1.1, 'f', transform=ax6.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax6.set_xlabel('Episode', **font_label)
        ax6.set_ylabel('Return', **font_label)
        # Increase y-axis upper limit to avoid legend overlap
        y_max = max(self.evaluation_results['episode_returns']) if self.evaluation_results['episode_returns'] else 500
        ax6.set_ylim(0, y_max * 1.5)
        legend6 = ax6.legend(loc='upper right', fontsize=11,
                             bbox_to_anchor=(1.0, 1.0),
                             fancybox=True, shadow=True,
                             frameon=True)
        legend6.get_frame().set_facecolor('white')
        legend6.get_frame().set_edgecolor('gray')
        legend6.get_frame().set_linewidth(1.5)
        legend6.get_frame().set_alpha(0.5)
        ax6.grid(True, linestyle='--', alpha=0.5)
        
        # ===== Row 3: Correlation analysis (scatter plots) =====
        # 7. Arbitrage score vs cost relationship
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(self.evaluation_results['avg_hourly_costs'], 
                             self.evaluation_results['peak_valley_arbitrage'],
                             c=self.evaluation_results['episode_returns'], 
                             cmap='viridis', s=100, alpha=0.7)
        ax7.set_xlabel('Average Hourly Cost', **font_label)
        ax7.set_ylabel('Arbitrage Score', **font_label)
        # ax7.set_title('Arbitrage vs Cost Relationship', fontsize=16)
        ax7.text(-0.2, 1.1, 'g', transform=ax7.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label('Episode Return', **font_label)
        ax7.grid(True, linestyle='--', alpha=0.5)
        
        # 8. Arbitrage score vs user satisfaction relationship
        ax8 = fig.add_subplot(gs[2, 1])
        scatter = ax8.scatter(self.evaluation_results['user_satisfaction'], 
                             self.evaluation_results['peak_valley_arbitrage'],
                             c=self.evaluation_results['temperature_comfort'], 
                             cmap='plasma', s=100, alpha=0.7)
        ax8.set_xlabel('User Satisfaction', **font_label)
        ax8.set_ylabel('Arbitrage Score', **font_label)
        # ax8.set_title('Arbitrage vs Satisfaction', fontsize=16)
        ax8.text(-0.2, 1.1, 'h', transform=ax8.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        cbar = plt.colorbar(scatter, ax=ax8)
        cbar.set_label('Temperature Comfort', **font_label)
        ax8.grid(True, linestyle='--', alpha=0.5)
        
        # 9. Arbitrage score vs SOC relationship
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.scatter(self.evaluation_results['ess_soc_mean'], 
                   self.evaluation_results['peak_valley_arbitrage'],
                   c='#4C72B0', s=100, alpha=0.7, label='ESS SOC')
        ax9.scatter(self.evaluation_results['ev_soc_mean'], 
                   self.evaluation_results['peak_valley_arbitrage'],
                   c='#E17C05', s=100, alpha=0.7, label='EV SOC')
        ax9.set_xlabel('SOC Mean', **font_label)
        ax9.set_ylabel('Arbitrage Score', **font_label)
        # ax9.set_title('Arbitrage vs SOC Levels', fontsize=16)
        ax9.text(-0.2, 1.1, 'i', transform=ax9.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax9.legend(loc='upper right', fontsize=10)
        ax9.grid(True, linestyle='--', alpha=0.5)
        
        # Global styling
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.tick_params(**font_tick)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        plt.savefig(f"figures/model_evaluation/arbitrage_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot separate arbitrage breakdown chart
        self.plot_arbitrage_breakdown()

    def plot_arbitrage_breakdown(self):
        """Plot separate arbitrage breakdown chart, distinguishing EV and ESS contributions"""
        plt.style.use('seaborn-darkgrid')
        
        # Create dedicated chart
        fig, ax = plt.subplots(figsize=(15, 8))
        
        font_title = {'fontsize': 18, 'fontweight': 'bold'}
        font_label = {'fontsize': 14}
        font_tick = {'labelsize': 12}
        
        episodes = range(1, len(self.evaluation_results['peak_valley_arbitrage']) + 1)
        
        # Use real arbitrage detail data, distinguish EV and ESS contributions
        # Data structure is now stored by episode
        ev_valley_charging = self.evaluation_results['arbitrage_details'].get('ev_valley_charging', [0] * len(episodes))
        ess_valley_charging = self.evaluation_results['arbitrage_details'].get('ess_valley_charging', [0] * len(episodes))
        ev_mid_arbitrage = self.evaluation_results['arbitrage_details'].get('ev_mid_arbitrage', [0] * len(episodes))
        ess_mid_arbitrage = self.evaluation_results['arbitrage_details'].get('ess_mid_arbitrage', [0] * len(episodes))
        ev_peak_discharging = self.evaluation_results['arbitrage_details'].get('ev_peak_discharging', [0] * len(episodes))
        ess_peak_discharging = self.evaluation_results['arbitrage_details'].get('ess_peak_discharging', [0] * len(episodes))
        
        # Ensure data length matches
        if len(ev_valley_charging) != len(episodes):
            ev_valley_charging = ev_valley_charging[:len(episodes)] + [0] * max(0, len(episodes) - len(ev_valley_charging))
        if len(ess_valley_charging) != len(episodes):
            ess_valley_charging = ess_valley_charging[:len(episodes)] + [0] * max(0, len(episodes) - len(ess_valley_charging))
        if len(ev_mid_arbitrage) != len(episodes):
            ev_mid_arbitrage = ev_mid_arbitrage[:len(episodes)] + [0] * max(0, len(episodes) - len(ev_mid_arbitrage))
        if len(ess_mid_arbitrage) != len(episodes):
            ess_mid_arbitrage = ess_mid_arbitrage[:len(episodes)] + [0] * max(0, len(episodes) - len(ess_mid_arbitrage))
        if len(ev_peak_discharging) != len(episodes):
            ev_peak_discharging = ev_peak_discharging[:len(episodes)] + [0] * max(0, len(episodes) - len(ev_peak_discharging))
        if len(ess_peak_discharging) != len(episodes):
            ess_peak_discharging = ess_peak_discharging[:len(episodes)] + [0] * max(0, len(episodes) - len(ess_peak_discharging))
        
        # Normalize to [0,1] range
        max_power = 11.0
        ev_valley_charging_norm = [v/max_power for v in ev_valley_charging]
        ess_valley_charging_norm = [v/max_power for v in ess_valley_charging]
        ev_mid_arbitrage_norm = [v/max_power for v in ev_mid_arbitrage]
        ess_mid_arbitrage_norm = [v/max_power for v in ess_mid_arbitrage]
        ev_peak_discharging_norm = [v/max_power for v in ev_peak_discharging]
        ess_peak_discharging_norm = [v/max_power for v in ess_peak_discharging]
        
        # Calculate total contribution for each episode
        ev_total_contribution = [ev_v + ev_m + ev_p for ev_v, ev_m, ev_p in 
                               zip(ev_valley_charging_norm, ev_mid_arbitrage_norm, ev_peak_discharging_norm)]
        ess_total_contribution = [ess_v + ess_m + ess_p for ess_v, ess_m, ess_p in 
                                zip(ess_valley_charging_norm, ess_mid_arbitrage_norm, ess_peak_discharging_norm)]
        
        # Create grouped bar chart showing separate EV and ESS contributions
        x = np.arange(len(episodes))
        width = 0.35
        
        # EV contribution (soft color scheme)
        ax.bar(x - width/2, ev_valley_charging_norm, width, label='EV Valley Charging', 
               color='#6B8E9E', alpha=0.8)  # Lighter blue
        ax.bar(x - width/2, ev_mid_arbitrage_norm, width, bottom=ev_valley_charging_norm,
               label='EV Mid Arbitrage', color='#E17C05', alpha=0.8)  # Orange, forms gradient with peak discharging
        ax.bar(x - width/2, ev_peak_discharging_norm, width, 
               bottom=[ev_v + ev_m for ev_v, ev_m in zip(ev_valley_charging_norm, ev_mid_arbitrage_norm)],
               label='EV Peak Discharging', color='#F4A261', alpha=0.8)  # Light orange
        
        # ESS contribution (soft color scheme)
        ax.bar(x + width/2, ess_valley_charging_norm, width, label='ESS Valley Charging', 
               color='#55A868', alpha=0.8)
        ax.bar(x + width/2, ess_mid_arbitrage_norm, width, bottom=ess_valley_charging_norm,
               label='ESS Mid Arbitrage', color='#6BCF7F', alpha=0.8)  # Light green
        ax.bar(x + width/2, ess_peak_discharging_norm, width, 
               bottom=[ess_v + ess_m for ess_v, ess_m in zip(ess_valley_charging_norm, ess_mid_arbitrage_norm)],
               label='ESS Peak Discharging', color='#7FBC8F', alpha=0.8)  # Lighter green
        
        # Add total arbitrage score line
        ax.plot(x, self.evaluation_results['peak_valley_arbitrage'], 
               color='#5A5A5A', linewidth=3, marker='o', markersize=8, label='Total Arbitrage Score')
        
        # Add EV and ESS total contribution lines
        ax.plot(x, ev_total_contribution, color='#6B8E9E', linestyle='--', linewidth=2, marker='s', markersize=6, 
               label='EV Total Contribution')
        ax.plot(x, ess_total_contribution, color='#55A868', linestyle='--', linewidth=2, marker='^', markersize=6, 
               label='ESS Total Contribution')
        
       
        ax.set_xlabel('Episode', fontsize=20, fontweight='bold')
        ax.set_ylabel('Normalized Arbitrage Score', fontsize=20, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(episodes)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        # Thicken borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(**font_tick)
        
        # Adjust layout to leave space for bottom legend
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        
        # Create legend at bottom, aligned with image width above
        legend = fig.legend(loc='lower center', ncol=3, fontsize=16,
                          bbox_to_anchor=(0.5, 0.02), frameon=True,
                          bbox_transform=fig.transFigure)
        
        # Set legend style
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_alpha(0.9)
        
        # Set legend width to match image width
        legend.get_frame().set_boxstyle("round,pad=0.02")
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        plt.savefig(f"figures/model_evaluation/arbitrage_breakdown_{timestamp}.png", dpi=600, bbox_inches='tight')
        plt.close()

    def plot_arbitrage_statistics(self):
        """Plot dedicated arbitrage statistics chart"""
        plt.style.use('seaborn-darkgrid')
        
        def safe_display(value):
            if np.isnan(value) or value == 0:
                return "0.000"
            return f"{value:.3f}"

        # Create statistics chart
        fig, axs = plt.subplots(2, 2, figsize=(16, 7), gridspec_kw={'height_ratios':[1,1], 'width_ratios':[1,1]})
        colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]
        titles = [
            "Arbitrage Statistics Summary",
            "Price Range Contributions",
            "Power Usage Analysis",
            "Correlation Analysis"
        ]
        contents = [
            [
                f"Mean Score: {self.statistics['mean_peak_valley_arbitrage']:.3f}",
                f"Standard Deviation: {np.std(self.evaluation_results['peak_valley_arbitrage']):.3f}",
                f"Maximum Score: {np.max(self.evaluation_results['peak_valley_arbitrage']):.3f}",
                f"Minimum Score: {np.min(self.evaluation_results['peak_valley_arbitrage']):.3f}",
                f"Score Range: {np.max(self.evaluation_results['peak_valley_arbitrage']) - np.min(self.evaluation_results['peak_valley_arbitrage']):.3f}"
            ],
            [
                f"Valley Charging (≤0.2): {safe_display(np.mean(self.evaluation_results['arbitrage_details']['valley_charging']))} ({np.mean(self.evaluation_results['arbitrage_details']['valley_charging']) / 11.0 * 100:.1f}%)",
                f"Mid Arbitrage (0.2-0.8): {safe_display(np.mean(self.evaluation_results['arbitrage_details']['mid_arbitrage']))} ({np.mean(self.evaluation_results['arbitrage_details']['mid_arbitrage']) / 11.0 * 100:.1f}%)",
                f"Peak Discharging (≥0.8): {safe_display(np.mean(self.evaluation_results['arbitrage_details']['peak_discharging']))} ({np.mean(self.evaluation_results['arbitrage_details']['peak_discharging']) / 11.0 * 100:.1f}%)"
            ],
            [
                f"Average Charging Power: {np.mean(self.evaluation_results['arbitrage_details']['charging_power']):.2f} kW",
                f"Average Discharging Power: {np.mean(self.evaluation_results['arbitrage_details']['discharging_power']):.2f} kW",
                f"Power Ratio (Charging/Discharging): {np.mean(self.evaluation_results['arbitrage_details']['charging_power']) / np.mean(self.evaluation_results['arbitrage_details']['discharging_power']):.2f}",
                f"Average Price Level: {np.mean(self.evaluation_results['arbitrage_details']['price_levels']):.3f}",
                f"Price Range: {np.min(self.evaluation_results['arbitrage_details']['price_levels']):.3f} - {np.max(self.evaluation_results['arbitrage_details']['price_levels']):.3f}"
            ],
            [
                f"Arbitrage vs Cost: {np.corrcoef(self.evaluation_results['peak_valley_arbitrage'], self.evaluation_results['avg_hourly_costs'])[0,1]:.3f}",
                f"Arbitrage vs User Satisfaction: {np.corrcoef(self.evaluation_results['peak_valley_arbitrage'], self.evaluation_results['user_satisfaction'])[0,1]:.3f}",
                f"Arbitrage vs Temperature Comfort: {np.corrcoef(self.evaluation_results['peak_valley_arbitrage'], self.evaluation_results['temperature_comfort'])[0,1]:.3f}",
                f"Cost vs Satisfaction: {np.corrcoef(self.evaluation_results['avg_hourly_costs'], self.evaluation_results['user_satisfaction'])[0,1]:.3f}",
                f"Cost vs Temperature: {np.corrcoef(self.evaluation_results['avg_hourly_costs'], self.evaluation_results['temperature_comfort'])[0,1]:.3f}"
            ]
        ]
        for ax, title, lines, color in zip(axs.flat, titles, contents, colors):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            # Draw rounded rectangle filling entire subplot
            ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor=color, edgecolor='gray', linewidth=2, alpha=0.85))
            # Center title
            ax.text(0.5, 0.82, title, ha='center', va='center', fontsize=22, color='black', fontweight='bold')
            # Left-align content
            y0 = 0.62
            for i, line in enumerate(lines):
                ax.text(0.05, y0 - i*0.12, line, ha='left', va='center', fontsize=18, color='black')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.08, hspace=0.08)
        fig.suptitle('Comprehensive Arbitrage Statistics', fontsize=22, fontweight='bold', y=0.97)
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        plt.savefig(f"figures/model_evaluation/arbitrage_statistics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_evaluation_results(self):
        """Plot beautiful and informative evaluation result charts, excluding arbitrage metrics - redesign layout"""
        import matplotlib.ticker as mticker
        plt.style.use('seaborn-darkgrid')
        
        # Create simpler chart layout focusing on non-arbitrage metrics - 2x3 layout
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        font_title = {'fontsize': 16, 'fontweight': 'bold'}
        font_label = {'fontsize': 14}
        font_tick = {'labelsize': 12}
        
        episodes = range(1, len(self.evaluation_results['episode_returns']) + 1)
        
        # ===== First row: Distribution analysis (bar charts) =====
        # 1. Total Cost distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.evaluation_results['avg_hourly_costs'], bins=10, alpha=0.7, color='#55A868', edgecolor='black')
        ax1.axvline(self.statistics['mean_cost'], color='red', linestyle='--', label=f'Mean: {self.statistics["mean_cost"]:.2f}')
        # ax1.set_title('Distribution of Average Hourly Cost', fontsize=16)
        ax1.text(-0.2, 1.1, 'a', transform=ax1.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax1.set_xlabel('Avg Hourly Cost', **font_label)
        ax1.set_ylabel('Count', **font_label)
        legend1 = ax1.legend(loc='upper right', fancybox=True, shadow=True, frameon=True)
        legend1.get_frame().set_facecolor('white')
        legend1.get_frame().set_edgecolor('gray')
        legend1.get_frame().set_linewidth(1.5)
        legend1.get_frame().set_alpha(0.5)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. User satisfaction distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.evaluation_results['user_satisfaction'], bins=10, alpha=0.7, color='#C44E52', edgecolor='black')
        ax2.axvline(self.statistics['mean_satisfaction'], color='red', linestyle='--', 
                    label=f'Mean: {self.statistics["mean_satisfaction"]:.3f}')
        # ax2.set_title('Distribution of User Satisfaction', fontsize=16)
        ax2.text(-0.2, 1.1, 'b', transform=ax2.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax2.set_xlabel('Satisfaction Score', **font_label)
        ax2.set_ylabel('Count', **font_label)
        legend2 = ax2.legend(loc='upper right', fancybox=True, shadow=True, frameon=True)
        legend2.get_frame().set_facecolor('white')
        legend2.get_frame().set_edgecolor('gray')
        legend2.get_frame().set_linewidth(1.5)
        legend2.get_frame().set_alpha(0.5)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Temperature comfort distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.evaluation_results['temperature_comfort'], bins=10, alpha=0.7, color='#E17C05', edgecolor='black')
        ax3.axvline(self.statistics['mean_temperature_comfort'], color='red', linestyle='--', 
                    label=f'Mean: {self.statistics["mean_temperature_comfort"]:.3f}')
        # ax3.set_title('Distribution of Temperature Comfort', fontsize=16)
        ax3.text(-0.2, 1.1, 'c', transform=ax3.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax3.set_xlabel('Comfort Score', **font_label)
        ax3.set_ylabel('Count', **font_label)
        legend3 = ax3.legend(loc='upper right', fancybox=True, shadow=True, frameon=True)
        legend3.get_frame().set_facecolor('white')
        legend3.get_frame().set_edgecolor('gray')
        legend3.get_frame().set_linewidth(1.5)
        legend3.get_frame().set_alpha(0.5)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # ===== Second row: Time series analysis (line charts) =====
        # 5. ESS/EV SOC Mean
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.plot(episodes, self.evaluation_results['ess_soc_mean'], label='ESS SOC Mean', color='#4C72B0', marker='o')
        ax5.plot(episodes, self.evaluation_results['ev_soc_mean'], label='EV SOC Mean', color='#E17C05', marker='s')
        # ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='SOC=0.5')
        # ax5.set_title('ESS/EV SOC Mean per Episode', fontsize=16)
        ax5.text(-0.2, 1.1, 'd', transform=ax5.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax5.set_xlabel('Episode', **font_label)
        ax5.set_ylabel('SOC Mean', **font_label)
        # Increase y-axis upper limit to avoid legend overlap
        y_max = max(max(self.evaluation_results['ess_soc_mean']), max(self.evaluation_results['ev_soc_mean'])) if self.evaluation_results['ess_soc_mean'] and self.evaluation_results['ev_soc_mean'] else 1.0
        ax5.set_ylim(0.6, 1)
        legend5 = ax5.legend(loc='upper right', fontsize=10, fancybox=True, shadow=True, frameon=True)
        legend5.get_frame().set_facecolor('white')
        legend5.get_frame().set_edgecolor('gray')
        legend5.get_frame().set_linewidth(1.5)
        legend5.get_frame().set_alpha(0.5)
        ax5.grid(True, linestyle='--', alpha=0.5)
        
        # 4. Washing machine behavior analysis
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(episodes, self.evaluation_results['wash_avg_deviation'], 
                label='Start Time Deviation', color='#4C72B0', marker='o')
        ax4.plot(episodes, self.evaluation_results['wash_in_pref_ratio'], 
                label='Preference Time Ratio', color='#E17C05', marker='s')
        # ax4.set_title('Wash Machine Behavior Analysis', fontsize=16)
        ax4.text(-0.2, 1.1, 'e', transform=ax4.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax4.set_xlabel('Episode', **font_label)
        ax4.set_ylabel('Score', **font_label)
        # Increase y-axis upper limit to avoid legend overlap
        y_max = max(max(self.evaluation_results['wash_avg_deviation']), max(self.evaluation_results['wash_in_pref_ratio'])) if self.evaluation_results['wash_avg_deviation'] and self.evaluation_results['wash_in_pref_ratio'] else 1.0
        ax4.set_ylim(0, y_max)
        legend4 = ax4.legend(loc='upper right', fancybox=True, shadow=True, frameon=True)
        legend4.get_frame().set_facecolor('white')
        legend4.get_frame().set_edgecolor('gray')
        legend4.get_frame().set_linewidth(1.5)
        legend4.get_frame().set_alpha(0.5)
        ax4.grid(True, linestyle='--', alpha=0.5)
        
        # 5. Temperature comfort time series
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(episodes, self.evaluation_results['ac1_temp_comfort'], label='AC1 Comfort', color='#4C72B0', marker='o')
        ax5.plot(episodes, self.evaluation_results['ac2_temp_comfort'], label='AC2 Comfort', color='#E17C05', marker='s')
        ax5.plot(episodes, self.evaluation_results['ewh_temp_comfort'], label='EWH Comfort', color='#55A868', marker='^')
        ax5.plot(episodes, self.evaluation_results['temperature_comfort'], label='Overall Comfort', color='#C44E52', marker='v', linewidth=2)
        # ax5.set_title('Temperature Comfort per Episode', fontsize=16)
        ax5.text(-0.2, 1.1, 'f', transform=ax5.transAxes, fontsize=24, fontweight='bold', 
                verticalalignment='top', horizontalalignment='left')
        ax5.set_xlabel('Episode', **font_label)
        ax5.set_ylabel('Comfort Score', **font_label)
        # Increase y-axis upper limit to avoid legend overlap
        y_max = max(max(self.evaluation_results['ac1_temp_comfort']), max(self.evaluation_results['ac2_temp_comfort']), 
                   max(self.evaluation_results['ewh_temp_comfort']), max(self.evaluation_results['temperature_comfort'])) if self.evaluation_results['ac1_temp_comfort'] else 1.0
        ax5.set_ylim(0.9, 1.05)
        legend5 = ax5.legend(loc='upper right', fontsize=9, fancybox=True, shadow=True, frameon=True)
        legend5.get_frame().set_facecolor('white')
        legend5.get_frame().set_edgecolor('gray')
        legend5.get_frame().set_linewidth(1.5)
        legend5.get_frame().set_alpha(0.5)
        ax5.grid(True, linestyle='--', alpha=0.5)
        
        # Global beautification
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.tick_params(**font_tick)
        
        # Adjust layout to avoid legend overlap
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        plt.savefig(f"figures/model_evaluation/evaluation_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_violation_rate(self):
        """Plot and save separate violation rate & mean violation per episode chart"""
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-darkgrid')
        font_title = {'fontsize': 18, 'fontweight': 'bold'}
        font_label = {'fontsize': 14}
        font_tick = {'labelsize': 12}
        episodes = range(1, len(self.evaluation_results['ess_violation_rate']) + 1)
        fig, ax = plt.subplots(figsize=(12, 7))
        width = 0.25
        # Bar chart: violation rate
        ax.bar([x - width for x in episodes], self.evaluation_results['ess_violation_rate'], 
               width, label='ESS Violation Rate', color='#4C72B0', alpha=0.7)
        ax.bar(episodes, self.evaluation_results['ev_violation_rate'], 
               width, label='EV Violation Rate', color='#E17C05', alpha=0.7)
        ax.bar([x + width for x in episodes], self.evaluation_results['total_violation_rate'], 
               width, label='Total Violation Rate', color='#55A868', alpha=0.7)
        ax.set_xlabel('Episode', fontsize=20, fontweight='bold')
        ax.set_ylabel('Violation Rate', fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='black', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(list(episodes))
        # Line chart: violation mean
        ax2 = ax.twinx()
        ax2.plot(episodes, self.evaluation_results['ess_violation_mean'], 
                label='ESS Violation Mean', color='#4C72B0', linestyle=':', marker='x', linewidth=2)
        ax2.plot(episodes, self.evaluation_results['ev_violation_mean'], 
                label='EV Violation Mean', color='#E17C05', linestyle=':', marker='+', linewidth=2)
        ax2.set_ylabel('Violation Mean (SOC deviation)', fontsize=20, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red', labelsize=18)
        # Thicken borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        for spine in ax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        ax.tick_params(**font_tick)
        ax2.tick_params(**font_tick)
        
        # Create legend, place directly below image
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        
        # Adjust layout to leave space for bottom legend
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        
        # Create legend at bottom, aligned with image width above
        legend = fig.legend(all_lines, all_labels, loc='lower center', ncol=3, fontsize=16,
                          bbox_to_anchor=(0.5, 0.02), frameon=True, 
                          bbox_transform=fig.transFigure)
        
        # Set legend style
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_alpha(0.9)
        
        # Set legend width to match image width
        legend.get_frame().set_boxstyle("round,pad=0.02")
        # Save
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        plt.savefig(f"figures/model_evaluation/violation_rate_{timestamp}.png", dpi=600, bbox_inches='tight')
        plt.close(fig)

    def evaluate_and_plot_all(self, num_episodes=10):
        self.evaluate_model(num_episodes=num_episodes)
        # self.plot_arbitrage_analysis()
        # self.plot_arbitrage_statistics()
        # self.plot_evaluation_results()
        self.plot_violation_rate()
        self.plot_arbitrage_breakdown()


def main():
    """Main function - Model evaluation example"""
    # Check if saved models exist
    model_dir = "model/saved_models"
    if not os.path.exists(model_dir):
        print("saved_models directory not found, please train models first!")
        return
    
    # Specify model file to evaluate
    model_filename = "proposed_rl.pth"
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        print(f"Model file {model_filename} not found in {model_dir}!")
        print("Available models:")
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        for f in model_files:
            print(f"  - {f}")
        return
    
    print(f"Using model: {model_path}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path)
    
    # One-click automatic evaluation and plotting
    evaluator.evaluate_and_plot_all(num_episodes=10)


if __name__ == "__main__":
    main() 