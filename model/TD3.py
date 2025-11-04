import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

# Add project root directory to Python path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from environment import HomeEnergyManagementEnv
from collections import OrderedDict

# ========== State Normalization Utility ==========
class RunningStats:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory (same as PPO)
results_dir = "model/results"
os.makedirs(results_dir, exist_ok=True)

# Define state keys (same as PPO)
state_keys = [
    'home_load', 'pv_generation', 'ess_state', 'ev_battery_state',
    'time_index', 'electricity_price', 'temperature',
    'wash_machine_state', 'Air_conditioner_power', 'Air_conditioner_power2', 'ewh_temp', 'ewh_power'
]

# Use PPO action space (same as PPO)
action_space_config = {
    'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],
    'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],
    'wash_machine_schedule': [0, 1, 2, 3, 4, 5, 6],
    'Air_conditioner_set_temp': [16,18, 20, 22, 24, 26, 28, 30],
    'Air_conditioner_set_temp2': [16,18, 20, 22, 24, 26, 28, 30],
    'ewh_set_temp': [40, 45, 50, 55, 60, 65, 70]
}

# Action key list
action_keys = list(action_space_config.keys())


class ActionConverter:
    """Convert continuous actions to mixed actions required by environment"""

    def __init__(self, action_space_config):
        self.action_map = action_space_config
        # Create reverse mapping: action value -> index
        self.value_to_index = {}
        for device, values in self.action_map.items():
            self.value_to_index[device] = {val: idx for idx, val in enumerate(values)}

    def _convert_single(self, value, options):
        scaled = (value + 1) / 2  # [-1,1] -> [0,1]
        idx = int(round(scaled * (len(options) - 1)))
        return options[max(0, min(idx, len(options) - 1))]

    def continuous_to_discrete(self, continuous_action):
        return {
            'ev_power': self._convert_single(continuous_action[0], self.action_map['ev_power']),
            'battery_power': self._convert_single(continuous_action[1], self.action_map['battery_power']),
            'wash_machine_schedule': self._convert_single(continuous_action[2],
                                                          self.action_map['wash_machine_schedule']),
            'Air_conditioner_set_temp': self._convert_single(continuous_action[3],
                                                             self.action_map['Air_conditioner_set_temp']),
            'Air_conditioner_set_temp2': self._convert_single(continuous_action[4],
                                                              self.action_map['Air_conditioner_set_temp2']),
            'ewh_set_temp': self._convert_single(continuous_action[5], self.action_map['ewh_set_temp'])
        }


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class TD3:
    def __init__(self, state_dim, action_dim, action_space_config):
        self.action_dim = action_dim
        self.converter = ActionConverter(action_space_config)

        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(OrderedDict(self.actor.state_dict()))

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(OrderedDict(self.critic1.state_dict()))
        self.critic2_target.load_state_dict(OrderedDict(self.critic2.state_dict()))

        # Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=3e-5
        )

        # Hyperparameters
        self.batch_size = 1024
        self.gamma = 0.96  # Same gamma as PPO
        self.tau = 0.005
        self.noise_std = 0.5  # Initial noise, dynamically adjusted later
        self.noise_clip = 0.5
        self.update_freq = 2
        self.replay_buffer = deque(maxlen=1000000)
        self.return_list = []  # Record return for each episode
        self.critic_loss_list = []  # Record critic loss
        self.actor_loss_list = []  # Record actor loss
        # New: reward normalization parameters
        self.reward_scale = 5.0  # Reward scaling factor
        self.q_clip = 500.0  # Q value clipping threshold
        # State normalization
        self.state_stats = RunningStats(state_dim)
        self.use_state_norm = True
        self.state_dim = state_dim

    def _state_to_list(self, state_dict):
        """Convert state dictionary to list"""
        return [state_dict[k] for k in state_keys]

    def _normalize_state(self, state_list):
        if self.use_state_norm:
            return self.state_stats.normalize(np.array(state_list))
        else:
            return np.array(state_list)

    def get_action(self, state_dict, action_mask=None, add_noise=True, return_continuous=False):
        """Modified action acquisition method, supports action mask, can return continuous actions"""
        state = self._state_to_list(state_dict)
        norm_state = self._normalize_state(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)

        with torch.no_grad():
            cont_action = self.actor(state_tensor).cpu().numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=cont_action.shape)
            cont_action = np.clip(cont_action + noise, -1, 1)

        # Convert to discrete action
        disc_action = self.converter.continuous_to_discrete(cont_action)

        # Apply action mask (if provided)
        if action_mask:
            for device_name, mask_list in action_mask.items():
                if device_name in disc_action:
                    current_val = disc_action[device_name]
                    if device_name in self.converter.value_to_index:
                        idx_map = self.converter.value_to_index[device_name]
                        if current_val in idx_map:
                            action_idx = idx_map[current_val]
                            if 0 <= action_idx < len(mask_list):
                                if not mask_list[action_idx]:
                                    allowed_indices = [i for i, allowed in enumerate(mask_list) if allowed]
                                    if allowed_indices:
                                        closest_idx = min(allowed_indices, key=lambda i: abs(i - action_idx))
                                        disc_action[device_name] = self.converter.action_map[device_name][closest_idx]

        if return_continuous:
            return cont_action, disc_action
        else:
            return disc_action


    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0
        batch = random.sample(self.replay_buffer, self.batch_size)
        states_raw = np.array([self._state_to_list(x[0]) for x in batch])
        if self.use_state_norm:
            self.state_stats.update(states_raw)
            states = torch.FloatTensor(self.state_stats.normalize(states_raw)).to(device)
        else:
            states = torch.FloatTensor(states_raw).to(device)
        # Directly use continuous action vector
        actions = torch.FloatTensor(np.array([x[1] for x in batch])).to(device)
        # Reward normalization
        raw_rewards = np.array([x[2] for x in batch])
        rewards = torch.FloatTensor(raw_rewards / self.reward_scale).to(device)
        next_states_raw = np.array([self._state_to_list(x[3]) for x in batch])
        if self.use_state_norm:
            next_states = torch.FloatTensor(self.state_stats.normalize(next_states_raw)).to(device)
        else:
            next_states = torch.FloatTensor(next_states_raw).to(device)
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).to(device)
        with torch.no_grad():
            target_cont_actions = self.actor_target(next_states)
            noise = torch.randn_like(target_cont_actions) * self.noise_std
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            target_cont_actions = torch.clamp(target_cont_actions + noise, -1, 1)
            target_Q1 = self.critic1_target(next_states, target_cont_actions)
            target_Q2 = self.critic2_target(next_states, target_cont_actions)
            # Q value clipping to prevent extreme values
            target_Q1 = torch.clamp(target_Q1, -self.q_clip, self.q_clip)
            target_Q2 = torch.clamp(target_Q2, -self.q_clip, self.q_clip)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_Q
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        # Also clip current Q values
        current_Q1 = torch.clamp(current_Q1, -self.q_clip, self.q_clip)
        current_Q2 = torch.clamp(current_Q2, -self.q_clip, self.q_clip)
        # Critic L2 regularization term
        l2_lambda = 5e-5
        l2_reg = torch.tensor(0., device=states.device)
        for param in self.critic1.parameters():
            l2_reg += torch.norm(param)
        for param in self.critic2.parameters():
            l2_reg += torch.norm(param)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q) + l2_lambda * l2_reg
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Stricter gradient clipping
        clip_grad_norm_(self.critic1.parameters(), 0.1)
        clip_grad_norm_(self.critic2.parameters(), 0.1)
        self.critic_optimizer.step()
        actor_loss = torch.tensor(0.0)
        if self.total_steps % self.update_freq == 0:
            cont_actions = self.actor(states)
            q_values = self.critic1(states, cont_actions)
            q_values = torch.clamp(q_values, -self.q_clip, self.q_clip)
            actor_loss = -q_values.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Stricter gradient clipping
            clip_grad_norm_(self.actor.parameters(), 0.1)
            self.actor_optimizer.step()
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            return critic_loss.item(), actor_loss.item(), q_values.mean().item()
        return critic_loss.item(), actor_loss.item(), 0

    def train(self, env, episodes=1000):
        self.total_steps = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "model/results"
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = os.path.join(results_dir, f"returns_td3_{timestamp}.csv")
        # Write TD3-style header (refer to Rainbow DQN format)
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Episode", "Return", "Actor_Loss", "Critic_Loss", "ESS_Violation_Rate", "EV_Violation_Rate", "Total_Violation_Rate",
                "Energy_Cost", "User_Satisfaction", "Temperature_Comfort",
                "AC1_Temp_Comfort", "AC2_Temp_Comfort", "EWH_Temp_Comfort", "Learning_Rate"
            ])
            episode_returns = []
            for ep in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                episode_critic_loss = 0
                episode_actor_loss = 0
                update_steps = 0
                # ========== Metric Collection ========== #
                episode_ess_violations = 0
                episode_ev_violations = 0
                episode_ess_violations_count = 0
                episode_ev_violations_count = 0
                episode_ess_socs = []
                episode_ev_socs = []
                episode_ess_safety_margins = []
                episode_ev_safety_margins = []
                episode_energy_costs = []
                episode_user_satisfactions = []
                episode_temperature_comforts = []
                episode_peak_valley_arbitrages = []
                step_count = 0
                # ========== State Normalization Warm-up ========== #
                if self.use_state_norm and ep < 10:
                    # Warm-up phase, collect state samples
                    for _ in range(100):
                        s = env.reset()
                        s_list = self._state_to_list(s)
                        self.state_stats.update(np.array([s_list]))
                # New: temperature comfort statistics
                ac1_temp_comforts = []
                ac2_temp_comforts = []
                ewh_temp_comforts = []
                while not done:
                    # First 2000 steps noise 0.5, later 0.2
                    self.noise_std = 0.5 if ep < 2000 else 0.2
                    action_mask = env.get_action_mask(state)
                    cont_action, disc_action = self.get_action(state, action_mask=action_mask, return_continuous=True)
                    next_state, reward, done = env.step(state, disc_action)
                    self.replay_buffer.append((state, cont_action, reward, next_state, done))
                    # ========== Metric Collection ========== #
                    state_dict = state if isinstance(state, dict) else {}
                    next_state_dict = next_state if isinstance(next_state, dict) else {}
                    ess_soc = state_dict.get('ess_state', 0) / getattr(env, 'ess_capacity', 24)
                    ev_soc = state_dict.get('ev_battery_state', 0) / getattr(env, 'ev_capacity', 24)
                    episode_ess_socs.append(ess_soc)
                    episode_ev_socs.append(ev_soc)
                    soc_lower = 0.1
                    soc_upper = 0.9
                    ess_violation = max(0, soc_lower - ess_soc) + max(0, ess_soc - soc_upper)
                    ev_violation = max(0, soc_lower - ev_soc) + max(0, ev_soc - soc_upper)
                    episode_ess_violations += ess_violation
                    episode_ev_violations += ev_violation
                    episode_ess_violations_count += int(ess_violation > 0)
                    episode_ev_violations_count += int(ev_violation > 0)
                    def calc_safety_margin(soc):
                        if soc_lower <= soc <= soc_upper:
                            return min(soc - soc_lower, soc_upper - soc) / (soc_upper - soc_lower)
                        if soc < soc_lower:
                            return -(soc_lower - soc) / soc_lower
                        return -(soc - soc_upper) / (1.0 - soc_upper)
                    episode_ess_safety_margins.append(calc_safety_margin(ess_soc))
                    episode_ev_safety_margins.append(calc_safety_margin(ev_soc))
                    episode_energy_costs.append(getattr(env, 'current_step_cost', 0))
                    if isinstance(disc_action, dict):
                        ev_power = disc_action.get('ev_power', 0)
                        battery_power = disc_action.get('battery_power', 0)
                    else:
                        ev_power = 0
                        battery_power = 0
                    price = state_dict.get('electricity_price', 0)
                    def calculate_peak_valley_arbitrage(electricity_price, ev_pwr, bat_pwr):
                        valley_threshold = 0.2
                        peak_threshold = 0.8
                        arbitrage_score = 0.0
                        if electricity_price <= valley_threshold:
                            total_charging = max(0, ev_pwr) + max(0, bat_pwr)
                            arbitrage_score = total_charging * 0.95
                        elif electricity_price >= peak_threshold:
                            total_discharging = abs(min(0, ev_pwr)) + abs(min(0, bat_pwr))
                            arbitrage_score = total_discharging * 0.95
                        max_arbitrage_power = 6.6 + 4.4
                        normalized_score = min(1.0, arbitrage_score / max_arbitrage_power)
                        return normalized_score
                    episode_peak_valley_arbitrages.append(calculate_peak_valley_arbitrage(price, ev_power, battery_power))
                    indoor_temp1 = getattr(env, 'indoor_temp', 22)
                    indoor_temp2 = getattr(env, 'indoor_temp2', 18)
                    user_pref1 = getattr(env, 'user_temp_preference', 22)
                    user_pref2 = getattr(env, 'user_temp_preference2', 18)
                    temp_diff1 = abs(indoor_temp1 - user_pref1)
                    temp_diff2 = abs(indoor_temp2 - user_pref2)
                    temp_comfort1 = max(0, 1 - max(0, temp_diff1 - 2) / 8)
                    temp_comfort2 = max(0, 1 - max(0, temp_diff2 - 2) / 8)
                    ac1_temp_comforts.append(temp_comfort1)
                    ac2_temp_comforts.append(temp_comfort2)
                    ewh_temp = state_dict.get('ewh_temp', 50)
                    hour = int(state_dict.get('time_index', 0) // 2)
                    if 6 <= hour <= 9 or 18 <= hour <= 22:
                        low_temp, high_temp = 50, 60
                    else:
                        low_temp, high_temp = 40, 50
                    if low_temp <= ewh_temp <= high_temp:
                        ewh_temp_comfort = 1.0
                    else:
                        deviation = max(low_temp - ewh_temp, ewh_temp - high_temp)
                        ewh_temp_comfort = max(0, 1 - deviation / 10)
                    ewh_temp_comforts.append(ewh_temp_comfort)
                    overall_comfort = (temp_comfort1 + temp_comfort2 + ewh_temp_comfort) / 3
                    episode_temperature_comforts.append(overall_comfort)
                    user_satisfaction = overall_comfort * 0.7 + 0.3
                    episode_user_satisfactions.append(user_satisfaction)
                    episode_reward += reward
                    state = next_state
                    self.total_steps += 1
                    critic_loss, actor_loss, q_mean = self.update()
                    if critic_loss > 0:
                        episode_critic_loss += critic_loss
                        episode_actor_loss += actor_loss
                        update_steps += 1
                    step_count += 1
                if update_steps > 0:
                    avg_critic = episode_critic_loss / update_steps
                    avg_actor = episode_actor_loss / update_steps
                else:
                    avg_critic, avg_actor = 0, 0
                self.return_list.append(episode_reward)
                self.critic_loss_list.append(avg_critic)
                self.actor_loss_list.append(avg_actor)
                episode_returns.append(episode_reward)
                # ========== Statistical Metrics Calculation ========== #
                ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
                ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0
                total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2 if step_count > 0 else 0
                energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
                user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
                temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0
                ac1_temp_comfort = np.mean(ac1_temp_comforts) if ac1_temp_comforts else 0
                ac2_temp_comfort = np.mean(ac2_temp_comforts) if ac2_temp_comforts else 0
                ewh_temp_comfort = np.mean(ewh_temp_comforts) if ewh_temp_comforts else 0
                current_lr = self.actor_optimizer.param_groups[0]['lr']
                
                # Write one row
                writer.writerow([
                    ep + 1, episode_reward, avg_actor, avg_critic, ess_violation_rate, ev_violation_rate, total_violation_rate,
                    energy_cost, user_satisfaction, temperature_comfort,
                    ac1_temp_comfort, ac2_temp_comfort, ewh_temp_comfort, current_lr
                ])
                file.flush()
                print(f"Episode {ep + 1}/{episodes}, Return: {episode_reward:.2f}, "
                      f"Actor Loss: {avg_actor:.4f}, Critic Loss: {avg_critic:.4f}, "
                      f"Violation: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}, LR: {current_lr:.6f}")
        
        # ========== TD3 Model Saving Logic ==========
        model_save_dir = "model/saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(model_save_dir, f"td3_model_{timestamp}.pth")
        
        model_save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_actor_state_dict': self.actor_target.state_dict(),
            'target_critic1_state_dict': self.critic1_target.state_dict(),
            'target_critic2_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_keys': state_keys,
            'action_space_config': action_space_config,
            'training_config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'lr_actor': self.actor_optimizer.param_groups[0]['lr'],
                'lr_critic': self.critic_optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'tau': self.tau,
                'buffer_size': self.replay_buffer.maxlen,
                'batch_size': self.batch_size,
                'noise_std': self.noise_std,
                'noise_clip': self.noise_clip,
                'update_freq': self.update_freq,
                'reward_scale': self.reward_scale,
                'q_clip': self.q_clip,
                'use_state_norm': self.use_state_norm
            }
        }
        
        # If state normalization is used, save state statistics
        if self.use_state_norm:
            model_save_dict['state_stats_mean'] = self.state_stats.mean
            model_save_dict['state_stats_var'] = self.state_stats.var
            model_save_dict['state_stats_count'] = self.state_stats.count
        
        torch.save(model_save_dict, model_filename)
        print(f"TD3 model saved to: {model_filename}")
        
        env.visualize()
        self.plot_returns()
        env.save_episode_costs()
        env.plot_reward_components()
        print(f"TD3 training completed! Returns data saved to: {csv_filename}")

    def load_model(self, model_path):
        """Load saved TD3 model"""
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load network state
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.actor_target.load_state_dict(checkpoint['target_actor_state_dict'])
            self.critic1_target.load_state_dict(checkpoint['target_critic1_state_dict'])
            self.critic2_target.load_state_dict(checkpoint['target_critic2_state_dict'])
            
            # Load optimizer state
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # Load state normalization statistics
            if 'state_stats_mean' in checkpoint and self.use_state_norm:
                self.state_stats.mean = checkpoint['state_stats_mean']
                self.state_stats.var = checkpoint['state_stats_var']
                self.state_stats.count = checkpoint['state_stats_count']
            
            print(f"TD3 model successfully loaded: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def plot_returns(self):
        """Plot return curve and save to results directory"""
        return_list = self.return_list
        window_size = 9

        episodes_list = list(range(len(return_list)))
        if len(return_list) >= window_size:
            mv_return = np.convolve(return_list, np.ones(window_size) / window_size, mode='valid')
        else:
            mv_return = np.array(return_list)

        plt.figure(figsize=(12, 6))
        plt.plot(episodes_list, return_list, label='Original Returns', color='thistle', alpha=0.7)

        if len(mv_return) > 0:
            plt.plot(np.arange(len(mv_return)) + window_size - 1, mv_return,
                     label=f'Moving Average (Window={window_size})', color='purple')

        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('TD3 Training Returns')
        plt.legend()
        plt.grid(True)

        # Save image to results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"figures/training_progress/td3_returns_{timestamp}.png")
        plt.close()

        # Plot loss curves
        plt.figure(figsize=(12, 6))
        plt.plot(self.critic_loss_list, label='Critic Loss', color='red')
        plt.plot(self.actor_loss_list, label='Actor Loss', color='blue')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('TD3 Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/training_progress/td3_losses_{timestamp}.png")
        plt.close()


if __name__ == "__main__":
    env = HomeEnergyManagementEnv()
    sample_state = env.reset()

    state_dim = len([sample_state[k] for k in state_keys])
    action_dim = len(action_keys)  # Action dimension equals number of action keys

    agent = TD3(state_dim, action_dim, action_space_config)
    agent.train(env, episodes=5000)