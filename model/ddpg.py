"""
DDPG continuous control solution, adapted for HEMS environment
"""
import os
import sys
import csv
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, OrderedDict

# Add project root directory to Python path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from environment import HomeEnergyManagementEnv
# Add evaluation directory to path
sys.path.append(os.path.join(project_root, 'evaluation'))
from plt import plot_returns
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== State Normalization ====================
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

# ==================== Action Converter ====================
class ActionConverter:
    def __init__(self, action_space_config):
        self.action_map = action_space_config
    def _convert_single(self, value, options):
        scaled = (value + 1) / 2  # [-1,1] -> [0,1]
        idx = int(round(scaled * (len(options) - 1)))
        return options[max(0, min(idx, len(options) - 1))]
    def continuous_to_discrete(self, continuous_action):
        return {
            'ev_power': self._convert_single(continuous_action[0], self.action_map['ev_power']),
            'battery_power': self._convert_single(continuous_action[1], self.action_map['battery_power']),
            'wash_machine_schedule': self._convert_single(continuous_action[2], self.action_map['wash_machine_schedule']),
            'Air_conditioner_set_temp': self._convert_single(continuous_action[3], self.action_map['Air_conditioner_set_temp']),
            'Air_conditioner_set_temp2': self._convert_single(continuous_action[4], self.action_map['Air_conditioner_set_temp2']),
            'ewh_set_temp': self._convert_single(continuous_action[5], self.action_map['ewh_set_temp'])
        }

# ==================== Network Structure ====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
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

# ==================== OU Noise ====================
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# ==================== DDPG Agent ====================
class DDPG:
    def __init__(self, state_dim, action_dim, action_space_config, lr_actor=1e-4, lr_critic=3e-4, gamma=0.96, tau=0.005, buffer_size=100000, batch_size=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.noise = OUNoise(action_dim)
        self.converter = ActionConverter(action_space_config)
        self.return_list = []
        self.loss_list = []
        self.state_stats = RunningStats(state_dim)
        self.use_state_norm = True
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        action = np.clip(action, -1, 1)
        return action
    def store(self, *transition):
        self.memory.append(transition)
    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        return zip(*batch)
    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        states, actions, rewards, next_states, dones = self.sample()
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(device)
        actions = torch.stack([torch.FloatTensor(a) for a in actions]).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.stack([torch.FloatTensor(s) for s in next_states]).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * q_target * (1 - dones)
        q_pred = self.critic(states, actions)
        critic_loss = F.mse_loss(q_pred, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        return actor_loss.item(), critic_loss.item()
    def soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1 - self.tau) * t_param.data)
    def reset_noise(self):
        self.noise.reset()
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

# ==================== Training Main Loop ====================
if __name__ == "__main__":
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    USE_STATE_NORMALIZATION = True
    state_keys = sorted(env.state_space.keys())
    state_dim = len(state_keys)
    action_space_config = env.action_space
    action_dim = 6  # Continuous action dimension
    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space_config=action_space_config,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.96,
        tau=0.005,
        buffer_size=100000,
        batch_size=128
    )
    if USE_STATE_NORMALIZATION:
        running_stats = agent.state_stats
        print("State normalization enabled")
    else:
        running_stats = None
        print("State normalization disabled")
    num_episodes = 2000
    max_steps = 200
    episode_returns = []
    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    csv_filename = os.path.join(results_dir, f"returns_ddpg_{timestamp}{norm_suffix}.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Episode", "Return", "Actor_Loss", "Critic_Loss", "ESS_Violation_Rate", "EV_Violation_Rate", "Total_Violation_Rate",
            "Energy_Cost", "User_Satisfaction", "Temperature_Comfort",
            "AC1_Temp_Comfort", "AC2_Temp_Comfort", "EWH_Temp_Comfort"
        ])
        for episode in range(num_episodes):
            state = env.reset()
            agent.reset_noise()
            episode_return = 0
            episode_states = []
            episode_ess_violations = 0
            episode_ev_violations = 0
            episode_ess_socs = []
            episode_ev_socs = []
            episode_energy_costs = []
            episode_user_satisfactions = []
            episode_temperature_comforts = []
            ac1_temp_comforts = []
            ac2_temp_comforts = []
            ewh_temp_comforts = []
            step_count = 0
            for t in range(max_steps):
                current_state_values = [state[k] for k in state_keys]
                episode_states.append(current_state_values)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    running_stats.update(np.array([current_state_values]))
                    norm_state = running_stats.normalize(np.array(current_state_values))
                else:
                    norm_state = np.array(current_state_values)
                action = agent.select_action(norm_state, add_noise=True)
                actions = agent.converter.continuous_to_discrete(action)
                next_state, reward, done = env.step(state, actions)
                next_state_values = [next_state[k] for k in state_keys]
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    norm_next_state = running_stats.normalize(np.array(next_state_values))
                else:
                    norm_next_state = np.array(next_state_values)
                agent.store(current_state_values, action, reward, next_state_values, float(done))
                episode_return += reward
                state = next_state
                step_count += 1
                # Constraint statistics
                ess_soc = state['ess_state'] / env.ess_capacity
                ev_soc = state['ev_battery_state'] / env.ev_capacity
                episode_ess_socs.append(ess_soc)
                episode_ev_socs.append(ev_soc)
                soc_lower = 0.1
                soc_upper = 0.9
                ess_violation = max(0, soc_lower - ess_soc) + max(0, ess_soc - soc_upper)
                ev_violation = max(0, soc_lower - ev_soc) + max(0, ev_soc - soc_upper)
                episode_ess_violations += ess_violation
                episode_ev_violations += ev_violation
                # Economics and comfort
                episode_energy_costs.append(env.current_step_cost)
                indoor_temp1 = env.indoor_temp
                indoor_temp2 = env.indoor_temp2
                user_pref1 = env.user_temp_preference
                user_pref2 = env.user_temp_preference2
                temp_diff1 = abs(indoor_temp1 - user_pref1)
                temp_diff2 = abs(indoor_temp2 - user_pref2)
                temp_comfort1 = max(0, 1 - max(0, temp_diff1 - 2) / 8)
                temp_comfort2 = max(0, 1 - max(0, temp_diff2 - 2) / 8)
                ac1_temp_comforts.append(temp_comfort1)
                ac2_temp_comforts.append(temp_comfort2)
                ewh_temp = env.state['ewh_temp']
                hour = int(env.state['time_index'] // 2)
                if 6 <= hour <= 9 or 18 <= hour <= 22:
                    target_temp, low_temp, high_temp = 55, 50, 60
                else:
                    target_temp, low_temp, high_temp = 45, 40, 50
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
                if done:
                    break
            # DDPG parameter update
            actor_loss, critic_loss = agent.update()
            episode_returns.append(episode_return)
            ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
            ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0
            total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2 if step_count > 0 else 0
            energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
            user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
            temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0
            ac1_temp_comfort = np.mean(ac1_temp_comforts) if ac1_temp_comforts else 0
            ac2_temp_comfort = np.mean(ac2_temp_comforts) if ac2_temp_comforts else 0
            ewh_temp_comfort = np.mean(ewh_temp_comforts) if ewh_temp_comforts else 0
            writer.writerow([
                episode + 1, episode_return, actor_loss, critic_loss, ess_violation_rate, ev_violation_rate, total_violation_rate,
                energy_cost, user_satisfaction, temperature_comfort,
                ac1_temp_comfort, ac2_temp_comfort, ewh_temp_comfort
            ])
            file.flush()
            print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Violation: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}")
        # Save model
        model_save_dir = "model/saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        model_filename = os.path.join(model_save_dir, f"ddpg_model_{timestamp}{norm_suffix}.pth")
        
        # ========== New: Complete DDPG model saving logic ==========
        model_save_dict = {
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'target_actor_state_dict': agent.actor_target.state_dict(),
            'target_critic_state_dict': agent.critic_target.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            'state_keys': state_keys,
            'training_config': {
                'state_dim': state_dim,
                'action_dim': action_dim,
                'action_space_config': action_space_config,
                'lr_actor': agent.actor_optimizer.param_groups[0]['lr'], # Get current learning rate
                'lr_critic': agent.critic_optimizer.param_groups[0]['lr'], # Get current learning rate
                'gamma': agent.gamma,
                'tau': agent.tau,
                'buffer_size': agent.memory.maxlen, # Use deque's maxlen
                'batch_size': agent.batch_size,
                'use_state_normalization': USE_STATE_NORMALIZATION
            }
        }
        # If state normalization is used, save running_stats
        if USE_STATE_NORMALIZATION and running_stats is not None:
            model_save_dict['running_stats_mean'] = running_stats.mean
            model_save_dict['running_stats_std'] = running_stats.var # Use variance as standard deviation
            model_save_dict['running_stats_count'] = running_stats.count
        
        torch.save(model_save_dict, model_filename)
        print(f"Model saved to: {model_filename}")
    # env.save_episode_costs()
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)
    print(f"Training completed! Returns data saved to: {csv_filename}")
