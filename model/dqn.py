"""
DQN多分支解决方案，适配HEMS环境，仿照PPO_3rd.py结构
"""
import os
import csv
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from environment import HomeEnergyManagementEnv
from plt import plot_returns
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 状态归一化 ====================
class RunningStats:
    def __init__(self, shape):
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 1e-4
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_std = x.std(dim=0)
        batch_count = x.size(0)
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        self.std = torch.sqrt(
            (self.std ** 2 * self.count + batch_std ** 2 * batch_count +
             delta ** 2 * self.count * batch_count / (self.count + batch_count)) / (self.count + batch_count)
        )
        self.count += batch_count
    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

# ==================== Q网络结构（多分支） ====================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
    def forward(self, x):
        return self.net(x)

class QBranch(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
    def forward(self, x):
        return self.net(x)

# ==================== DQN智能体 ====================
class MultiBranchDQN:
    def __init__(self, state_dim, hidden_dim, action_space_config, lr=1e-4, gamma=0.96, tau=0.01, 
                 buffer_size=100000, batch_size=128, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.action_mapping = {
            name: {idx: val for idx, val in enumerate(values)}
            for name, values in action_space_config.items()
        }
        self.action_dims = {name: len(values) for name, values in action_space_config.items()}
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        # Q网络和目标网络
        self.shared_backbone = SharedFeatureExtractor(state_dim, hidden_dim).to(device)
        self.q_branches = nn.ModuleDict({
            name: QBranch(hidden_dim, dim).to(device)
            for name, dim in self.action_dims.items()
        })
        self.target_shared_backbone = SharedFeatureExtractor(state_dim, hidden_dim).to(device)
        self.target_q_branches = nn.ModuleDict({
            name: QBranch(hidden_dim, dim).to(device)
            for name, dim in self.action_dims.items()
        })
        self.update_target_network(1.0)
        self.optimizer = torch.optim.Adam(
            list(self.shared_backbone.parameters()) + list(self.q_branches.parameters()), lr=lr)
    def select_action(self, state_tensor, action_mask=None, explore=True):
        actions = {}
        if explore and random.random() < self.epsilon:
            for name, mapping in self.action_mapping.items():
                valid_indices = list(mapping.keys())
                if action_mask and name in action_mask:
                    valid_indices = [i for i, valid in enumerate(action_mask[name]) if valid]
                idx = random.choice(valid_indices)
                actions[name] = mapping[idx]
            return actions
        with torch.no_grad():
            features = self.shared_backbone(state_tensor)
            for name, branch in self.q_branches.items():
                q_values = branch(features)
                if action_mask and name in action_mask:
                    mask_tensor = torch.tensor(action_mask[name], dtype=torch.bool, device=q_values.device)
                    masked_q = torch.where(mask_tensor, q_values, torch.tensor(-1e9, dtype=q_values.dtype, device=q_values.device))
                else:
                    masked_q = q_values
                idx = masked_q.argmax().item()
                actions[name] = self.action_mapping[name][idx]
        return actions
    def store(self, *transition):
        self.memory.append(transition)
    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        return zip(*batch)
    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.sample()
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        loss = 0.0
        features = self.shared_backbone(states)
        next_features = self.target_shared_backbone(next_states)
        for name, branch in self.q_branches.items():
            action_indices = []
            action_values = list(self.action_mapping[name].values())
            for val in [a[name] for a in actions]:
                close_idx = np.where(np.isclose(action_values, val, atol=1e-4))[0]
                if len(close_idx) > 0:
                    action_indices.append(close_idx[0])
                else:
                    idx = int(np.argmin(np.abs(np.array(action_values) - val)))
                    action_indices.append(idx)
            action_indices = torch.tensor(action_indices, dtype=torch.long, device=device)
            q_pred = branch(features).gather(1, action_indices.unsqueeze(1)).squeeze()
            with torch.no_grad():
                target_q = self.target_q_branches[name](next_features).max(1)[0]
                q_target = rewards + self.gamma * target_q * (1 - dones)
            loss += F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network(self.tau)
        return loss.item()
    def update_target_network(self, tau):
        for target_param, param in zip(self.target_shared_backbone.parameters(), self.shared_backbone.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for name in self.q_branches:
            for target_param, param in zip(self.target_q_branches[name].parameters(), self.q_branches[name].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)
    def save(self, path, state_keys):
        torch.save({
            'shared_backbone': self.shared_backbone.state_dict(),
            'q_branches': self.q_branches.state_dict(),
            'state_keys': state_keys,
            'epsilon': self.epsilon
        }, path)
    def load(self, path):
        checkpoint = torch.load(path)
        self.shared_backbone.load_state_dict(checkpoint['shared_backbone'])
        self.q_branches.load_state_dict(checkpoint['q_branches'])
        self.epsilon = checkpoint.get('epsilon', 0.05)

# ==================== 训练主循环 ====================
if __name__ == "__main__":
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    USE_STATE_NORMALIZATION = True
    state_dim = len(env.state_space)
    hidden_dim = 128
    action_space_config = env.action_space
    agent = MultiBranchDQN(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_space_config=action_space_config,
        lr=1e-4,
        gamma=0.96,
        tau=0.01,
        buffer_size=50000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )
    if USE_STATE_NORMALIZATION:
        running_stats = RunningStats(shape=state_dim)
        print("状态归一化已启用")
    else:
        running_stats = None
        print("状态归一化已禁用")
    state_keys = sorted(env.state_space.keys())
    num_episodes = 3000
    max_steps = 200
    episode_returns = []
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    csv_filename = os.path.join(results_dir, f"returns_dqn_{timestamp}{norm_suffix}.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Episode", "Return", "Loss", "ESS_Violation_Rate", "EV_Violation_Rate", "Total_Violation_Rate",
            "Energy_Cost", "User_Satisfaction", "Temperature_Comfort",
            "AC1_Temp_Comfort", "AC2_Temp_Comfort", "EWH_Temp_Comfort"
        ])
        for episode in range(num_episodes):
            state = env.reset()
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
                state_tensor = torch.FloatTensor(current_state_values).unsqueeze(0).to(device)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_state = running_stats.normalize(state_tensor).clamp(-5, 5)
                else:
                    normalized_state = state_tensor
                action_mask = env.get_action_mask(state)
                actions = agent.select_action(normalized_state, action_mask=action_mask, explore=True)
                next_state, reward, done = env.step(state, actions)
                next_state_values = [next_state[k] for k in state_keys]
                next_state_tensor = torch.FloatTensor(next_state_values).unsqueeze(0).to(device)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_next_state = running_stats.normalize(next_state_tensor).clamp(-5, 5)
                else:
                    normalized_next_state = next_state_tensor
                agent.store(
                    torch.FloatTensor(current_state_values),
                    actions,
                    reward,
                    torch.FloatTensor(next_state_values),
                    float(done)
                )
                episode_return += reward
                state = next_state
                step_count += 1
                # 约束统计
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
                # 经济与舒适度
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
            # 更新running_stats
            if USE_STATE_NORMALIZATION and running_stats is not None and episode_states:
                states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
                running_stats.update(states_tensor)
            # DQN参数更新
            loss = agent.update()
            agent.decay_epsilon()
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
                episode + 1, episode_return, loss, ess_violation_rate, ev_violation_rate, total_violation_rate,
                energy_cost, user_satisfaction, temperature_comfort,
                ac1_temp_comfort, ac2_temp_comfort, ewh_temp_comfort
            ])
            file.flush()
            print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Loss: {loss:.4f}, Violation: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}, Epsilon: {agent.epsilon:.3f}")
        # 保存模型
        model_save_dir = "saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        model_filename = os.path.join(model_save_dir, f"dqn_model_{timestamp}{norm_suffix}.pth")
        
        # ========== 新增：完善DQN模型保存逻辑 ==========
        model_save_dict = {
            'shared_backbone_state_dict': agent.shared_backbone.state_dict(),
            'q_branches_state_dict': agent.q_branches.state_dict(),
            'target_shared_backbone_state_dict': agent.target_shared_backbone.state_dict(),
            'target_q_branches_state_dict': agent.target_q_branches.state_dict(),
            'state_keys': state_keys,
            'epsilon': agent.epsilon,
            'training_config': {
                'state_dim': state_dim,
                'hidden_dim': hidden_dim,
                'action_space_config': action_space_config,
                'gamma': agent.gamma,
                'tau': agent.tau,
                'batch_size': agent.batch_size,
                'epsilon_start': agent.epsilon,
                'epsilon_end': agent.epsilon_end,
                'epsilon_decay': agent.epsilon_decay,
                'use_state_normalization': USE_STATE_NORMALIZATION
            }
        }
        # 如果使用了状态归一化，保存running_stats
        if USE_STATE_NORMALIZATION and running_stats is not None:
            model_save_dict['running_stats_mean'] = running_stats.mean
            model_save_dict['running_stats_std'] = running_stats.std
            model_save_dict['running_stats_count'] = running_stats.count
        
        torch.save(model_save_dict, model_filename)
        print(f"模型已保存到: {model_filename}")
    env.save_episode_costs()
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)
    print(f"训练完成！Returns数据已保存到: {csv_filename}")
