"""
在PPO_1st的基础上，实现running状态归一化
"""
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from environment import HomeEnergyManagementEnv
from plt import plot_returns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RunningStats:
    def __init__(self, shape):
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 1e-4  # 防止除以零

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

# ==================== 网络结构 ====================
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
        # 正交初始化
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class ActionBranch(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # 正交初始化
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 正交初始化
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 关键修改：压缩最后一个维度


# ==================== PPO智能体 ====================
class HomeEnergyPPO:
    def __init__(self, env, state_dim, hidden_dim, action_space_config,
                 gamma=0.96, lmbda=0.98, eps=0.2, epochs=4,
                 ent_coef=0.1, max_grad_norm=10, device=device, constraint_config=None):

        # 处理约束配置
        if constraint_config is None:
            constraint_config = {
                'use_constraint': False,  # False表示不使用约束
                'lambda_lr': 1e-4,
                'soc_lower': 0.1,
                'soc_upper': 0.9,
            }
        self.constraint_config = constraint_config.copy()  # 防止外部修改

        # 创建动作映射表
        self.action_mapping = {
            name: {idx: val for idx, val in enumerate(values)}
            for name, values in action_space_config.items()
        }

        # 每个设备的离散动作数量
        self.action_dims = {
            name: len(values)
            for name, values in action_space_config.items()
        }

        # 网络初始化
        self.shared_backbone = SharedFeatureExtractor(state_dim, hidden_dim).to(device)
        self.actor_branches = nn.ModuleDict({
            name: ActionBranch(hidden_dim, dim).to(device)
            for name, dim in self.action_dims.items()
        })
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # 优化器
        self.actor_optim = torch.optim.AdamW([
            {'params': self.shared_backbone.parameters()},
            {'params': self.actor_branches.parameters()}
        ], lr=5e-6)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=3e-5)

        # 学习率调度
        self.actor_scheduler = CosineAnnealingLR(self.actor_optim, T_max=1000, eta_min=3e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optim, T_max=1000, eta_min=5e-5)

        if self.constraint_config['use_constraint']:
            # 约束相关参数
            self.ess_capacity = env.ess_capacity  # 从环境获取ESS容量
            self.ev_capacity = env.ev_capacity  # 从环境获取EV容量
            self.lambda_ess = torch.tensor(1.0, requires_grad=True, device=device)
            self.lambda_ev = torch.tensor(1.0, requires_grad=True, device=device)
            self.lambda_optim = torch.optim.Adam([self.lambda_ess, self.lambda_ev],
                                                 lr=self.constraint_config['lambda_lr'])

        # 动态目标熵设置
        self.initial_target_entropy = np.mean([
            1 * np.log(len(actions)) for actions in action_space_config.values()
        ])
        self.final_target_entropy = self.initial_target_entropy * 0.01
        self.target_entropy = self.initial_target_entropy

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-5)

        # 算法参数
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def take_action(self, state_tensor, action_mask=None):
        shared_features = self.shared_backbone(state_tensor)

        actions = {}
        log_probs = []
        entropies = []

        for name, branch in self.actor_branches.items():
            logits = branch(shared_features)

            # 应用动作掩码（如果提供）
            if action_mask and name in action_mask:
                # 转换为张量并确保正确形状
                mask_tensor = torch.tensor(action_mask[name],
                                           dtype=torch.bool,
                                           device=logits.device)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)  # 添加批次维度

                # 应用掩码
                masked_logits = torch.where(mask_tensor, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
            else:
                masked_logits = logits

            # 创建概率分布
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample()

            # 将索引转换为实际值
            actions[name] = self.action_mapping[name][int(action_idx.item())]

            # 收集日志概率和熵
            log_probs.append(dist.log_prob(action_idx))
            entropies.append(dist.entropy())

        total_log_prob = torch.stack(log_probs).sum(dim=0)
        total_entropy = torch.stack(entropies).sum()
        value = self.critic(state_tensor).item()

        return actions, total_log_prob, total_entropy, value


    def compute_gae(self, rewards, values, next_values, dones):
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lmbda * next_non_terminal * last_gae

        return advantages

    def update(self, batch_data_list, running_stats, state_keys, episode, total_episodes):
        # ==================== 数据准备 ====================
        batch_data = {
            'states': [t['state'] for t in batch_data_list],
            'actions': {
                name: [t['actions'][name] for t in batch_data_list]
                for name in self.action_mapping.keys()
            },
            'rewards': [t['rewards'] for t in batch_data_list],
            'next_states': [t['next_state'] for t in batch_data_list],
            'dones': [t['dones'] for t in batch_data_list],
            'log_probs': [t['log_probs'] for t in batch_data_list],
            'values': [t['values'] for t in batch_data_list],
            'next_values': [t['next_values'] for t in batch_data_list]
        }

        # 状态张量转换（确保顺序一致）
        states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['states']
        ]).to(device)

        # 使用running_stats归一化状态
        normalized_states = running_stats.normalize(states).clamp(-5, 5)

        # 归一化下一个状态
        next_states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['next_states']
        ]).to(device)
        normalized_next_states = running_stats.normalize(next_states).clamp(-5, 5)

        # 动作索引转换（处理浮点精度）
        action_indices = {}
        for name in self.action_mapping:
            indices = [
                np.where(np.isclose(list(self.action_mapping[name].values()), val, atol=1e-4))[0][0]
                for val in batch_data['actions'][name]
            ]
            action_indices[name] = torch.tensor(indices, dtype=torch.long, device=device)

        # 张量转换（显式指定device）
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32, device=device)
        dones = torch.tensor(batch_data['dones'], dtype=torch.float32, device=device)
        old_log_probs = torch.stack([lp.squeeze() for lp in batch_data['log_probs']]).to(device)
        values = torch.tensor(batch_data['values'], dtype=torch.float32, device=device)
        next_values = torch.tensor(batch_data['next_values'], dtype=torch.float32, device=device)

        # ==================== GAE计算 ====================
        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 存储每个epoch的损失用于返回
        epoch_losses = []

        for _ in range(self.epochs):
            # ==================== Critic更新 ====================
            critic_values = self.critic(normalized_states)  # 使用归一化状态
            with torch.no_grad():
                # 使用归一化的下一个状态计算目标值
                target_values = rewards + self.gamma * next_values * (1 - dones)
            critic_loss = F.mse_loss(critic_values, target_values)

            # ==================== Actor更新 ====================
            shared_features = self.shared_backbone(normalized_states)  # 使用归一化状态
            new_log_probs = []
            entropies = []

            # 各动作分支的概率计算
            for name, branch in self.actor_branches.items():
                logits = branch(shared_features)
                # 添加数值安全处理
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("Warning: Invalid logits detected in action branch")
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(action_indices[name]))
                entropies.append(dist.entropy())

            total_entropy = torch.stack(entropies).mean(dim=0).mean()
            new_log_probs = torch.stack(new_log_probs).sum(dim=0).squeeze()

            # 维度一致性检查
            assert new_log_probs.shape == old_log_probs.shape, \
                f"维度不匹配: new_log_probs {new_log_probs.shape} vs old_log_probs {old_log_probs.shape}"

            # PPO核心损失计算
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 动态调整目标熵
            progress = min(1.0, episode / total_episodes)
            self.target_entropy = self.initial_target_entropy * (1 - progress) + self.final_target_entropy * progress

            # ==================== 自动熵调整 ====================
            alpha_loss = -(self.log_alpha * (total_entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)  # 保留计算图用于后续梯度
            self.alpha_optim.step()
            alpha = self.log_alpha.exp().detach().clamp(max=1.0)

            # ==================== 约束处理（条件执行）====================
            constraint_loss = torch.tensor(0.0, device=device)
            if self.constraint_config['use_constraint']:
                # 约束违反计算
                ess_states = states[:, 2]
                ev_states = states[:, 3]
                soc_lower = self.constraint_config['soc_lower']
                soc_upper = self.constraint_config['soc_upper']

                ess_violation = (
                        torch.relu(soc_lower * self.ess_capacity - ess_states) ** 2 * 50 +
                        torch.relu(ess_states - soc_upper * self.ess_capacity) ** 2 * 50
                )
                ev_violation = (
                        torch.relu(soc_lower * self.ev_capacity - ev_states) ** 2 * 50 +
                        torch.relu(ev_states - soc_upper * self.ev_capacity) ** 2 * 50
                )

                # 约束损失计算
                constraint_loss = (
                        self.lambda_ess * ess_violation.mean() +
                        self.lambda_ev * ev_violation.mean()
                )

            # ==================== 总损失计算 ====================
            total_loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    # - alpha * total_entropy  # 使用自适应熵系数
                    + constraint_loss  # 根据use_constraint自动为0或实际值
            )

            # ==================== 统一梯度更新 ====================
            # 梯度清零
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            if self.constraint_config['use_constraint']:
                self.lambda_optim.zero_grad()

            # 反向传播（单次）
            total_loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.shared_backbone.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_branches.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # 参数更新
            self.actor_optim.step()
            self.critic_optim.step()

            # 约束相关更新（条件执行）
            if self.constraint_config['use_constraint']:
                self.lambda_optim.step()
                self.lambda_ess.data.clamp_(min=0)
                self.lambda_ev.data.clamp_(min=0)

            # 记录损失
            epoch_losses.append(total_loss.item())

        # # 学习率调度
        # self.actor_scheduler.step()
        # self.critic_scheduler.step()

        # 返回平均损失
        return np.mean(epoch_losses)


# ==================== 训练循环 ====================
if __name__ == "__main__":
    # 环境初始化
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # 智能体初始化
    agent = HomeEnergyPPO(
        env=env,
        state_dim=len(env.state_space),
        hidden_dim=128,
        action_space_config=env.action_space,
        device=device
    )

    # 初始化 running_stats - 添加形状参数
    running_stats = RunningStats(shape=len(env.state_space))

    # 定义动作空间用于预热
    action_spaces = {
            'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],  # 电动汽车充放电功率范围
            'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],  # 储能电池充电功率范围
            'wash_machine_schedule': [0, 1, 2, 3, 4, 5, 6],  # 洗衣机调度动作
            'Air_conditioner_set_temp': [18, 20, 22, 24, 26, 28, 30],  # 空调设定温度
            'Air_conditioner_set_temp2': [16, 18, 20, 22, 24, 26, 28, 30],
            'ewh_set_temp': [40, 45, 50, 55, 60, 65, 70]  # 离散温度设定动作
    }

    # 预热阶段：收集初始数据
    print("Warming up running_stats...")
    warmup_states = []
    state = env.reset()
    state_keys = sorted(env.state_space.keys())  # 确保状态键顺序一致

    for _ in range(1000):  # 收集1000个状态样本
        warmup_states.append([state[k] for k in state_keys])

        # 使用正确的动作空间
        actions = {name: np.random.choice(space) for name, space in action_spaces.items()}

        next_state, _, _ = env.step(state, actions)
        state = next_state

    # 用预热数据初始化 running_stats
    if warmup_states:
        states_tensor = torch.tensor(warmup_states, dtype=torch.float32, device=device)
        running_stats.update(states_tensor)
    print(f"Running stats initialized: mean={running_stats.mean.cpu().numpy()}, std={running_stats.std.cpu().numpy()}")

    num_episodes = 50
    episode_returns = []

    # 创建结果目录
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 创建唯一的文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"returns_ppo_{timestamp}.csv")

    # 打开CSV文件用于写入
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行 - 扩展评估指标
        writer.writerow([
            "Episode", "Return", "Loss", 
            "ESS_Violation_Rate", "EV_Violation_Rate", "Total_Violation_Rate",
            "Energy_Cost", "User_Satisfaction", "Temperature_Comfort",
            "ESS_SOC_Mean", "ESS_SOC_Std", "EV_SOC_Mean", "EV_SOC_Std",
            "Renewable_Utilization", "Peak_Valley_Arbitrage",
            "Training_Stability", "Sample_Efficiency"
        ])

        for episode in range(num_episodes):
            state = env.reset()
            batch = []
            episode_return = 0
            episode_states = []  # 收集整个episode的状态用于更新running_stats
            
            # 新增：episode级别的指标收集
            episode_ess_violations = 0
            episode_ev_violations = 0
            episode_ess_socs = []
            episode_ev_socs = []
            episode_energy_costs = []
            episode_user_satisfactions = []
            episode_temperature_comforts = []
            episode_renewable_utilizations = []
            episode_peak_valley_arbitrages = []
            step_count = 0

            while True:
                # 收集当前状态（原始值）
                current_state_values = [state[k] for k in state_keys]
                episode_states.append(current_state_values)

                # 归一化当前状态
                state_tensor = torch.FloatTensor(current_state_values).unsqueeze(0).to(device)
                normalized_state = running_stats.normalize(state_tensor).clamp(-5, 5)  # 添加裁剪防止极端值

                # 从环境获取动作掩码
                action_mask = env.get_action_mask(state)

                # 智能体使用掩码选择动作
                actions, log_prob, _, value = agent.take_action(
                    normalized_state,
                    action_mask=action_mask
                )

                # 环境执行动作
                next_state, reward, done = env.step(state, actions)

                # 收集下一个状态（原始值）
                next_state_values = [next_state[k] for k in state_keys]
                episode_states.append(next_state_values)

                # 计算下一个状态的价值（使用归一化状态）
                next_state_tensor = torch.FloatTensor(next_state_values).unsqueeze(0).to(device)
                normalized_next_state = running_stats.normalize(next_state_tensor).clamp(-5, 5)
                next_value = agent.critic(normalized_next_state).item()

                # 新增：收集详细指标
                # 1. 约束违反检查
                ess_soc = state['ess_state'] / env.ess_capacity
                ev_soc = state['ev_battery_state'] / env.ev_capacity
                episode_ess_socs.append(ess_soc)
                episode_ev_socs.append(ev_soc)
                
                if ess_soc < 0.1 or ess_soc > 0.9:
                    episode_ess_violations += 1
                if ev_soc < 0.1 or ev_soc > 0.9:
                    episode_ev_violations += 1
                
                # 2. 经济指标
                episode_energy_costs.append(env.current_step_cost)
                
                # 3. 可再生能源利用率
                pv_generation = state['pv_generation']
                total_consumption = state['home_load'] + actions.get('ev_power', 0) + actions.get('battery_power', 0)
                if total_consumption > 0:
                    renewable_utilization = min(pv_generation / total_consumption, 1.0)
                else:
                    renewable_utilization = 0.0
                episode_renewable_utilizations.append(renewable_utilization)
                
                # 4. 峰谷电价套利效果（简化计算）
                price = state['electricity_price']
                if price < 0.4:  # 谷电价
                    peak_valley_arbitrage = 1.0 if actions.get('ev_power', 0) > 0 or actions.get('battery_power', 0) > 0 else 0.0
                elif price > 0.6:  # 峰电价
                    peak_valley_arbitrage = 1.0 if actions.get('ev_power', 0) < 0 or actions.get('battery_power', 0) < 0 else 0.0
                else:
                    peak_valley_arbitrage = 0.0
                episode_peak_valley_arbitrages.append(peak_valley_arbitrage)
                
                # 5. 用户满意度（简化计算）
                # 温度舒适度
                indoor_temp = getattr(env, 'indoor_temp', 25)
                user_pref = getattr(env, 'user_temp_preference', 22)
                temp_comfort = max(0, 1 - abs(indoor_temp - user_pref) / 10)
                episode_temperature_comforts.append(temp_comfort)
                
                # 综合用户满意度
                user_satisfaction = temp_comfort * 0.7 + 0.3  # 简化计算
                episode_user_satisfactions.append(user_satisfaction)

                # 存储transition（使用原始状态值）
                batch.append({
                    'state': state,  # 原始状态字典
                    'actions': actions,
                    'rewards': reward,
                    'next_state': next_state,  # 原始状态字典
                    'dones': done,
                    'log_probs': log_prob.detach().squeeze(),
                    'values': value,
                    'next_values': next_value
                })

                episode_return += reward
                state = next_state  # 更新为原始状态
                step_count += 1

                if done:
                    break

            # 更新running_stats：使用整个episode的状态
            if episode_states:
                states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
                running_stats.update(states_tensor)

            # 更新参数
            loss = agent.update(batch, running_stats, state_keys, episode, num_episodes)  # 传入额外的参数
            episode_returns.append(episode_return)

            # 计算episode级别的统计指标
            ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
            ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0
            total_violation_rate = (episode_ess_violations + episode_ev_violations) / step_count if step_count > 0 else 0
            
            energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
            user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
            temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0
            
            ess_soc_mean = np.mean(episode_ess_socs) if episode_ess_socs else 0.5
            ess_soc_std = np.std(episode_ess_socs) if episode_ess_socs else 0
            ev_soc_mean = np.mean(episode_ev_socs) if episode_ev_socs else 0.5
            ev_soc_std = np.std(episode_ev_socs) if episode_ev_socs else 0
            
            renewable_utilization = np.mean(episode_renewable_utilizations) if episode_renewable_utilizations else 0
            peak_valley_arbitrage = np.mean(episode_peak_valley_arbitrages) if episode_peak_valley_arbitrages else 0
            
            # 训练稳定性（使用最近10个episode的回报方差）
            if len(episode_returns) >= 10:
                recent_returns = episode_returns[-10:]
                training_stability = 1.0 / (1.0 + np.std(recent_returns))  # 方差越小，稳定性越高
            else:
                training_stability = 0.0
            
            # 样本效率（简化计算）
            sample_efficiency = episode_return / step_count if step_count > 0 else 0

            # 写入当前episode的详细数据到CSV
            writer.writerow([
                episode + 1, episode_return, loss,
                ess_violation_rate, ev_violation_rate, total_violation_rate,
                energy_cost, user_satisfaction, temperature_comfort,
                ess_soc_mean, ess_soc_std, ev_soc_mean, ev_soc_std,
                renewable_utilization, peak_valley_arbitrage,
                training_stability, sample_efficiency
            ])
            file.flush()  # 确保数据立即写入文件

            print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Loss: {loss:.4f}, "
                  f"Violation: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}")

            # 在每个episode结束时记录总成本
            env.episode_costs.append(env.total_cost)

    # 训练结束后保存成本数据
    env.save_episode_costs()


# ***********************************************************************************
    # 保存训练好的模型
    model_save_dir = "saved_models"
    os.makedirs(model_save_dir, exist_ok=True)
    model_filename = os.path.join(model_save_dir, f"ppo_model_{timestamp}.pth")
    
    # 保存模型状态字典
    torch.save({
        'shared_backbone_state_dict': agent.shared_backbone.state_dict(),
        'actor_branches_state_dict': agent.actor_branches.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optim.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optim.state_dict(),
        'running_stats_mean': running_stats.mean,
        'running_stats_std': running_stats.std,
        'running_stats_count': running_stats.count,
        'action_mapping': agent.action_mapping,
        'state_keys': state_keys,
        'training_config': {
            'state_dim': len(env.state_space),
            'hidden_dim': 128,
            'action_space_config': env.action_space,
            'gamma': agent.gamma,
            'lmbda': agent.lmbda,
            'eps': agent.eps,
            'epochs': agent.epochs,
            'ent_coef': agent.ent_coef,
            'max_grad_norm': agent.max_grad_norm
        }
    }, model_filename)
    
    print(f"模型已保存到: {model_filename}")
# **************************************************************************

    # 训练结束后关闭文件（with语句会自动关闭）
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)

    print(f"训练完成！Returns数据已保存到: {csv_filename}")