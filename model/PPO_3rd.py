"""
在PPO_1st的基础上，实现running状态归一化
"""
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from environment import HomeEnergyManagementEnv
# 添加evaluation目录到路径（使用相对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'evaluation'))
from plt import plot_returns
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import deque  # 新增：用于保存最近50个模型

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
                 ent_coef=0.1, max_grad_norm=10, device=device,
                 constraint_mode="none",  # 'none' or 'lagrangian'
                 use_state_normalization=True,
                 use_advantage_normalization=True,
                 use_dynamic_mask=True,
                 constraint_config=None):
        """
        旗舰版PPO3，支持消融实验开关：
        - constraint_mode: 'none'（无约束）或 'lagrangian'（PPO-Lagrangian）
        - use_state_normalization: 状态归一化开关
        - use_advantage_normalization: 优势函数归一化开关
        - use_dynamic_mask: 动态掩码开关
        """
        self.constraint_mode = constraint_mode
        self.use_state_normalization = use_state_normalization
        self.use_advantage_normalization = use_advantage_normalization
        self.use_dynamic_mask = use_dynamic_mask
        # 处理约束配置
        if constraint_config is None:
            constraint_config = {
                'soc_lower': 0.1,  # 修改下界约束到20%
                'soc_upper': 0.9,  # 修改上界约束到80%
                'lambda_init': 1.0,  # 增加初始值
                'lambda_max': 50.0,  # 增加最大值
                'dual_ascent_rate': 0.1,  # 增加更新率
                'constraint_weight': 5.0,  # 增加早期权重
                'final_constraint_weight': 2.0  # 增加最终权重
            }
        self.constraint_config = constraint_config.copy()

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

        # 优化器 - 降低学习率
        self.actor_optim = torch.optim.AdamW([
            {'params': self.shared_backbone.parameters()},
            {'params': self.actor_branches.parameters()}
        ], lr=1e-5)  # 降低学习率
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=1e-5)  # 降低学习率

        # 学习率调度
        self.actor_scheduler = CosineAnnealingLR(self.actor_optim, T_max=1000, eta_min=3e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optim, T_max=1000, eta_min=5e-5)

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

        # 约束相关参数初始化 - 在所有模式下都初始化容量
        self.ess_capacity = env.ess_capacity
        self.ev_capacity = env.ev_capacity

        if self.constraint_mode == "lagrangian":
            self.lambda_ess = torch.tensor(self.constraint_config.get('lambda_init', 0.1), requires_grad=False, device=device)
            self.lambda_ev = torch.tensor(self.constraint_config.get('lambda_init', 0.1), requires_grad=False, device=device)

    def take_action(self, state_tensor, action_mask=None):
        shared_features = self.shared_backbone(state_tensor)
        actions = {}
        log_probs = []
        entropies = []
        for name, branch in self.actor_branches.items():
            logits = branch(shared_features)
            # 动态掩码开关
            if self.use_dynamic_mask and action_mask and name in action_mask:
                mask_tensor = torch.tensor(action_mask[name], dtype=torch.bool, device=logits.device)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)
                masked_logits = torch.where(mask_tensor, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
            else:
                masked_logits = logits
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample()
            actions[name] = self.action_mapping[name][int(action_idx.item())]
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
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
            'next_values': [t['next_values'] for t in batch_data_list],
            'ess_violation': [t['ess_violation'] for t in batch_data_list],
            'ev_violation': [t['ev_violation'] for t in batch_data_list]
        }

        # 状态张量转换（确保顺序一致）
        states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['states']
        ]).to(device)

        # 使用running_stats归一化状态（如果启用）
        if self.use_state_normalization and running_stats is not None:
            normalized_states = running_stats.normalize(states).clamp(-5, 5)
        else:
            normalized_states = states

        # 归一化下一个状态
        next_states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['next_states']
        ]).to(device)
        if self.use_state_normalization and running_stats is not None:
            normalized_next_states = running_stats.normalize(next_states).clamp(-5, 5)
        else:
            normalized_next_states = next_states

        # 动作索引转换（处理浮点精度）
        action_indices = {}
        for name in self.action_mapping:
            indices = []
            action_values = list(self.action_mapping[name].values())
            for val in batch_data['actions'][name]:
                close_idx = np.where(np.isclose(action_values, val, atol=1e-4))[0]
                if len(close_idx) > 0:
                    indices.append(close_idx[0])
                else:
                    # 找不到完全匹配时，取最接近的
                    idx = int(np.argmin(np.abs(np.array(action_values) - val)))
                    indices.append(idx)
            action_indices[name] = torch.tensor(indices, dtype=torch.long, device=device)

        # 张量转换（显式指定device）
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32, device=device)
        dones = torch.tensor(batch_data['dones'], dtype=torch.float32, device=device)
        old_log_probs = torch.stack([lp.squeeze() for lp in batch_data['log_probs']]).to(device)
        values = torch.tensor(batch_data['values'], dtype=torch.float32, device=device)
        next_values = torch.tensor(batch_data['next_values'], dtype=torch.float32, device=device)
        ess_violations = torch.tensor(batch_data['ess_violation'], dtype=torch.float32, device=device)
        ev_violations = torch.tensor(batch_data['ev_violation'], dtype=torch.float32, device=device)

        # ==================== GAE计算 ====================
        advantages = self.compute_gae(rewards, values, next_values, dones)

        # ==================== 约束计算重构 ====================
        constraint_loss = torch.tensor(0.0, device=device)
        lagrangian_terms = torch.tensor(0.0, device=device)
        ess_violation_mean = 0.0
        ev_violation_mean = 0.0
        constraint_weight = 1.0
        # 新增：步数违反率
        ess_violation_rate = 0.0
        ev_violation_rate = 0.0

        # 获取状态向量中ess_state和ev_battery_state的索引
        ess_index = state_keys.index('ess_state')
        ev_index = state_keys.index('ev_battery_state')
        ess_soc = states[:, ess_index] / self.ess_capacity
        ev_soc = states[:, ev_index] / self.ev_capacity

        soc_lower = self.constraint_config['soc_lower']
        soc_upper = self.constraint_config['soc_upper']

        # 计算约束违反量（始终为正）- 在所有模式下都计算
        ess_violation = torch.relu(soc_lower - ess_soc) + torch.relu(ess_soc - soc_upper)
        ev_violation = torch.relu(soc_lower - ev_soc) + torch.relu(ev_soc - soc_upper)

        # 计算平均违反量（用于监控）- 在所有模式下都计算
        ess_violation_mean = ess_violation.mean().item()
        ev_violation_mean = ev_violation.mean().item()

        # 计算违反率张量（用于优化）
        ess_violation_rate_tensor = (ess_violation > 0).float().mean()
        ev_violation_rate_tensor = (ev_violation > 0).float().mean()
        ess_violation_rate = ess_violation_rate_tensor.item()
        ev_violation_rate = ev_violation_rate_tensor.item()

        if self.constraint_mode == "lagrangian":
            # 约束loss用违反率
            lagrangian_terms = self.lambda_ess * ess_violation_rate_tensor + \
                              self.lambda_ev * ev_violation_rate_tensor
            # 新增：违反量惩罚项
            violation_amount_penalty = (ess_violation.mean() + ev_violation.mean()) * 10.0
            constraint_loss = lagrangian_terms + violation_amount_penalty

            # 计算动态约束权重 - 保持原有权重
            progress = min(1.0, episode / total_episodes)
            constraint_weight = self.constraint_config['constraint_weight'] * (1 - progress) + \
                               self.constraint_config['final_constraint_weight'] * progress
            constraint_loss *= constraint_weight * 100.0

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

            # ==================== 约束更新（在epoch循环内）====================
            if self.constraint_mode == "lagrangian":
                with torch.no_grad():
                    # 动态调整约束强度：基于违反率设定目标
                    progress = min(1.0, episode / total_episodes)
                    if progress < 0.3:
                        constraint_strength = 3.0
                        target_ess_violation_rate = 0.3
                        target_ev_violation_rate = 0.3
                    elif progress < 0.7:
                        constraint_strength = 2.0
                        target_ess_violation_rate = 0.25
                        target_ev_violation_rate = 0.25
                    else:
                        constraint_strength = 1.5
                        target_ess_violation_rate = 0.2
                        target_ev_violation_rate = 0.2
                    # 基于违反率计算误差
                    ess_error = ess_violation_rate - target_ess_violation_rate
                    ev_error = ev_violation_rate - target_ev_violation_rate
                    kp = constraint_strength
                    new_lambda_ess = self.lambda_ess + 0.1 * kp * ess_error
                    new_lambda_ev = self.lambda_ev + 0.1 * kp * ev_error
                    lambda_max = self.constraint_config['lambda_max']
                    self.lambda_ess = torch.clamp(new_lambda_ess, 0, lambda_max)
                    self.lambda_ev = torch.clamp(new_lambda_ev, 0, lambda_max)
                    # 重新计算约束损失（使用更新后的lambda）
                    constraint_loss = self.lambda_ess * ess_violation_rate_tensor + \
                                    self.lambda_ev * ev_violation_rate_tensor
                    constraint_loss *= constraint_weight

            # ==================== 计算总损失 ====================
            if self.constraint_mode == "lagrangian":
                # 大幅增加约束损失权重
                total_loss = actor_loss + 0.1 * critic_loss + 1.0 * constraint_loss
            else:
                total_loss = actor_loss + 0.1 * critic_loss

            # ==================== 统一梯度更新 ====================
            # 梯度清零
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()

            # 反向传播（单次）
            total_loss.backward()

            # 梯度裁剪
            clip_grad_norm_(self.shared_backbone.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_branches.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # 参数更新
            self.actor_optim.step()
            self.critic_optim.step()

            # 记录损失
            epoch_losses.append(total_loss.item())

        # 学习率调度更新
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 返回具体的损失值而不是平均损失
        # 计算最后一次epoch的详细损失（用于监控）
        # 新增：返回步数违反率和平均违反量
        return actor_loss.item(), critic_loss.item(), constraint_loss, total_loss.item(), constraint_weight, ess_violation_rate, ev_violation_rate, ess_violation_mean, ev_violation_mean


# ==================== 训练循环 ====================
if __name__ == "__main__":
    # 环境初始化
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # ==================== 配置参数 ====================
    USE_STATE_NORMALIZATION = True  # 设置为False可以关闭状态归一化
    USE_ADVANTAGE_NORMALIZATION = True  # 新增：优势函数归一化开关
    CONSTRAINT_MODE = "none"  # "none" 或 "lagrangian"
    USE_DYNAMIC_MASK = True  # 动态掩码开关

    # 智能体初始化
    agent = HomeEnergyPPO(
        env=env,
        state_dim=len(env.state_space),
        hidden_dim=128,
        action_space_config=env.action_space,
        device=device,
        constraint_mode=CONSTRAINT_MODE,
        use_state_normalization=USE_STATE_NORMALIZATION,
        use_advantage_normalization=USE_ADVANTAGE_NORMALIZATION,
        use_dynamic_mask=USE_DYNAMIC_MASK,
        constraint_config={
            'soc_lower': 0.1,  # 修改下界约束到20%
            'soc_upper': 0.9,  # 修改上界约束到80%
            'lambda_init': 1.0,  # 增加初始值
            'lambda_max': 50.0,  # 增加最大值
            'dual_ascent_rate': 0.1,  # 增加更新率
            'constraint_weight': 5.0,  # 增加早期权重
            'final_constraint_weight': 2.0  # 增加最终权重
        }
    )

    # 初始化 running_stats - 根据配置决定是否使用
    if USE_STATE_NORMALIZATION:
        running_stats = RunningStats(shape=len(env.state_space))
        print("状态归一化已启用")
    else:
        running_stats = None
        print("状态归一化已禁用")

    # 定义动作空间用于预热
    action_spaces = {
            'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],  # 电动汽车充放电功率范围
            'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],  # 储能电池充电功率范围
            'wash_machine_schedule': [0, 1, 2, 3, 4, 5, 6],  # 洗衣机调度动作
            'Air_conditioner_set_temp': [16, 18, 20, 22, 24, 26, 28, 30],  # 空调设定温度
            'Air_conditioner_set_temp2': [16, 18, 20, 22, 24, 26, 28, 30],
            'ewh_set_temp': [40, 45, 50, 55, 60, 65, 70]  # 离散温度设定动作
    }

    # 预热阶段：收集初始数据
    if USE_STATE_NORMALIZATION:
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
        if warmup_states and USE_STATE_NORMALIZATION and running_stats is not None:
            states_tensor = torch.tensor(warmup_states, dtype=torch.float32, device=device)
            running_stats.update(states_tensor)
        if USE_STATE_NORMALIZATION and running_stats is not None:
            print(f"Running stats initialized: mean={running_stats.mean.cpu().numpy()}, std={running_stats.std.cpu().numpy()}")
    else:
        print("跳过状态归一化预热阶段")
        state = env.reset()
        state_keys = sorted(env.state_space.keys())  # 确保状态键顺序一致

    num_episodes = 5000
    episode_returns = []

    # 创建结果目录
    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)

    # 新增：创建一个长度为50的队列，保存最近50个episode的回报和模型参数
    recent_models = deque(maxlen=50)  # 每次只保留最新的50个

    # 创建唯一的文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    constraint_suffix = "_constrained" if CONSTRAINT_MODE == "lagrangian" else "_unconstrained"
    csv_filename = os.path.join(results_dir, f"returns_ppo_{timestamp}{norm_suffix}{constraint_suffix}.csv")

    # 打开CSV文件用于写入
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行 - 扩展评估指标
        writer.writerow([
            "Episode", "Return", "Actor_Loss", "Critic_Loss", "Constraint_Loss", "Total_Loss",
            "ESS_Violation_Rate", "EV_Violation_Rate", "Total_Violation_Rate",
            "ESS_Violation_Count_Rate", "EV_Violation_Count_Rate", "Total_Violation_Count_Rate",
            "ESS_Safety_Margin_Mean", "ESS_Safety_Margin_Std", "EV_Safety_Margin_Mean", "EV_Safety_Margin_Std",
            "Energy_Cost", "User_Satisfaction", "Temperature_Comfort",
            "AC1_Temp_Comfort", "AC2_Temp_Comfort", "EWH_Temp_Comfort",
            "AC1_Indoor_Temp", "AC2_Indoor_Temp", "EWH_Temp",
            "ESS_SOC_Mean", "ESS_SOC_Std", "EV_SOC_Mean", "EV_SOC_Std",
            "Peak_Valley_Arbitrage",
            "Training_Stability", "Sample_Efficiency",
            "Lambda_ESS", "Lambda_EV", "Constraint_Weight",
            "ESS_Violation_Mean", "EV_Violation_Mean"
        ])

        # ==================== 分阶段训练与鲁棒机制参数 ====================
        rollback_patience = 30  # 连续多少episode回报低于阈值就回滚
        rollback_threshold = 0.7  # 回报低于历史均值70%就触发回滚
        best_model = None
        best_return = -float('inf')
        rollback_counter = 0

        def get_value_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.item(), x.device
            else:
                return float(x), torch.device('cpu')  # 或者你的默认device

        for episode in range(num_episodes):
            # === 动态设置软约束reward shaping参数 ===
            if episode < 1000:
                target_violation_rate = 0.3
            elif episode < 3000:
                target_violation_rate = 0.2
            else:
                target_violation_rate = 0.1
            setattr(env, 'constraint_mode', CONSTRAINT_MODE)
            setattr(env, 'target_violation_rate', target_violation_rate)

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
            episode_peak_valley_arbitrages = []
            step_count = 0

            # 新增：违反类型统计
            episode_ess_mild_violations = 0
            episode_ess_severe_violations = 0
            episode_ev_mild_violations = 0
            episode_ev_severe_violations = 0

            # 新增：保留原有的次数统计用于对比
            episode_ess_violations_count = 0
            episode_ev_violations_count = 0

            # 新增：安全边界分析（Constraint Loss无法提供）
            episode_ess_safety_margins = []
            episode_ev_safety_margins = []

            # 新增：温度舒适度统计
            ac1_temp_comforts = []
            ac2_temp_comforts = []
            ewh_temp_comforts = []

            while True:
                # 收集当前状态（原始值）
                current_state_values = [state[k] for k in state_keys]
                episode_states.append(current_state_values)

                # 归一化当前状态
                state_tensor = torch.FloatTensor(current_state_values).unsqueeze(0).to(device)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_state = running_stats.normalize(state_tensor).clamp(-5, 5)  # 添加裁剪防止极端值
                else:
                    normalized_state = state_tensor  # 不使用归一化

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
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_next_state = running_stats.normalize(next_state_tensor).clamp(-5, 5)
                else:
                    normalized_next_state = next_state_tensor
                next_value = agent.critic(normalized_next_state).item()

                # 新增：收集详细指标
                # 1. 约束违反检查 - 改进版本
                ess_soc = state['ess_state'] / env.ess_capacity
                ev_soc = state['ev_battery_state'] / env.ev_capacity
                episode_ess_socs.append(ess_soc)
                episode_ev_socs.append(ev_soc)

                # 改进的约束违反计算：专注于Constraint Loss无法提供的分析
                def calculate_violation_metrics(soc, lower_bound=None, upper_bound=None):
                    """使用与约束损失相同的边界进行计算"""
                    # 从智能体配置中获取边界，如果没有传递参数
                    if lower_bound is None:
                        lower_bound = agent.constraint_config['soc_lower']
                    if upper_bound is None:
                        upper_bound = agent.constraint_config['soc_upper']

                    if lower_bound <= soc <= upper_bound:
                        return {
                            'violation_count': 0,
                            'violation_severity': 0.0,
                            'safety_margin': min(soc - lower_bound, upper_bound - soc) / (upper_bound - lower_bound)
                        }

                    # 计算违反程度
                    if soc < lower_bound:
                        violation_severity = (lower_bound - soc) / lower_bound
                        safety_margin = -violation_severity
                    else:  # soc > upper_bound
                        violation_severity = (soc - upper_bound) / (1.0 - upper_bound)
                        safety_margin = -violation_severity

                    return {
                        'violation_count': 1,
                        'violation_severity': violation_severity,
                        'safety_margin': safety_margin
                    }

                # 计算违反指标
                ess_metrics = calculate_violation_metrics(ess_soc)
                ev_metrics = calculate_violation_metrics(ev_soc)

                # 累加指标
                episode_ess_violations += ess_metrics['violation_severity']
                episode_ev_violations += ev_metrics['violation_severity']
                episode_ess_violations_count += ess_metrics['violation_count']
                episode_ev_violations_count += ev_metrics['violation_count']

                # 新增：安全边界分析（Constraint Loss无法提供）
                episode_ess_safety_margins.append(ess_metrics['safety_margin'])
                episode_ev_safety_margins.append(ev_metrics['safety_margin'])

                # 2. 经济指标
                episode_energy_costs.append(env.current_step_cost)

                # 3. 峰谷电价套利效果（改进版）
                price = state['electricity_price']
                ev_power = actions.get('ev_power', 0)
                battery_power = actions.get('battery_power', 0)

                def calculate_peak_valley_arbitrage(electricity_price, ev_pwr, bat_pwr):
                    """修正版：只要低谷充电/高峰放电就有套利分数，分数按功率归一化"""
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

                peak_valley_arbitrage = calculate_peak_valley_arbitrage(price, ev_power, battery_power)
                episode_peak_valley_arbitrages.append(peak_valley_arbitrage)

                # 5. 用户满意度（修正计算）
                # 温度舒适度 - 考虑两个空调的不同偏好
                indoor_temp1 = env.indoor_temp  # 第一台空调的室内温度
                indoor_temp2 = env.indoor_temp2  # 第二台空调的室内温度
                user_pref1 = env.user_temp_preference  # 第一台空调偏好22°C
                user_pref2 = env.user_temp_preference2  # 第二台空调偏好18°C

                # 计算两个空调的温度舒适度（±2°C舒适范围）
                temp_diff1 = abs(indoor_temp1 - user_pref1)
                temp_diff2 = abs(indoor_temp2 - user_pref2)
                temp_comfort1 = max(0, 1 - max(0, temp_diff1 - 2) / 8)  # 超出2°C后线性下降
                temp_comfort2 = max(0, 1 - max(0, temp_diff2 - 2) / 8)  # 超出2°C后线性下降

                # 计算热水器温度舒适度（使用episode结束时的状态）
                ewh_temp = env.state['ewh_temp']
                hour = int(env.state['time_index'] // 2)  # 确保hour是整数
                if 6 <= hour <= 9 or 18 <= hour <= 22:  # 用水高峰时段
                    target_temp, low_temp, high_temp = 55, 50, 60
                else:  # 非高峰时段
                    target_temp, low_temp, high_temp = 45, 40, 50
                if low_temp <= ewh_temp <= high_temp:
                    ewh_temp_comfort = 1.0
                else:
                    deviation = max(low_temp - ewh_temp, ewh_temp - high_temp)
                    ewh_temp_comfort = max(0, 1 - deviation / 10)

                # 综合温度舒适度（两个空调+热水器，等权平均）
                overall_comfort = (temp_comfort1 + temp_comfort2 + ewh_temp_comfort) / 3
                episode_temperature_comforts.append(overall_comfort)

                # 综合用户满意度（基于综合舒适度）
                user_satisfaction = overall_comfort * 0.7 + 0.3  # 简化计算
                episode_user_satisfactions.append(user_satisfaction)

                # 计算违反量（用于PPO-Lagrangian梯度）
                soc_lower = agent.constraint_config['soc_lower']
                soc_upper = agent.constraint_config['soc_upper']
                ess_violation = max(0, soc_lower - ess_soc) + max(0, ess_soc - soc_upper)
                ev_violation = max(0, soc_lower - ev_soc) + max(0, ev_soc - soc_upper)

                # 存储transition（使用原始状态值）
                batch.append({
                    'state': state,  # 原始状态字典
                    'actions': actions,
                    'rewards': reward,
                    'next_state': next_state,  # 原始状态字典
                    'dones': done,
                    'log_probs': log_prob.detach().squeeze(),
                    'values': value,
                    'next_values': next_value,
                    'ess_violation': ess_violation,
                    'ev_violation': ev_violation
                })

                episode_return += reward
                state = next_state  # 更新为原始状态
                step_count += 1

                # 新增：在每步循环里，收集每步的舒适度
                ac1_temp_comforts.append(temp_comfort1)
                ac2_temp_comforts.append(temp_comfort2)
                ewh_temp_comforts.append(ewh_temp_comfort)

                if done:
                    break

            # 更新running_stats：使用整个episode的状态
            if USE_STATE_NORMALIZATION and running_stats is not None and episode_states:
                states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
                running_stats.update(states_tensor)

            # 更新参数（获取额外的返回值：constraint_weight、步数违反率、平均违反量）
            actor_loss, critic_loss, constraint_loss, total_loss, constraint_weight, ess_violation_rate, ev_violation_rate, ess_violation_mean, ev_violation_mean = agent.update(
                batch, running_stats, state_keys, episode, num_episodes
            )
            episode_returns.append(episode_return)

            # ========== 动态约束权重调整 ==========
            # 只有在lagrangian模式下才调整约束权重
            if agent.constraint_mode == "lagrangian":
                # 使用更温和的权重策略
                if episode < 1000:
                    constraint_weight = 2.0  # 早期：中等约束权重
                elif episode < 2000:
                    constraint_weight = 1.5  # 中期：降低约束权重
                elif episode < 3000:
                    constraint_weight = 1.0  # 后期：进一步降低约束权重
                else:
                    constraint_weight = 0.5  # 最终：最小约束权重
                agent.constraint_config['constraint_weight'] = constraint_weight
                agent.constraint_config['final_constraint_weight'] = constraint_weight
            else:
                # 在"none"模式下，设置默认值
                constraint_weight = 0.0

            # # ========== 自动回滚机制 ==========
            # if episode > 50:
            #     recent_returns = episode_returns[-50:]
            #     mean_return = np.mean(recent_returns)
            #     if episode_return < rollback_threshold * mean_return:
            #         rollback_counter += 1
            #     else:
            #         rollback_counter = 0
            #     if rollback_counter >= rollback_patience:
            #         print(f"[回滚] Episode {episode+1}: 回报崩溃，回滚到上一个健康模型，lambda/权重减半")
            #         # 回滚模型参数
            #         if best_model is not None:
            #             # 使用类型转换解决OrderedDict问题
            #             from collections import OrderedDict
            #             agent.shared_backbone.load_state_dict(OrderedDict(best_model['shared_backbone']))
            #             agent.actor_branches.load_state_dict(OrderedDict(best_model['actor_branches']))
            #             agent.critic.load_state_dict(OrderedDict(best_model['critic']))
            #             # 只有在lagrangian模式下才回滚lambda值
            #             if agent.constraint_mode == "lagrangian":
            #                 # 简化lambda值处理 - 直接重置为默认值
            #                 agent.lambda_ess = torch.tensor(0.5, device=device)
            #                 agent.lambda_ev = torch.tensor(0.5, device=device)
            #                 agent.constraint_config['constraint_weight'] *= 0.5
            #         rollback_counter = 0
            # # 保存当前最优模型
            # if episode_return > best_return:
            #     best_return = episode_return
            #     best_model = {
            #         'shared_backbone': agent.shared_backbone.state_dict(),
            #         'actor_branches': agent.actor_branches.state_dict(),
            #         'critic': agent.critic.state_dict(),
            #     }
            #     # 只有在lagrangian模式下才保存lambda值到分离的变量
            #     if agent.constraint_mode == "lagrangian":
            #         # 使用分离的变量存储lambda值，避免类型冲突
            #         best_lambda_ess = 0.0
            #         best_lambda_ev = 0.0

            # 计算episode级别的统计指标
            ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
            ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0

            # 修改总违反率计算方式：使用两个设备违反率的平均值，而不是总和
            # 这样可以确保总违反率在[0, 1]范围内，更符合直觉
            total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2 if step_count > 0 else 0

            energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
            user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
            temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0

            # 计算两个空调的独立温度舒适度（改为全episode均值）
            ac1_temp_comfort = np.mean(ac1_temp_comforts) if ac1_temp_comforts else 0
            ac2_temp_comfort = np.mean(ac2_temp_comforts) if ac2_temp_comforts else 0
            ewh_temp_comfort = np.mean(ewh_temp_comforts) if ewh_temp_comforts else 0

            # 计算热水器温度舒适度（使用episode结束时的状态）
            ewh_temp = env.state['ewh_temp']
            hour = int(env.state['time_index'] // 2)  # 确保hour是整数
            if 6 <= hour <= 9 or 18 <= hour <= 22:  # 用水高峰时段
                target_temp, low_temp, high_temp = 55, 50, 60
            else:  # 非高峰时段
                target_temp, low_temp, high_temp = 45, 40, 50

            if low_temp <= ewh_temp <= high_temp:
                ewh_temp_comfort = 1.0
            else:
                deviation = max(low_temp - ewh_temp, ewh_temp - high_temp)
                ewh_temp_comfort = max(0, 1 - deviation / 10)

            ess_soc_mean = np.mean(episode_ess_socs) if episode_ess_socs else 0.5
            ess_soc_std = np.std(episode_ess_socs) if episode_ess_socs else 0
            ev_soc_mean = np.mean(episode_ev_socs) if episode_ev_socs else 0.5
            ev_soc_std = np.std(episode_ev_socs) if episode_ev_socs else 0

            # 新增：安全边界分析（Constraint Loss无法提供）
            ess_safety_margin_mean = np.mean(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_mean = np.mean(episode_ev_safety_margins) if episode_ev_safety_margins else 0
            ess_safety_margin_std = np.std(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_std = np.std(episode_ev_safety_margins) if episode_ev_safety_margins else 0

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
                episode + 1, episode_return, actor_loss, critic_loss, constraint_loss, total_loss,
                ess_violation_rate, ev_violation_rate, total_violation_rate,
                episode_ess_violations_count / step_count if step_count > 0 else 0,
                episode_ev_violations_count / step_count if step_count > 0 else 0,
                (episode_ess_violations_count + episode_ev_violations_count) / (2 * step_count) if step_count > 0 else 0,
                ess_safety_margin_mean, ess_safety_margin_std, ev_safety_margin_mean, ev_safety_margin_std,
                energy_cost, user_satisfaction, temperature_comfort,
                ac1_temp_comfort, ac2_temp_comfort, ewh_temp_comfort,
                env.indoor_temp, env.indoor_temp2, ewh_temp,
                ess_soc_mean, ess_soc_std, ev_soc_mean, ev_soc_std,
                peak_valley_arbitrage,
                training_stability, sample_efficiency,
                float(agent.lambda_ess) if agent.constraint_mode == "lagrangian" else 0.0,
                float(agent.lambda_ev) if agent.constraint_mode == "lagrangian" else 0.0,
                constraint_weight,
                # 新增：保存平均违反量
                ess_violation_mean, ev_violation_mean
            ])
            file.flush()  # 确保数据立即写入文件

            print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Actor Loss: {actor_loss:.4f}, "
                  f"Critic Loss: {critic_loss:.4f}, Constraint Loss: {constraint_loss:.4f}, "
                  f"Total Loss: {total_loss:.4f}, Violation Score: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}, "
                  f"λ_ESS: {float(agent.lambda_ess) if agent.constraint_mode == 'lagrangian' else 0:.2f}, "
                  f"λ_EV: {float(agent.lambda_ev) if agent.constraint_mode == 'lagrangian' else 0:.2f}, "
                  f"ESS Violation Rate: {ess_violation_rate:.3f}, EV Violation Rate: {ev_violation_rate:.3f}, "
                  f"ESS Violation Mean: {ess_violation_mean:.4f}, EV Violation Mean: {ev_violation_mean:.4f}")



            # 在每个episode结束时记录总成本
            env.episode_costs.append(env.total_cost)

            # ========== 保存模拟数据到CSV ==========
            # 只保存最后50个episode的数据
            if episode >= num_episodes - 50:
                data_filename = f"simulation_data_episode_{episode + 1}.csv"
                env.save_simulation_data(data_filename)
                print(f"Episode {episode + 1}: 模拟数据已保存")

            # ========== episode结束，保存当前模型到recent_models队列 ==========
            model_snapshot = {
                'episode': episode,
                'return': episode_return,
                'shared_backbone': agent.shared_backbone.state_dict(),
                'actor_branches': agent.actor_branches.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_optimizer': agent.actor_optim.state_dict(),
                'critic_optimizer': agent.critic_optim.state_dict(),
        'action_mapping': agent.action_mapping,
        'state_keys': state_keys,
                # 补全training_config字段，确保评估脚本兼容
        'training_config': {
                    'state_dim': len(env.state_space),  # 状态维度
                    'hidden_dim': 128,  # 隐藏层维度
                    'action_space_config': env.action_space,  # 动作空间配置
            'gamma': agent.gamma,
            'lmbda': agent.lmbda,
            'eps': agent.eps,
            'epochs': agent.epochs,
            'ent_coef': agent.ent_coef,
            'max_grad_norm': agent.max_grad_norm,
            'use_state_normalization': USE_STATE_NORMALIZATION,
                    'constraint_mode': CONSTRAINT_MODE,
                    'use_dynamic_mask': USE_DYNAMIC_MASK,
                    'constraint_config': agent.constraint_config.copy() if agent.constraint_mode == "lagrangian" else None
        }
    }
            # 如果使用了状态归一化，保存running_stats
    if USE_STATE_NORMALIZATION and running_stats is not None:
            model_snapshot['running_stats_mean'] = running_stats.mean.clone()
            model_snapshot['running_stats_std'] = running_stats.std.clone()
            model_snapshot['running_stats_count'] = running_stats.count
            # 如果是lagrangian模式，保存lambda值
    if CONSTRAINT_MODE == "lagrangian":
            model_snapshot['lambda_ess'] = agent.lambda_ess.clone()
            model_snapshot['lambda_ev'] = agent.lambda_ev.clone()
            # 关键：每个episode都要保存快照，保证recent_models不为空
    recent_models.append(model_snapshot)

    # 训练结束后保存成本数据
    env.save_episode_costs()

    # ========== 训练结束后保存最终模拟数据 ==========
    print("训练完成，保存最终模拟数据...")
    final_data_filename = f"final_simulation_data_episode_{num_episodes}.csv"
    env.save_simulation_data(final_data_filename)
    print(f"最终模拟数据已保存: {final_data_filename}")

    # ========== 生成最终可视化图片 ==========
    print("生成最终可视化图片...")
    env.visualize()
    print("最终可视化图片已生成")

# ***********************************************************************************

    # 保存训练好的模型
    # ==================== 训练结束后，保存最近50个episode中回报最高的模型 ====================
    # 在recent_models队列中找到回报（return）最高的那个
    if recent_models:
        best_model_snapshot = max(recent_models, key=lambda x: x['return'])
        # 重新组织保存字典，便于后续加载
        model_save_dict = {
            'shared_backbone_state_dict': best_model_snapshot['shared_backbone'],
            'actor_branches_state_dict': best_model_snapshot['actor_branches'],
            'critic_state_dict': best_model_snapshot['critic'],
            'actor_optimizer_state_dict': best_model_snapshot['actor_optimizer'],
            'critic_optimizer_state_dict': best_model_snapshot['critic_optimizer'],
            'action_mapping': best_model_snapshot['action_mapping'],
            'state_keys': best_model_snapshot['state_keys'],
            'training_config': best_model_snapshot['training_config']
        }
        if USE_STATE_NORMALIZATION and 'running_stats_mean' in best_model_snapshot:
            model_save_dict.update({
                'running_stats_mean': best_model_snapshot['running_stats_mean'],
                'running_stats_std': best_model_snapshot['running_stats_std'],
                'running_stats_count': best_model_snapshot['running_stats_count'],
        })
        if CONSTRAINT_MODE == "lagrangian" and 'lambda_ess' in best_model_snapshot:
            model_save_dict.update({
                'lambda_ess': best_model_snapshot['lambda_ess'],
                'lambda_ev': best_model_snapshot['lambda_ev'],
                'constraint_config': best_model_snapshot['constraint_config']
        })
        # 保存模型
        model_save_dir = "model/saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
        constraint_suffix = "_constrained" if CONSTRAINT_MODE == "lagrangian" else "_unconstrained"
        model_filename = os.path.join(model_save_dir, f"ppo_model_{timestamp}{norm_suffix}{constraint_suffix}.pth")
        torch.save(model_save_dict, model_filename)
        print(f"已自动选择最近50个episode中回报最高的模型并保存到: {model_filename}")
    else:
        print("警告：没有可保存的模型快照！")

# **************************************************************************

    # 训练结束后关闭文件（with语句会自动关闭）
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)

    print(f"训练完成！Returns数据已保存到: {csv_filename}")