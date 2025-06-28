"""
拉格朗日约束、自适应熵、共享网络层与独立动作头、梯度裁剪、学习率调度、优势函数归一化、层归一化、正交初始化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from environment import HomeEnergyManagementEnv
from plt import plot_returns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                 ent_coef=0.1, max_grad_norm=5, device=device, constraint_config=None):

        # 处理约束配置
        if constraint_config is None:
            constraint_config = {
                'use_constraint': False,  # False表示不使用约束
                'lambda_lr': 1e-4
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
        ], lr=2.5e-4)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=2.5e-4)

        # 学习率调度
        self.actor_scheduler = CosineAnnealingLR(self.actor_optim, T_max=5000)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optim, T_max=5000)

        if self.constraint_config['use_constraint']:
            # 约束相关参数
            self.ess_capacity = env.ess_capacity  # 从环境获取ESS容量
            self.ev_capacity = env.ev_capacity  # 从环境获取EV容量
            self.lambda_ess = torch.tensor(1.0, requires_grad=True, device=device)
            self.lambda_ev = torch.tensor(1.0, requires_grad=True, device=device)
            self.lambda_optim = torch.optim.Adam([self.lambda_ess, self.lambda_ev],
                                                 lr=self.constraint_config['lambda_lr'])

        # # 熵相关参数
        self.target_entropy = np.mean([
            0.5 * np.log(len(actions)) for actions in action_space_config.values()])  # 每个动作分支的熵

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
                masked_logits = torch.where(mask_tensor, logits,
                                            torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
            else:
                masked_logits = logits

            # 创建概率分布
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample()

            # 将索引转换为实际值
            actions[name] = self.action_mapping[name][action_idx.item()]

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

    def update(self, batch_data_list):
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
        state_keys = sorted(batch_data['states'][0].keys())
        states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['states']
        ]).to(device)

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

        for _ in range(self.epochs):
            # ==================== Critic更新 ====================
            critic_values = self.critic(states)
            critic_loss = F.mse_loss(critic_values, rewards + self.gamma * next_values * (1 - dones))

            # ==================== Actor更新 ====================
            shared_features = self.shared_backbone(states)
            new_log_probs = []
            entropies = []

            # 各动作分支的概率计算
            for name, branch in self.actor_branches.items():
                logits = branch(shared_features)
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
                        torch.relu(soc_lower * self.ess_capacity - ess_states) +
                        torch.relu(ess_states - soc_upper * self.ess_capacity)
                )
                ev_violation = (
                        torch.relu(soc_lower * self.ev_capacity - ev_states) +
                        torch.relu(ev_states - soc_upper * self.ev_capacity)
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
                    # - 0.5 * alpha * total_entropy
                    # + constraint_loss  # 根据use_constraint自动为0或实际值
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

            # 参数更新
            self.actor_optim.step()
            self.critic_optim.step()

            # 约束相关更新（条件执行）
            if self.constraint_config['use_constraint']:
                self.lambda_optim.step()
                self.lambda_ess.data.clamp_(min=0)
                self.lambda_ev.data.clamp_(min=0)

            self.actor_scheduler.step()
            self.critic_scheduler.step()

            return total_loss.item()


# ==================== 训练循环 ====================
if __name__ == "__main__":
    # 环境初始化（使用您提供的环境类）
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    agent = HomeEnergyPPO(
        env=env,
        state_dim=len(env.state_space),
        hidden_dim=128,
        action_space_config=env.action_space,
        device=device
    )

    num_episodes = 5000
    episode_returns = []

    for episode in range(num_episodes):
        state = env.reset()
        batch = []
        episode_return = 0
        state_tensor = torch.FloatTensor(list(state.values())).unsqueeze(0).to(device)

        while True:
            # 智能体生成动作（实际值）
            actions, log_prob, _, value = agent.take_action(state_tensor)

            # 动作约束
            if not env.is_ev_at_home():
                actions['ev_power'] = 0.0

            if state['ess_state'] > 23.5:
                actions['battery_power'] = min(actions['battery_power'], 0)
            elif state['ess_state'] < 0.5:
                actions['battery_power'] = max(actions['battery_power'], 0)

            if state['ev_battery_state'] > 23.5:
                actions['ev_power'] = min(actions['ev_power'], 0)
            elif state['ev_battery_state'] < 0.5:
                actions['ev_power'] = max(actions['ev_power'], 0)

            # 环境执行动作
            next_state, reward, done = env.step(state, actions)

            # 计算下一个状态的价值
            next_state_tensor = torch.FloatTensor(list(next_state.values())).unsqueeze(0).to(device)
            next_value = agent.critic(next_state_tensor).item()

            # 存储transition
            batch.append({
                'state': state,
                'actions': actions,  # 存储实际动作值
                'rewards': reward,
                'next_state': next_state,
                'dones': done,
                'log_probs': log_prob.detach().squeeze(),
                'values': value,
                'next_values': next_value
            })

            episode_return += reward
            state = next_state

            if done:
                break

        # 更新参数
        loss = agent.update(batch)
        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Loss: {loss:.4f}")

    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)