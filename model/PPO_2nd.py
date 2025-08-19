"""
独立网络层
"""
import csv
import os
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from environment import HomeEnergyManagementEnv
from model import rl_utils
# 添加evaluation目录到路径（使用相对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'evaluation'))
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


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        
        # 正交初始化
        for layer in self.shared_backbone.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
        nn.init.orthogonal_(self.head.weight)
        self.head.bias.data.zero_()

    def forward(self, x):
        shared_features = self.shared_backbone(x)
        return self.head(shared_features)

# 改进后的价值网络（增加深度和Dropout）
class ValueNet(torch.nn.Module):
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

class MultiAgentPPO:
    def __init__(self, env, state_dim, hidden_dim, action_dim_ev, action_dim_ess, action_dim_wash, action_dim_ac, action_dim_water,
                  epochs, lmbda, eps, gamma, device, constraints, ent_coef, max_grad_norm, constraint_config=None, constraint_mode="none",
                  use_state_normalization=False, use_advantage_normalization=False, use_dynamic_mask=False):
        
        self.lr_actor = 2e-5  # 调整为与PPO3一致
        self.lr_critic = 3e-5  # 调整为与PPO3一致
        self.env = env
        self.actor_ev = PolicyNet(state_dim, hidden_dim, action_dim_ev).to(device)
        self.actor_ess = PolicyNet(state_dim, hidden_dim, action_dim_ess).to(device)
        self.actor_wash = PolicyNet(state_dim, hidden_dim, action_dim_wash).to(device)
        self.actor_ac = PolicyNet(state_dim, hidden_dim, action_dim_ac).to(device)
        self.actor_ac2 = PolicyNet(state_dim, hidden_dim, action_dim_ac).to(device)
        self.actor_water = PolicyNet(state_dim, hidden_dim, action_dim_water).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # 使用AdamW优化器
        self.actor_ev_optimizer = torch.optim.AdamW(self.actor_ev.parameters(), lr=self.lr_actor)
        self.actor_ess_optimizer = torch.optim.AdamW(self.actor_ess.parameters(), lr=self.lr_actor)
        self.actor_wash_optimizer = torch.optim.AdamW(self.actor_wash.parameters(), lr=self.lr_actor)
        self.actor_ac_optimizer = torch.optim.AdamW(self.actor_ac.parameters(), lr=self.lr_actor)
        self.actor_ac2_optimizer = torch.optim.AdamW(self.actor_ac2.parameters(), lr=self.lr_actor)
        self.actor_water_optimizer = torch.optim.AdamW(self.actor_water.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_critic)
        
        # 学习率调度
        self.actor_schedulers = {
            'ev': CosineAnnealingLR(self.actor_ev_optimizer, T_max=1000, eta_min=3e-5),
            'ess': CosineAnnealingLR(self.actor_ess_optimizer, T_max=1000, eta_min=3e-5),
            'wash': CosineAnnealingLR(self.actor_wash_optimizer, T_max=1000, eta_min=3e-5),
            'ac': CosineAnnealingLR(self.actor_ac_optimizer, T_max=1000, eta_min=3e-5),
            'ac2': CosineAnnealingLR(self.actor_ac2_optimizer, T_max=1000, eta_min=3e-5),
            'water': CosineAnnealingLR(self.actor_water_optimizer, T_max=1000, eta_min=3e-5),
        }
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=1000, eta_min=5e-5)
        
        self.constraint_mode = constraint_mode
        if constraint_config is None:
            constraint_config = {
                'lambda_lr': 1e-4,
                'soc_lower': 0.1,
                'soc_upper': 0.9,
            }
        self.constraint_config = constraint_config.copy()
        
        # 添加缺失的属性
        self.ess_capacity = env.ess_capacity
        self.ev_capacity = env.ev_capacity
        self.lambda_ess = torch.tensor(0.0, device=device)
        self.lambda_ev = torch.tensor(0.0, device=device)
        
        # 动态目标熵设置
        self.initial_target_entropy = np.mean([
            1 * np.log(action_dim_ev), 1 * np.log(action_dim_ess), 1 * np.log(action_dim_wash),
            1 * np.log(action_dim_ac), 1 * np.log(action_dim_ac), 1 * np.log(action_dim_water)
        ])
        self.final_target_entropy = self.initial_target_entropy * 0.01
        self.target_entropy = self.initial_target_entropy

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-5)
        
        self.gamma = gamma
        self.epochs = epochs
        self.lmbda = lmbda
        self.eps = eps
        self.device = device
        self.constraints = constraints
        self.lambda_ = torch.tensor([1.0] * len(constraints), device=device)
        self.ent_coef = ent_coef  # 新增熵系数
        self.max_grad_norm = max_grad_norm
        self.ess_charge_from_home_record = []
        self.ess_charge_from_grid_record = []
        self.ess_discharge_to_home_record = []
        self.ess_discharge_to_grid_record = []

        self.use_state_normalization = use_state_normalization
        self.use_advantage_normalization = use_advantage_normalization
        self.use_dynamic_mask = use_dynamic_mask

    def reset(self):
        self.ess_charge_from_home_record = []
        self.ess_charge_from_grid_record = []
        self.ess_discharge_to_home_record = []
        self.ess_discharge_to_grid_record = []

    def take_action(self, state, action_mask=None, normalized_state=None):
        ev_at_home = self.env.is_ev_at_home()
        
        # 使用归一化状态或原始状态
        if normalized_state is not None:
            state_tensor = normalized_state
        else:
            state_values = [state[key] for key in sorted(state.keys()) if isinstance(state[key], (int, float))]
            state_tensor = torch.tensor([state_values], dtype=torch.float).to(self.device)

        # EV 动作选择
        ev_actions = self.actor_ev(state_tensor)
        if action_mask and 'ev_power' in action_mask:
            mask_tensor = torch.tensor(action_mask['ev_power'], dtype=torch.bool, device=ev_actions.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            ev_actions = torch.where(mask_tensor, ev_actions, torch.tensor(-1e9, dtype=ev_actions.dtype, device=ev_actions.device))
        
        probs_ev = F.softmax(ev_actions, dim=1)
        action_dist_ev = torch.distributions.Categorical(probs_ev)
        ev_action_index = action_dist_ev.sample().item()

        if ev_at_home:
            # 获取当前SOC和电价
            current_soc = state['ev_battery_state'] / 24
            price = state['electricity_price']
            action_ev_power = self.env.action_space['ev_power'][ev_action_index]
            # 动态限制充电行为
            if current_soc > 0.95:  # 接近满电时禁止充电
                action_ev_power = min(self.env.action_space['ev_power'][ev_action_index], 0)
            # 动态限制放电行为
            if current_soc < 0.05:
                action_ev_power = max(self.env.action_space['ev_power'][ev_action_index], 0)
        else:
            action_ev_power = 0.0

        # ESS 动作选择
        ess_actions = self.actor_ess(state_tensor)
        if action_mask and 'battery_power' in action_mask:
            mask_tensor = torch.tensor(action_mask['battery_power'], dtype=torch.bool, device=ess_actions.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            ess_actions = torch.where(mask_tensor, ess_actions, torch.tensor(-1e9, dtype=ess_actions.dtype, device=ess_actions.device))
        
        probs_ess = F.softmax(ess_actions, dim=1)
        action_dist_ess = torch.distributions.Categorical(probs_ess)
        ess_action_index = action_dist_ess.sample().item()
        action_ess_power = self.env.action_space['battery_power'][ess_action_index]

        if state['ess_state'] > 23.5:
            action_ess_power = min(action_ess_power, 0)
        if state['ess_state'] < 0.5:
            action_ess_power = max(action_ess_power, 0)

        # 洗衣机动作选择
        wash_actions = self.actor_wash(state_tensor)
        if action_mask and 'wash_machine_schedule' in action_mask:
            mask_tensor = torch.tensor(action_mask['wash_machine_schedule'], dtype=torch.bool, device=wash_actions.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            wash_actions = torch.where(mask_tensor, wash_actions, torch.tensor(-1e9, dtype=wash_actions.dtype, device=wash_actions.device))
        
        probs_wash = F.softmax(wash_actions, dim=1)
        action_dist_wash = torch.distributions.Categorical(probs_wash)
        action_wash_machine = action_dist_wash.sample().item()

        # 空调动作选择
        ac_actions = self.actor_ac(state_tensor)
        if action_mask and 'Air_conditioner_set_temp' in action_mask:
            mask_tensor = torch.tensor(action_mask['Air_conditioner_set_temp'], dtype=torch.bool, device=ac_actions.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            ac_actions = torch.where(mask_tensor, ac_actions, torch.tensor(-1e9, dtype=ac_actions.dtype, device=ac_actions.device))
        
        probs_ac = F.softmax(ac_actions, dim=1)
        action_dist_ac = torch.distributions.Categorical(probs_ac)
        ac_action_index = action_dist_ac.sample().item()
        action_ac_set_temp = self.env.action_space['Air_conditioner_set_temp'][ac_action_index]

        # 空调2动作选择
        ac_actions2 = self.actor_ac2(state_tensor)
        if action_mask and 'Air_conditioner_set_temp2' in action_mask:
            mask_tensor = torch.tensor(action_mask['Air_conditioner_set_temp2'], dtype=torch.bool, device=ac_actions2.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            ac_actions2 = torch.where(mask_tensor, ac_actions2, torch.tensor(-1e9, dtype=ac_actions2.dtype, device=ac_actions2.device))
        
        probs_ac2 = F.softmax(ac_actions2, dim=1)
        action_dist_ac2 = torch.distributions.Categorical(probs_ac2)
        ac_action_index2 = action_dist_ac2.sample().item()
        action_ac_set_temp2 = self.env.action_space['Air_conditioner_set_temp'][ac_action_index2]

        # 热水器动作选择
        water_actions = self.actor_water(state_tensor)
        if action_mask and 'ewh_set_temp' in action_mask:
            mask_tensor = torch.tensor(action_mask['ewh_set_temp'], dtype=torch.bool, device=water_actions.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            water_actions = torch.where(mask_tensor, water_actions, torch.tensor(-1e9, dtype=water_actions.dtype, device=water_actions.device))
        
        probs_water = F.softmax(water_actions, dim=1)
        action_dist_water = torch.distributions.Categorical(probs_water)
        water_action_index = action_dist_water.sample().item()
        action_water_set_temp = self.env.action_space['ewh_set_temp'][water_action_index]

        # 收集所有动作
        actions = {
            'ev_power': action_ev_power,
            'battery_power': action_ess_power,
            'wash_machine_schedule': action_wash_machine,
            'Air_conditioner_set_temp': action_ac_set_temp,
            'Air_conditioner_set_temp2': action_ac_set_temp2,
            'ewh_set_temp': action_water_set_temp
        }

        # 计算总日志概率和熵
        log_probs = [
            action_dist_ev.log_prob(torch.tensor(ev_action_index, device=self.device)),
            action_dist_ess.log_prob(torch.tensor(ess_action_index, device=self.device)),
            action_dist_wash.log_prob(torch.tensor(action_wash_machine, device=self.device)),
            action_dist_ac.log_prob(torch.tensor(ac_action_index, device=self.device)),
            action_dist_ac2.log_prob(torch.tensor(ac_action_index2, device=self.device)),
            action_dist_water.log_prob(torch.tensor(water_action_index, device=self.device))
        ]
        total_log_prob = torch.stack(log_probs).sum()
        
        entropies = [
            action_dist_ev.entropy(),
            action_dist_ess.entropy(),
            action_dist_wash.entropy(),
            action_dist_ac.entropy(),
            action_dist_ac2.entropy(),
            action_dist_water.entropy()
        ]
        total_entropy = torch.stack(entropies).sum()
        
        # 计算价值
        value = self.critic(state_tensor).item()

        return actions, total_log_prob, total_entropy, value

    def update(self, transition_dict, running_stats=None, state_keys=None, episode=None, total_episodes=None, USE_STATE_NORMALIZATION=False):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # 状态归一化处理
        if USE_STATE_NORMALIZATION and running_stats is not None:
            normalized_states = running_stats.normalize(states).clamp(-5, 5)
            normalized_next_states = running_stats.normalize(next_states).clamp(-5, 5)
        else:
            normalized_states = states
            normalized_next_states = next_states
        
        # 计算当前价值和下一个价值
        values = self.critic(normalized_states)
        with torch.no_grad():
            next_values = self.critic(normalized_next_states)
        
        # 计算GAE
        advantages = self.compute_gae2(rewards.squeeze(), values, next_values, dones.squeeze())
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算目标价值
        with torch.no_grad():
            # 确保所有张量都是正确的形状
            rewards_1d = rewards.squeeze(-1)  # 确保是1维
            next_values_1d = next_values.squeeze(-1)  # 确保是1维
            dones_1d = dones.squeeze(-1)  # 确保是1维
            
            td_target = rewards_1d + self.gamma * next_values_1d * (1 - dones_1d)
            # 确保td_target是1维张量
            if td_target.dim() == 2:
                td_target = td_target.squeeze(-1)
        
        # 获取旧的动作概率（只计算一次，用于比率计算）
        with torch.no_grad():
            old_ev_actions = self.actor_ev(states)
            old_ess_actions = self.actor_ess(states)
            old_wash_actions = self.actor_wash(states)
            old_ac_actions = self.actor_ac(states)
            old_ac_actions2 = self.actor_ac2(states)
            old_water_actions = self.actor_water(states)
        
        # 确保动作索引在有效范围内
        ev_indices = torch.clamp(actions[:, 0].long(), 0, old_ev_actions.shape[1] - 1)
        ess_indices = torch.clamp(actions[:, 1].long(), 0, old_ess_actions.shape[1] - 1)
        wash_indices = torch.clamp(actions[:, 2].long(), 0, old_wash_actions.shape[1] - 1)
        ac_indices = torch.clamp(actions[:, 3].long(), 0, old_ac_actions.shape[1] - 1)
        ac_indices2 = torch.clamp(actions[:, 4].long(), 0, old_ac_actions2.shape[1] - 1)
        water_indices = torch.clamp(actions[:, 5].long(), 0, old_water_actions.shape[1] - 1)
        
        log_probs_ev_old = F.log_softmax(old_ev_actions, dim=1).gather(1, ev_indices.unsqueeze(-1))
        log_probs_ess_old = F.log_softmax(old_ess_actions, dim=1).gather(1, ess_indices.unsqueeze(-1))
        log_probs_wash_old = F.log_softmax(old_wash_actions, dim=1).gather(1, wash_indices.unsqueeze(-1))
        log_probs_ac_old = F.log_softmax(old_ac_actions, dim=1).gather(1, ac_indices.unsqueeze(-1))
        log_probs_ac_old2 = F.log_softmax(old_ac_actions2, dim=1).gather(1, ac_indices2.unsqueeze(-1))
        log_probs_water_old = F.log_softmax(old_water_actions, dim=1).gather(1, water_indices.unsqueeze(-1))
        
        # 存储每个epoch的损失用于返回
        epoch_losses = []
        
        for _ in range(self.epochs):
            # 每次epoch重新计算新的动作概率（避免重复使用计算图）
            new_ev_actions = self.actor_ev(normalized_states)
            new_ess_actions = self.actor_ess(normalized_states)
            new_wash_actions = self.actor_wash(normalized_states)
            new_ac_actions = self.actor_ac(normalized_states)
            new_ac_actions2 = self.actor_ac2(normalized_states)
            new_water_actions = self.actor_water(normalized_states)

            # 计算新的日志概率
            log_probs_ev = F.log_softmax(new_ev_actions, dim=1).gather(1, ev_indices.unsqueeze(-1))
            log_probs_ess = F.log_softmax(new_ess_actions, dim=1).gather(1, ess_indices.unsqueeze(-1))
            log_probs_wash = F.log_softmax(new_wash_actions, dim=1).gather(1, wash_indices.unsqueeze(-1))
            log_probs_ac = F.log_softmax(new_ac_actions, dim=1).gather(1, ac_indices.unsqueeze(-1))
            log_probs_ac2 = F.log_softmax(new_ac_actions2, dim=1).gather(1, ac_indices2.unsqueeze(-1))
            log_probs_water = F.log_softmax(new_water_actions, dim=1).gather(1, water_indices.unsqueeze(-1))

            # 计算比率和PPO损失 - 确保advantages被detach
            advantages_detached = advantages.detach()
            ratio_ev = torch.exp(log_probs_ev - log_probs_ev_old.detach())
            surr1_ev = ratio_ev * advantages_detached
            surr2_ev = torch.clamp(ratio_ev, 1 - self.eps, 1 + self.eps) * advantages_detached
            actor_ev_loss = -torch.mean(torch.min(surr1_ev, surr2_ev))

            ratio_ess = torch.exp(log_probs_ess - log_probs_ess_old.detach())
            surr1_ess = ratio_ess * advantages_detached
            surr2_ess = torch.clamp(ratio_ess, 1 - self.eps, 1 + self.eps) * advantages_detached
            actor_ess_loss = -torch.mean(torch.min(surr1_ess, surr2_ess))

            ratio_wash = torch.exp(log_probs_wash - log_probs_wash_old.detach())
            surr1_wash = ratio_wash * advantages_detached
            surr2_wash = torch.clamp(ratio_wash, 1 - self.eps, 1 + self.eps) * advantages_detached
            actor_wash_loss = -torch.mean(torch.min(surr1_wash, surr2_wash))

            ratio_ac = torch.exp(log_probs_ac - log_probs_ac_old.detach())
            surr1_ac = ratio_ac * advantages_detached
            surr2_ac = torch.clamp(ratio_ac, 1 - self.eps, 1 + self.eps) * advantages_detached
            actor_ac_loss = -torch.mean(torch.min(surr1_ac, surr2_ac))

            ratio_ac2 = torch.exp(log_probs_ac2 - log_probs_ac_old2.detach())
            surr1_ac2 = ratio_ac2 * advantages_detached
            surr2_ac2 = torch.clamp(ratio_ac2, 1 - self.eps, 1 + self.eps) * advantages_detached
            actor_ac_loss2 = -torch.mean(torch.min(surr1_ac2, surr2_ac2))

            ratio_water = torch.exp(log_probs_water - log_probs_water_old.detach())
            surr1_water = ratio_water * advantages_detached
            surr2_water = torch.clamp(ratio_water, 1 - self.eps, 1 + self.eps) * advantages_detached
            actor_water_loss = -torch.mean(torch.min(surr1_water, surr2_water))

            # 计算熵
            entropy_ev = torch.distributions.Categorical(logits=new_ev_actions).entropy().mean()
            entropy_ess = torch.distributions.Categorical(logits=new_ess_actions).entropy().mean()
            entropy_wash = torch.distributions.Categorical(logits=new_wash_actions).entropy().mean()
            entropy_ac = torch.distributions.Categorical(logits=new_ac_actions).entropy().mean()
            entropy_ac2 = torch.distributions.Categorical(logits=new_ac_actions2).entropy().mean()
            entropy_water = torch.distributions.Categorical(logits=new_water_actions).entropy().mean()

            total_entropy = (
                    0.2 * entropy_ev + 0.3 * entropy_ess + 0.1 * entropy_wash + 
                    0.1 * entropy_ac + 0.1 * entropy_ac2 + 0.2 * entropy_water
            )

            # 动态调整目标熵
            if episode is not None and total_episodes is not None:
                progress = min(1.0, episode / total_episodes)
                self.target_entropy = self.initial_target_entropy * (1 - progress) + self.final_target_entropy * progress

            # 自动熵调整
            alpha_loss = -(self.log_alpha * (total_entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            alpha = self.log_alpha.exp().detach().clamp(max=1.0)

            # 约束处理
            constraint_loss = torch.tensor(0.0, device=self.device)
            if self.constraint_mode == "lagrangian":
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

                constraint_loss = (
                        self.lambda_ess * ess_violation.mean() +
                        self.lambda_ev * ev_violation.mean()
                )

            # Critic损失 - 彻底修复维度不匹配问题
            critic_values = self.critic(normalized_states)
            
            # 确保td_target和critic_values都是1维张量，并且td_target被detach
            td_target_detached = td_target.detach()
            if td_target_detached.dim() == 2:
                td_target_detached = td_target_detached.squeeze(-1)
            if critic_values.dim() == 2:
                critic_values = critic_values.squeeze(-1)
            
            # 确保两个张量形状完全一致
            assert td_target_detached.shape == critic_values.shape, f"Shape mismatch: td_target {td_target_detached.shape}, critic_values {critic_values.shape}"
            
            critic_loss = F.mse_loss(critic_values, td_target_detached)

            # 总Actor损失
            actor_loss = actor_ev_loss + actor_ess_loss + actor_wash_loss + actor_ac_loss + actor_ac_loss2 + actor_water_loss

            # 总损失
            total_loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    # + 2 * constraint_loss
            )

            # 梯度清零
            self.actor_ev_optimizer.zero_grad()
            self.actor_ess_optimizer.zero_grad()
            self.actor_wash_optimizer.zero_grad()
            self.actor_ac_optimizer.zero_grad()
            self.actor_ac2_optimizer.zero_grad()
            self.actor_water_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            clip_grad_norm_(self.actor_ev.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_ess.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_wash.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_ac.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_ac2.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_water.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # 参数更新
            self.actor_ev_optimizer.step()
            self.actor_ess_optimizer.step()
            self.actor_wash_optimizer.step()
            self.actor_ac_optimizer.step()
            self.actor_ac2_optimizer.step()
            self.actor_water_optimizer.step()
            self.critic_optimizer.step()

            # 记录损失
            epoch_losses.append(total_loss.item())

        # 返回具体的损失值
        if self.constraint_mode == "lagrangian":
            constraint_loss_value = constraint_loss.item()
        else:
            constraint_loss_value = 0.0
        
        return actor_loss.item(), critic_loss.item(), constraint_loss_value, total_loss.item()

    def compute_gae(self, td_delta):
        advantages = []
        advantage = 0
        for delta in reversed(td_delta):
            advantage = self.gamma * self.lmbda * advantage + delta.item()
            advantages.append(advantage)
        advantages.reverse()
        return torch.tensor(advantages, dtype=torch.float).view(-1, 1).to(self.device)

    def compute_gae2(self, rewards, values, next_values, dones):
        batch_size = rewards.size(0)
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

    def compute_current_costs(self, states):
        ess_states = states[:, 2]
        ev_states = states[:, 3]

        ess_violation = torch.relu(0.1 * 24 - ess_states) + torch.relu(ess_states - 0.9 * 24)
        ev_violation = torch.relu(0.1 * 24 - ev_states) + torch.relu(ev_states - 0.9 * 24)

        return torch.stack([ess_violation, ev_violation], dim=1)

    # 在训练过程中可以动态衰减熵系数
    def update_ent_coef(self, progress):
        initial_ent_coef = 0.1
        min_ent_coef = 0.01
        self.ent_coef = max(initial_ent_coef * (1 - progress), min_ent_coef)


# ==================== 训练循环 ====================
if __name__ == "__main__":
    # ==================== 配置参数（全部集中，便于实验和可读性） ====================
    USE_STATE_NORMALIZATION = True  # 状态归一化开关
    USE_ADVANTAGE_NORMALIZATION = True  # 优势归一化开关
    USE_DYNAMIC_MASK = True  # 动态掩码开关
    NUM_EPISODES = 5000  # 训练轮数
    HIDDEN_DIM = 128
    EPOCHS = 4
    LAMBDA = 0.98
    EPS = 0.2
    GAMMA = 0.96
    MAX_GRAD_NORM = 10
    ENT_COEF = 0.1
    
    # 环境初始化
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # 智能体初始化
    state_dim = len(env.state_space)
    action_dim_ev = len(env.action_space['ev_power'])
    action_dim_ess = len(env.action_space['battery_power'])
    action_dim_wash = len(env.action_space['wash_machine_schedule'])
    action_dim_ac = len(env.action_space['Air_conditioner_set_temp'])
    action_dim_water = len(env.action_space['ewh_set_temp'])
    
    agent = MultiAgentPPO(
        env=env,
        state_dim=state_dim,
        hidden_dim=HIDDEN_DIM,
        action_dim_ev=action_dim_ev,
        action_dim_ess=action_dim_ess,
        action_dim_wash=action_dim_wash,
        action_dim_ac=action_dim_ac,
        action_dim_water=action_dim_water,
        epochs=EPOCHS,
        lmbda=LAMBDA,
        eps=EPS,
        gamma=GAMMA,
        device=device,
        constraints=[24.0, 24.0],
        ent_coef=ENT_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        constraint_config={
            'lambda_lr': 1e-4,
            'soc_lower': 0.1,
            'soc_upper': 0.9,
        },
        constraint_mode="none",
        use_state_normalization=USE_STATE_NORMALIZATION,
        use_advantage_normalization=USE_ADVANTAGE_NORMALIZATION,
        use_dynamic_mask=USE_DYNAMIC_MASK
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
        'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],
        'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],
        'wash_machine_schedule': [0, 1, 2, 3, 4, 5, 6],
        'Air_conditioner_set_temp': [16, 18, 20, 22, 24, 26, 28, 30],
        'Air_conditioner_set_temp2': [16, 18, 20, 22, 24, 26, 28, 30],
        'ewh_set_temp': [40, 45, 50, 55, 60, 65, 70]
    }

    # 预热阶段：收集初始数据
    if USE_STATE_NORMALIZATION:
        print("Warming up running_stats...")
        warmup_states = []
        state = env.reset()
        state_keys = sorted(env.state_space.keys())

        for _ in range(1000):
            warmup_states.append([state[k] for k in state_keys])
            actions = {name: np.random.choice(space) for name, space in action_spaces.items()}
            next_state, _, _ = env.step(state, actions)
            state = next_state

        if warmup_states and USE_STATE_NORMALIZATION and running_stats is not None:
            states_tensor = torch.tensor(warmup_states, dtype=torch.float32, device=device)
            running_stats.update(states_tensor)
        if USE_STATE_NORMALIZATION and running_stats is not None:
            print(f"Running stats initialized: mean={running_stats.mean.cpu().numpy()}, std={running_stats.std.cpu().numpy()}")
    else:
        print("跳过状态归一化预热阶段")
        state = env.reset()
        state_keys = sorted(env.state_space.keys())

    num_episodes = NUM_EPISODES
    episode_returns = []

    # 创建结果目录
    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)

    # 创建唯一的文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    csv_filename = os.path.join(results_dir, f"returns_ppo2_{timestamp}{norm_suffix}.csv")

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

        for episode in range(num_episodes):
            state = env.reset()
            batch = []
            episode_return = 0
            episode_states = []
            
            # episode级别的指标收集
            episode_ess_violations = 0
            episode_ev_violations = 0
            episode_ess_socs = []
            episode_ev_socs = []
            episode_energy_costs = []
            episode_user_satisfactions = []
            episode_temperature_comforts = []
            episode_peak_valley_arbitrages = []
            step_count = 0
            
            # 违反类型统计
            episode_ess_mild_violations = 0
            episode_ess_severe_violations = 0
            episode_ev_mild_violations = 0
            episode_ev_severe_violations = 0
            
            # 保留原有的次数统计用于对比
            episode_ess_violations_count = 0
            episode_ev_violations_count = 0

            # 安全边界分析
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
                    normalized_state = running_stats.normalize(state_tensor).clamp(-5, 5)
                else:
                    normalized_state = state_tensor

                # 从环境获取动作掩码
                action_mask = env.get_action_mask(state)

                # 智能体使用掩码选择动作
                actions, log_prob, _, value = agent.take_action(
                    state,
                    action_mask=action_mask,
                    normalized_state=normalized_state
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

                # 收集详细指标
                # 1. 约束违反检查
                ess_soc = state['ess_state'] / env.ess_capacity
                ev_soc = state['ev_battery_state'] / env.ev_capacity
                episode_ess_socs.append(ess_soc)
                episode_ev_socs.append(ev_soc)
                
                def calculate_violation_metrics(soc, lower_bound=0.1, upper_bound=0.9):
                    """计算约束违反指标"""
                    if lower_bound <= soc <= upper_bound:
                        return {
                            'violation_count': 0,
                            'violation_severity': 0.0,
                            'safety_margin': min(soc - lower_bound, upper_bound - soc) / (upper_bound - lower_bound)
                        }
                    
                    if soc < lower_bound:
                        violation_severity = (lower_bound - soc) / lower_bound
                        safety_margin = -violation_severity
                    else:
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
                
                # 安全边界分析
                episode_ess_safety_margins.append(ess_metrics['safety_margin'])
                episode_ev_safety_margins.append(ev_metrics['safety_margin'])
                
                # 2. 经济指标
                episode_energy_costs.append(env.current_step_cost)
                
                # 3. 峰谷电价套利效果
                price = state['electricity_price']
                ev_power = actions.get('ev_power', 0)
                battery_power = actions.get('battery_power', 0)
                
                def calculate_peak_valley_arbitrage(electricity_price, ev_pwr, bat_pwr):
                    """计算峰谷电价套利效果"""
                    valley_threshold = 0.2
                    peak_threshold = 0.8
                    
                    arbitrage_score = 0.0
                    
                    if electricity_price <= valley_threshold:
                        total_charging = max(0, ev_pwr) + max(0, bat_pwr)
                        if total_charging > 0:
                            price_discount = (valley_threshold - electricity_price) / valley_threshold
                            arbitrage_score = total_charging * price_discount * 0.95
                    
                    elif electricity_price >= peak_threshold:
                        total_discharging = abs(min(0, ev_pwr)) + abs(min(0, bat_pwr))
                        if total_discharging > 0:
                            price_premium = (electricity_price - peak_threshold) / peak_threshold
                            arbitrage_score = total_discharging * price_premium * 0.95
                    
                    max_arbitrage_power = 6.6 + 4.4
                    normalized_score = min(1.0, arbitrage_score / max_arbitrage_power)
                    
                    return normalized_score
                
                peak_valley_arbitrage = calculate_peak_valley_arbitrage(price, ev_power, battery_power)
                episode_peak_valley_arbitrages.append(peak_valley_arbitrage)
                
                # 4. 用户满意度
                indoor_temp1 = env.indoor_temp
                indoor_temp2 = env.indoor_temp2
                user_pref1 = env.user_temp_preference
                user_pref2 = env.user_temp_preference2
                
                temp_diff1 = abs(indoor_temp1 - user_pref1)
                temp_diff2 = abs(indoor_temp2 - user_pref2)
                
                temp_comfort1 = max(0, 1 - max(0, temp_diff1 - 2) / 8)
                temp_comfort2 = max(0, 1 - max(0, temp_diff2 - 2) / 8)
                # 新增：每步都记录ac1/ac2/ewh舒适度
                ac1_temp_comforts.append(temp_comfort1)
                ac2_temp_comforts.append(temp_comfort2)
                # 计算热水器舒适度
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
                
                temp_comfort = (temp_comfort1 + temp_comfort2) / 2
                episode_temperature_comforts.append(temp_comfort)
                
                user_satisfaction = temp_comfort * 0.7 + 0.3
                episode_user_satisfactions.append(user_satisfaction)

                # 存储transition
                batch.append({
                    'state': state,
                    'actions': actions,
                    'rewards': reward,
                    'next_state': next_state,
                    'dones': done,
                    'log_probs': log_prob.detach().squeeze(),
                    'values': value,
                    'next_values': next_value
                })

                episode_return += reward
                state = next_state
                step_count += 1

                if done:
                    break

            # 更新running_stats
            if USE_STATE_NORMALIZATION and running_stats is not None and episode_states:
                states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
                running_stats.update(states_tensor)

            # 准备数据用于更新
            batch_data = {
                'states': [t['state'] for t in batch],
                'actions': {
                    name: [t['actions'][name] for t in batch]
                    for name in ['ev_power', 'battery_power', 'wash_machine_schedule', 
                               'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp']
                },
                'rewards': [t['rewards'] for t in batch],
                'next_states': [t['next_state'] for t in batch],
                'dones': [t['dones'] for t in batch],
                'log_probs': [t['log_probs'] for t in batch],
                'values': [t['values'] for t in batch],
                'next_values': [t['next_values'] for t in batch]
            }

            # 状态张量转换
            states = torch.stack([
                torch.FloatTensor([s[key] for key in state_keys])
                for s in batch_data['states']
            ]).to(device)

            # 动作索引转换
            action_indices = []
            for t in batch:
                indices = []
                for name in ['ev_power', 'battery_power', 'wash_machine_schedule', 
                           'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp']:
                    if name in t['actions']:
                        # 找到动作在动作空间中的索引
                        action_space = env.action_space[name]
                        action_value = t['actions'][name]
                        try:
                            idx = action_space.index(action_value)
                        except ValueError:
                            # 如果找不到精确匹配，找到最接近的值
                            idx = min(range(len(action_space)), 
                                    key=lambda i: abs(action_space[i] - action_value))
                        indices.append(idx)
                    else:
                        indices.append(0)
                action_indices.append(indices)

            # 更新transition_dict格式
            transition_dict = {
                'states': states.cpu().numpy(),
                'actions': np.array(action_indices),
                'rewards': np.array(batch_data['rewards']),
                'next_states': torch.stack([
                    torch.FloatTensor([s[key] for key in state_keys])
                    for s in batch_data['next_states']
                ]).cpu().numpy(),
                'dones': np.array(batch_data['dones'])
            }

            # 更新参数
            actor_loss, critic_loss, constraint_loss, total_loss = agent.update(
                transition_dict, running_stats, state_keys, episode, num_episodes, USE_STATE_NORMALIZATION
            )
            episode_returns.append(episode_return)

            # 计算episode级别的统计指标
            ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
            ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0
            total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2 if step_count > 0 else 0
            
            energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
            user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
            temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0
            
            # 计算两个空调和热水器的独立温度舒适度（改为全episode均值）
            ac1_temp_comfort = np.mean(ac1_temp_comforts) if ac1_temp_comforts else 0
            ac2_temp_comfort = np.mean(ac2_temp_comforts) if ac2_temp_comforts else 0
            ewh_temp_comfort = np.mean(ewh_temp_comforts) if ewh_temp_comforts else 0
            
            # 计算热水器温度舒适度
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
            
            ess_soc_mean = np.mean(episode_ess_socs) if episode_ess_socs else 0.5
            ess_soc_std = np.std(episode_ess_socs) if episode_ess_socs else 0
            ev_soc_mean = np.mean(episode_ev_socs) if episode_ev_socs else 0.5
            ev_soc_std = np.std(episode_ev_socs) if episode_ev_socs else 0
            
            # 安全边界分析
            ess_safety_margin_mean = np.mean(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_mean = np.mean(episode_ev_safety_margins) if episode_ev_safety_margins else 0
            ess_safety_margin_std = np.std(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_std = np.std(episode_ev_safety_margins) if episode_ev_safety_margins else 0
            
            peak_valley_arbitrage = np.mean(episode_peak_valley_arbitrages) if episode_peak_valley_arbitrages else 0
            
            # 训练稳定性
            if len(episode_returns) >= 10:
                recent_returns = episode_returns[-10:]
                training_stability = 1.0 / (1.0 + np.std(recent_returns))
            else:
                training_stability = 0.0
            
            # 样本效率
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
                0.0, 0.0, 0.0, # Lambda_ESS, Lambda_EV, Constraint_Weight
                ess_violation_rate, ev_violation_rate # ESS_Violation_Mean, EV_Violation_Mean
            ])
            file.flush()

            print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Actor Loss: {actor_loss:.4f}, "
                  f"Critic Loss: {critic_loss:.4f}, Constraint Loss: {constraint_loss:.4f}, "
                  f"Total Loss: {total_loss:.4f}, Violation Score: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}")
            
            # 添加详细的违反类型分析
            if episode % 10 == 0:
                ess_violation_count_rate = episode_ess_violations_count / step_count if step_count > 0 else 0
                ev_violation_count_rate = episode_ev_violations_count / step_count if step_count > 0 else 0
                total_violation_count_rate = (ess_violation_count_rate + ev_violation_count_rate) / 2
                
                print(f"  ESS SOC: {ess_soc_mean:.3f}±{ess_soc_std:.3f}, "
                      f"EV SOC: {ev_soc_mean:.3f}±{ev_soc_std:.3f}")
                print(f"  Violation Score - ESS: {ess_violation_rate:.3f}, EV: {ev_violation_rate:.3f}")
                print(f"  Violation Count - ESS: {ess_violation_count_rate:.3f}, EV: {ev_violation_count_rate:.3f}")
                print(f"  Total - Score: {total_violation_rate:.3f}, Count: {total_violation_count_rate:.3f}")
                
                print(f"  Safety Margin - ESS: {ess_safety_margin_mean:.3f}±{ess_safety_margin_std:.3f}, "
                      f"EV: {ev_safety_margin_mean:.3f}±{ev_safety_margin_std:.3f}")
                
                if agent.constraint_mode == "lagrangian":
                    print(f"  Lambda ESS: {agent.lambda_ess.item():.3f}, "
                          f"Lambda EV: {agent.lambda_ev.item():.3f}")

            # 在每个episode结束时记录总成本
            env.episode_costs.append(env.total_cost)

    # 训练结束后保存成本数据
    env.save_episode_costs()

    # 保存训练好的模型
    model_save_dir = "model/saved_models"
    os.makedirs(model_save_dir, exist_ok=True)
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    model_filename = os.path.join(model_save_dir, f"ppo2_model_{timestamp}{norm_suffix}.pth")
    
    # 保存模型状态字典
    model_save_dict = {
        'actor_ev_state_dict': agent.actor_ev.state_dict(),
        'actor_ess_state_dict': agent.actor_ess.state_dict(),
        'actor_wash_state_dict': agent.actor_wash.state_dict(),
        'actor_ac_state_dict': agent.actor_ac.state_dict(),
        'actor_ac2_state_dict': agent.actor_ac2.state_dict(),
        'actor_water_state_dict': agent.actor_water.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_ev_optimizer_state_dict': agent.actor_ev_optimizer.state_dict(),
        'actor_ess_optimizer_state_dict': agent.actor_ess_optimizer.state_dict(),
        'actor_wash_optimizer_state_dict': agent.actor_wash_optimizer.state_dict(),
        'actor_ac_optimizer_state_dict': agent.actor_ac_optimizer.state_dict(),
        'actor_ac2_optimizer_state_dict': agent.actor_ac2_optimizer.state_dict(),
        'actor_water_optimizer_state_dict': agent.actor_water_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'state_keys': state_keys,
        'training_config': {
            'state_dim': len(env.state_space),
            'hidden_dim': HIDDEN_DIM,
            'action_space_config': env.action_space,
            'gamma': GAMMA,
            'lmbda': LAMBDA,
            'eps': EPS,
            'epochs': EPOCHS,
            'ent_coef': ENT_COEF,
            'max_grad_norm': MAX_GRAD_NORM,
            'use_state_normalization': USE_STATE_NORMALIZATION,
            'use_advantage_normalization': USE_ADVANTAGE_NORMALIZATION,
            'use_dynamic_mask': USE_DYNAMIC_MASK
        }
    }
    
    # 根据配置决定是否保存running_stats
    if USE_STATE_NORMALIZATION and running_stats is not None:
        model_save_dict.update({
            'running_stats_mean': running_stats.mean,
            'running_stats_std': running_stats.std,
            'running_stats_count': running_stats.count,
        })
    
    torch.save(model_save_dict, model_filename)
    
    print(f"模型已保存到: {model_filename}")

    # 训练结束后关闭文件
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)

    print(f"训练完成！Returns数据已保存到: {csv_filename}")
