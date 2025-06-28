import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils import replay_buffers
from environment import HomeEnergyManagementEnv
from model import rl_utils
from plt import plot_returns
from torch.optim.lr_scheduler import CosineAnnealingLR

# 处理连续动作
# class PolicyNetContinuous(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNetContinuous, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim * 2)
#         self.ln1 = torch.nn.LayerNorm(hidden_dim * 2)
#         self.fc2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
#         self.ln2 = torch.nn.LayerNorm(hidden_dim)
#         self.mean_layer = torch.nn.Linear(hidden_dim, action_dim)
#         # 初始化标准差为较小值
#         self.log_std = torch.nn.Parameter(torch.full((1, action_dim), -0.5))
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         mean = self.mean_layer(x)
#         log_std = self.log_std.expand_as(mean)
#         std = torch.exp(log_std)
#         return mean, std

# 处理离散动作
# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.output = torch.nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         actions = self.output(x)
#         return actions
#
# class ValueNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super(ValueNet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = torch.nn.Linear(hidden_dim, 1)
#         # self.dropout = torch.nn.Dropout(p=0.2)  # 新增Dropout层
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)
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
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MultiAgentPPO:
    def __init__(self, env, state_dim, hidden_dim, action_dim_ev, action_dim_ess, action_dim_wash, action_dim_ac, action_dim_water,
                  epochs, lmbda, eps, gamma, device, constraints,ent_coef, max_grad_norm):
        self.lr_actor = 1e-6
        self.lr_critic = 1e-6
        self.env = env
        self.actor_ev = PolicyNet(state_dim, hidden_dim, action_dim_ev).to(device)
        self.actor_ess = PolicyNet(state_dim, hidden_dim, action_dim_ess).to(device)
        self.actor_wash = PolicyNet(state_dim, hidden_dim, action_dim_wash).to(device)
        self.actor_ac = PolicyNet(state_dim, hidden_dim, action_dim_ac).to(device)
        self.actor_ac2 = PolicyNet(state_dim, hidden_dim, action_dim_ac).to(device)
        self.actor_water = PolicyNet(state_dim, hidden_dim, action_dim_water).to(device)
        # self.actor_water = PolicyNetContinuous(state_dim, hidden_dim, 1).to(device)  # 热水器使用连续策略网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_ev_optimizer = torch.optim.Adam(self.actor_ev.parameters(), lr=self.lr_actor)
        self.actor_ess_optimizer = torch.optim.Adam(self.actor_ess.parameters(), lr=self.lr_actor)
        self.actor_wash_optimizer = torch.optim.Adam(self.actor_wash.parameters(), lr=self.lr_actor)
        self.actor_ac_optimizer = torch.optim.Adam(self.actor_ac.parameters(), lr=self.lr_actor)
        self.actor_ac2_optimizer = torch.optim.Adam(self.actor_ac2.parameters(), lr=self.lr_actor)
        self.actor_water_optimizer = torch.optim.Adam(self.actor_water.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
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

    def reset(self):
        self.ess_charge_from_home_record = []
        self.ess_charge_from_grid_record = []
        self.ess_discharge_to_home_record = []
        self.ess_discharge_to_grid_record = []

    def take_action(self, state):
        ev_at_home = self.env.is_ev_at_home()
        state_values = [state[key] for key in sorted(state.keys()) if isinstance(state[key], (int, float))]
        state_tensor = torch.tensor([state_values], dtype=torch.float).to(self.device)

        # EV 动作选择
        ev_actions = self.actor_ev(state_tensor)
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
        probs_ess = F.softmax(ess_actions, dim=1)
        action_dist_ess = torch.distributions.Categorical(probs_ess)
        ess_action_index = action_dist_ess.sample().item()
        action_ess_power = self.env.action_space['battery_power'][ess_action_index]

        # diff = state['pv_generation'] - env.total_load_compute()
        # if diff > 0:  # 如果光伏供电有盈余，充入ess，此时限制ess动作为充电
        #     action_ess_power = max(self.env.action_space['battery_power'][ess_action_index], 0) + diff
            # action_ess_power = diff
        # else:   # 此时光伏无法满足家庭负载用电，ess向家庭供电以及向电网放电
        #     if state['ess_state'] > -diff and state['electricity_price'] >= 0.5:  # 如果ess电量够用，且此时是高电价时期，ess不充电
        #         action_ess_power = min(self.env.action_space['battery_power'][ess_action_index], 0) + diff
        #
        #     elif state['ess_state'] > -diff and state['electricity_price'] < 0.5:  # 此时ess电量足用，且不在电价高峰期，不做限制
        #         action_ess_power = self.env.action_space['battery_power'][ess_action_index]
        #
        #     elif state['ess_state'] < -diff and state['electricity_price'] > 0.5:  # 此时ess电量不足以供给家庭使用,且在电价高峰期
        #         action_ess_power = 0
        #     else:  # 此时ess电量不足以供给家庭使用 ， ess不放电
        #         action_ess_power = max(self.env.action_space['battery_power'][ess_action_index], 0)

        if state['ess_state'] > 23.5:
            action_ess_power = min(action_ess_power, 0)
        if state['ess_state'] < 0.5:
            action_ess_power = max(action_ess_power, 0)

        # 洗衣机动作选择
        wash_actions = self.actor_wash(state_tensor)
        probs_wash = F.softmax(wash_actions, dim=1)
        action_dist_wash = torch.distributions.Categorical(probs_wash)
        action_wash_machine = action_dist_wash.sample().item()

        # 空调动作选择
        ac_actions = self.actor_ac(state_tensor)
        probs_ac = F.softmax(ac_actions, dim=1)
        action_dist_ac = torch.distributions.Categorical(probs_ac)
        ac_action_index = action_dist_ac.sample().item()
        action_ac_set_temp = self.env.action_space['Air_conditioner_set_temp'][ac_action_index]

        # 第二台空调动作选择
        ac_actions2 = self.actor_ac2(state_tensor)
        probs_ac2 = F.softmax(ac_actions2, dim=1)
        action_dist_ac2 = torch.distributions.Categorical(probs_ac2)
        ac_action_index2 = action_dist_ac2.sample().item()
        action_ac_set_temp2 = self.env.action_space['Air_conditioner_set_temp'][ac_action_index2]

        # 热水器动作选择（离散）
        water_actions = self.actor_water(state_tensor)
        probs_water = F.softmax(water_actions, dim=1)
        action_dist_water = torch.distributions.Categorical(probs_water)
        water_action_index = action_dist_water.sample().item()
        action_water_tem = self.env.action_space['ewh_set_temp'][water_action_index]

        # # 热水器动作选择（连续）
        # water_mean, water_std = self.actor_water(state_tensor)
        # water_dist = torch.distributions.Normal(water_mean, water_std)
        # action_water_power = water_dist.sample().item()
        # action_water_power = torch.clamp(torch.tensor(action_water_power),
        #                                  self.env.action_space['ewh_power'][0],
        #                                  self.env.action_space['ewh_power'][1]).item()

        action = {
            'ev_power': action_ev_power,
            'battery_power': action_ess_power,
            'wash_machine_schedule': action_wash_machine,
            'Air_conditioner_set_temp': action_ac_set_temp,
            'Air_conditioner_set_temp2': action_ac_set_temp2,
            'ewh_set_temp': action_water_tem
        }
        return action

    def update(self, transition_dict):
        states = torch.tensor(
            [[state[key] for key in sorted(state.keys()) if isinstance(state[key], (int, float))] for state in
             transition_dict["states"]], dtype=torch.float).to(self.device)

        actions = torch.tensor(
            [[action[key] for key in sorted(action.keys())] for action in transition_dict["actions"]],
            dtype=torch.float).to(self.device)

        next_states = torch.tensor(
            [[next_state[key] for key in sorted(next_state.keys()) if isinstance(next_state[key], (int, float))] for
             next_state in transition_dict["next_states"]], dtype=torch.float).to(self.device)

        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
        advantages = self.compute_gae(td_delta)
        # advantages = self.compute_gae2(rewards, values, next_values, dones)


        with torch.no_grad():
            old_ev_actions = self.actor_ev(states)
            old_ess_actions = self.actor_ess(states)
            old_wash_actions = self.actor_wash(states)
            old_ac_actions = self.actor_ac(states)
            old_ac_actions2 = self.actor_ac2(states)
            old_water_actions = self.actor_water(states)    # 离散

            # 连续
            # old_water_mean, old_water_std = self.actor_water(states)
            # old_water_dist = torch.distributions.Normal(old_water_mean, old_water_std)
            # old_log_probs_water = old_water_dist.log_prob(actions[:, 4].unsqueeze(-1))

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

        current_costs = self.compute_current_costs(states)

        for _ in range(self.epochs):
            new_ev_actions = self.actor_ev(states)
            new_ess_actions = self.actor_ess(states)
            new_wash_actions = self.actor_wash(states)
            new_ac_actions = self.actor_ac(states)
            new_ac_actions2 = self.actor_ac2(states)
            new_water_actions = self.actor_water(states)

            # 连续
            # new_water_mean, new_water_std = self.actor_water(states)
            # new_water_dist = torch.distributions.Normal(new_water_mean, new_water_std)
            # log_probs_water = new_water_dist.log_prob(actions[:, 4].unsqueeze(-1))

            # 确保动作索引在有效范围内
            ev_indices = torch.clamp(actions[:, 0].long(), 0, new_ev_actions.shape[1] - 1)
            ess_indices = torch.clamp(actions[:, 1].long(), 0, new_ess_actions.shape[1] - 1)
            wash_indices = torch.clamp(actions[:, 2].long(), 0, new_wash_actions.shape[1] - 1)
            ac_indices = torch.clamp(actions[:, 3].long(), 0, new_ac_actions.shape[1] - 1)
            ac_indices2 = torch.clamp(actions[:, 4].long(), 0, new_ac_actions2.shape[1] - 1)
            water_indices = torch.clamp(actions[:, 5].long(), 0, new_water_actions.shape[1] - 1)  # 离散

            log_probs_ev = F.log_softmax(new_ev_actions, dim=1).gather(1, ev_indices.unsqueeze(-1))
            ratio_ev = torch.exp(log_probs_ev - log_probs_ev_old.detach())
            surr1_ev = ratio_ev * advantages
            surr2_ev = torch.clamp(ratio_ev, 1 - self.eps, 1 + self.eps) * advantages
            actor_ev_loss = -torch.mean(torch.min(surr1_ev, surr2_ev))

            log_probs_ess = F.log_softmax(new_ess_actions, dim=1).gather(1, ess_indices.unsqueeze(-1))
            ratio_ess = torch.exp(log_probs_ess - log_probs_ess_old.detach())
            surr1_ess = ratio_ess * advantages
            surr2_ess = torch.clamp(ratio_ess, 1 - self.eps, 1 + self.eps) * advantages
            actor_ess_loss = -torch.mean(torch.min(surr1_ess, surr2_ess))

            log_probs_wash = F.log_softmax(new_wash_actions, dim=1).gather(1, wash_indices.unsqueeze(-1))
            ratio_wash = torch.exp(log_probs_wash - log_probs_wash_old.detach())
            surr1_wash = ratio_wash * advantages
            surr2_wash = torch.clamp(ratio_wash, 1 - self.eps, 1 + self.eps) * advantages
            actor_wash_loss = -torch.mean(torch.min(surr1_wash, surr2_wash))

            log_probs_ac = F.log_softmax(new_ac_actions, dim=1).gather(1, ac_indices.unsqueeze(-1))
            ratio_ac = torch.exp(log_probs_ac - log_probs_ac_old.detach())
            surr1_ac = ratio_ac * advantages
            surr2_ac = torch.clamp(ratio_ac, 1 - self.eps, 1 + self.eps) * advantages
            actor_ac_loss = -torch.mean(torch.min(surr1_ac, surr2_ac))

            log_probs_ac2 = F.log_softmax(new_ac_actions2, dim=1).gather(1, ac_indices2.unsqueeze(-1))
            ratio_ac2 = torch.exp(log_probs_ac2 - log_probs_ac_old2.detach())
            surr1_ac2 = ratio_ac2 * advantages
            surr2_ac2 = torch.clamp(ratio_ac2, 1 - self.eps, 1 + self.eps) * advantages
            actor_ac_loss2 = -torch.mean(torch.min(surr1_ac2, surr2_ac2))

            # 离散
            log_probs_water = F.log_softmax(new_water_actions, dim=1).gather(1, water_indices.unsqueeze(-1))
            ratio_water = torch.exp(log_probs_water - log_probs_water_old.detach())
            surr1_water = ratio_water * advantages
            surr2_water = torch.clamp(ratio_water, 1 - self.eps, 1 + self.eps) * advantages
            actor_water_loss = -torch.mean(torch.min(surr1_water, surr2_water))

            # # 连续
            # ratio_water = torch.exp(log_probs_water - old_log_probs_water.detach())
            # surr1_water = ratio_water * advantages
            # surr2_water = torch.clamp(ratio_water, 1 - self.eps, 1 + self.eps) * advantages
            # actor_water_loss = -torch.mean(torch.min(surr1_water, surr2_water))

            constraint_targets = torch.tensor([
                0.1 * 24,  # ESS允许的边界偏离量
                16 * 0.2  # EV允许的边界偏离量
            ], device=self.device)

            log_barriers = torch.zeros_like(current_costs)
            for i in range(current_costs.shape[1]):
                log_barriers[:, i] = -torch.log(1 - current_costs[:, i] / constraint_targets[i])

            entropy_ev = torch.distributions.Categorical(logits=new_ev_actions).entropy().mean()
            entropy_ess = torch.distributions.Categorical(logits=new_ess_actions).entropy().mean()
            entropy_wash = torch.distributions.Categorical(logits=new_wash_actions).entropy().mean()
            entropy_ac = torch.distributions.Categorical(logits=new_ac_actions).entropy().mean()
            entropy_ac2 = torch.distributions.Categorical(logits=new_ac_actions2).entropy().mean()
            entropy_water = torch.distributions.Categorical(logits=new_water_actions).entropy().mean()

            # 总熵奖励（可根据不同设备调整权重）
            total_entropy = (
                    0.2 * entropy_ev +  # EV动作探索权重
                    0.3 * entropy_ess +  # ESS需要更多探索
                    0.1 * entropy_wash +  # 洗衣机动作简单，减少探索
                    0.1 * entropy_ac +  # 空调
                    0.1 * entropy_ac2 +  # 空调
                    0.2 * entropy_water  # 热水器
            ) * self.ent_coef

            actor_loss = actor_ev_loss + actor_ess_loss + actor_wash_loss + actor_ac_loss+ actor_ac_loss2 + actor_water_loss
            # + torch.mean(log_barriers @ self.lambda_)  # 对数障碍约束
            # - total_entropy)               # 熵

            with torch.no_grad():
                violation_degree = current_costs.mean(dim=0) / constraint_targets
                self.lambda_ *= torch.exp(0.05 * violation_degree)
                self.lambda_ = torch.clamp(self.lambda_, min=1.0, max=1000)

            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_ev_optimizer.zero_grad()
            self.actor_ess_optimizer.zero_grad()
            self.actor_wash_optimizer.zero_grad()
            self.actor_ac_optimizer.zero_grad()
            self.actor_ac2_optimizer.zero_grad()
            self.actor_water_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.actor_ev.parameters(), self.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.actor_ess.parameters(), self.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.actor_wash.parameters(), self.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.actor_ac.parameters(), self.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.actor_ac2.parameters(), self.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.actor_water.parameters(), self.max_grad_norm)
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.actor_ev_optimizer.step()
            self.actor_ess_optimizer.step()
            self.actor_wash_optimizer.step()
            self.actor_ac_optimizer.step()
            self.actor_ac2_optimizer.step()
            self.actor_water_optimizer.step()
            self.critic_optimizer.step()

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


# lr_actor = 1e-6
# lr_critic = 1e-5
num_episodes = 2000
hidden_dim = 256
gamma = 0.99
lmbda = 0.99
epochs = 10
eps = 0.2
constraints = [24.0, 24.0]
ent_coef = 0.1  # 熵系数
max_grad_norm = 0.5  # 梯度裁剪
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# replay_buffer = replay_buffers.ReplayBuffer(capacity=1000000)
env = HomeEnergyManagementEnv()

env.seed(0)
torch.manual_seed(0)
state_dim = len(env.state_space)
action_dim_ev = len(env.action_space['ev_power'])  # EV 动作空间维度
action_dim_ess = len(env.action_space['battery_power'])  # ESS 动作空间维度
action_dim_wash = len(env.action_space['wash_machine_schedule'])  # 洗衣机动做空间维度
action_dim_ac = len(env.action_space['Air_conditioner_set_temp'])  # 空调动作空间维度
action_dim_water = len(env.action_space['ewh_set_temp'])
agent = MultiAgentPPO(env, state_dim, hidden_dim, action_dim_ev, action_dim_ess, action_dim_wash, action_dim_ac,
                      action_dim_water, epochs, lmbda, eps, gamma, device, constraints, ent_coef,max_grad_norm)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

# return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,replay_buffer,minimal_size=100000,batch_size=512)

env.visualize()
env.plot_ewh_analysis()
plot_returns(return_list)