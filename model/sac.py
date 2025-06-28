"""
使用了重要性采样的sac
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
from environment import HomeEnergyManagementEnv
import matplotlib.pyplot as plt

def state_to_vector(state):
    """兼容处理字典和向量两种输入格式"""
    if isinstance(state, dict):
        # 原始字典处理逻辑
        return np.array([
            state['home_load'],  # 0-10kW
            state['pv_generation'],
            state['ess_state'],
            state['ev_battery_state'],
            state['time_index'],
            state['electricity_price'],
            state['temperature'],
            state['wash_machine_state'],
            state['Air_conditioner_power'],
            state['Air_conditioner_power2'],
            state['ewh_temp'],
            state['ewh_power']
        ], dtype=np.float32)
    else:
        # 直接返回已向量化的状态（假设输入已经是处理好的向量）
        return np.asarray(state, dtype=np.float32)

class ActionConverter:
    """将连续动作转换为环境需要的混合动作"""

    def __init__(self):
        self.action_map = {
            'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],
            'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],
            'wash_machine': (0, 1, 2, 3, 4, 5, 6),
            'ac_temp': [18, 20, 22, 24, 26, 28, 30],
            'ac_temp2': (18, 20, 22, 24, 26, 28, 30),
            'ewh_temp': [40, 45, 50, 55, 60, 65, 70]
        }

    def _convert_single(self, value, options):
        scaled = (value + 1) / 2  # [-1,1] -> [0,1]
        idx = int(round(scaled * (len(options) - 1)))
        return options[max(0, min(idx, len(options) - 1))]

    def continuous_to_discrete(self, continuous_action):
        return {
            'ev_power': self._convert_single(continuous_action[0], self.action_map['ev_power']),
            'battery_power': self._convert_single(continuous_action[1], self.action_map['battery_power']),
            'wash_machine_schedule': self._convert_single(continuous_action[2], self.action_map['wash_machine']),
            'Air_conditioner_set_temp': self._convert_single(continuous_action[3], self.action_map['ac_temp']),
            'Air_conditioner_set_temp2': self._convert_single(continuous_action[4], self.action_map['ac_temp2']),
            'ewh_set_temp': self._convert_single(continuous_action[5], self.action_map['ewh_temp'])
        }
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_scale = torch.tensor([1.0] * action_dim, device=device)
        self.action_bias = torch.tensor([0.0] * action_dim, device=device)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

class EnhancedSAC:
    def __init__(self, state_dim, action_dim, device,
                 lr=1e-6, gamma=0.995, tau=0.005, alpha=0.1):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_dim, action_dim, device=device).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # self.target_entropy = -action_dim
        # self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        # self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()*0.8
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        # self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr * 0.5)
        self.converter = ActionConverter()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            continuous_action, _ = self.actor.sample(state)
        return self.converter.continuous_to_discrete(continuous_action.cpu().numpy()[0])

    def update_parameters(self, batch):
        # states, actions, next_states, rewards, dones = batch
        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.FloatTensor(actions).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        #
        # # 更新Critic
        # with torch.no_grad():
        #     next_actions, next_log_probs = self.actor.sample(next_states)
        #     target_q1, target_q2 = self.target_critic(next_states, next_actions)
        #     target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
        #     target_value = rewards + (1 - dones) * self.gamma * target_q
        #
        # current_q1, current_q2 = self.critic(states, actions)
        # critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        #
        # self.critic_optim.zero_grad()
        # critic_loss.backward()
        # self.critic_optim.step()
        states, actions, next_states, rewards, dones, indices, weights = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Critic更新（带重要性采样权重）
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        td_loss1 = F.mse_loss(current_q1, target_value, reduction='none')
        td_loss2 = F.mse_loss(current_q2, target_value, reduction='none')
        critic_loss = (weights * td_loss1).mean() + (weights * td_loss2).mean()

        # 计算TD误差用于优先级更新
        with torch.no_grad():
            td_errors = 0.5 * ((current_q1 - target_value).abs() +
                               (current_q2 - target_value).abs())
            td_errors = td_errors.cpu().numpy().flatten()

        # 后续更新步骤保持不变...
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.log_alpha.exp().detach() * log_probs - torch.min(q1_new, q2_new)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 更新温度参数
        # alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        # self.alpha_optim.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optim.step()
        # 在alpha loss计算中加入clip防止突变
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)  # 新增梯度裁剪
        self.alpha_optim.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return td_errors
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            np.array([state_to_vector(s) for s in states]),  # 统一向量化
            np.array(actions),
            np.array([state_to_vector(s) for s in next_states]),
            np.array(rewards),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, device, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha  # 优先级强度（0=均匀采样）
        self.beta = beta  # 重要性采样调整系数
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # 防止零优先级
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0  # 当前写入位置
        self.size = 0  # 当前存储量

    def push(self, state, action, next_state, reward, done):
        # 新样本初始优先级设为当前最大优先级（若为空则设为1）
        max_prio = self.priorities.max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, action, next_state, reward, done))
            self.size += 1
        else:
            self.buffer[self.pos] = (state, action, next_state, reward, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if self.size == 0:
            return None

        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 按概率采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        self.beta = min(self.beta + self.beta_increment, 1.0)  # 逐步增加beta
        weights /= weights.max()  # 归一化

        # 解包样本数据
        states, actions, next_states, rewards, dones = zip(*samples)
        return (
            np.array([state_to_vector(s) for s in states]),
            np.array(actions),
            np.array([state_to_vector(s) for s in next_states]),
            np.array(rewards),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + self.epsilon

    def __len__(self):
        return self.size

def plot_episode_returns(returns, window=50):
    plt.figure(figsize=(10, 5))

    # 原始回报曲线（淡紫色）
    plt.plot(returns,
             color='#E0B0FF',  # 淡紫色
             alpha=0.3,
             label='Episode Reward')

    # 滑动平均曲线（深紫色）
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    plt.plot(smoothed,
             color='#6A0DAD',  # 深紫色
             linewidth=2,
             label=f'{window}-Episode Average')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def train_enhanced_sac(env, episodes=500, batch_size=1024, buffer_size=1e6, min_size = 100000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 9  # 更新后的状态维度
    action_dim = 4  # 连续动作维度

    agent = EnhancedSAC(state_dim, action_dim, device)
    # buffer = ReplayBuffer(int(buffer_size), device)
    buffer = PrioritizedReplayBuffer(int(buffer_size), device)  # 改用优先缓存
    returns = []

    for episode in range(episodes):
        # 动态调整探索率
        # exploration_factor = max(0.1, 1 - episode / 500)
        state = env.reset()
        # state = torch.FloatTensor(state_to_vector(env.reset())).to(device)
        episode_reward = 0
        done = False

        while not done:
            # 生成并执行动作
            action = agent.select_action(state_to_vector(state))

            # 应用物理约束
            if not env.is_ev_at_home():
                action['ev_power'] = 0.0  # EV不在家时禁止充放电

            if state['ess_state'] > 23.5:
                action['battery_power'] = min(action['battery_power'], 0)
            elif state['ess_state'] < 0.5:
                action['battery_power'] = max(action['battery_power'], 0)

            if state['ev_battery_state'] > 23.5:
                action['ev_power'] = min(action['ev_power'], 0)
            elif state['ev_battery_state'] < 0.5:
                action['ev_power'] = max(action['ev_power'], 0)
            next_state, reward, done = env.step(state, action)

            # 存储转换过程
            buffer.push(
                state_to_vector(state),
                np.array([
                    action['ev_power'],
                    action['battery_power'],
                    action['wash_machine_schedule'],
                    action['Air_conditioner_set_temp'],
                    action['Air_conditioner_set_temp2'],
                    action['ewh_set_temp']
                ]),
                state_to_vector(next_state),
                reward,
                done
            )

            # 更新网络参数
            # if len(buffer) > batch_size:
            #     agent.update_parameters(buffer.sample(batch_size))
            if len(buffer) > min_size:
                batch = buffer.sample(batch_size)
                if batch is not None:
                    td_errors = agent.update_parameters(batch)
                    states_idx = batch[5]  # 获取样本索引
                    buffer.update_priorities(states_idx, td_errors)  # 更新优先级

            episode_reward += reward
            state = next_state

        returns.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes} | Reward: {episode_reward:.1f}")

        # 每50轮可视化一次
        if (episode + 1) % 100 == 0:
            env.visualize()
            plot_episode_returns(returns)

    # 最终可视化
    env.visualize()
    plot_episode_returns(returns)
    return returns


if __name__ == "__main__":
    env = HomeEnergyManagementEnv()
    returns = train_enhanced_sac(env, episodes=5000)