import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

from environment import HomeEnergyManagementEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_keys = [  # 按环境状态字典顺序定义键
    'home_load', 'pv_generation', 'ess_state', 'ev_battery_state',
    'time_index', 'electricity_price', 'temperature',
    'wash_machine_state', 'Air_conditioner_power', 'ewh_temp', 'ewh_power'
]

action_keys = [  # 定义动作字典的键
    'ev_power',
    'battery_power',
    'wash_machine_schedule',
    'Air_conditioner_set_temp',
    'ewh_set_temp'
]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dims, action_options):
        super(Actor, self).__init__()
        self.action_options = action_options
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(action_dims))
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
    def __init__(self, state_dim, action_dims, action_options):
        self.action_dims = action_dims
        self.action_options = action_options
        self.action_dim = len(action_dims)

        # 初始化网络
        self.actor = Actor(state_dim, action_dims, action_options).to(device)
        self.actor_target = Actor(state_dim, action_dims, action_options).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, self.action_dim).to(device)
        self.critic2 = Critic(state_dim, self.action_dim).to(device)
        self.critic1_target = Critic(state_dim, self.action_dim).to(device)
        self.critic2_target = Critic(state_dim, self.action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=3e-5
        )

        # 超参数
        self.batch_size = 512
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std = 0.2
        self.noise_clip = 0.5
        self.update_freq = 2
        self.replay_buffer = deque(maxlen=10000000)
        self.return_list = []  # 添加一个列表来记录每个episode的回报

    def _state_to_list(self, state_dict):
        """将状态字典转换为列表"""
        return [state_dict[k] for k in state_keys]

    def get_action(self, state_dict, add_noise=True):
        """修改后的动作获取方法"""
        state = self._state_to_list(state_dict)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            cont_action = self.actor(state_tensor).cpu().numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=cont_action.shape)
            cont_action = np.clip(cont_action + noise, -1, 1)

        # 转换为离散动作
        disc_action = []
        for i, value in enumerate(cont_action):
            options = self.action_options[i]
            closest = min(options, key=lambda x: abs(x - value))
            disc_action.append(closest)

        # 将离散动作列表转换为字典
        action_dict = {key: value for key, value in zip(action_keys, disc_action)}

        # 应用物理约束
        if not env.is_ev_at_home():  # 假设 env 有一个方法 is_ev_at_home()
            action_dict['ev_power'] = 0.0  # EV 不在家时禁止充放电

        ess_state = state_dict['ess_state']
        ev_state = state_dict['ev_battery_state']
        if ess_state > 23.5:
            action_dict['battery_power'] = min(action_dict['battery_power'], 0)  # 只允许放电
        elif ess_state < 0.5:
            action_dict['battery_power'] = max(action_dict['battery_power'], 0)  # 只允许充电

        if ev_state > 23.5:
            action_dict['ev_power'] = min(action_dict['ev_power'], 0)
        elif ev_state < 0.5:
            action_dict['ev_power'] = max(action_dict['ev_power'], 0)
        return action_dict

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从回放缓冲区采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor([self._state_to_list(x[0]) for x in batch]).to(device)
        actions = torch.FloatTensor([list(x[1].values()) for x in batch]).to(device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(device)
        next_states = torch.FloatTensor([self._state_to_list(x[3]) for x in batch]).to(device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(device)

        # 更新Critic
        with torch.no_grad():
            # 生成目标动作并添加噪声
            target_cont_actions = self.actor_target(next_states)
            noise = torch.randn_like(target_cont_actions) * self.noise_std
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            target_cont_actions = torch.clamp(target_cont_actions + noise, -1, 1)

            # 计算目标Q值
            target_Q1 = self.critic1_target(next_states, target_cont_actions)
            target_Q2 = self.critic2_target(next_states, target_cont_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_Q

        # 计算当前Q值
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        # Critic损失
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新Actor和目标网络
        if self.total_steps % self.update_freq == 0:
            # 更新Actor
            cont_actions = self.actor(states)
            actor_loss = -self.critic1(states, cont_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)   # 梯度裁剪
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, env, episodes=1000):
        self.total_steps = 0
        for ep in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done = env.step(state, action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                self.update()

            self.return_list.append(episode_reward)  # 记录每个episode的回报
            print(f"Episode {ep + 1}, Reward: {episode_reward:.2f}")


def plot_returns(return_list, window_size=9):
    """
    绘制原始回报曲线和移动平均回报曲线。

    参数:
        return_list (list): 每个episode的回报列表。
        window_size (int): 移动平均窗口大小。
    """
    episodes_list = list(range(len(return_list)))

    # 计算移动平均回报
    if len(return_list) >= window_size:
        mv_return = np.convolve(return_list, np.ones(window_size)/window_size, mode='valid')
    else:
        mv_return = np.array(return_list)  # 如果数据点少于窗口大小，直接返回原始数据

    # 绘制原始回报曲线和移动平均回报曲线
    plt.figure(figsize=(12, 6))

    # 绘制原始回报曲线，淡紫色
    plt.plot(episodes_list, return_list, label='Original Returns', color='thistle', alpha=0.7)

    # 绘制移动平均回报曲线，稍微深一点的颜色
    if len(mv_return) > 0:
        plt.plot(np.arange(len(mv_return)) + window_size - 1, mv_return,
                 label=f'Moving Average Returns (Window={window_size})', color='purple')

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Original and Moving Average Returns Over Episodes')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    env = HomeEnergyManagementEnv()
    sample_state = env.reset()

    state_dim = len([sample_state[k] for k in state_keys])

    action_options = [
        [-6.6, -3.3, 0, 3.3, 6.6],  # ev_power
        [-4.4, -2.2, 0, 2.2, 4.4],  # battery_power
        [0, 1],  # wash_machine_schedule
        [18, 20, 22, 24, 26, 28, 30],  # Air_conditioner_set_temp
        [40, 45, 50, 55, 60, 65, 70]  # ewh_set_temp
    ]
    action_dims = [len(opt) for opt in action_options]

    agent = TD3(state_dim, action_dims, action_options)
    agent.train(env, episodes=10)

    plot_returns(agent.return_list)
    env.visualize()
    env.plot_ewh_analysis()