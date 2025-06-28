# import random
# from collections import deque
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from matplotlib import pyplot as plt
#
# from environment import HomeEnergyManagementEnv
#
#
# class DQN(nn.Module):
#     """定义深度Q网络结构"""
#
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
#
#
# class DQNAgent:
#     """DQN算法代理"""
#
#     def __init__(self, env):
#         self.env = env
#         self.state_dim = 9  # 根据环境状态空间确定
#         self.action_dim = len(env.actions)
#
#         # 初始化网络
#         self.model = DQN(self.state_dim, self.action_dim)
#         self.target_model = DQN(self.state_dim, self.action_dim)
#         self.target_model.load_state_dict(self.model.state_dict())
#
#         # 优化器和超参数设置
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.memory = deque(maxlen=10000)
#         self.batch_size = 64
#         self.gamma = 0.95
#         self.epsilon = 1.0
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#
#     def preprocess_state(self, state):
#         """状态预处理：归一化处理"""
#         processed = [
#             state['home_load'] / 10.0,
#             state['pv_generation'] / 5.0,
#             state['ess_state'] / self.env.ess_capacity,
#             state['ev_battery_state'] / self.env.ev_capacity,
#             state['time_index'] / 47.0,  # 假设时间索引是0-47
#             state['electricity_price'] / 2.0,
#             state['temperature'] / 40.0,
#             state['wash_machine_state'],
#             state['Air_conditioner_power'] / 5.0
#         ]
#         return torch.tensor(processed, dtype=torch.float32).numpy()
#
#     def act(self, state, valid_actions):
#         """ε-greedy策略选择动作（修正索引错误版本）"""
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(valid_actions)
#         else:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 q_values = self.model(state_tensor)
#             q_values_np = q_values.numpy().flatten()
#
#             # 获取有效动作的Q值
#             valid_q_values = q_values_np[valid_actions]
#
#             if len(valid_q_values) == 0:
#                 return np.random.choice(valid_actions)
#
#             # 找到最大值索引（在valid_actions中的相对位置）
#             max_q = valid_q_values.max()
#             best_mask = (valid_q_values == max_q)
#             best_indices = np.where(best_mask)[0]  # 在valid_actions中的索引
#
#             # 映射回原始动作索引
#             best_actions = [valid_actions[i] for i in best_indices]
#
#             return np.random.choice(best_actions)
#
#     def remember(self, state, action, reward, next_state, done, next_valid_actions):
#         """存储经验到记忆池"""
#         self.memory.append((state, action, reward, next_state, done, next_valid_actions))
#
#     def replay(self):
#         """经验回放训练（修正维度错误版本）"""
#         if len(self.memory) < self.batch_size:
#             return
#
#         minibatch = random.sample(self.memory, self.batch_size)
#
#         # 数据准备（保持原有代码）
#         states = torch.FloatTensor([x[0] for x in minibatch])
#         actions = torch.LongTensor([x[1] for x in minibatch])
#         rewards = torch.FloatTensor([x[2] for x in minibatch])
#         next_states = torch.FloatTensor([x[3] for x in minibatch])
#         dones = torch.FloatTensor([x[4] for x in minibatch])
#         next_valids = [x[5] for x in minibatch]  # 获取所有next_valids
#
#         current_q = self.model(states).gather(1, actions.unsqueeze(1))
#         next_q = torch.zeros(len(minibatch))  # 使用实际batch大小
#
#         for i in range(len(minibatch)):
#             if dones[i]:
#                 next_q[i] = 0.0
#             else:
#                 with torch.no_grad():
#                     target_q = self.target_model(next_states[i].unsqueeze(0))
#
#                 # 关键修改点1：使用numel()代替len()
#                 valid_actions = next_valids[i]
#                 if not valid_actions:  # 空动作列表处理
#                     next_q[i] = 0.0
#                     continue
#
#                 # 关键修改点2：正确的二维索引方式
#                 valid_q = target_q[0, valid_actions]  # shape: (n_valid_actions,)
#
#                 # 关键修改点3：使用numel判断元素数量
#                 if valid_q.numel() > 0:  # numel()返回张量元素总数
#                     next_q[i] = valid_q.max()
#                 else:
#                     next_q[i] = 0.0
#
#         target = rewards + self.gamma * next_q
#         loss = F.mse_loss(current_q.squeeze(), target)
#
#         # 保持原有优化步骤
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # ε衰减（保持原有代码）
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#     def update_target_model(self):
#         """更新目标网络"""
#         self.target_model.load_state_dict(self.model.state_dict())
#
# def plot_episode_returns(returns, window=50):
#     plt.figure(figsize=(10, 5))
#
#     # 原始回报曲线（淡紫色）
#     plt.plot(returns,
#              color='#E0B0FF',  # 淡紫色
#              alpha=0.3,
#              label='Episode Reward')
#
#     # 滑动平均曲线（深紫色）
#     smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
#     plt.plot(smoothed,
#              color='#6A0DAD',  # 深紫色
#              linewidth=2,
#              label=f'{window}-Episode Average')
#
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.title('Training Progress')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.show()
#
# def train_dqn(env, episodes=500):
#     agent = DQNAgent(env)
#     returns = []
#     # 预验证动作空间一致性
#     # 修正断言：对比实际动作数量与模型输出维度
#     assert len(env.actions) == agent.action_dim, \
#         f"Action space mismatch: {len(env.actions)} vs {agent.action_dim}"
#
#     for episode in range(episodes):
#         state = env.reset()
#         state_processed = agent.preprocess_state(state)
#         total_reward = 0
#         done = False
#
#         while not done:
#             valid_actions = env.get_valid_actions()
#             if not valid_actions:
#                 action_idx = 0  # 安全默认动作
#             else:
#                 action_idx = agent.act(state_processed, valid_actions)
#
#             # 动作索引安全验证
#             if action_idx >= agent.action_dim:
#                 action_idx = action_idx % agent.action_dim
#                 print(f"Corrected invalid action index: {action_idx}")
#
#             action = env.actions[action_idx]
#             next_state, reward, done = env.step(state, action)
#
#             agent.remember(state_processed, action_idx, reward,
#                            agent.preprocess_state(next_state), done,
#                            env.get_valid_actions())
#
#             if len(agent.memory) > agent.batch_size:
#                 agent.replay()
#             total_reward += reward
#         # 定期同步目标网络
#         if episode % 10 == 0:
#             agent.update_target_model()
#
#         returns.append(total_reward)
#         print(f"Episode {episode} completed. Total reward: {total_reward:.2f}")
#
#     env.visualize()
#     plot_episode_returns(returns)
#
# # 环境初始化时需要添加动作生成
# class EnhancedHomeEnv(HomeEnergyManagementEnv):
#     def __init__(self):
#         super().__init__()
#         self.actions = self._generate_all_actions()  # 生成实际动作列表
#
#     def _generate_all_actions(self):
#         """正确生成所有可能的动作组合"""
#         actions = []
#         ev_options = self.action_space['ev_power']
#         battery_options = self.action_space['battery_power']
#         wash_options = self.action_space['wash_machine_schedule']
#         ac_temp_options = self.action_space['Air_conditioner_set_temp']
#
#         for ev_p in ev_options:
#             for bat_p in battery_options:
#                 for wash in wash_options:
#                     for ac_temp in ac_temp_options:
#                         actions.append({
#                             'ev_power': ev_p,
#                             'battery_power': bat_p,
#                             'wash_machine_schedule': wash,
#                             'Air_conditioner_set_temp': ac_temp
#                         })
#         print(f"Generated {len(actions)} actions")  # 调试输出
#         return actions
#
#     def get_valid_actions(self):
#         """修正后的有效动作筛选"""
#         valid_indices = []
#         current_hour = self.current_time_index // 2
#
#         for idx, action in enumerate(self.actions):  # 遍历实际动作列表
#             # EV约束
#             if not self.is_ev_at_home() and action['ev_power'] != 0:
#                 continue
#
#             # 洗衣机约束
#             if action['wash_machine_schedule'] == 1:
#                 if self.wash_machine_used_today or not (10 <= current_hour <= 22):
#                     continue
#
#             # 空调温度有效性
#             if action['Air_conditioner_set_temp'] not in self.air_conditioner_power_map:
#                 continue
#
#             valid_indices.append(idx)
#
#         return valid_indices
# # 训练执行
# if __name__ == "__main__":
#     env = EnhancedHomeEnv()
#     train_dqn(env, episodes=500)



