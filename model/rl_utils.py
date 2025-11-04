from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import collections
import random

from interface import DataInterface
from environment import HomeEnergyManagementEnv



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        transitions = random.sample(self.buffer, batch_size)
        b_s, b_a, b_r, b_ns, b_d = zip(*transitions)
        return np.array(b_s), np.array(b_a), np.array(b_r), np.array(b_ns), np.array(b_d)

    def __len__(self):
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    plot_count = 0  # Initialize plot counter

    # for i in range(10):
        # with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
    for i_episode in range(num_episodes):
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state = env.reset()
                # agent.reset()    # Reset list for recording ESS charge/discharge data sources and destinations
            done = False
            step_count = 0  # For counting steps

            while not done:
                    # env.reward_shape(plot_count / num_episodes)   # Dynamically adjust penalty coefficient
                action = agent.take_action(state)
                next_state, reward, done = env.step(state, action)
                    # agent.update_ent_coef(plot_count / num_episodes)   # Dynamically adaptive entropy
                    # agent.update_lr(plot_count)                  # Learning rate periodic restart
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return = episode_return + reward

            return_list.append(episode_return)
            agent.update(transition_dict)

            plot_count += 1
            if plot_count % 500 == 0:
                env.plot_reward_components()

                # if (i_episode+1) % 10 == 0:
            print(f'Episode {i_episode + 1}, Return: {episode_return:.2f}')
                    # pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                # pbar.update(1)

    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    plot_count = 0  # Initialize plot counter
    # curriculum_wrapper = CurriculumWrapper(env)

    for i in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = False
        ev_battery_episode = []
        ess_state_episode = []
        home_load_episode = []
        pv_generation_episode = []
        electricity_price_episode = []
        ev_at_home_episode = []  # Save EV at home status for current episode
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(state, action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward

            ev_battery_episode.append(state['ev_battery_state'] if env.is_ev_at_home() else None)
            ess_state_episode.append(env.state['ess_state'])
            home_load_episode.append(env.state['home_load'])
            pv_generation_episode.append(env.state['pv_generation'])
            electricity_price_episode.append(env.state['electricity_price'])
            ev_at_home_episode.append(env.ev_at_home_record[-1])

            # Ensure replay buffer has enough samples before starting training
            if len(replay_buffer) > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)

        return_list.append(episode_return)

        # # Update curriculum stage
        # curriculum_wrapper.episode_count += 1  # Automatically manage episode_count inside CurriculumWrapper
        # if curriculum_wrapper.episode_count % 80 == 0:
        #     curriculum_wrapper.curriculum_stage = min(4, curriculum_wrapper.curriculum_stage + 1)
        plot_count += 1
        if plot_count % 500 == 0:
            env.plot_reward_components()

        if (i + 1) % 10 == 0:
            print(f'Episode {i + 1}, Return: {episode_return}')
    return return_list


# def compute_advantage(gamma, lmbda, td_delta):
#     td_delta = td_delta.detach().numpy()
#     advantage_list = []
#     advantage = 0.0
#     for delta in td_delta[::-1]:
#         advantage = gamma * lmbda * advantage + delta
#         advantage_list.append(advantage)
#     advantage_list.reverse()  # reverse() is used to reverse the list
#     return torch.tensor(advantage_list, dtype=torch.float)


def compute_advantage(gamma, lmbda, td_delta):
    advantage_list = []
    advantage = 0
    for delta in reversed(td_delta):
        advantage = delta + gamma * lmbda * advantage
        advantage_list.append(advantage)
    advantage_list.reverse()

    # Convert list to NumPy array, then to Tensor
    # Note: Use .detach() to detach Tensor before conversion
    advantage_array = np.array([a.detach().cpu().numpy() for a in advantage_list])
    advantages = torch.tensor(advantage_array, dtype=torch.float)
    return advantages
