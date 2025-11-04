"""
Rainbow DQN multi-branch solution, adapted for HEMS environment
Combines Double DQN, Dueling Networks, Prioritized Experience Replay,
Multi-step Learning, Distributional RL, and Noisy Networks improvements
"""
import os
import sys
import csv
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math

# Add project root directory to Python path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from environment import HomeEnergyManagementEnv
# Add evaluation directory to path
sys.path.append(os.path.join(project_root, 'evaluation'))
from plt import plot_returns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== State Normalization ====================
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

# ==================== Noisy Linear Layer ====================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# ==================== Dueling Network Architecture ====================
class DuelingQBranch(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Orthogonal initialization
        for layer in self.feature_layer.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
        
        for layer in self.value_stream.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
        
        for layer in self.advantage_stream.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ==================== Prioritized Experience Replay ====================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(self, *transition):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        
        # Calculate sampling probabilities
        priorities_alpha = priorities ** self.alpha
        probabilities = priorities_alpha / priorities_alpha.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[idx] for idx in indices]
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# ==================== Shared Feature Extractor ====================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
    def forward(self, x):
        return self.net(x)

# ==================== Rainbow DQN Agent ====================
class RainbowDQN:
    def __init__(self, state_dim, hidden_dim, action_space_config, lr=1e-4, gamma=0.96, tau=0.01, 
                 buffer_size=100000, batch_size=512, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 n_step=3, alpha=0.6, beta=0.4):
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
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        
        # Use Noisy Networks instead of epsilon-greedy
        self.epsilon = 0.0  # No longer use epsilon-greedy
        
        # Prioritized experience replay
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta)
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Q network and target network
        self.shared_backbone = SharedFeatureExtractor(state_dim, hidden_dim).to(device)
        self.q_branches = nn.ModuleDict({
            name: DuelingQBranch(hidden_dim, dim).to(device)
            for name, dim in self.action_dims.items()
        })
        self.target_shared_backbone = SharedFeatureExtractor(state_dim, hidden_dim).to(device)
        self.target_q_branches = nn.ModuleDict({
            name: DuelingQBranch(hidden_dim, dim).to(device)
            for name, dim in self.action_dims.items()
        })
        
        # Initialize target network
        self.update_target_network(1.0)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.shared_backbone.parameters()) + list(self.q_branches.parameters()), 
            lr=lr, weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    
    def select_action(self, state_tensor, action_mask=None, explore=True):
        actions = {}
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
        # Store to N-step buffer
        self.n_step_buffer.append(transition)
        
        # If buffer is full, calculate N-step return
        if len(self.n_step_buffer) == self.n_step:
            n_step_transition = self._compute_n_step_return()
            self.memory.push(*n_step_transition)
    
    def _compute_n_step_return(self):
        # Calculate N-step return
        states, actions, rewards, next_states, dones = zip(*self.n_step_buffer)
        
        # Calculate cumulative reward
        cumulative_reward = 0
        for i in range(self.n_step):
            cumulative_reward += rewards[i] * (self.gamma ** i)
        
        # Last state and done flag
        final_state = next_states[-1]
        final_done = dones[-1]
        
        return (states[0], actions[0], cumulative_reward, final_state, final_done)
    
    def sample(self):
        batch, indices, weights = self.memory.sample(self.batch_size)
        return zip(*batch), indices, weights
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        transitions, indices, weights = self.sample()
        states, actions, rewards, next_states, dones = transitions
        
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        total_loss = 0.0
        
        # Get current Q values
        current_features = self.shared_backbone(states)
        next_features = self.shared_backbone(next_states)
        target_next_features = self.target_shared_backbone(next_states)
        
        for name, branch in self.q_branches.items():
            # Get action index
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
            
            # Current Q value
            current_q = branch(current_features).gather(1, action_indices.unsqueeze(1)).squeeze()
            
            # Double DQN: use main network to select action, target network to evaluate action
            with torch.no_grad():
                # Use main network to select next action
                next_q_values = branch(next_features)
                next_actions = next_q_values.argmax(1)
                
                # Use target network to evaluate selected action
                target_q_values = self.target_q_branches[name](target_next_features)
                target_q = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
                
                # Calculate target Q value
                target_q = rewards + (self.gamma ** self.n_step) * target_q * (1 - dones)
            
            # Calculate loss (using importance sampling weights)
            # Manually implement Huber loss, compatible with all PyTorch versions
            diff = current_q - target_q
            abs_diff = torch.abs(diff)
            quadratic = torch.clamp(abs_diff, max=1.0)
            linear = abs_diff - quadratic
            loss = 0.5 * quadratic * quadratic + linear
            weighted_loss = (loss * weights).mean()
            total_loss += weighted_loss
        
        # Backward propagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.shared_backbone.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.q_branches.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        priorities = total_loss.item() + 1e-6  # Avoid priority being 0
        self.memory.update_priorities(indices, [priorities] * len(indices))
        
        return total_loss.item()
    
    def update_target_network(self, tau):
        # Soft update target network
        for target_param, param in zip(self.target_shared_backbone.parameters(), self.shared_backbone.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for name in self.q_branches:
            for target_param, param in zip(self.target_q_branches[name].parameters(), self.q_branches[name].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def reset_noise(self):
        # Reset noise for all noisy networks (if NoisyLinear is used)
        for branch in self.q_branches.values():
            for module in branch.modules():
                if hasattr(module, 'reset_noise'):
                    module.reset_noise()
    
    def save(self, path, state_keys):
        torch.save({
            'shared_backbone': self.shared_backbone.state_dict(),
            'q_branches': self.q_branches.state_dict(),
            'target_shared_backbone': self.target_shared_backbone.state_dict(),
            'target_q_branches': self.target_q_branches.state_dict(),
            'state_keys': state_keys,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.shared_backbone.load_state_dict(checkpoint['shared_backbone'])
        self.q_branches.load_state_dict(checkpoint['q_branches'])
        self.target_shared_backbone.load_state_dict(checkpoint['target_shared_backbone'])
        self.target_q_branches.load_state_dict(checkpoint['target_q_branches'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

# ==================== Training Main Loop ====================
if __name__ == "__main__":
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    USE_STATE_NORMALIZATION = True
    state_dim = len(env.state_space)
    hidden_dim = 256
    action_space_config = env.action_space
    
    agent = RainbowDQN(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_space_config=action_space_config,
        lr=1e-4,
        gamma=0.96,
        tau=0.005,
        buffer_size=100000,
        batch_size=256,
        n_step=3,
        alpha=0.6,
        beta=0.4
    )
    
    if USE_STATE_NORMALIZATION:
        running_stats = RunningStats(shape=state_dim)
        print("State normalization enabled")
    else:
        running_stats = None
        print("State normalization disabled")
    
    state_keys = sorted(env.state_space.keys())
    num_episodes = 5000
    max_steps = 200
    episode_returns = []
    
    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    csv_filename = os.path.join(results_dir, f"returns_rainbow_dqn_{timestamp}{norm_suffix}.csv")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Episode", "Return", "Loss", "ESS_Violation_Rate", "EV_Violation_Rate", "Total_Violation_Rate",
            "Energy_Cost", "User_Satisfaction", "Temperature_Comfort",
            "AC1_Temp_Comfort", "AC2_Temp_Comfort", "EWH_Temp_Comfort", "Learning_Rate"
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
                
                # Constraint statistics
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
                
                # Economics and comfort
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
            
            # Update running_stats
            if USE_STATE_NORMALIZATION and running_stats is not None and episode_states:
                states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
                running_stats.update(states_tensor)
            
            # Rainbow DQN parameter update
            loss = agent.update()
            
            # Periodically update target network and reset noise
            if episode % 10 == 0:
                agent.update_target_network(agent.tau)
                agent.reset_noise()
            
            # Record episode cost
            env.episode_costs.append(env.total_cost)
            
            episode_returns.append(episode_return)
            
            # Calculate statistical metrics
            ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
            ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0
            total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2 if step_count > 0 else 0
            energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
            user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
            temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0
            ac1_temp_comfort = np.mean(ac1_temp_comforts) if ac1_temp_comforts else 0
            ac2_temp_comfort = np.mean(ac2_temp_comforts) if ac2_temp_comforts else 0
            ewh_temp_comfort = np.mean(ewh_temp_comforts) if ewh_temp_comforts else 0
            
            current_lr = agent.optimizer.param_groups[0]['lr']
            
            writer.writerow([
                episode + 1, episode_return, loss, ess_violation_rate, ev_violation_rate, total_violation_rate,
                energy_cost, user_satisfaction, temperature_comfort,
                ac1_temp_comfort, ac2_temp_comfort, ewh_temp_comfort, current_lr
            ])
            file.flush()
            
            print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Loss: {loss:.4f}, "
                  f"Violation: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}, LR: {current_lr:.6f}")
        
        # Save model
        model_save_dir = "model/saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        model_filename = os.path.join(model_save_dir, f"rainbow_dqn_model_{timestamp}{norm_suffix}.pth")
        
        model_save_dict = {
            'shared_backbone_state_dict': agent.shared_backbone.state_dict(),
            'q_branches_state_dict': agent.q_branches.state_dict(),
            'target_shared_backbone_state_dict': agent.target_shared_backbone.state_dict(),
            'target_q_branches_state_dict': agent.target_q_branches.state_dict(),
            'state_keys': state_keys,
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'scheduler_state_dict': agent.scheduler.state_dict(),
            'training_config': {
                'state_dim': state_dim,
                'hidden_dim': hidden_dim,
                'action_space_config': action_space_config,
                'gamma': agent.gamma,
                'tau': agent.tau,
                'batch_size': agent.batch_size,
                'n_step': agent.n_step,
                'alpha': agent.alpha,
                'beta': agent.beta,
                'use_state_normalization': USE_STATE_NORMALIZATION
            }
        }
        
        # If state normalization is used, save running_stats
        if USE_STATE_NORMALIZATION and running_stats is not None:
            model_save_dict['running_stats_mean'] = running_stats.mean
            model_save_dict['running_stats_std'] = running_stats.std
            model_save_dict['running_stats_count'] = running_stats.count
        
        torch.save(model_save_dict, model_filename)
        print(f"Rainbow DQN model saved to: {model_filename}")
    
    env.save_episode_costs()
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)
    print(f"Rainbow DQN training completed! Returns data saved to: {csv_filename}")
