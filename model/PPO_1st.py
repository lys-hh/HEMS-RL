"""
Lagrangian constraints, adaptive entropy, shared network layers with independent action heads,
gradient clipping, learning rate scheduling, advantage function normalization, layer normalization, orthogonal initialization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.clip_grad import clip_grad_norm_

from environment import HomeEnergyManagementEnv
# Add evaluation directory to path (using relative path)
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'evaluation'))
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
        # Orthogonal initialization
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
        # Orthogonal initialization
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
        # Orthogonal initialization
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Key modification: squeeze last dimension


# ==================== PPO Agent ====================
class HomeEnergyPPO:
    def __init__(self, env, state_dim, hidden_dim, action_space_config,
                 gamma=0.96, lmbda=0.98, eps=0.2, epochs=4,
                 ent_coef=0.1, max_grad_norm=5, device=device, constraint_config=None, constraint_mode="none"):

        # Create action mapping table
        self.action_mapping = {
            name: {idx: val for idx, val in enumerate(values)}
            for name, values in action_space_config.items()
        }

        # Number of discrete actions for each device
        self.action_dims = {
            name: len(values)
            for name, values in action_space_config.items()
        }

        # Network initialization
        self.shared_backbone = SharedFeatureExtractor(state_dim, hidden_dim).to(device)
        self.actor_branches = nn.ModuleDict({
            name: ActionBranch(hidden_dim, dim).to(device)
            for name, dim in self.action_dims.items()
        })
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # Optimizer
        self.actor_optim = torch.optim.AdamW([
            {'params': self.shared_backbone.parameters()},
            {'params': self.actor_branches.parameters()}
        ], lr=2.5e-4)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=2.5e-4)

        # Learning rate scheduling
        self.actor_scheduler = CosineAnnealingLR(self.actor_optim, T_max=5000)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optim, T_max=5000)

        self.constraint_mode = constraint_mode
        if constraint_config is None:
            constraint_config = {
                'lambda_lr': 1e-4
            }
        self.constraint_config = constraint_config.copy()

        # # Entropy related parameters
        self.target_entropy = np.mean([
            0.5 * np.log(len(actions)) for actions in action_space_config.values()])  # Entropy for each action branch

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-5)

        # Algorithm parameters
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        if self.constraint_mode == "lagrangian":
            self.ess_capacity = env.ess_capacity
            self.ev_capacity = env.ev_capacity
            self.lambda_ess = torch.tensor(self.constraint_config.get('lambda_init', 1.0), requires_grad=False, device=device)
            self.lambda_ev = torch.tensor(self.constraint_config.get('lambda_init', 1.0), requires_grad=False, device=device)

    def take_action(self, state_tensor, action_mask=None):
        shared_features = self.shared_backbone(state_tensor)

        actions = {}
        log_probs = []
        entropies = []

        for name, branch in self.actor_branches.items():
            logits = branch(shared_features)

            # Apply action mask (if provided)
            if action_mask and name in action_mask:
                # Convert to tensor and ensure correct shape
                mask_tensor = torch.tensor(action_mask[name],
                                           dtype=torch.bool,
                                           device=logits.device)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

                # Apply mask
                masked_logits = torch.where(mask_tensor, logits,
                                            torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
            else:
                masked_logits = logits

            # Create probability distribution
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample()

            # Convert indices to actual values
            actions[name] = self.action_mapping[name][action_idx.item()]

            # Collect log probabilities and entropy
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
        # ==================== Data Preparation ====================
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

        # State tensor conversion (ensure consistent order)
        state_keys = sorted(batch_data['states'][0].keys())
        states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['states']
        ]).to(device)

        # Action index conversion (handle floating point precision)
        action_indices = {}
        for name in self.action_mapping:
            indices = [
                np.where(np.isclose(list(self.action_mapping[name].values()), val, atol=1e-4))[0][0]
                for val in batch_data['actions'][name]
            ]
            action_indices[name] = torch.tensor(indices, dtype=torch.long, device=device)

        # Tensor conversion (explicitly specify device)
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32, device=device)
        dones = torch.tensor(batch_data['dones'], dtype=torch.float32, device=device)
        old_log_probs = torch.stack([lp.squeeze() for lp in batch_data['log_probs']]).to(device)
        values = torch.tensor(batch_data['values'], dtype=torch.float32, device=device)
        next_values = torch.tensor(batch_data['next_values'], dtype=torch.float32, device=device)

        # ==================== GAE Calculation ====================
        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            # ==================== Critic Update ====================
            critic_values = self.critic(states)
            critic_loss = F.mse_loss(critic_values, rewards + self.gamma * next_values * (1 - dones))

            # ==================== Actor Update ====================
            shared_features = self.shared_backbone(states)
            new_log_probs = []
            entropies = []

            # Probability calculation for each action branch
            for name, branch in self.actor_branches.items():
                logits = branch(shared_features)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(action_indices[name]))
                entropies.append(dist.entropy())

            total_entropy = torch.stack(entropies).mean(dim=0).mean()
            new_log_probs = torch.stack(new_log_probs).sum(dim=0).squeeze()

            # Dimension consistency check
            assert new_log_probs.shape == old_log_probs.shape, \
                f"Dimension mismatch: new_log_probs {new_log_probs.shape} vs old_log_probs {old_log_probs.shape}"

            # PPO core loss calculation
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # ==================== Automatic Entropy Adjustment ====================
            alpha_loss = -(self.log_alpha * (total_entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)  # Retain computation graph for subsequent gradients
            self.alpha_optim.step()
            alpha = self.log_alpha.exp().detach().clamp(max=1.0)

            # ==================== Constraint Handling (Conditional Execution) ====================
            constraint_loss = torch.tensor(0.0, device=device)
            if self.constraint_mode == "lagrangian":
                # Constraint violation calculation
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

                # Constraint loss calculation
                constraint_loss = (
                        self.lambda_ess * ess_violation.mean() +
                        self.lambda_ev * ev_violation.mean()
                )

            # ==================== Total Loss Calculation ====================
            total_loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    # - 0.5 * alpha * total_entropy
                    # + constraint_loss  # Automatically 0 or actual value based on use_constraint
            )

            # ==================== Unified Gradient Update ====================
            # Zero gradients
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()

            # Backward propagation (single pass)
            total_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.shared_backbone.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_branches.parameters(), self.max_grad_norm)

            # Parameter update
            self.actor_optim.step()
            self.critic_optim.step()

            self.actor_scheduler.step()
            self.critic_scheduler.step()

            return total_loss.item()


# ==================== Training Loop ====================
if __name__ == "__main__":
    # Environment initialization (using provided environment class)
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
            # Agent generates action (actual value)
            actions, log_prob, _, value = agent.take_action(state_tensor)

            # Action constraints
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

            # Environment executes action
            next_state, reward, done = env.step(state, actions)

            # Calculate value of next state
            next_state_tensor = torch.FloatTensor(list(next_state.values())).unsqueeze(0).to(device)
            next_value = agent.critic(next_state_tensor).item()

            # Store transition
            batch.append({
                'state': state,
                'actions': actions,  # Store actual action values
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

        # Update parameters
        loss = agent.update(batch)
        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Loss: {loss:.4f}")

    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)