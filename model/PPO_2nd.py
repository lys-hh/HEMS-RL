"""
Independent network layers
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
# Add evaluation directory to path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'evaluation'))
from plt import plot_returns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RunningStats:
    def __init__(self, shape):
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 1e-4  # Prevent division by zero

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
        
        # Orthogonal initialization
        for layer in self.shared_backbone.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
        nn.init.orthogonal_(self.head.weight)
        self.head.bias.data.zero_()

    def forward(self, x):
        shared_features = self.shared_backbone(x)
        return self.head(shared_features)

# Improved value network (increased depth and Dropout)
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
        
        # Orthogonal initialization
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Key modification: squeeze last dimension

class MultiAgentPPO:
    def __init__(self, env, state_dim, hidden_dim, action_dim_ev, action_dim_ess, action_dim_wash, action_dim_ac, action_dim_water,
                  epochs, lmbda, eps, gamma, device, constraints, ent_coef, max_grad_norm, constraint_config=None, constraint_mode="none",
                  use_state_normalization=False, use_advantage_normalization=False, use_dynamic_mask=False):
        
        self.lr_actor = 2e-5  # Adjusted to match PPO3
        self.lr_critic = 3e-5  # Adjusted to match PPO3
        self.env = env
        self.actor_ev = PolicyNet(state_dim, hidden_dim, action_dim_ev).to(device)
        self.actor_ess = PolicyNet(state_dim, hidden_dim, action_dim_ess).to(device)
        self.actor_wash = PolicyNet(state_dim, hidden_dim, action_dim_wash).to(device)
        self.actor_ac = PolicyNet(state_dim, hidden_dim, action_dim_ac).to(device)
        self.actor_ac2 = PolicyNet(state_dim, hidden_dim, action_dim_ac).to(device)
        self.actor_water = PolicyNet(state_dim, hidden_dim, action_dim_water).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        
        # Use AdamW optimizer
        self.actor_ev_optimizer = torch.optim.AdamW(self.actor_ev.parameters(), lr=self.lr_actor)
        self.actor_ess_optimizer = torch.optim.AdamW(self.actor_ess.parameters(), lr=self.lr_actor)
        self.actor_wash_optimizer = torch.optim.AdamW(self.actor_wash.parameters(), lr=self.lr_actor)
        self.actor_ac_optimizer = torch.optim.AdamW(self.actor_ac.parameters(), lr=self.lr_actor)
        self.actor_ac2_optimizer = torch.optim.AdamW(self.actor_ac2.parameters(), lr=self.lr_actor)
        self.actor_water_optimizer = torch.optim.AdamW(self.actor_water.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_critic)
        
        # Learning rate scheduling
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
        
        # Add missing attributes
        self.ess_capacity = env.ess_capacity
        self.ev_capacity = env.ev_capacity
        self.lambda_ess = torch.tensor(0.0, device=device)
        self.lambda_ev = torch.tensor(0.0, device=device)
        
        # Dynamic target entropy setting
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
        self.ent_coef = ent_coef  # New entropy coefficient
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
        
        # Use normalized state or original state
        if normalized_state is not None:
            state_tensor = normalized_state
        else:
            state_values = [state[key] for key in sorted(state.keys()) if isinstance(state[key], (int, float))]
            state_tensor = torch.tensor([state_values], dtype=torch.float).to(self.device)

        # EV action selection
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
            # Get current SOC and electricity price
            current_soc = state['ev_battery_state'] / 24
            price = state['electricity_price']
            action_ev_power = self.env.action_space['ev_power'][ev_action_index]
            # Dynamically limit charging behavior
            if current_soc > 0.95:  # Prohibit charging when near full charge
                action_ev_power = min(self.env.action_space['ev_power'][ev_action_index], 0)
            # Dynamically limit discharging behavior
            if current_soc < 0.05:
                action_ev_power = max(self.env.action_space['ev_power'][ev_action_index], 0)
        else:
            action_ev_power = 0.0

        # ESS action selection
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

        # Washing machine action selection
        wash_actions = self.actor_wash(state_tensor)
        if action_mask and 'wash_machine_schedule' in action_mask:
            mask_tensor = torch.tensor(action_mask['wash_machine_schedule'], dtype=torch.bool, device=wash_actions.device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            wash_actions = torch.where(mask_tensor, wash_actions, torch.tensor(-1e9, dtype=wash_actions.dtype, device=wash_actions.device))
        
        probs_wash = F.softmax(wash_actions, dim=1)
        action_dist_wash = torch.distributions.Categorical(probs_wash)
        action_wash_machine = action_dist_wash.sample().item()

        # Air conditioner action selection
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

        # Air conditioner 2 action selection
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

        # Water heater action selection
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

        # Collect all actions
        actions = {
            'ev_power': action_ev_power,
            'battery_power': action_ess_power,
            'wash_machine_schedule': action_wash_machine,
            'Air_conditioner_set_temp': action_ac_set_temp,
            'Air_conditioner_set_temp2': action_ac_set_temp2,
            'ewh_set_temp': action_water_set_temp
        }

        # Calculate total log probability and entropy
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
        
        # Calculate value
        value = self.critic(state_tensor).item()

        return actions, total_log_prob, total_entropy, value

    def update(self, transition_dict, running_stats=None, state_keys=None, episode=None, total_episodes=None, USE_STATE_NORMALIZATION=False):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # State normalization processing
        if USE_STATE_NORMALIZATION and running_stats is not None:
            normalized_states = running_stats.normalize(states).clamp(-5, 5)
            normalized_next_states = running_stats.normalize(next_states).clamp(-5, 5)
        else:
            normalized_states = states
            normalized_next_states = next_states
        
        # Calculate current value and next value
        values = self.critic(normalized_states)
        with torch.no_grad():
            next_values = self.critic(normalized_next_states)
        
        # Calculate GAE
        advantages = self.compute_gae2(rewards.squeeze(), values, next_values, dones.squeeze())
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate target value
        with torch.no_grad():
            # Ensure all tensors have correct shape
            rewards_1d = rewards.squeeze(-1)  # Ensure 1D
            next_values_1d = next_values.squeeze(-1)  # Ensure 1D
            dones_1d = dones.squeeze(-1)  # Ensure 1D
            
            td_target = rewards_1d + self.gamma * next_values_1d * (1 - dones_1d)
            # Ensure td_target is 1D tensor
            if td_target.dim() == 2:
                td_target = td_target.squeeze(-1)
        
        # Get old action probabilities (calculate once, used for ratio calculation)
        with torch.no_grad():
            old_ev_actions = self.actor_ev(states)
            old_ess_actions = self.actor_ess(states)
            old_wash_actions = self.actor_wash(states)
            old_ac_actions = self.actor_ac(states)
            old_ac_actions2 = self.actor_ac2(states)
            old_water_actions = self.actor_water(states)
        
        # Ensure action indices are within valid range
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
        
        # Store losses for each epoch for return
        epoch_losses = []
        
        for _ in range(self.epochs):
            # Recalculate new action probabilities for each epoch (avoid reusing computation graph)
            new_ev_actions = self.actor_ev(normalized_states)
            new_ess_actions = self.actor_ess(normalized_states)
            new_wash_actions = self.actor_wash(normalized_states)
            new_ac_actions = self.actor_ac(normalized_states)
            new_ac_actions2 = self.actor_ac2(normalized_states)
            new_water_actions = self.actor_water(normalized_states)

            # Calculate new log probabilities
            log_probs_ev = F.log_softmax(new_ev_actions, dim=1).gather(1, ev_indices.unsqueeze(-1))
            log_probs_ess = F.log_softmax(new_ess_actions, dim=1).gather(1, ess_indices.unsqueeze(-1))
            log_probs_wash = F.log_softmax(new_wash_actions, dim=1).gather(1, wash_indices.unsqueeze(-1))
            log_probs_ac = F.log_softmax(new_ac_actions, dim=1).gather(1, ac_indices.unsqueeze(-1))
            log_probs_ac2 = F.log_softmax(new_ac_actions2, dim=1).gather(1, ac_indices2.unsqueeze(-1))
            log_probs_water = F.log_softmax(new_water_actions, dim=1).gather(1, water_indices.unsqueeze(-1))

            # Calculate ratio and PPO loss - ensure advantages are detached
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

            # Calculate entropy
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

            # Dynamically adjust target entropy
            if episode is not None and total_episodes is not None:
                progress = min(1.0, episode / total_episodes)
                self.target_entropy = self.initial_target_entropy * (1 - progress) + self.final_target_entropy * progress

            # Automatic entropy adjustment
            alpha_loss = -(self.log_alpha * (total_entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            alpha = self.log_alpha.exp().detach().clamp(max=1.0)

            # Constraint handling
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

            # Critic loss - completely fix dimension mismatch issue
            critic_values = self.critic(normalized_states)
            
            # Ensure td_target and critic_values are both 1D tensors, and td_target is detached
            td_target_detached = td_target.detach()
            if td_target_detached.dim() == 2:
                td_target_detached = td_target_detached.squeeze(-1)
            if critic_values.dim() == 2:
                critic_values = critic_values.squeeze(-1)
            
            # Ensure both tensors have exactly the same shape
            assert td_target_detached.shape == critic_values.shape, f"Shape mismatch: td_target {td_target_detached.shape}, critic_values {critic_values.shape}"
            
            critic_loss = F.mse_loss(critic_values, td_target_detached)

            # Total Actor loss
            actor_loss = actor_ev_loss + actor_ess_loss + actor_wash_loss + actor_ac_loss + actor_ac_loss2 + actor_water_loss

            # Total loss
            total_loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    # + 2 * constraint_loss
            )

            # Zero gradients
            self.actor_ev_optimizer.zero_grad()
            self.actor_ess_optimizer.zero_grad()
            self.actor_wash_optimizer.zero_grad()
            self.actor_ac_optimizer.zero_grad()
            self.actor_ac2_optimizer.zero_grad()
            self.actor_water_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Backward propagation
            total_loss.backward()

            # Gradient clipping
            clip_grad_norm_(self.actor_ev.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_ess.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_wash.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_ac.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_ac2.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_water.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # Parameter update
            self.actor_ev_optimizer.step()
            self.actor_ess_optimizer.step()
            self.actor_wash_optimizer.step()
            self.actor_ac_optimizer.step()
            self.actor_ac2_optimizer.step()
            self.actor_water_optimizer.step()
            self.critic_optimizer.step()

            # Record loss
            epoch_losses.append(total_loss.item())

        # Return specific loss values
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

    # Can dynamically decay entropy coefficient during training
    def update_ent_coef(self, progress):
        initial_ent_coef = 0.1
        min_ent_coef = 0.01
        self.ent_coef = max(initial_ent_coef * (1 - progress), min_ent_coef)


# ==================== Training Loop ====================
if __name__ == "__main__":
    # ==================== Configuration Parameters (All centralized for easy experimentation and readability) ====================
    USE_STATE_NORMALIZATION = True  # State normalization switch
    USE_ADVANTAGE_NORMALIZATION = True  # Advantage normalization switch
    USE_DYNAMIC_MASK = True  # Dynamic mask switch
    NUM_EPISODES = 5000  # Number of training episodes
    HIDDEN_DIM = 128
    EPOCHS = 4
    LAMBDA = 0.98
    EPS = 0.2
    GAMMA = 0.96
    MAX_GRAD_NORM = 10
    ENT_COEF = 0.1
    
    # Environment initialization
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # Agent initialization
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

    # Initialize running_stats - decide whether to use based on configuration
    if USE_STATE_NORMALIZATION:
        running_stats = RunningStats(shape=len(env.state_space))
        print("State normalization enabled")
    else:
        running_stats = None
        print("State normalization disabled")

    # Define action space for warm-up
    action_spaces = {
        'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],
        'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],
        'wash_machine_schedule': [0, 1, 2, 3, 4, 5, 6],
        'Air_conditioner_set_temp': [16, 18, 20, 22, 24, 26, 28, 30],
        'Air_conditioner_set_temp2': [16, 18, 20, 22, 24, 26, 28, 30],
        'ewh_set_temp': [40, 45, 50, 55, 60, 65, 70]
    }

    # Warm-up phase: collect initial data
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
        print("Skipping state normalization warm-up phase")
        state = env.reset()
        state_keys = sorted(env.state_space.keys())

    num_episodes = NUM_EPISODES
    episode_returns = []

    # Create results directory
    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)

    # Create unique filename (include timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    csv_filename = os.path.join(results_dir, f"returns_ppo2_{timestamp}{norm_suffix}.csv")

    # Open CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header row - extended evaluation metrics
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
            
            # Episode-level metric collection
            episode_ess_violations = 0
            episode_ev_violations = 0
            episode_ess_socs = []
            episode_ev_socs = []
            episode_energy_costs = []
            episode_user_satisfactions = []
            episode_temperature_comforts = []
            episode_peak_valley_arbitrages = []
            step_count = 0
            
            # Violation type statistics
            episode_ess_mild_violations = 0
            episode_ess_severe_violations = 0
            episode_ev_mild_violations = 0
            episode_ev_severe_violations = 0
            
            # Keep original count statistics for comparison
            episode_ess_violations_count = 0
            episode_ev_violations_count = 0

            # Safety margin analysis
            episode_ess_safety_margins = []
            episode_ev_safety_margins = []

            # New: temperature comfort statistics
            ac1_temp_comforts = []
            ac2_temp_comforts = []
            ewh_temp_comforts = []

            while True:
                # Collect current state (original values)
                current_state_values = [state[k] for k in state_keys]
                episode_states.append(current_state_values)

                # Normalize current state
                state_tensor = torch.FloatTensor(current_state_values).unsqueeze(0).to(device)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_state = running_stats.normalize(state_tensor).clamp(-5, 5)
                else:
                    normalized_state = state_tensor

                # Get action mask from environment
                action_mask = env.get_action_mask(state)

                # Agent uses mask to select action
                actions, log_prob, _, value = agent.take_action(
                    state,
                    action_mask=action_mask,
                    normalized_state=normalized_state
                )

                # Environment executes action
                next_state, reward, done = env.step(state, actions)

                # Collect next state (original values)
                next_state_values = [next_state[k] for k in state_keys]
                episode_states.append(next_state_values)

                # Calculate value of next state (using normalized state)
                next_state_tensor = torch.FloatTensor(next_state_values).unsqueeze(0).to(device)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_next_state = running_stats.normalize(next_state_tensor).clamp(-5, 5)
                else:
                    normalized_next_state = next_state_tensor
                next_value = agent.critic(normalized_next_state).item()

                # Collect detailed metrics
                # 1. Constraint violation check
                ess_soc = state['ess_state'] / env.ess_capacity
                ev_soc = state['ev_battery_state'] / env.ev_capacity
                episode_ess_socs.append(ess_soc)
                episode_ev_socs.append(ev_soc)
                
                def calculate_violation_metrics(soc, lower_bound=0.1, upper_bound=0.9):
                    """Calculate constraint violation metrics"""
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
                
                # Calculate violation metrics
                ess_metrics = calculate_violation_metrics(ess_soc)
                ev_metrics = calculate_violation_metrics(ev_soc)
                
                # Accumulate metrics
                episode_ess_violations += ess_metrics['violation_severity']
                episode_ev_violations += ev_metrics['violation_severity']
                episode_ess_violations_count += ess_metrics['violation_count']
                episode_ev_violations_count += ev_metrics['violation_count']
                
                # Safety margin analysis
                episode_ess_safety_margins.append(ess_metrics['safety_margin'])
                episode_ev_safety_margins.append(ev_metrics['safety_margin'])
                
                # 2. Economic metrics
                episode_energy_costs.append(env.current_step_cost)
                
                # 3. Peak-valley electricity price arbitrage effect
                price = state['electricity_price']
                ev_power = actions.get('ev_power', 0)
                battery_power = actions.get('battery_power', 0)
                
                def calculate_peak_valley_arbitrage(electricity_price, ev_pwr, bat_pwr):
                    """Calculate peak-valley electricity price arbitrage effect"""
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
                
                # 4. User satisfaction
                indoor_temp1 = env.indoor_temp
                indoor_temp2 = env.indoor_temp2
                user_pref1 = env.user_temp_preference
                user_pref2 = env.user_temp_preference2
                
                temp_diff1 = abs(indoor_temp1 - user_pref1)
                temp_diff2 = abs(indoor_temp2 - user_pref2)
                
                temp_comfort1 = max(0, 1 - max(0, temp_diff1 - 2) / 8)
                temp_comfort2 = max(0, 1 - max(0, temp_diff2 - 2) / 8)
                # New: record ac1/ac2/ewh comfort at each step
                ac1_temp_comforts.append(temp_comfort1)
                ac2_temp_comforts.append(temp_comfort2)
                # Calculate water heater comfort
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

                # Store transition
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

            # Update running_stats
            if USE_STATE_NORMALIZATION and running_stats is not None and episode_states:
                states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
                running_stats.update(states_tensor)

            # Prepare data for update
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

            # State tensor conversion
            states = torch.stack([
                torch.FloatTensor([s[key] for key in state_keys])
                for s in batch_data['states']
            ]).to(device)

            # Action index conversion
            action_indices = []
            for t in batch:
                indices = []
                for name in ['ev_power', 'battery_power', 'wash_machine_schedule', 
                           'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp']:
                    if name in t['actions']:
                        # Find index of action in action space
                        action_space = env.action_space[name]
                        action_value = t['actions'][name]
                        try:
                            idx = action_space.index(action_value)
                        except ValueError:
                            # If exact match not found, find closest value
                            idx = min(range(len(action_space)), 
                                    key=lambda i: abs(action_space[i] - action_value))
                        indices.append(idx)
                    else:
                        indices.append(0)
                action_indices.append(indices)

            # Update transition_dict format
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

            # Update parameters
            actor_loss, critic_loss, constraint_loss, total_loss = agent.update(
                transition_dict, running_stats, state_keys, episode, num_episodes, USE_STATE_NORMALIZATION
            )
            episode_returns.append(episode_return)

            # Calculate episode-level statistical metrics
            ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
            ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0
            total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2 if step_count > 0 else 0
            
            energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
            user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
            temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0
            
            # Calculate independent temperature comfort for two air conditioners and water heater (changed to full episode mean)
            ac1_temp_comfort = np.mean(ac1_temp_comforts) if ac1_temp_comforts else 0
            ac2_temp_comfort = np.mean(ac2_temp_comforts) if ac2_temp_comforts else 0
            ewh_temp_comfort = np.mean(ewh_temp_comforts) if ewh_temp_comforts else 0
            
            # Calculate water heater temperature comfort
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
            
            # Safety margin analysis
            ess_safety_margin_mean = np.mean(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_mean = np.mean(episode_ev_safety_margins) if episode_ev_safety_margins else 0
            ess_safety_margin_std = np.std(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_std = np.std(episode_ev_safety_margins) if episode_ev_safety_margins else 0
            
            peak_valley_arbitrage = np.mean(episode_peak_valley_arbitrages) if episode_peak_valley_arbitrages else 0
            
            # Training stability
            if len(episode_returns) >= 10:
                recent_returns = episode_returns[-10:]
                training_stability = 1.0 / (1.0 + np.std(recent_returns))
            else:
                training_stability = 0.0
            
            # Sample efficiency
            sample_efficiency = episode_return / step_count if step_count > 0 else 0

            # Write current episode detailed data to CSV
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
            
            # Add detailed violation type analysis
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

            # Record total cost at end of each episode
            env.episode_costs.append(env.total_cost)

    # Save cost data after training completion
    env.save_episode_costs()

    # Save trained model
    model_save_dir = "model/saved_models"
    os.makedirs(model_save_dir, exist_ok=True)
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    model_filename = os.path.join(model_save_dir, f"ppo2_model_{timestamp}{norm_suffix}.pth")
    
    # Save model state dictionary
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
    
    # Decide whether to save running_stats based on configuration
    if USE_STATE_NORMALIZATION and running_stats is not None:
        model_save_dict.update({
            'running_stats_mean': running_stats.mean,
            'running_stats_std': running_stats.std,
            'running_stats_count': running_stats.count,
        })
    
    torch.save(model_save_dict, model_filename)
    
    print(f"Model saved to: {model_filename}")

    # Close file after training completion
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)

    print(f"Training completed! Returns data saved to: {csv_filename}")
