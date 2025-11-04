"""
Final implementation version
"""
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.clip_grad import clip_grad_norm_
from environment import HomeEnergyManagementEnv
# Add evaluation directory to path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'evaluation'))
from plt import plot_returns
import matplotlib
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import deque  # New: for saving the most recent 50 models

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

# ==================== Network Structure ====================
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
                 ent_coef=0.1, max_grad_norm=10, device=device,
                 constraint_mode="none",  # 'none' or 'lagrangian'
                 use_state_normalization=True,
                 use_advantage_normalization=True,
                 use_dynamic_mask=True,
                 constraint_config=None):
        """
        Flagship PPO3, supports ablation experiment switches:
        - constraint_mode: 'none' (no constraints) or 'lagrangian' (PPO-Lagrangian)
        - use_state_normalization: state normalization switch
        - use_advantage_normalization: advantage function normalization switch
        - use_dynamic_mask: dynamic mask switch
        """
        self.constraint_mode = constraint_mode
        self.use_state_normalization = use_state_normalization
        self.use_advantage_normalization = use_advantage_normalization
        self.use_dynamic_mask = use_dynamic_mask
        # Handle constraint configuration
        if constraint_config is None:
            constraint_config = {
                'soc_lower': 0.1,  # Modify lower bound constraint to 20%
                'soc_upper': 0.9,  # Modify upper bound constraint to 80%
                'lambda_init': 1.0,  # Increase initial value
                'lambda_max': 50.0,  # Increase maximum value
                'dual_ascent_rate': 0.1,  # Increase update rate
                'constraint_weight': 5.0,  # Increase early weight
                'final_constraint_weight': 2.0  # Increase final weight
            }
        self.constraint_config = constraint_config.copy()

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

        # Optimizer - reduce learning rate
        self.actor_optim = torch.optim.AdamW([
            {'params': self.shared_backbone.parameters()},
            {'params': self.actor_branches.parameters()}
        ], lr=1e-5)  # Reduce learning rate
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=1e-5)  # Reduce learning rate

        # Learning rate scheduling
        self.actor_scheduler = CosineAnnealingLR(self.actor_optim, T_max=1000, eta_min=3e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optim, T_max=1000, eta_min=5e-5)

        # Dynamic target entropy setting
        self.initial_target_entropy = np.mean([
            1 * np.log(len(actions)) for actions in action_space_config.values()
        ])
        self.final_target_entropy = self.initial_target_entropy * 0.01
        self.target_entropy = self.initial_target_entropy

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

        # Constraint-related parameter initialization - initialize capacity in all modes
        self.ess_capacity = env.ess_capacity
        self.ev_capacity = env.ev_capacity

        if self.constraint_mode == "lagrangian":
            self.lambda_ess = torch.tensor(self.constraint_config.get('lambda_init', 0.1), requires_grad=False, device=device)
            self.lambda_ev = torch.tensor(self.constraint_config.get('lambda_init', 0.1), requires_grad=False, device=device)

    def take_action(self, state_tensor, action_mask=None):
        shared_features = self.shared_backbone(state_tensor)
        actions = {}
        log_probs = []
        entropies = []
        for name, branch in self.actor_branches.items():
            logits = branch(shared_features)
            # Dynamic mask switch
            if self.use_dynamic_mask and action_mask and name in action_mask:
                mask_tensor = torch.tensor(action_mask[name], dtype=torch.bool, device=logits.device)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)
                masked_logits = torch.where(mask_tensor, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
            else:
                masked_logits = logits
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample()
            actions[name] = self.action_mapping[name][int(action_idx.item())]
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
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update(self, batch_data_list, running_stats, state_keys, episode, total_episodes):
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
            'next_values': [t['next_values'] for t in batch_data_list],
            'ess_violation': [t['ess_violation'] for t in batch_data_list],
            'ev_violation': [t['ev_violation'] for t in batch_data_list]
        }

        # State tensor conversion (ensure consistent order)
        states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['states']
        ]).to(device)

        # Normalize state using running_stats (if enabled)
        if self.use_state_normalization and running_stats is not None:
            normalized_states = running_stats.normalize(states).clamp(-5, 5)
        else:
            normalized_states = states

        # Normalize next state
        next_states = torch.stack([
            torch.FloatTensor([s[key] for key in state_keys])
            for s in batch_data['next_states']
        ]).to(device)
        if self.use_state_normalization and running_stats is not None:
            normalized_next_states = running_stats.normalize(next_states).clamp(-5, 5)
        else:
            normalized_next_states = next_states

        # Action index conversion (handle floating point precision)
        action_indices = {}
        for name in self.action_mapping:
            indices = []
            action_values = list(self.action_mapping[name].values())
            for val in batch_data['actions'][name]:
                close_idx = np.where(np.isclose(action_values, val, atol=1e-4))[0]
                if len(close_idx) > 0:
                    indices.append(close_idx[0])
                else:
                    # If exact match not found, take closest one
                    idx = int(np.argmin(np.abs(np.array(action_values) - val)))
                    indices.append(idx)
            action_indices[name] = torch.tensor(indices, dtype=torch.long, device=device)

        # Tensor conversion (explicitly specify device)
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32, device=device)
        dones = torch.tensor(batch_data['dones'], dtype=torch.float32, device=device)
        old_log_probs = torch.stack([lp.squeeze() for lp in batch_data['log_probs']]).to(device)
        values = torch.tensor(batch_data['values'], dtype=torch.float32, device=device)
        next_values = torch.tensor(batch_data['next_values'], dtype=torch.float32, device=device)
        ess_violations = torch.tensor(batch_data['ess_violation'], dtype=torch.float32, device=device)
        ev_violations = torch.tensor(batch_data['ev_violation'], dtype=torch.float32, device=device)

        # ==================== GAE Calculation ====================
        advantages = self.compute_gae(rewards, values, next_values, dones)

        # ==================== Constraint Calculation Refactoring ====================
        constraint_loss = torch.tensor(0.0, device=device)
        lagrangian_terms = torch.tensor(0.0, device=device)
        ess_violation_mean = 0.0
        ev_violation_mean = 0.0
        constraint_weight = 1.0
        # New: step violation rate
        ess_violation_rate = 0.0
        ev_violation_rate = 0.0

        # Get indices of ess_state and ev_battery_state in state vector
        ess_index = state_keys.index('ess_state')
        ev_index = state_keys.index('ev_battery_state')
        ess_soc = states[:, ess_index] / self.ess_capacity
        ev_soc = states[:, ev_index] / self.ev_capacity

        soc_lower = self.constraint_config['soc_lower']
        soc_upper = self.constraint_config['soc_upper']

        # Calculate constraint violation amount (always positive) - calculated in all modes
        ess_violation = torch.relu(soc_lower - ess_soc) + torch.relu(ess_soc - soc_upper)
        ev_violation = torch.relu(soc_lower - ev_soc) + torch.relu(ev_soc - soc_upper)

        # Calculate average violation amount (for monitoring) - calculated in all modes
        ess_violation_mean = ess_violation.mean().item()
        ev_violation_mean = ev_violation.mean().item()

        # Calculate violation rate tensor (for optimization)
        ess_violation_rate_tensor = (ess_violation > 0).float().mean()
        ev_violation_rate_tensor = (ev_violation > 0).float().mean()
        ess_violation_rate = ess_violation_rate_tensor.item()
        ev_violation_rate = ev_violation_rate_tensor.item()

        if self.constraint_mode == "lagrangian":
            # Constraint loss uses violation rate
            lagrangian_terms = self.lambda_ess * ess_violation_rate_tensor + \
                              self.lambda_ev * ev_violation_rate_tensor
            # New: violation amount penalty term
            violation_amount_penalty = (ess_violation.mean() + ev_violation.mean()) * 10.0
            constraint_loss = lagrangian_terms + violation_amount_penalty

            # Calculate dynamic constraint weight - keep original weight
            progress = min(1.0, episode / total_episodes)
            constraint_weight = self.constraint_config['constraint_weight'] * (1 - progress) + \
                               self.constraint_config['final_constraint_weight'] * progress
            constraint_loss *= constraint_weight * 100.0

        # Store losses for each epoch for return
        epoch_losses = []

        for _ in range(self.epochs):
            # ==================== Critic Update ====================
            critic_values = self.critic(normalized_states)  # Use normalized state
            with torch.no_grad():
                # Calculate target value using normalized next state
                target_values = rewards + self.gamma * next_values * (1 - dones)
            critic_loss = F.mse_loss(critic_values, target_values)

            # ==================== Actor Update ====================
            shared_features = self.shared_backbone(normalized_states)  # Use normalized state
            new_log_probs = []
            entropies = []

            # Probability calculation for each action branch
            for name, branch in self.actor_branches.items():
                logits = branch(shared_features)
                # Add numerical safety handling
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("Warning: Invalid logits detected in action branch")
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

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

            # Dynamically adjust target entropy
            progress = min(1.0, episode / total_episodes)
            self.target_entropy = self.initial_target_entropy * (1 - progress) + self.final_target_entropy * progress

            # ==================== Automatic Entropy Adjustment ====================
            alpha_loss = -(self.log_alpha * (total_entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)  # Retain computation graph for subsequent gradients
            self.alpha_optim.step()
            alpha = self.log_alpha.exp().detach().clamp(max=1.0)

            # ==================== Constraint Update (within epoch loop) ====================
            if self.constraint_mode == "lagrangian":
                with torch.no_grad():
                    # Dynamically adjust constraint strength: set target based on violation rate
                    progress = min(1.0, episode / total_episodes)
                    if progress < 0.3:
                        constraint_strength = 3.0
                        target_ess_violation_rate = 0.3
                        target_ev_violation_rate = 0.3
                    elif progress < 0.7:
                        constraint_strength = 2.0
                        target_ess_violation_rate = 0.25
                        target_ev_violation_rate = 0.25
                    else:
                        constraint_strength = 1.5
                        target_ess_violation_rate = 0.2
                        target_ev_violation_rate = 0.2
                    # Calculate error based on violation rate
                    ess_error = ess_violation_rate - target_ess_violation_rate
                    ev_error = ev_violation_rate - target_ev_violation_rate
                    kp = constraint_strength
                    new_lambda_ess = self.lambda_ess + 0.1 * kp * ess_error
                    new_lambda_ev = self.lambda_ev + 0.1 * kp * ev_error
                    lambda_max = self.constraint_config['lambda_max']
                    self.lambda_ess = torch.clamp(new_lambda_ess, 0, lambda_max)
                    self.lambda_ev = torch.clamp(new_lambda_ev, 0, lambda_max)
                    # Recalculate constraint loss (using updated lambda)
                    constraint_loss = self.lambda_ess * ess_violation_rate_tensor + \
                                    self.lambda_ev * ev_violation_rate_tensor
                    constraint_loss *= constraint_weight

            # ==================== Calculate Total Loss ====================
            if self.constraint_mode == "lagrangian":
                # Significantly increase constraint loss weight
                total_loss = actor_loss + 0.1 * critic_loss + 1.0 * constraint_loss
            else:
                total_loss = actor_loss + 0.1 * critic_loss

            # ==================== Unified Gradient Update ====================
            # Zero gradients
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()

            # Backward propagation (single pass)
            total_loss.backward()

            # Gradient clipping
            clip_grad_norm_(self.shared_backbone.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_branches.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # Parameter update
            self.actor_optim.step()
            self.critic_optim.step()

            # Record loss
            epoch_losses.append(total_loss.item())

        # Learning rate scheduler update
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Return specific loss values instead of average loss
        # Calculate detailed loss for last epoch (for monitoring)
        # New: return step violation rate and average violation amount
        return actor_loss.item(), critic_loss.item(), constraint_loss, total_loss.item(), constraint_weight, ess_violation_rate, ev_violation_rate, ess_violation_mean, ev_violation_mean


# ==================== Training Loop ====================
if __name__ == "__main__":
    # Environment initialization
    env = HomeEnergyManagementEnv()
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # ==================== Configuration Parameters ====================
    USE_STATE_NORMALIZATION = True  # Set to False to disable state normalization
    USE_ADVANTAGE_NORMALIZATION = True  # New: advantage function normalization switch
    CONSTRAINT_MODE = "none"  # "none" or "lagrangian"
    USE_DYNAMIC_MASK = True  # Dynamic mask switch

    # Agent initialization
    agent = HomeEnergyPPO(
        env=env,
        state_dim=len(env.state_space),
        hidden_dim=128,
        action_space_config=env.action_space,
        device=device,
        constraint_mode=CONSTRAINT_MODE,
        use_state_normalization=USE_STATE_NORMALIZATION,
        use_advantage_normalization=USE_ADVANTAGE_NORMALIZATION,
        use_dynamic_mask=USE_DYNAMIC_MASK,
        constraint_config={
            'soc_lower': 0.1,  # Modify lower bound constraint to 20%
            'soc_upper': 0.9,  # Modify upper bound constraint to 80%
            'lambda_init': 1.0,  # Increase initial value
            'lambda_max': 50.0,  # Increase maximum value
            'dual_ascent_rate': 0.1,  # Increase update rate
            'constraint_weight': 5.0,  # Increase early weight
            'final_constraint_weight': 2.0  # Increase final weight
        }
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
            'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],  # EV charge/discharge power range
            'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],  # ESS charge/discharge power range
            'wash_machine_schedule': [0, 1, 2, 3, 4, 5, 6],  # Washing machine scheduling actions
            'Air_conditioner_set_temp': [16, 18, 20, 22, 24, 26, 28, 30],  # AC set temperature
            'Air_conditioner_set_temp2': [16, 18, 20, 22, 24, 26, 28, 30],
            'ewh_set_temp': [40, 45, 50, 55, 60, 65, 70]  # Discrete temperature setting actions
    }

    # Warm-up phase: collect initial data
    if USE_STATE_NORMALIZATION:
        print("Warming up running_stats...")
        warmup_states = []
        state = env.reset()
        state_keys = sorted(env.state_space.keys())  # Ensure state key order is consistent

        for _ in range(1000):  # Collect 1000 state samples
            warmup_states.append([state[k] for k in state_keys])

            # Use correct action space
            actions = {name: np.random.choice(space) for name, space in action_spaces.items()}

            next_state, _, _ = env.step(state, actions)
            state = next_state

        # Initialize running_stats with warm-up data
        if warmup_states and USE_STATE_NORMALIZATION and running_stats is not None:
            states_tensor = torch.tensor(warmup_states, dtype=torch.float32, device=device)
            running_stats.update(states_tensor)
        if USE_STATE_NORMALIZATION and running_stats is not None:
            print(f"Running stats initialized: mean={running_stats.mean.cpu().numpy()}, std={running_stats.std.cpu().numpy()}")
    else:
        print("Skipping state normalization warm-up phase")
        state = env.reset()
        state_keys = sorted(env.state_space.keys())  # Ensure state key order is consistent

    num_episodes = 5000
    episode_returns = []

    # Create results directory
    results_dir = "model/results"
    os.makedirs(results_dir, exist_ok=True)

    # New: create a queue of length 50 to save returns and model parameters for the most recent 50 episodes
    recent_models = deque(maxlen=50)  # Only keep the latest 50 each time

    # Create unique filename (include timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
    constraint_suffix = "_constrained" if CONSTRAINT_MODE == "lagrangian" else "_unconstrained"
    csv_filename = os.path.join(results_dir, f"returns_ppo_{timestamp}{norm_suffix}{constraint_suffix}.csv")

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

        # ==================== Phased Training and Robustness Mechanism Parameters ====================
        rollback_patience = 30  # Rollback if returns are below threshold for this many consecutive episodes
        rollback_threshold = 0.7  # Trigger rollback if returns are below 70% of historical mean
        best_model = None
        best_return = -float('inf')
        rollback_counter = 0

        def get_value_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.item(), x.device
            else:
                return float(x), torch.device('cpu')  # Or your default device

        for episode in range(num_episodes):
            # === Dynamically set soft constraint reward shaping parameters ===
            if episode < 1000:
                target_violation_rate = 0.3
            elif episode < 3000:
                target_violation_rate = 0.2
            else:
                target_violation_rate = 0.1
            setattr(env, 'constraint_mode', CONSTRAINT_MODE)
            setattr(env, 'target_violation_rate', target_violation_rate)

            state = env.reset()
            batch = []
            episode_return = 0
            episode_states = []  # Collect states for entire episode to update running_stats

            # New: episode-level metric collection
            episode_ess_violations = 0
            episode_ev_violations = 0
            episode_ess_socs = []
            episode_ev_socs = []
            episode_energy_costs = []
            episode_user_satisfactions = []
            episode_temperature_comforts = []
            episode_peak_valley_arbitrages = []
            step_count = 0

            # New: violation type statistics
            episode_ess_mild_violations = 0
            episode_ess_severe_violations = 0
            episode_ev_mild_violations = 0
            episode_ev_severe_violations = 0

            # New: retain original count statistics for comparison
            episode_ess_violations_count = 0
            episode_ev_violations_count = 0

            # New: safety margin analysis (not provided by Constraint Loss)
            episode_ess_safety_margins = []
            episode_ev_safety_margins = []

            # New: temperature comfort statistics
            ac1_temp_comforts = []
            ac2_temp_comforts = []
            ewh_temp_comforts = []

            while True:
                # Collect current state (raw values)
                current_state_values = [state[k] for k in state_keys]
                episode_states.append(current_state_values)

                # Normalize current state
                state_tensor = torch.FloatTensor(current_state_values).unsqueeze(0).to(device)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_state = running_stats.normalize(state_tensor).clamp(-5, 5)  # Add clipping to prevent extreme values
                else:
                    normalized_state = state_tensor  # Do not use normalization

                # Get action mask from environment
                action_mask = env.get_action_mask(state)

                # Agent uses mask to select action
                actions, log_prob, _, value = agent.take_action(
                    normalized_state,
                    action_mask=action_mask
                )

                # Environment executes action
                next_state, reward, done = env.step(state, actions)

                # Collect next state (raw values)
                next_state_values = [next_state[k] for k in state_keys]
                episode_states.append(next_state_values)

                # Calculate value of next state (using normalized state)
                next_state_tensor = torch.FloatTensor(next_state_values).unsqueeze(0).to(device)
                if USE_STATE_NORMALIZATION and running_stats is not None:
                    normalized_next_state = running_stats.normalize(next_state_tensor).clamp(-5, 5)
                else:
                    normalized_next_state = next_state_tensor
                next_value = agent.critic(normalized_next_state).item()

                # New: collect detailed metrics
                # 1. Constraint violation check - improved version
                ess_soc = state['ess_state'] / env.ess_capacity
                ev_soc = state['ev_battery_state'] / env.ev_capacity
                episode_ess_socs.append(ess_soc)
                episode_ev_socs.append(ev_soc)

                # Improved constraint violation calculation: focus on analysis not provided by Constraint Loss
                def calculate_violation_metrics(soc, lower_bound=None, upper_bound=None):
                    """Use the same boundaries as constraint loss for calculation"""
                    # Get boundaries from agent configuration if not passed as parameters
                    if lower_bound is None:
                        lower_bound = agent.constraint_config['soc_lower']
                    if upper_bound is None:
                        upper_bound = agent.constraint_config['soc_upper']

                    if lower_bound <= soc <= upper_bound:
                        return {
                            'violation_count': 0,
                            'violation_severity': 0.0,
                            'safety_margin': min(soc - lower_bound, upper_bound - soc) / (upper_bound - lower_bound)
                        }

                    # Calculate violation severity
                    if soc < lower_bound:
                        violation_severity = (lower_bound - soc) / lower_bound
                        safety_margin = -violation_severity
                    else:  # soc > upper_bound
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

                # New: safety margin analysis (not provided by Constraint Loss)
                episode_ess_safety_margins.append(ess_metrics['safety_margin'])
                episode_ev_safety_margins.append(ev_metrics['safety_margin'])

                # 2. Economic metrics
                episode_energy_costs.append(env.current_step_cost)

                # 3. Peak-valley price arbitrage effect (improved version)
                price = state['electricity_price']
                ev_power = actions.get('ev_power', 0)
                battery_power = actions.get('battery_power', 0)

                def calculate_peak_valley_arbitrage(electricity_price, ev_pwr, bat_pwr):
                    """Revised version: arbitrage score is given whenever charging during low price or discharging during high price, score normalized by power"""
                    valley_threshold = 0.2
                    peak_threshold = 0.8
                    arbitrage_score = 0.0
                    if electricity_price <= valley_threshold:
                        total_charging = max(0, ev_pwr) + max(0, bat_pwr)
                        arbitrage_score = total_charging * 0.95
                    elif electricity_price >= peak_threshold:
                        total_discharging = abs(min(0, ev_pwr)) + abs(min(0, bat_pwr))
                        arbitrage_score = total_discharging * 0.95
                    max_arbitrage_power = 6.6 + 4.4
                    normalized_score = min(1.0, arbitrage_score / max_arbitrage_power)
                    return normalized_score

                peak_valley_arbitrage = calculate_peak_valley_arbitrage(price, ev_power, battery_power)
                episode_peak_valley_arbitrages.append(peak_valley_arbitrage)

                # 5. User satisfaction (revised calculation)
                # Temperature comfort - consider different preferences for two ACs
                indoor_temp1 = env.indoor_temp  # Indoor temperature of first AC
                indoor_temp2 = env.indoor_temp2  # Indoor temperature of second AC
                user_pref1 = env.user_temp_preference  # First AC prefers 22°C
                user_pref2 = env.user_temp_preference2  # Second AC prefers 18°C

                # Calculate temperature comfort for both ACs (±2°C comfort range)
                temp_diff1 = abs(indoor_temp1 - user_pref1)
                temp_diff2 = abs(indoor_temp2 - user_pref2)
                temp_comfort1 = max(0, 1 - max(0, temp_diff1 - 2) / 8)  # Linear decrease after exceeding 2°C
                temp_comfort2 = max(0, 1 - max(0, temp_diff2 - 2) / 8)  # Linear decrease after exceeding 2°C

                # Calculate water heater temperature comfort (using state at end of episode)
                ewh_temp = env.state['ewh_temp']
                hour = int(env.state['time_index'] // 2)  # Ensure hour is integer
                if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak water usage hours
                    target_temp, low_temp, high_temp = 55, 50, 60
                else:  # Non-peak hours
                    target_temp, low_temp, high_temp = 45, 40, 50
                if low_temp <= ewh_temp <= high_temp:
                    ewh_temp_comfort = 1.0
                else:
                    deviation = max(low_temp - ewh_temp, ewh_temp - high_temp)
                    ewh_temp_comfort = max(0, 1 - deviation / 10)

                # Overall temperature comfort (two ACs + water heater, equal weight average)
                overall_comfort = (temp_comfort1 + temp_comfort2 + ewh_temp_comfort) / 3
                episode_temperature_comforts.append(overall_comfort)

                # Overall user satisfaction (based on overall comfort)
                user_satisfaction = overall_comfort * 0.7 + 0.3  # Simplified calculation
                episode_user_satisfactions.append(user_satisfaction)

                # Calculate violation amount (for PPO-Lagrangian gradient)
                soc_lower = agent.constraint_config['soc_lower']
                soc_upper = agent.constraint_config['soc_upper']
                ess_violation = max(0, soc_lower - ess_soc) + max(0, ess_soc - soc_upper)
                ev_violation = max(0, soc_lower - ev_soc) + max(0, ev_soc - soc_upper)

                # Store transition (using raw state values)
                batch.append({
                    'state': state,  # Raw state dictionary
                    'actions': actions,
                    'rewards': reward,
                    'next_state': next_state,  # Raw state dictionary
                    'dones': done,
                    'log_probs': log_prob.detach().squeeze(),
                    'values': value,
                    'next_values': next_value,
                    'ess_violation': ess_violation,
                    'ev_violation': ev_violation
                })

                episode_return += reward
                state = next_state  # Update to raw state
                step_count += 1

                # New: collect comfort at each step in the loop
                ac1_temp_comforts.append(temp_comfort1)
                ac2_temp_comforts.append(temp_comfort2)
                ewh_temp_comforts.append(ewh_temp_comfort)

                if done:
                    break

            # Update running_stats: use states from entire episode
            if USE_STATE_NORMALIZATION and running_stats is not None and episode_states:
                states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=device)
                running_stats.update(states_tensor)

            # Update parameters (get additional return values: constraint_weight, step violation rate, mean violation amount)
            actor_loss, critic_loss, constraint_loss, total_loss, constraint_weight, ess_violation_rate, ev_violation_rate, ess_violation_mean, ev_violation_mean = agent.update(
                batch, running_stats, state_keys, episode, num_episodes
            )
            episode_returns.append(episode_return)

            # ========== Dynamic Constraint Weight Adjustment ==========
            # Only adjust constraint weight in lagrangian mode
            if agent.constraint_mode == "lagrangian":
                # Use a more moderate weight strategy
                if episode < 1000:
                    constraint_weight = 2.0  # Early: medium constraint weight
                elif episode < 2000:
                    constraint_weight = 1.5  # Mid: reduce constraint weight
                elif episode < 3000:
                    constraint_weight = 1.0  # Late: further reduce constraint weight
                else:
                    constraint_weight = 0.5  # Final: minimum constraint weight
                agent.constraint_config['constraint_weight'] = constraint_weight
                agent.constraint_config['final_constraint_weight'] = constraint_weight
            else:
                # Set default value in "none" mode
                constraint_weight = 0.0

            # # ========== Automatic Rollback Mechanism ==========
            # if episode > 50:
            #     recent_returns = episode_returns[-50:]
            #     mean_return = np.mean(recent_returns)
            #     if episode_return < rollback_threshold * mean_return:
            #         rollback_counter += 1
            #     else:
            #         rollback_counter = 0
            #     if rollback_counter >= rollback_patience:
            #         print(f"[Rollback] Episode {episode+1}: Return collapse, rollback to previous healthy model, lambda/weight halved")
            #         # Rollback model parameters
            #         if best_model is not None:
            #             # Use type conversion to solve OrderedDict problem
            #             from collections import OrderedDict
            #             agent.shared_backbone.load_state_dict(OrderedDict(best_model['shared_backbone']))
            #             agent.actor_branches.load_state_dict(OrderedDict(best_model['actor_branches']))
            #             agent.critic.load_state_dict(OrderedDict(best_model['critic']))
            #             # Only rollback lambda values in lagrangian mode
            #             if agent.constraint_mode == "lagrangian":
            #                 # Simplify lambda value handling - directly reset to default value
            #                 agent.lambda_ess = torch.tensor(0.5, device=device)
            #                 agent.lambda_ev = torch.tensor(0.5, device=device)
            #                 agent.constraint_config['constraint_weight'] *= 0.5
            #         rollback_counter = 0
            # # Save current best model
            # if episode_return > best_return:
            #     best_return = episode_return
            #     best_model = {
            #         'shared_backbone': agent.shared_backbone.state_dict(),
            #         'actor_branches': agent.actor_branches.state_dict(),
            #         'critic': agent.critic.state_dict(),
            #     }
            #     # Only save lambda values to separate variables in lagrangian mode
            #     if agent.constraint_mode == "lagrangian":
            #         # Use separate variables to store lambda values, avoid type conflicts
            #         best_lambda_ess = 0.0
            #         best_lambda_ev = 0.0

            # Calculate episode-level statistics
            ess_violation_rate = episode_ess_violations / step_count if step_count > 0 else 0
            ev_violation_rate = episode_ev_violations / step_count if step_count > 0 else 0

            # Modify total violation rate calculation: use average of two device violation rates instead of sum
            # This ensures total violation rate is in [0, 1] range, more intuitive
            total_violation_rate = (ess_violation_rate + ev_violation_rate) / 2 if step_count > 0 else 0

            energy_cost = np.mean(episode_energy_costs) if episode_energy_costs else 0
            user_satisfaction = np.mean(episode_user_satisfactions) if episode_user_satisfactions else 0
            temperature_comfort = np.mean(episode_temperature_comforts) if episode_temperature_comforts else 0

            # Calculate independent temperature comfort for both ACs (changed to full episode mean)
            ac1_temp_comfort = np.mean(ac1_temp_comforts) if ac1_temp_comforts else 0
            ac2_temp_comfort = np.mean(ac2_temp_comforts) if ac2_temp_comforts else 0
            ewh_temp_comfort = np.mean(ewh_temp_comforts) if ewh_temp_comforts else 0

            # Calculate water heater temperature comfort (using state at end of episode)
            ewh_temp = env.state['ewh_temp']
            hour = int(env.state['time_index'] // 2)  # Ensure hour is integer
            if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak water usage hours
                target_temp, low_temp, high_temp = 55, 50, 60
            else:  # Non-peak hours
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

            # New: safety margin analysis (not provided by Constraint Loss)
            ess_safety_margin_mean = np.mean(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_mean = np.mean(episode_ev_safety_margins) if episode_ev_safety_margins else 0
            ess_safety_margin_std = np.std(episode_ess_safety_margins) if episode_ess_safety_margins else 0
            ev_safety_margin_std = np.std(episode_ev_safety_margins) if episode_ev_safety_margins else 0

            peak_valley_arbitrage = np.mean(episode_peak_valley_arbitrages) if episode_peak_valley_arbitrages else 0

            # Training stability (use variance of returns from last 10 episodes)
            if len(episode_returns) >= 10:
                recent_returns = episode_returns[-10:]
                training_stability = 1.0 / (1.0 + np.std(recent_returns))  # Lower variance means higher stability
            else:
                training_stability = 0.0

            # Sample efficiency (simplified calculation)
            sample_efficiency = episode_return / step_count if step_count > 0 else 0

            # Write detailed data for current episode to CSV
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
                float(agent.lambda_ess) if agent.constraint_mode == "lagrangian" else 0.0,
                float(agent.lambda_ev) if agent.constraint_mode == "lagrangian" else 0.0,
                constraint_weight,
                # New: save mean violation amount
                ess_violation_mean, ev_violation_mean
            ])
            file.flush()  # Ensure data is written to file immediately

            print(f"Episode {episode + 1}, Return: {episode_return:.2f}, Actor Loss: {actor_loss:.4f}, "
                  f"Critic Loss: {critic_loss:.4f}, Constraint Loss: {constraint_loss:.4f}, "
                  f"Total Loss: {total_loss:.4f}, Violation Score: {total_violation_rate:.3f}, Cost: {energy_cost:.2f}, "
                  f"λ_ESS: {float(agent.lambda_ess) if agent.constraint_mode == 'lagrangian' else 0:.2f}, "
                  f"λ_EV: {float(agent.lambda_ev) if agent.constraint_mode == 'lagrangian' else 0:.2f}, "
                  f"ESS Violation Rate: {ess_violation_rate:.3f}, EV Violation Rate: {ev_violation_rate:.3f}, "
                  f"ESS Violation Mean: {ess_violation_mean:.4f}, EV Violation Mean: {ev_violation_mean:.4f}")



            # Record total cost at end of each episode
            env.episode_costs.append(env.total_cost)

            # ========== Save simulation data to CSV ==========
            # Only save data from last 50 episodes
            if episode >= num_episodes - 50:
                data_filename = f"simulation_data_episode_{episode + 1}.csv"
                env.save_simulation_data(data_filename)
                print(f"Episode {episode + 1}: Simulation data saved")

            # ========== Episode end, save current model to recent_models queue ==========
            model_snapshot = {
                'episode': episode,
                'return': episode_return,
                'shared_backbone': agent.shared_backbone.state_dict(),
                'actor_branches': agent.actor_branches.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_optimizer': agent.actor_optim.state_dict(),
                'critic_optimizer': agent.critic_optim.state_dict(),
        'action_mapping': agent.action_mapping,
        'state_keys': state_keys,
                # Complete training_config field to ensure evaluation script compatibility
        'training_config': {
                    'state_dim': len(env.state_space),  # State dimension
                    'hidden_dim': 128,  # Hidden layer dimension
                    'action_space_config': env.action_space,  # Action space configuration
            'gamma': agent.gamma,
            'lmbda': agent.lmbda,
            'eps': agent.eps,
            'epochs': agent.epochs,
            'ent_coef': agent.ent_coef,
            'max_grad_norm': agent.max_grad_norm,
            'use_state_normalization': USE_STATE_NORMALIZATION,
                    'constraint_mode': CONSTRAINT_MODE,
                    'use_dynamic_mask': USE_DYNAMIC_MASK,
                    'constraint_config': agent.constraint_config.copy() if agent.constraint_mode == "lagrangian" else None
        }
    }
            # If state normalization is used, save running_stats
    if USE_STATE_NORMALIZATION and running_stats is not None:
            model_snapshot['running_stats_mean'] = running_stats.mean.clone()
            model_snapshot['running_stats_std'] = running_stats.std.clone()
            model_snapshot['running_stats_count'] = running_stats.count
            # If in lagrangian mode, save lambda values
    if CONSTRAINT_MODE == "lagrangian":
            model_snapshot['lambda_ess'] = agent.lambda_ess.clone()
            model_snapshot['lambda_ev'] = agent.lambda_ev.clone()
            # Key: save snapshot for each episode to ensure recent_models is not empty
    recent_models.append(model_snapshot)

    # Save cost data after training ends
    env.save_episode_costs()

    # ========== Save final simulation data after training ends ==========
    print("Training completed, saving final simulation data...")
    final_data_filename = f"final_simulation_data_episode_{num_episodes}.csv"
    env.save_simulation_data(final_data_filename)
    print(f"Final simulation data saved: {final_data_filename}")

    # ========== Generate final visualization images ==========
    print("Generating final visualization images...")
    env.visualize()
    print("Final visualization images generated")

# ***********************************************************************************

    # Save trained model
    # ==================== After training ends, save the model with highest return from recent 50 episodes ====================
    # Find the one with highest return in the recent_models queue
    if recent_models:
        best_model_snapshot = max(recent_models, key=lambda x: x['return'])
        # Reorganize save dictionary for easier loading later
        model_save_dict = {
            'shared_backbone_state_dict': best_model_snapshot['shared_backbone'],
            'actor_branches_state_dict': best_model_snapshot['actor_branches'],
            'critic_state_dict': best_model_snapshot['critic'],
            'actor_optimizer_state_dict': best_model_snapshot['actor_optimizer'],
            'critic_optimizer_state_dict': best_model_snapshot['critic_optimizer'],
            'action_mapping': best_model_snapshot['action_mapping'],
            'state_keys': best_model_snapshot['state_keys'],
            'training_config': best_model_snapshot['training_config']
        }
        if USE_STATE_NORMALIZATION and 'running_stats_mean' in best_model_snapshot:
            model_save_dict.update({
                'running_stats_mean': best_model_snapshot['running_stats_mean'],
                'running_stats_std': best_model_snapshot['running_stats_std'],
                'running_stats_count': best_model_snapshot['running_stats_count'],
        })
        if CONSTRAINT_MODE == "lagrangian" and 'lambda_ess' in best_model_snapshot:
            model_save_dict.update({
                'lambda_ess': best_model_snapshot['lambda_ess'],
                'lambda_ev': best_model_snapshot['lambda_ev'],
                'constraint_config': best_model_snapshot['constraint_config']
        })
        # Save model
        model_save_dir = "model/saved_models"
        os.makedirs(model_save_dir, exist_ok=True)
        norm_suffix = "_norm" if USE_STATE_NORMALIZATION else "_no_norm"
        constraint_suffix = "_constrained" if CONSTRAINT_MODE == "lagrangian" else "_unconstrained"
        model_filename = os.path.join(model_save_dir, f"ppo_model_{timestamp}{norm_suffix}{constraint_suffix}.pth")
        torch.save(model_save_dict, model_filename)
        print(f"Automatically selected and saved the model with highest return from recent 50 episodes to: {model_filename}")
    else:
        print("Warning: No model snapshots available to save!")

# **************************************************************************

    # Close file after training ends (with statement will automatically close)
    env.visualize()
    env.plot_reward_components()
    plot_returns(episode_returns)

    print(f"Training completed! Returns data saved to: {csv_filename}")