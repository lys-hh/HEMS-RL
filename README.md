# HEMS (Home Energy Management System) - Reinforcement Learning Project

## Project Overview

This project uses reinforcement learning techniques to optimize home energy management systems, enabling intelligent energy scheduling and cost optimization. It supports multiple advanced RL algorithms including PPO, DQN, Rainbow DQN, DDPG, SAC, and TD3.

## 🚀 Quick Start

### Requirements
- Python 3.8+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib
- Gym/Gymnasium

### Installation
```bash
# Install all dependencies using requirements.txt
pip install -r requirements.txt

# Or manually install main dependencies
pip install torch numpy pandas matplotlib gym
```

### Quick Run
```bash
# Train Rainbow DQN
python model/dqn.py

# Train PPO
python model/PPO_3rd.py

# Generate visualization results
python evaluation/plot_from_csv.py
```

## Project Structure

```
HEMS_project/
├── model/                    # Reinforcement learning algorithm implementations
│   ├── PPO_1st.py           # Basic PPO implementation
│   ├── PPO_2nd.py           # Improved PPO version
│   ├── PPO_3rd.py           # Flagship PPO (currently in use)
│   ├── sac.py               # SAC algorithm implementation
│   ├── sac2.py              # SAC improved version
│   ├── TD3.py               # TD3 algorithm implementation
│   ├── dqn.py               # Rainbow DQN algorithm implementation
│   ├── double_dqn.py        # Double DQN algorithm implementation
│   ├── ddpg.py              # DDPG continuous control algorithm
│   ├── evaluation.py        # Model evaluation tools
│   └── rl_utils.py          # Reinforcement learning utility functions
├── evaluation/              # Evaluation and visualization modules
│   ├── model_evaluation.py  # Model performance evaluation
│   ├── plot_from_csv.py     # CSV data visualization
│   ├── strategy_carbon_comparison.py  # Strategy comparison experiments
│   └── README.md            # Evaluation module documentation
├── figures/                 # Figure organization
│   ├── algorithm_comparison/    # Algorithm comparison plots
│   ├── environment_plots/       # Environment state plots
│   ├── experiment_results/      # Experiment result plots
│   └── README.md            # Figure organization documentation
├── simulation_data/         # Simulation data storage
├── environment.py           # Home energy management environment
├── interface.py             # Data interface
├── README.md               # Project documentation
└── README_data_saving.md   # Data saving functionality documentation
```

## Core Features

### 1. Intelligent Energy Scheduling
- **Electric Vehicle Charging/Discharging**: Smart scheduling based on electricity prices and demand
- **Energy Storage System Management**: Optimize charging/discharging timing
- **Appliance Scheduling**: Intelligent scheduling for washing machines, air conditioners, and other devices
- **Water Heater Control**: Optimize heating strategy based on water usage needs

### 2. Constraint Optimization
- **SOC Constraints**: Ensure battery SOC stays within safe range (10%-90%)
- **Power Constraints**: Limit charging/discharging power
- **Temperature Comfort**: Guarantee user temperature comfort
- **Cost Optimization**: Minimize energy costs

### 3. Algorithm Features
- **PPO-Lagrangian**: Supports constrained reinforcement learning
- **State Normalization**: Improves training stability
- **Dynamic Masking**: Ensures action validity
- **Automatic Rollback**: Prevents training crashes

## Supported Algorithms

### 1. Rainbow DQN Algorithm (dqn.py)
- **Rainbow DQN**: Combines multiple DQN improvements including Double DQN, Dueling Networks, Prioritized Experience Replay, Multi-step Learning, Distributional RL, and Noisy Networks.
- **Improvements**:
  - **Double DQN**: Separates action selection and evaluation to reduce Q-value overestimation
  - **Dueling Networks**: Separates state value and advantage functions to improve learning efficiency
  - **Prioritized Experience Replay**: Prioritizes important experiences to improve sample efficiency
  - **Multi-step Learning**: Multi-step learning reduces bias
  - **Noisy Networks**: Replaces epsilon-greedy with noisy networks for better exploration
  - **Huber Loss**: Uses Huber loss to improve training stability

#### Run Command
```bash
python model/dqn.py
```

#### Key Parameters
- `buffer_size`: Experience replay buffer size (default: 100000)
- `batch_size`: Batch size for each update (default: 256)
- `n_step`: Multi-step learning steps (default: 3)
- `alpha`: Prioritized experience replay exponent (default: 0.6)
- `beta`: Importance sampling exponent (default: 0.4)
- `gamma`: Discount factor (default: 0.96)
- `tau`: Target network soft update coefficient (default: 0.005)
- `USE_STATE_NORMALIZATION`: Enable state normalization

### 2. Double DQN Algorithm (double_dqn.py)
- **Double DQN**: Reduces Q-value overestimation by separating action selection and evaluation, improving training stability and performance.
- **Improvements**:
  - Uses main network for action selection, target network for evaluation
  - Adds Dropout layers to prevent overfitting
  - Uses Huber loss for stability
  - Adds gradient clipping and learning rate scheduling
  - Includes L2 regularization

#### Run Command
```bash
python model/double_dqn.py
```

#### Key Parameters
- `buffer_size`: Experience replay buffer size (default: 100000)
- `batch_size`: Batch size for each update (default: 256)
- `epsilon_start/epsilon_end/epsilon_decay`: Exploration rate parameters
- `gamma`: Discount factor (default: 0.96)
- `tau`: Target network soft update coefficient (default: 0.005)
- `USE_STATE_NORMALIZATION`: Enable state normalization

### 3. DDPG Continuous Control Algorithm (ddpg.py)
- **DDPG**: Adapted for multi-device, multi-action spaces, supports continuous action output, OU noise exploration, state normalization, constraint statistics, and experience replay.
- **Purpose**: Provides DDPG-based reinforcement learning solution for HEMS environment, suitable for continuous action spaces.

#### Run Command
```bash
python model/ddpg.py
```

#### Key Parameters
- `buffer_size`: Experience replay buffer size (default: 100000)
- `batch_size`: Batch size for each update (default: 128)
- `lr_actor/lr_critic`: Actor/Critic learning rates
- `gamma`: Discount factor (default: 0.96)
- `tau`: Target network soft update coefficient (default: 0.005)
- `USE_STATE_NORMALIZATION`: Enable state normalization

### 4. PPO Algorithm Series
- **PPO_1st.py**: Basic PPO implementation
- **PPO_2nd.py**: Improved PPO with constraint handling
- **PPO_3rd.py**: Flagship PPO supporting PPO-Lagrangian constrained reinforcement learning

#### Run Command
```bash
python model/PPO_3rd.py
```

## Usage

### Training Models
```bash
# Train Rainbow DQN
python model/dqn.py

# Train PPO
python model/PPO_3rd.py

# Train DDPG
python model/ddpg.py
```

### Evaluating Models
```bash
python evaluation/model_evaluation.py
```

### Visualizing Results
```bash
# Generate plots from CSV data
python evaluation/plot_from_csv.py

# Algorithm comparison experiments
python evaluation/strategy_carbon_comparison.py
```

## Configuration Parameters

### Constraint Configuration
- `soc_lower`: SOC lower bound (default: 0.2)
- `soc_upper`: SOC upper bound (default: 0.8)
- `lambda_init`: Initial Lagrangian multiplier (default: 0.5)
- `lambda_max`: Maximum Lagrangian multiplier (default: 15.0)

### Training Configuration
- `num_episodes`: Training episodes (default: 5000)
- `learning_rate`: Learning rate (default: 5e-6)
- `gamma`: Discount factor (default: 0.96)
- `eps`: PPO clipping parameter (default: 0.2)

## Monitoring Metrics

### Constraint Metrics
- **ESS_Violation_Rate**: ESS violation rate
- **EV_Violation_Rate**: EV violation rate
- **Lambda_ESS**: ESS Lagrangian multiplier
- **Lambda_EV**: EV Lagrangian multiplier

### Performance Metrics
- **Return**: Total return
- **Energy_Cost**: Energy cost
- **User_Satisfaction**: User satisfaction
- **Peak_Valley_Arbitrage**: Peak-valley arbitrage effect
- **Carbon_Emission**: Carbon emissions (kg CO2)
- **Carbon_Reduction**: Carbon emission reduction effect

### Training Metrics
- **Actor_Loss**: Policy loss
- **Critic_Loss**: Value loss
- **Constraint_Loss**: Constraint loss
- **Training_Stability**: Training stability

## Data Saving

The project supports detailed training data saving functionality, including:
- Simulation data CSV files
- Reward component breakdown
- Automatic visualization generation

For detailed documentation, see: [README_data_saving.md](README_data_saving.md)

## Notes

1. **Constraint Training**: Currently uses PPO-Lagrangian method, requires careful parameter tuning
2. **State Normalization**: Recommended to enable for improved training stability
3. **Dynamic Weights**: Constraint weights dynamically adjust with training progress
4. **Automatic Rollback**: Automatically rolls back to healthy state when returns collapse
5. **File Paths**: Ensure all dependency file paths are correct

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please contact through:
- Submit an Issue
- Email the project maintainer

## Changelog

### v2.0.0 (2025-08-05)
- Added Rainbow DQN algorithm implementation
- Enhanced data saving and visualization functionality
- Optimized project documentation structure

### v1.0.0 (2025-04-30)
- Initial release
- Support for basic algorithms: PPO, DQN, DDPG
- Home energy management environment implementation
