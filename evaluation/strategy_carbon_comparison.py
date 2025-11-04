"""
Strategy carbon emission comparison experiment
Objective: Compare carbon emission effects of different strategies (random strategy, various trained models) under different configurations
Extensible: Can easily add more algorithm models for comparison
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import importlib
import types

# Add project root directory to Python path (using relative path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# Add model directory to path
sys.path.append(os.path.join(project_root, 'model'))

from environment import HomeEnergyManagementEnv
import random
from datetime import datetime
import warnings
import multiprocessing as mp

# Set matplotlib backend and warnings
warnings.filterwarnings('ignore')  # Ignore warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Set font to support English
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Compatibility shim: Some serialized files depend on numpy._core (NumPy 2.x naming),
# which doesn't exist in current environment (NumPy 1.21.x), causing torch.load deserialization to fail.
# If numpy._core is not detected, proxy numpy.core as numpy._core.
try:
    if importlib.util.find_spec('numpy._core') is None:
        import numpy.core as _np_core
        _proxy = types.ModuleType('numpy._core')
        _proxy.__dict__.update(_np_core.__dict__)
        sys.modules['numpy._core'] = _proxy
except Exception:
    pass

class StrategyCarbonComparison:
    def __init__(self):
        # Get relative path of project root (define first)
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.current_dir)
        
        # Carbon intensity data loading
        self.carbon_intensity_data = None
        self.carbon_intensity_start_date = None
        self.load_carbon_intensity_data()
        
        # Experiment parameters
        self.num_episodes = 4
        self.episode_length = 48  # 7 days
        
        # Experiment time period selection (optional)
        self.experiment_start_date = None  # Experiment start date, if None use data start date
        model_dir = os.path.join(self.project_root, 'model', 'saved_models')
        
        # Ensure directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Strategy configuration: supports extending multiple strategies
        self.strategies = {
            'baseline_grid': {
                'name': 'Grid Baseline Strategy',
                'model_path': None,
                'strategy_func': self.get_baseline_grid_action
            },
            'proposed_rl': {
                'name': 'PPO Algorithm',
                'model_path': os.path.join(model_dir, 'proposed_rl.pth'),
                'strategy_func': self.get_model_action
            },
            'rainbow_dqn': {
                'name': 'Rainbow DQN',
                'model_path': os.path.join(model_dir, 'rainbow_dqn_model_20250805_022938_norm.pth'),
                'strategy_func': self.get_model_action
            },
            # 'ddpg': {
            #     'name': 'DDPG Algorithm',
            #     'model_path': os.path.join(model_dir, 'ddpg.pth'),
            #     'strategy_func': self.get_model_action
            # },
            'td3': {
                'name': 'TD3 Algorithm',
                'model_path': os.path.join(model_dir, 'td3_model_min.pth'),
                'strategy_func': self.get_model_action
            },
            'sac': {
                'name': 'SAC Algorithm',
                'model_path': os.path.join(model_dir, 'sac2_model_min.pth'),
                'strategy_func': self.get_model_action
            }
        }
        
        # Experiment configuration: unified use of full configuration
        self.configurations = {
            'full_optimization': {
                'name': 'Full Optimization Configuration',
                'mask_type': 'full'
            }
        }
        
        # Result storage: {strategy}_{config} format
        self.results = {}
        
        # Load all models
        self.loaded_models = {}
        self.load_all_models()
    
    def load_carbon_intensity_data(self):
        """Load carbon intensity data"""
        try:
            carbon_data_path = os.path.join(self.project_root, 'data', 'nsw_carbon_intensity.csv')
            print(f"📊 Loading carbon intensity data: {carbon_data_path}")
            
            # Read CSV file
            df = pd.read_csv(carbon_data_path)
            
            # Convert timestamp
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
            
            # Convert carbon intensity from gCO₂/kWh to kgCO₂/kWh
            df['carbon_intensity_kgco2_per_kwh'] = df['carbon_intensity_gco2_per_kwh_nsw'] / 1000.0
            
            # Store data
            self.carbon_intensity_data = df
            self.carbon_intensity_start_date = df['timestamp_utc'].min()
            
            print(f"✅ Carbon intensity data loaded successfully")
            print(f"   Data range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
            print(f"   Data points: {len(df)}")
            print(f"   Carbon intensity range: {df['carbon_intensity_kgco2_per_kwh'].min():.3f} - {df['carbon_intensity_kgco2_per_kwh'].max():.3f} kgCO₂/kWh")
            
        except Exception as e:
            print(f"❌ Carbon intensity data loading failed: {e}")
            print("   Will use default price-based carbon intensity mapping")
            # Fallback to original method
            self.carbon_intensity_data = None
            self.carbon_intensity_mapping = {
                'low_valley': 0.3,    # Low valley: dominated by clean energy
                'flat': 0.6,          # Flat period: mix of clean energy and thermal power  kgCO2/kWh
                'peak': 0.9           # Peak period: dominated by thermal power  kgCO2/kWh
            }
    
    def set_experiment_period(self, start_date_str):
        """Set experiment time period
        
        Args:
            start_date_str: Start date string in format like '2019-01-01' or '2019-01-01 00:00:00'
        """
        try:
            # Parse date and ensure timezone information (UTC)
            self.experiment_start_date = pd.to_datetime(start_date_str)
            if self.experiment_start_date.tz is None:
                self.experiment_start_date = self.experiment_start_date.tz_localize('UTC')
            print(f"📅 Experiment time period set: {self.experiment_start_date}")
            
            # Validate date is within data range
            if self.carbon_intensity_data is not None:
                data_start = self.carbon_intensity_data['timestamp_utc'].min()
                data_end = self.carbon_intensity_data['timestamp_utc'].max()
                
                if self.experiment_start_date < data_start:
                    print(f"⚠️ Warning: Experiment start date {self.experiment_start_date} is earlier than data start date {data_start}")
                elif self.experiment_start_date > data_end:
                    print(f"⚠️ Warning: Experiment start date {self.experiment_start_date} is later than data end date {data_end}")
                else:
                    print(f"✅ Experiment time period set successfully")
                    
        except Exception as e:
            print(f"❌ Failed to set experiment time period: {e}")
            self.experiment_start_date = None
    
    def _safe_torch_load(self, model_path):
        """Compatibility-enhanced torch.load, try to avoid deserialization failures due to environment differences"""
        # Try weights_only first (if PyTorch version supports)
        try:
            return torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            pass
        except Exception:
            pass
        # Regular loading
        try:
            return torch.load(model_path, map_location='cpu')
        except Exception:
            pass
        # Try using pickle with latin1 encoding fallback
        try:
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e:
            raise e

    def _extract_checkpoint_keys_worker(self, model_path, required_keys, conn):
        """Worker function for subprocess: only extract required keys, avoid main process hanging"""
        try:
            import sys as _sys
            import importlib as _importlib
            import types as _types
            import os as _os
            # Limit subprocess thread count to avoid BLAS/OMP hanging
            try:
                _os.environ.setdefault('OMP_NUM_THREADS', '1')
                _os.environ.setdefault('MKL_NUM_THREADS', '1')
            except Exception:
                pass
            # Inject numpy._core compatibility shim in subprocess to avoid structseq-related deserialization crashes
            try:
                if _importlib.util.find_spec('numpy._core') is None:
                    import numpy.core as _np_core
                    _proxy = _types.ModuleType('numpy._core')
                    _proxy.__dict__.update(_np_core.__dict__)
                    _sys.modules['numpy._core'] = _proxy
            except Exception:
                pass

            import torch
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            ckpt = None
            # 1) Try weights_only first (if available)
            try:
                ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            except TypeError:
                pass
            except Exception:
                pass
            # 2) Regular loading
            if ckpt is None:
                try:
                    ckpt = torch.load(model_path, map_location='cpu')
                except Exception:
                    ckpt = None
            # 3) Pickle latin1 fallback
            if ckpt is None:
                try:
                    import pickle as _pickle
                    with open(model_path, 'rb') as f:
                        ckpt = _pickle.load(f, encoding='latin1')
                except Exception as e:
                    conn.send((False, str(e)))
                    return
            slim = {}
            if isinstance(required_keys, dict):
                # Allow (key -> default) form
                for k, default in required_keys.items():
                    slim[k] = ckpt.get(k, default)
            else:
                for k in required_keys:
                    if k in ckpt:
                        slim[k] = ckpt[k]
            conn.send((True, slim))
        except Exception as e:
            conn.send((False, str(e)))
        finally:
            conn.close()

    def _load_checkpoint_minimal(self, model_path, required_keys, timeout_sec=180):
        """Minimally load checkpoint in subprocess, only return specified keys, give up if timeout"""
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = mp.Process(target=self._extract_checkpoint_keys_worker, args=(model_path, required_keys, child_conn))
        proc.start()
        proc.join(timeout=timeout_sec)
        if proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            return None, f"load timeout (> {timeout_sec}s)"
        if parent_conn.poll():
            ok, payload = parent_conn.recv()
            if ok:
                return payload, None
            return None, payload
        return None, "unknown load failure"
        
    def load_all_models(self):
        """Load all configured models"""
        print("🔍 Starting to load models...")
        for strategy_key, strategy_config in self.strategies.items():
            model_path = strategy_config['model_path']
            print(f"📁 Checking strategy {strategy_key}: path = {model_path}")
            
            if model_path and os.path.exists(model_path):
                print(f"✅ File exists, starting to load {strategy_config['name']} model...")
                model = self.load_model(model_path, strategy_key)
                self.loaded_models[strategy_key] = model
                if model:
                    print(f"✅ {strategy_config['name']} model loaded successfully")
                else:
                    print(f"❌ {strategy_config['name']} model loading failed")
            else:
                self.loaded_models[strategy_key] = None
                if model_path:
                    print(f"❌ {strategy_config['name']} model file does not exist: {model_path}")
                    # Debug: show directory contents
                    model_dir = os.path.dirname(model_path)
                    if os.path.exists(model_dir):
                        print(f"   📂 Directory contents: {os.listdir(model_dir)}")
                    else:
                        print(f"   📂 Directory does not exist: {model_dir}")
                else:
                    print(f"ℹ️ {strategy_config['name']} does not need to load model")
    
    def load_model(self, model_path, strategy_key):
        """Load single model (according to strategy type)"""
        try:
            print(f"   ⏳ Reading checkpoint({strategy_key}) -> {model_path}")
            # SAC/TD3 use subprocess minimal loading to avoid main process hanging
            if strategy_key in ('sac', 'td3'):
                # For already minimized files, directly safe load in main process to avoid subprocess handle issues on Windows
                if os.path.basename(model_path).endswith('_min.pth'):
                    checkpoint = self._safe_torch_load(model_path)
                    print(f"   ✅ Checkpoint read({strategy_key}), contains keys: {list(checkpoint.keys())}")
                else:
                    required = ['training_config', 'state_keys', 'actor_state_dict']
                    checkpoint, err = self._load_checkpoint_minimal(model_path, required, timeout_sec=180)
                    if checkpoint is None:
                        print(f"   ❌ {strategy_key} subprocess minimal loading failed: {err}")
                        return None
                    print(f"   ✅ Checkpoint read({strategy_key}), contains keys: {list(checkpoint.keys())}")
            else:
                # Use compatibility-enhanced safe loading to avoid deserialization issues in different environments
                checkpoint = self._safe_torch_load(model_path)
                print(f"   ✅ Checkpoint read({strategy_key}), contains keys: {list(checkpoint.keys())[:6]} ...")
            
            if strategy_key == 'proposed_rl':
                return self.load_ppo_model(checkpoint)
            elif strategy_key == 'rainbow_dqn':
                return self.load_rainbow_dqn_model(checkpoint)
            # elif strategy_key == 'ddpg':
            #     return self.load_ddpg_model(checkpoint)
            elif strategy_key == 'td3':
                return self.load_td3_model(checkpoint)
            elif strategy_key == 'sac':
                return self.load_sac_model(checkpoint)
            else:
                print(f"⚠️ Unknown strategy type: {strategy_key}")
                return None
                
        except Exception as e:
            print(f"❌ {strategy_key} model loading failed: {e}")
            return None
    
    def load_ppo_model(self, checkpoint):
        """Load PPO model"""
        try:
            if isinstance(checkpoint, dict) and 'shared_backbone_state_dict' in checkpoint:
                # Ensure model module can be found
                model_path = os.path.join(self.project_root, 'model')
                if model_path not in sys.path:
                    sys.path.append(model_path)
                from PPO_3rd import HomeEnergyPPO
                
                # Create environment instance
                env = HomeEnergyManagementEnv()
                
                # Get training configuration from checkpoint
                training_config = checkpoint['training_config']
                
                # Rebuild complete agent (using PPO_3rd parameters)
                agent = HomeEnergyPPO(
                    env=env,
                    state_dim=training_config['state_dim'],
                    hidden_dim=training_config['hidden_dim'],
                    action_space_config=training_config['action_space_config'],
                    gamma=training_config['gamma'],
                    lmbda=training_config['lmbda'],
                    eps=training_config['eps'],
                    epochs=training_config['epochs'],
                    ent_coef=training_config['ent_coef'],
                    max_grad_norm=training_config['max_grad_norm'],
                    device='cpu',
                    constraint_mode=training_config.get('constraint_mode', 'none'),
                    use_state_normalization=training_config.get('use_state_normalization', True),
                    use_advantage_normalization=True,
                    use_dynamic_mask=training_config.get('use_dynamic_mask', True),
                    constraint_config=training_config.get('constraint_config', None)
                )
                
                # Load model weights
                agent.shared_backbone.load_state_dict(checkpoint['shared_backbone_state_dict'])
                agent.actor_branches.load_state_dict(checkpoint['actor_branches_state_dict'])
                
                # Set to evaluation mode
                agent.shared_backbone.eval()
                for branch in agent.actor_branches.values():
                    branch.eval()
                
                # Load running statistics (for state normalization)
                running_stats = None
                if 'running_stats_mean' in checkpoint:
                    from PPO_3rd import RunningStats
                    running_stats = RunningStats(shape=training_config['state_dim'])
                    running_stats.mean = checkpoint['running_stats_mean']
                    running_stats.std = checkpoint['running_stats_std']
                    running_stats.count = checkpoint['running_stats_count']
                
                return {
                    'agent': agent,
                    'running_stats': running_stats,
                    'state_keys': checkpoint['state_keys']
                }
            else:
                print("❌ PPO model format not supported")
                print(f"   Expected key: shared_backbone_state_dict")
                print(f"   Actual keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict type'}")
                return None
        except Exception as e:
            print(f"❌ PPO model loading failed: {e}")
            return None
    
    def load_rainbow_dqn_model(self, checkpoint):
        """Load Rainbow DQN model"""
        try:
            # Ensure model module can be found
            model_path = os.path.join(self.project_root, 'model')
            if model_path not in sys.path:
                sys.path.append(model_path)
            from dqn import RainbowDQN
            
            # Create environment instance
            env = HomeEnergyManagementEnv()
            
            # Get configuration from checkpoint
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                
                # Rebuild agent (force CPU usage)
                import torch
                original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                
                agent = RainbowDQN(
                    state_dim=config['state_dim'],
                    hidden_dim=config['hidden_dim'],
                    action_space_config=config['action_space_config'],
                    lr=1e-4,  # Use default value, training parameters not saved in model file
                    gamma=config.get('gamma', 0.96),
                    tau=config.get('tau', 0.01),
                    buffer_size=100000,  # Use default value
                    batch_size=config.get('batch_size', 512),
                    epsilon_start=1.0,  # Use default value
                    epsilon_end=0.05,   # Use default value
                    epsilon_decay=0.995,  # Use default value
                    n_step=config.get('n_step', 3),
                    alpha=config.get('alpha', 0.6),
                    beta=config.get('beta', 0.4)
                )
                
                # Ensure all components are on CPU
                agent.device = torch.device('cpu')
                
                # Load model weights
                agent.shared_backbone.load_state_dict(checkpoint['shared_backbone_state_dict'])
                agent.q_branches.load_state_dict(checkpoint['q_branches_state_dict'])
                
                # Set to evaluation mode and ensure on CPU
                agent.shared_backbone.eval()
                agent.shared_backbone.to('cpu')
                for branch in agent.q_branches.values():
                    branch.eval()
                    branch.to('cpu')
                
                # Load running statistics (if exists)
                running_stats = None
                if 'running_stats_mean' in checkpoint:
                    from PPO_3rd import RunningStats
                    running_stats = RunningStats(shape=config['state_dim'])
                    running_stats.mean = checkpoint['running_stats_mean']
                    running_stats.std = checkpoint['running_stats_std']
                    running_stats.count = checkpoint['running_stats_count']
                
                return {
                    'agent': agent,
                    'running_stats': running_stats,
                    'state_keys': checkpoint.get('state_keys', None)
                }
            else:
                print("❌ Rainbow DQN model format not supported")
                return None
                
        except Exception as e:
            print(f"❌ Rainbow DQN model loading failed: {e}")
            return None
    
    # def load_ddpg_model(self, checkpoint):
    #     """Load DDPG model"""
    #     try:
    #         # Ensure model module can be found
    #         model_path = os.path.join(self.project_root, 'model')
    #         if model_path not in sys.path:
    #             sys.path.append(model_path)
    #         from ddpg import DDPG, Actor, Critic, ActionConverter
            
    #         # Get configuration from checkpoint
    #         if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
    #             config = checkpoint['training_config']
                
    #             # Reconstruct agent
    #             agent = DDPG(config['state_dim'], config['action_dim'], config['action_space_config'])
                
    #             # Load model weights
    #             agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    #             agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                
    #             # Set to evaluation mode
    #             agent.actor.eval()
    #             agent.critic.eval()
                
    #             # Load running statistics (if exists)
    #             running_stats = None
    #             if 'running_stats_mean' in checkpoint:
    #                 from ddpg import RunningStats
    #                 running_stats = RunningStats(shape=config['state_dim'])
    #                 running_stats.mean = checkpoint['running_stats_mean']
    #                 running_stats.std = checkpoint.get('running_stats_std', checkpoint.get('running_stats_var', 1.0))
    #                 running_stats.count = checkpoint['running_stats_count']
                
    #             return {
    #                 'agent': agent,
    #                 'running_stats': running_stats,
    #                 'state_keys': checkpoint.get('state_keys', None)
    #             }
    #         else:
    #             print("❌ DDPG model format not supported")
    #             return None
                
    #     except Exception as e:
    #         print(f"❌ DDPG model loading failed: {e}")
    #         return None
    
    def load_td3_model(self, checkpoint):
        """Load TD3 model"""
        try:
            # Directly use lightweight inference wrapper to avoid hanging caused by importing TD3 module
            # Get configuration from checkpoint
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                
                # Lightweight action converter (consistent mapping with TD3)
                class _TD3ActionConverter:
                    def __init__(self, action_space_config):
                        self.action_map = action_space_config
                    def _convert_single(self, value, options):
                        scaled = (value + 1) / 2
                        idx = int(round(scaled * (len(options) - 1)))
                        return options[max(0, min(idx, len(options) - 1))]
                    def continuous_to_discrete(self, continuous_action):
                        return {
                            'ev_power': self._convert_single(continuous_action[0], self.action_map['ev_power']),
                            'battery_power': self._convert_single(continuous_action[1], self.action_map['battery_power']),
                            'wash_machine_schedule': self._convert_single(continuous_action[2], self.action_map['wash_machine_schedule']),
                            'Air_conditioner_set_temp': self._convert_single(continuous_action[3], self.action_map['Air_conditioner_set_temp']),
                            'Air_conditioner_set_temp2': self._convert_single(continuous_action[4], self.action_map['Air_conditioner_set_temp2']),
                            'ewh_set_temp': self._convert_single(continuous_action[5], self.action_map['ewh_set_temp'])
                        }

                # Lightweight network structure (consistent with TD3 Actor)
                class _TD3Actor(torch.nn.Module):
                    def __init__(self, state_dim, action_dim):
                        super().__init__()
                        self.net = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, 512),
                            torch.nn.ReLU(),
                            torch.nn.Linear(512, 256),
                            torch.nn.ReLU(),
                            torch.nn.Linear(256, action_dim)
                        )
                    def forward(self, state):
                        return self.net(state)

                class _TD3Agent:
                    def __init__(self, state_dim, action_dim, action_space_config):
                        self.actor = _TD3Actor(state_dim, action_dim)
                        self.converter = _TD3ActionConverter(action_space_config)

                agent = _TD3Agent(config['state_dim'], config['action_dim'], config['action_space_config'])

                # Load weights (only for inference)
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.actor.eval()
                
                # Load running statistics (if exists)
                running_stats = None
                if 'running_stats_mean' in checkpoint:
                    # Lightweight RunningStats
                    class _RunningStats:
                        def __init__(self, shape):
                            self.mean = np.zeros(shape, dtype=np.float32)
                            self.std = np.ones(shape, dtype=np.float32)
                            self.count = 1.0
                        def normalize(self, x):
                            return (x - self.mean) / (self.std + 1e-8)
                    running_stats = _RunningStats(shape=config['state_dim'])
                    running_stats.mean = checkpoint['running_stats_mean']
                    running_stats.std = checkpoint.get('running_stats_std', checkpoint.get('running_stats_var', 1.0))
                    running_stats.count = checkpoint['running_stats_count']
                
                return {
                    'agent': agent,
                    'running_stats': running_stats,
                    'state_keys': checkpoint.get('state_keys', None)
                }
            else:
                print("❌ TD3 model format not supported")
                return None
                
        except Exception as e:
            print(f"❌ TD3 model loading failed: {e}")
            return None
    
    def load_sac_model(self, checkpoint):
        """Load SAC model"""
        try:
            print("   🔧 SAC: Entering lightweight loading logic")
            # Use lightweight inference wrapper to avoid hanging caused by importing sac2 module
            # Get configuration from checkpoint
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                print(f"   🔧 SAC: Reading training_config: state_dim={config.get('state_dim')}, action_dim={config.get('action_dim')}")
                
                # Lightweight SAC components (consistent inference interface with sac2)
                class _SACActionConverter:
                    def __init__(self):
                        self.action_map = {
                            'ev_power': [-6.6, -3.3, 0, 3.3, 6.6],
                            'battery_power': [-4.4, -2.2, 0, 2.2, 4.4],
                            'wash_machine_schedule': [0, 1, 2, 3, 4, 5, 6],
                            'Air_conditioner_set_temp': [16, 18, 20, 22, 24, 26, 28, 30],
                            'Air_conditioner_set_temp2': [16, 18, 20, 22, 24, 26, 28, 30],
                            'ewh_set_temp': [40, 45, 50, 55, 60, 65, 70]
                        }
                    def _convert_single(self, value, options):
                        scaled = (value + 1) / 2
                        idx = int(round(scaled * (len(options) - 1)))
                        return options[max(0, min(idx, len(options) - 1))]
                    def continuous_to_discrete(self, continuous_action):
                        return {
                            'ev_power': self._convert_single(continuous_action[0], self.action_map['ev_power']),
                            'battery_power': self._convert_single(continuous_action[1], self.action_map['battery_power']),
                            'wash_machine_schedule': self._convert_single(continuous_action[2], self.action_map['wash_machine_schedule']),
                            'Air_conditioner_set_temp': self._convert_single(continuous_action[3], self.action_map['Air_conditioner_set_temp']),
                            'Air_conditioner_set_temp2': self._convert_single(continuous_action[4], self.action_map['Air_conditioner_set_temp2']),
                            'ewh_set_temp': self._convert_single(continuous_action[5], self.action_map['ewh_set_temp'])
                        }

                # Infer hidden dimension from weights to ensure structure alignment
                try:
                    inferred_hidden = checkpoint['actor_state_dict']['net.0.weight'].shape[0]
                except Exception:
                    inferred_hidden = config.get('hidden_dim', 512)

                class _SACActor(torch.nn.Module):
                    def __init__(self, state_dim, action_dim, hidden_dim):
                        super().__init__()
                        self.net = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.LayerNorm(hidden_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim),
                            torch.nn.LayerNorm(hidden_dim),
                            torch.nn.ReLU()
                        )
                        self.mean = torch.nn.Linear(hidden_dim, action_dim)
                        self.log_std = torch.nn.Linear(hidden_dim, action_dim)
                        # Consistent with training side: don't include action_scale/bias as part of state_dict
                        self.action_scale = torch.tensor([1.0] * action_dim)
                        self.action_bias = torch.tensor([0.0] * action_dim)
                    def forward(self, state):
                        x = self.net(state)
                        mean = self.mean(x)
                        log_std = self.log_std(x)
                        log_std = torch.clamp(log_std, min=-20, max=2)
                        return mean, log_std
                    def sample(self, state):
                        mean, log_std = self.forward(state)
                        std = torch.exp(log_std)
                        normal = torch.distributions.Normal(mean, std)
                        x_t = normal.rsample()
                        y_t = torch.tanh(x_t)
                        action = y_t * self.action_scale + self.action_bias
                        log_prob = normal.log_prob(x_t)
                        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
                        return action, log_prob.sum(1, keepdim=True)

                class _SACAgent:
                    def __init__(self, state_dim, action_dim):
                        self.actor = _SACActor(state_dim, action_dim, inferred_hidden)
                        self.converter = _SACActionConverter()
                    def select_action(self, state_tensor):
                        with torch.no_grad():
                            if isinstance(state_tensor, np.ndarray):
                                state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)
                            action_cont, _ = self.actor.sample(state_tensor)
                        return self.converter.continuous_to_discrete(action_cont.cpu().numpy()[0])

                print("   🔧 SAC: Building lightweight Actor")
                agent = _SACAgent(config['state_dim'], config['action_dim'])
                print("   🔧 SAC: Loading actor_state_dict")
                agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
                print("   🔧 SAC: Setting eval mode")
                agent.actor.eval()
                
                return {
                    'agent': agent,
                    'running_stats': None,  # SAC usually doesn't use running statistics
                    'state_keys': checkpoint.get('state_keys', None)
                }
            else:
                print("❌ SAC model format not supported")
                return None
                
        except Exception as e:
            print(f"❌ SAC model loading failed: {e}")
            return None
    
    def get_carbon_intensity(self, state, step_index):
        """Get corresponding carbon intensity based on state and time step"""
        if self.carbon_intensity_data is not None:
            try:
                # Use real carbon intensity data
                # Calculate current time point (based on step_index and start date)
                start_date = self.experiment_start_date if self.experiment_start_date is not None else self.carbon_intensity_start_date
                current_time = start_date + pd.Timedelta(minutes=step_index * 30)
                
                # Find closest time point (carbon intensity data every 5 minutes, environment every 30 minutes per step)
                # Select nearest data point before current time point
                mask = self.carbon_intensity_data['timestamp_utc'] <= current_time
                available_data = self.carbon_intensity_data[mask]
                
                if len(available_data) > 0:
                    # Take last available data point
                    carbon_intensity = available_data.iloc[-1]['carbon_intensity_kgco2_per_kwh']
                    return carbon_intensity
                else:
                    # If no available data, use first data point
                    carbon_intensity = self.carbon_intensity_data.iloc[0]['carbon_intensity_kgco2_per_kwh']
                    return carbon_intensity
                    
            except Exception as e:
                print(f"⚠️ Failed to get real carbon intensity data: {e}")
                # Fallback to price-based method
                electricity_price = state.get('electricity_price', 0.5)
                return self._get_carbon_intensity_by_price(electricity_price)
        else:
            # Use price-based method
            electricity_price = state.get('electricity_price', 0.5)
            return self._get_carbon_intensity_by_price(electricity_price)
    
    def _get_carbon_intensity_by_price(self, electricity_price):
        """Get corresponding carbon intensity based on electricity price (fallback method)"""
        if electricity_price <= 0.2:  # Low valley period
            return self.carbon_intensity_mapping['low_valley']
        elif electricity_price >= 0.8:  # Peak period
            return self.carbon_intensity_mapping['peak']
        else:  # Flat period
            return self.carbon_intensity_mapping['flat']
    
    
    def get_baseline_grid_action(self, state, env, episode_num=0):
        """Grid baseline strategy: traditional scheduling approach"""
        electricity_price = state.get('electricity_price', 0.5)
        hour = int(state.get('time_index', 0) // 2)  # Convert to hours
        day = int(state.get('time_index', 0) // 48)  # Convert to days
        
        action = {}
        
        # 1. EV strategy: consider away status and energy consumption
        # Check if EV is at home
        ev_at_home = env.is_ev_at_home()
        ev_soc = state.get('ev_battery_state', 0) / 24  # Convert to 0-1 range
        
        if ev_at_home:
            # EV at home: fixed maximum power charging, no V2G function
            if ev_soc < 0.9:  # Charge when SOC below 90%
                action['ev_power'] = 6.6  # Maximum charging power
            else:
                action['ev_power'] = 0
        else:
            # EV away: cannot charge or discharge
            action['ev_power'] = 0
        
        # 2. ESS strategy: completely off
        action['battery_power'] = 0
        
        # 3. AC strategy: temperature control
        indoor_temp = state.get('indoor_temp', 24)
        indoor_temp2 = state.get('indoor_temp2', 18)
        
        # AC1: control in 20-24°C range
        if indoor_temp > 24:  # Temperature too high
            action['Air_conditioner_set_temp'] = 20  # Cool down
        elif indoor_temp < 20:  # Temperature too low
            action['Air_conditioner_set_temp'] = 24  # Heat up
        else:
            action['Air_conditioner_set_temp'] = 22  # Maintain comfort
        
        # AC2: control in 16-20°C range
        if indoor_temp2 > 20:  # Temperature too high
            action['Air_conditioner_set_temp2'] = 16  # Cool down
        elif indoor_temp2 < 16:  # Temperature too low
            action['Air_conditioner_set_temp2'] = 20  # Heat up
        else:
            action['Air_conditioner_set_temp2'] = 18  # Maintain comfort
        
        # 4. Washing machine strategy: use once at random time each day
        import random
        random.seed(day * 1000 + episode_num)  # Ensure reproducible randomness for each day
        daily_random_hour = random.randint(6, 22)  # Randomly select between 6-22 hours
        
        if hour == daily_random_hour:  # Random time of the day
            action['wash_machine_schedule'] = 1  # Use immediately
        else:
            action['wash_machine_schedule'] = 0  # Don't use
        
        # 5. Water heater strategy: temperature control (50-60°C range)
        ewh_temp = state.get('ewh_temp', 55)
        
        if ewh_temp < 50:  # Temperature too low
            action['ewh_set_temp'] = 65  # Maximum power heating to high temperature
        elif ewh_temp > 60:  # Temperature too high
            action['ewh_set_temp'] = 45  # Stop heating
        else:
            action['ewh_set_temp'] = 55  # Maintain target temperature
        
        return action
    
    def get_model_action(self, state, env, strategy_key='proposed_rl'):
        """Get action using trained model (using environment action mask)"""
        model = self.loaded_models.get(strategy_key)
        if model is None:
            raise ValueError(f"❌ {strategy_key} model not loaded, cannot execute model action")
        
        try:
            if strategy_key == 'baseline_grid':
                action = self.get_baseline_grid_action(state, env)
            elif strategy_key == 'proposed_rl':
                action = self.get_ppo_action(state, env, model)
            elif strategy_key == 'rainbow_dqn':
                action = self.get_rainbow_dqn_action(state, env, model)
            # elif strategy_key == 'ddpg':
            #     action = self.get_ddpg_action(state, env, model)
            elif strategy_key == 'td3':
                action = self.get_td3_action(state, env, model)
            elif strategy_key == 'sac':
                action = self.get_sac_action(state, env, model)
            else:
                raise ValueError(f"❌ Unknown strategy type: {strategy_key}")
            
            # Check if action is valid
            if action is None:
                raise ValueError(f"❌ {strategy_key} model returned None action, model inference failed")
            
            return action
                
        except Exception as e:
            print(f"❌ {strategy_key} model prediction failed: {e}")
            raise  # Re-raise exception to let caller know the real problem
    
    def get_ppo_action(self, state, env, model):
        """Get PPO model action"""
        agent = model['agent']
        
        # Get environment action mask
        action_mask = env.get_action_mask(state)
        
        # Convert state to model expected format
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # State normalization
        if model['running_stats'] is not None:
            normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
        else:
            normalized_state = state_tensor
        
        # Ensure model components are on CPU and set to float32
        agent.shared_backbone.to('cpu').float()
        for branch in agent.actor_branches.values():
            branch.to('cpu').float()
        normalized_state = normalized_state.float()
        
        # Use agent to predict action (pass action mask)
        with torch.no_grad():
            action_result = agent.take_action(normalized_state, action_mask)
            action_dict = action_result[0]  # Get actions dictionary
            
            # Validate action dictionary validity
            if not isinstance(action_dict, dict):
                raise ValueError(f"❌ PPO model returned invalid action format: {type(action_dict)}")
            
            # Check if necessary action keys exist
            required_keys = ['ev_power', 'battery_power', 'wash_machine_schedule', 
                           'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp']
            for key in required_keys:
                if key not in action_dict:
                    raise ValueError(f"❌ PPO action dictionary missing necessary key: {key}")
        
        return action_dict
    
    def get_rainbow_dqn_action(self, state, env, model):
        """Get Rainbow DQN model action"""
        agent = model['agent']
        
        # Get environment action mask
        action_mask = env.get_action_mask(state)
        
        # Convert state to model expected format
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # State normalization
        if model['running_stats'] is not None:
            normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
        else:
            normalized_state = state_tensor
        
        # Ensure all model components are on CPU and set to float32
        agent.shared_backbone.to('cpu').float()
        for branch in agent.q_branches.values():
            branch.to('cpu').float()
        normalized_state = normalized_state.float()
        
        # Use DQN agent to predict action (pass action mask, no exploration)
        with torch.no_grad():
            action_dict = agent.select_action(normalized_state.cpu(), action_mask, explore=False)
        
        return action_dict
    
    # def get_ddpg_action(self, state, env, model):
    #     """Get DDPG model action"""
    #     agent = model['agent']
        
    #     # Convert state to model expected format
    #     state_vector = self.state_to_vector(state, model['state_keys'])
    #     state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
    #     # State normalization
    #     if model['running_stats'] is not None:
    #         normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
    #     else:
    #         normalized_state = state_tensor
        
    #     # Ensure all model components are on CPU and set to float32
    #     agent.actor.to('cpu').float()
    #     agent.critic.to('cpu').float()
    #     normalized_state = normalized_state.float()
        
    #     # Use DDPG agent to predict action (no noise added, for evaluation)
    #     with torch.no_grad():
    #         continuous_action = agent.actor(normalized_state).cpu().numpy().flatten()
    #         # Use action converter to convert continuous action to discrete action
    #         action_dict = agent.converter.continuous_to_discrete(continuous_action)
            
    #         # Validate action dictionary validity
    #         if not isinstance(action_dict, dict):
    #             raise ValueError(f"❌ DDPG action converter returned invalid action format: {type(action_dict)}")
            
    #         # Check if necessary action keys exist
    #         required_keys = ['ev_power', 'battery_power', 'wash_machine_schedule', 
    #                        'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp']
    #         for key in required_keys:
    #             if key not in action_dict:
    #                 raise ValueError(f"❌ DDPG action dictionary missing necessary key: {key}")
        
    #     return action_dict
    
    def get_td3_action(self, state, env, model):
        """Get TD3 model action"""
        agent = model['agent']
        
        # Convert state to model expected format
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # State normalization
        if model['running_stats'] is not None:
            normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
        else:
            normalized_state = state_tensor
        
        # Ensure all model components are on CPU and set to float32
        agent.actor.to('cpu').float()
        normalized_state = normalized_state.float()
        
        # Use TD3 agent to predict action (no noise added, for evaluation)
        with torch.no_grad():
            continuous_action = agent.actor(normalized_state).cpu().numpy().flatten()
            # Use action converter to convert continuous action to discrete action
            action_dict = agent.converter.continuous_to_discrete(continuous_action)
        
        return action_dict
    
    def get_sac_action(self, state, env, model):
        """Get SAC model action"""
        agent = model['agent']
        
        # Convert state to model expected format
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # Ensure all model components are on CPU and set to float32
        agent.actor.to('cpu').float()
        state_tensor = state_tensor.float()
        
        # Use SAC agent to predict action (no noise added, for evaluation)
        with torch.no_grad():
            action_dict = agent.select_action(state_tensor)
        
        return action_dict
    
    def state_to_vector(self, state, state_keys):
        """Convert state dictionary to vector"""
        if state_keys:
            state_vector = [state[k] for k in state_keys]
        else:
            # Fallback: sort by alphabetical order
            ordered_keys = sorted(state.keys())
            state_vector = [state[k] for k in ordered_keys]
        return state_vector
    
    def apply_device_mask(self, action, mask_type):
        """Apply device mask (enable/disable energy storage devices)"""
        masked_action = action.copy()
        
        # To ensure strict control variable, basic electrical devices use fixed strategy
        masked_action['Air_conditioner_set_temp'] = 24     # Fixed temperature
        masked_action['Air_conditioner_set_temp2'] = 24    # Fixed temperature
        masked_action['ewh_set_temp'] = 55                 # Fixed temperature
        
        if mask_type == 'baseline':
            # Control group: disable energy storage function
            masked_action['battery_power'] = 0
            if masked_action['ev_power'] < 0:
                masked_action['ev_power'] = 0
                
        elif mask_type == 'v2g_only':
            # V2G group: partial energy storage function
            masked_action['battery_power'] = 0
            
        elif mask_type == 'full':
            # Full optimization group: allow all functions
            pass
        
        return masked_action
    
    def apply_comfort_mask(self, action, state, strategy_key, env=None):
        """Apply comfort mask for algorithms except grid baseline and proposed_rl"""
        # Grid baseline and proposed_rl don't need mask
        if strategy_key in ['baseline_grid', 'proposed_rl']:
            return action
        
        masked_action = action.copy()
        mask_applied = False
        
        # Get current state
        indoor_temp = state.get('indoor_temp', 24)
        indoor_temp2 = state.get('indoor_temp2', 24)
        ewh_temp = state.get('ewh_temp', 50)
        ev_soc = state.get('ev_battery_state', 12) / 24  # Convert to 0-1 range
        ess_soc = state.get('battery_soc', 0.5)  # Assume already in 0-1 range
        
        # Check if EV is at home
        ev_at_home = True  # Default at home
        if env is not None:
            ev_at_home = env.is_ev_at_home()
        
        # AC1 mask: more "clumsy" mask, force fixed temperature
        if indoor_temp > 26:  # Relax temperature range, more clumsy
            masked_action['Air_conditioner_set_temp'] = 22  # Fixed temperature, no choice
            mask_applied = True
        elif indoor_temp < 18:  # Relax temperature range, more clumsy
            masked_action['Air_conditioner_set_temp'] = 22  # Fixed temperature, no choice
            mask_applied = True
        
        # AC2 mask: more "clumsy" mask, force fixed temperature
        if indoor_temp2 > 22:  # Relax temperature range, more clumsy
            masked_action['Air_conditioner_set_temp2'] = 18  # Fixed temperature, no choice
            mask_applied = True
        elif indoor_temp2 < 14:  # Relax temperature range, more clumsy
            masked_action['Air_conditioner_set_temp2'] = 18  # Fixed temperature, no choice
            mask_applied = True
        
        # Water heater mask: more "clumsy" mask, force fixed temperature
        if ewh_temp < 45:  # Relax temperature range, more clumsy
            masked_action['ewh_set_temp'] = 55  # Fixed temperature, no choice
            mask_applied = True
        elif ewh_temp > 65:  # Relax temperature range, more clumsy
            masked_action['ewh_set_temp'] = 55  # Fixed temperature, no choice
            mask_applied = True
        
        # EV SOC mask: more "clumsy" mask, force fixed power
        if ev_at_home:
            # EV at home: apply SOC constraints, but more clumsy
            if ev_soc < 0.1:  # SOC too low, force charging (stricter threshold)
                masked_action['ev_power'] = 6.6  # Fixed maximum charging power, no choice
                mask_applied = True
            elif ev_soc > 0.9:  # SOC too high, force discharging (stricter threshold)
                masked_action['ev_power'] = -6.6  # Fixed maximum discharging power, no choice
                mask_applied = True
        else:
            # EV away: force prohibit charging/discharging
            masked_action['ev_power'] = 0
            mask_applied = True
        
        # ESS SOC mask: force guide to meet control objectives (relax constraints, allow normal charging/discharging)
        if ess_soc < 0.05:  # SOC too low, must charge (lower threshold)
            original_power = action['battery_power']
            masked_action['battery_power'] = max(original_power, 0)  # Force charging or stop discharging
            if original_power != masked_action['battery_power']:
                mask_applied = True
        elif ess_soc > 0.95:  # SOC too high, must discharge or stop charging (raise threshold)
            original_power = action['battery_power']
            masked_action['battery_power'] = min(original_power, 0)  # Force discharging or stop charging
            if original_power != masked_action['battery_power']:
                mask_applied = True
        
        # Debug info: record mask application
        # if mask_applied:
        #     ev_status = "at home" if ev_at_home else "away"
        #     print(f"🔧 Comfort mask applied to {strategy_key}: "
        #           f"AC1_temp={indoor_temp:.1f}°C(20-24°C), AC2_temp={indoor_temp2:.1f}°C(16-20°C), "
        #           f"EWH_temp={ewh_temp:.1f}°C(50-60°C), EV_SOC={ev_soc:.2f}({ev_status}), ESS_SOC={ess_soc:.2f}")
        
        return masked_action
    
    def calculate_home_total_carbon(self, env, state, step_index):
        """Calculate carbon emissions for total home load"""
        carbon_intensity = self.get_carbon_intensity(state, step_index)
        
        # Get total home load
        total_home_load = env.total_load_compute()
        
        # Correction: calculate carbon emissions based on net purchase (selling electricity reduces carbon emissions from purchased electricity)
        if total_home_load > 0:  # Home needs to purchase electricity from grid
            grid_purchase = total_home_load * 0.5
            grid_sale = 0
            # Purchase generates carbon emissions: purchase amount × carbon intensity at that time
            step_carbon = grid_purchase * carbon_intensity
        else:  # Home sells electricity to grid
            grid_purchase = 0
            grid_sale = abs(total_home_load) * 0.5
            # Selling reduces carbon emissions: sale amount × carbon intensity at that time (negative value indicates emission reduction)
            step_carbon = -grid_sale * carbon_intensity
        
        return step_carbon, grid_purchase, grid_sale, carbon_intensity, total_home_load
    
    def run_single_episode(self, strategy_key, config_key, episode_num):
        """Run single episode"""
        env = HomeEnergyManagementEnv()
        state = env.reset()
        
        strategy_config = self.strategies[strategy_key]
        config = self.configurations[config_key]
        
        # Initialize episode data records
        episode_data = {
            'episode': episode_num,
            'strategy': strategy_config['name'],
            'configuration': config['name'],
            'total_carbon': 0,
            'total_cost': 0,
            'total_grid_purchase': 0,
            'total_grid_sale': 0,
            'ev_charge_energy': 0,
            'ev_discharge_energy': 0,
            'ess_charge_energy': 0,
            'ess_discharge_energy': 0,
            'vehicle_storage_actions': 0,
            'battery_storage_actions': 0,
            'carbon_intensity_avg': 0,
            'home_load_avg': 0,
            'comfort_score_sum': 0,
            'ev_initial_soc': state['ev_battery_state'] / 24,
            'ess_initial_soc': state['ess_state'] / 24,
            'ev_final_soc': 0,
            'ess_final_soc': 0
        }
        
        for step in range(self.episode_length):
            # Get strategy action
            if strategy_key == 'baseline_grid':
                raw_action = self.get_baseline_grid_action(state, env, episode_num)
            else:
                try:
                    raw_action = self.get_model_action(state, env, strategy_key)
                except Exception as e:
                    print(f"❌ Episode {episode_num}, Step {step}: {strategy_key} model failed: {e}")
                    raise  # Re-raise exception to stop experiment
            
            # Apply device mask
            action = self.apply_device_mask(raw_action, config['mask_type'])
            
            # Disable comfort mask to allow all algorithms to compete fairly
            # action = self.apply_comfort_mask(action, state, strategy_key, env)
            
            # Execute action, update environment state
            next_state, reward, done = env.step(state, action)
            
            # Calculate total home carbon emissions for current step
            step_carbon, grid_purchase, grid_sale, carbon_intensity, total_home_load = self.calculate_home_total_carbon(
                env, state, step
            )
            
            # Record carbon emissions and grid interaction data
            episode_data['total_carbon'] += step_carbon
            episode_data['total_grid_purchase'] += grid_purchase
            episode_data['total_grid_sale'] += grid_sale
            episode_data['carbon_intensity_avg'] += carbon_intensity
            episode_data['home_load_avg'] += abs(total_home_load)

            # Cost (priced by purchase amount and electricity price; grid_purchase already includes 0.5h step coefficient)
            step_cost = grid_purchase * state['electricity_price']
            episode_data['total_cost'] += step_cost
            
            
            # Record energy data
            if action['ev_power'] > 0:
                episode_data['ev_charge_energy'] += action['ev_power'] * 0.5
            elif action['ev_power'] < 0:
                episode_data['ev_discharge_energy'] += abs(action['ev_power']) * 0.5
                episode_data['vehicle_storage_actions'] += 1
            
            if action['battery_power'] > 0:
                episode_data['ess_charge_energy'] += action['battery_power'] * 0.5
            elif action['battery_power'] < 0:
                episode_data['ess_discharge_energy'] += abs(action['battery_power']) * 0.5
                episode_data['battery_storage_actions'] += 1
            
            # Update state
            state = next_state

            # Comfort (simple indicator based on room temperature preference and hot water temperature range)
            try:
                temp_diff1 = abs(state.get('indoor_temp', 24) - state.get('user_temp_preference', 24))
                temp_diff2 = abs(state.get('indoor_temp2', 24) - state.get('user_temp_preference2', 24))
                temp_comfort1 = max(0, 1 - max(0, temp_diff1 - 2) / 8)
                temp_comfort2 = max(0, 1 - max(0, temp_diff2 - 2) / 8)

                ewh_temp = state.get('ewh_temp', 50)
                hour = int(state.get('time_index', 0) // 2)
                if 6 <= hour <= 9 or 18 <= hour <= 22:
                    low_temp, high_temp = 50, 60
                else:
                    low_temp, high_temp = 40, 50
                if low_temp <= ewh_temp <= high_temp:
                    ewh_temp_comfort = 1.0
                else:
                    deviation = max(low_temp - ewh_temp, ewh_temp - high_temp)
                    ewh_temp_comfort = max(0, 1 - deviation / 10)

                comfort_step = (temp_comfort1 + temp_comfort2 + ewh_temp_comfort) / 3.0
                episode_data['comfort_score_sum'] += comfort_step
            except Exception:
                pass
        
        # Record final SOC
        episode_data['ev_final_soc'] = state['ev_battery_state'] / 24
        episode_data['ess_final_soc'] = state['ess_state'] / 24
        
        # Calculate average values
        episode_data['carbon_intensity_avg'] /= self.episode_length
        episode_data['home_load_avg'] /= self.episode_length
        episode_data['comfort_score_avg'] = episode_data['comfort_score_sum'] / self.episode_length
        
        return episode_data
    
    def run_experiment_group(self, strategy_key, config_key):
        """Run a group of experiments"""
        strategy_name = self.strategies[strategy_key]['name']
        config_name = self.configurations[config_key]['name']
        group_key = f"{strategy_key}_{config_key}"
        
        print(f"🔄 Running {strategy_name} - {config_name}...")
        
        results = []
        for episode in range(self.num_episodes):
            episode_data = self.run_single_episode(strategy_key, config_key, episode)
            results.append(episode_data)
        
        self.results[group_key] = results
        print(f"✅ {strategy_name} - {config_name} completed")
        
        return results
    
    def run_all_experiments(self):
        """Run all experiments"""
        print("🚀 Starting strategy carbon emission comparison experiment...")
        
        # Run all combinations of strategies and configurations
        for strategy_key in self.strategies.keys():
            for config_key in self.configurations.keys():
                self.run_experiment_group(strategy_key, config_key)
        
        print("\n🎉 All experiment groups completed!")
    
    def analyze_results(self):
        """Analyze experiment results"""
        # Convert to DataFrame
        all_results = []
        for group_key, results in self.results.items():
            for result in results:
                result['group_key'] = group_key
                all_results.append(result)
        
        df_all = pd.DataFrame(all_results)
        
        # Calculate average values for each group
        results_summary = {}
        for group_key, results in self.results.items():
            df = pd.DataFrame(results)
            results_summary[group_key] = {
                'avg_carbon': df['total_carbon'].mean(),
                'avg_cost': df['total_cost'].mean(),
                'avg_grid_purchase': df['total_grid_purchase'].mean(),
                'avg_grid_sale': df['total_grid_sale'].mean(),
                'avg_ev_charge': df['ev_charge_energy'].mean(),
                'avg_ev_discharge': df['ev_discharge_energy'].mean(),
                'avg_ess_charge': df['ess_charge_energy'].mean(),
                'avg_ess_discharge': df['ess_discharge_energy'].mean(),
                'avg_vehicle_storage_actions': df['vehicle_storage_actions'].mean(),
                'avg_battery_storage_actions': df['battery_storage_actions'].mean(),
                'avg_comfort': df['comfort_score_avg'].mean(),
                'avg_ess_final_soc': df['ess_final_soc'].mean(),
                'avg_ev_final_soc': df['ev_final_soc'].mean(),
                'strategy': df['strategy'].iloc[0],
                'configuration': df['configuration'].iloc[0]
            }
        
        # Print result analysis
        print(f"\n{'='*70}")
        print(f"📊 Algorithm strategy carbon emission comparison experiment results analysis (corrected electricity sales emission reduction logic)")
        print(f"{'='*70}")
        print(f"💡 Correction note: selling electricity can reduce carbon emissions from purchased electricity, calculate carbon emissions based on net purchase")
        
        # Carbon emission comparison for each strategy
        print(f"\n🔍 Algorithm strategy carbon emission comparison under full configuration:")
        print(f"{'Algorithm Strategy':<20} {'Average Carbon Emission(kg CO2)':<20}")
        print(f"{'-'*45}")
        
        carbon_data = {}
        for strategy_key, strategy in self.strategies.items():
            group_key = f"{strategy_key}_full_optimization"
            if group_key in results_summary:
                summary = results_summary[group_key]
                carbon_data[strategy['name']] = summary['avg_carbon']
                print(f"{strategy['name']:<20} {summary['avg_carbon']:<20.2f}")
        
        # Print detailed carbon emission comparison data
        print(f"\n📊 Detailed carbon emission comparison data:")
        for strategy_name, carbon_value in carbon_data.items():
            print(f"  {strategy_name}: {carbon_value:.4f} kg CO2")
        
        # Cancel ESS remaining energy carbon emission correction to compare all algorithms on same baseline
        # results_summary = self.calculate_ess_remaining_carbon_correction(results_summary)
        
        # Plot all algorithms ESS SOC change comparison
        self.plot_algorithms_ess_soc_comparison(results_summary)
        
        # Plot all algorithms EV SOC change comparison
        self.plot_algorithms_ev_soc_comparison(results_summary)
        
        # If there's a baseline strategy, calculate improvement magnitude
        if 'baseline_grid' in [k for k in self.strategies.keys()]:
            baseline_key = "baseline_grid_full_optimization"
            if baseline_key in results_summary:
                baseline_carbon = results_summary[baseline_key]['avg_carbon']
                print(f"\n📈 Improvement relative to grid baseline strategy (original carbon emission comparison):")
                print(f"  Grid baseline strategy: {baseline_carbon:.4f} kg CO2")
                
                for strategy_key, strategy in self.strategies.items():
                    if strategy_key != 'baseline_grid':
                        model_key = f"{strategy_key}_full_optimization"
                        if model_key in results_summary:
                            model_carbon = results_summary[model_key]['avg_carbon']
                            improvement = baseline_carbon - model_carbon
                            improvement_pct = (improvement / baseline_carbon * 100) if baseline_carbon > 0 else 0
                            print(f"  {strategy['name']}: Reduction {improvement:.4f} kg CO2 ({improvement_pct:.2f}%)")

        
        return results_summary, df_all
    
    def calculate_ess_remaining_carbon_correction(self, results_summary):
        """Calculate carbon emission correction for ESS remaining energy"""
        print(f"\n🔋 ESS remaining energy carbon emission correction analysis:")
        print(f"{'='*60}")
        
        # Calculate average carbon intensity (for estimating carbon emissions from remaining energy)
        if self.carbon_intensity_data is not None:
            avg_carbon_intensity = self.carbon_intensity_data['carbon_intensity_kgco2_per_kwh'].mean()
        else:
            avg_carbon_intensity = 0.5  # Default average carbon intensity
        
        print(f"Average carbon intensity: {avg_carbon_intensity:.4f} kg CO2/kWh")
        
        # Grid baseline strategy ESS initial SOC (as baseline)
        baseline_key = "baseline_grid_full_optimization"
        baseline_ess_final_soc = 0.1  # Grid baseline strategy ESS completely off, SOC stays at initial 10%
        
        corrected_results = {}
        
        for group_key, summary in results_summary.items():
            strategy_name = summary['strategy']
            original_carbon = summary['avg_carbon']
            ess_final_soc = summary.get('avg_ess_final_soc', 0.1)
            
            # Calculate ESS remaining energy change relative to baseline
            ess_soc_change = ess_final_soc - baseline_ess_final_soc
            ess_remaining_energy = ess_soc_change * 24  # Convert to kWh (ESS capacity 24kWh)
            
            # Calculate carbon emission impact of remaining energy
            # Remaining energy means can avoid purchasing from grid in future, thus reducing carbon emissions
            carbon_correction = -ess_remaining_energy * avg_carbon_intensity
            
            # Corrected carbon emissions
            corrected_carbon = original_carbon + carbon_correction
            
            # Save corrected results
            corrected_summary = summary.copy()
            corrected_summary['avg_carbon_corrected'] = corrected_carbon
            corrected_summary['ess_soc_change'] = ess_soc_change
            corrected_summary['ess_remaining_energy'] = ess_remaining_energy
            corrected_summary['carbon_correction'] = carbon_correction
            corrected_results[group_key] = corrected_summary
            
            # Print correction information
            print(f"\n{strategy_name}:")
            print(f"  Original carbon emissions: {original_carbon:.4f} kg CO2")
            print(f"  ESS final SOC: {ess_final_soc:.3f}")
            print(f"  ESS SOC change: {ess_soc_change:+.3f}")
            print(f"  Remaining energy: {ess_remaining_energy:+.3f} kWh")
            print(f"  Carbon emission correction: {carbon_correction:+.4f} kg CO2")
            print(f"  Corrected carbon emissions: {corrected_carbon:.4f} kg CO2")
            
            if strategy_name == 'Grid Baseline Strategy':
                print(f"  📌 Baseline strategy, ESS completely off")
            elif ess_soc_change > 0:
                print(f"  💡 ESS stored energy, can reduce carbon emissions in future")
            elif ess_soc_change < 0:
                print(f"  💡 ESS released stored energy, increased current carbon emissions")
        
        return corrected_results
    
    def plot_algorithms_ess_soc_comparison(self, results_summary):
        """Plot ESS SOC change comparison chart for all algorithms"""
        print(f"\n🔍 All algorithms ESS SOC change comparison analysis:")
        
        # List of algorithms to analyze
        algorithms = ['proposed_rl', 'rainbow_dqn', 'td3', 'sac']
        algorithm_names = ['PPO', 'Rainbow DQN', 'TD3', 'SAC']
        colors = ['#4ECDC4', '#45B7D1', '#FECA57', '#9B59B6']
        
        # Store data for all algorithms
        all_traces = {}
        
        # Run one episode for each algorithm
        for i, (algo_key, algo_name) in enumerate(zip(algorithms, algorithm_names)):
            print(f"🧪 Running {algo_name} algorithm episode...")
            
            env = HomeEnergyManagementEnv()
            state = env.reset()
            
            # Record data
            time_steps = []
            ess_soc_trace = []
            ess_power_trace = []
            
            strategy_config = self.strategies[algo_key]
            config = self.configurations['full_optimization']
            
            for step in range(self.episode_length):
                # Get algorithm action
                try:
                    if algo_key == 'proposed_rl':
                        raw_action = self.get_model_action(state, env, algo_key)
                    else:
                        raw_action = self.get_model_action(state, env, algo_key)
                except Exception as e:
                    print(f"❌ {algo_name} model failed: {e}")
                    continue
                
                # Apply device mask and comfort mask
                action = self.apply_device_mask(raw_action, config['mask_type'])
                # Disable comfort mask to allow all algorithms to compete fairly
                # if algo_key != 'proposed_rl':  # Other algorithms need comfort mask
                #     action = self.apply_comfort_mask(action, state, algo_key, env)
                
                # Record current state
                time_steps.append(step)
                ess_soc_trace.append(state['ess_state'] / 24)  # Convert to 0-1
                ess_power_trace.append(action['battery_power'])
                
                # Execute action
                next_state, reward, done = env.step(state, action)
                state = next_state
            
            # Store data
            all_traces[algo_name] = {
                'time_steps': time_steps,
                'ess_soc_trace': ess_soc_trace,
                'ess_power_trace': ess_power_trace,
                'color': colors[i]
            }
        
        # Plot comparison charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. ESS SOC change comparison
        for algo_name, data in all_traces.items():
            ax1.plot(data['time_steps'], data['ess_soc_trace'], 
                    color=data['color'], linewidth=2, label=f'{algo_name}')
        
        ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Initial SOC (10%)')
        ax1.set_ylabel('ESS SOC', fontsize=12)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_title('All Algorithms ESS SOC Change Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. ESS power change comparison
        for algo_name, data in all_traces.items():
            ax2.plot(data['time_steps'], data['ess_power_trace'], 
                    color=data['color'], linewidth=2, label=f'{algo_name}')
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('ESS Power (kW)', fontsize=12)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_title('All Algorithms ESS Power Change Comparison', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save image
        diagnosis_path = os.path.join(self.project_root, 'figures', 'diagnosis')
        os.makedirs(diagnosis_path, exist_ok=True)
        diagnosis_file = os.path.join(diagnosis_path, 'algorithms_ess_soc_comparison.png')
        plt.savefig(diagnosis_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print comparison analysis results
        print(f"\n📊 All algorithms ESS SOC change comparison results:")
        print(f"{'Algorithm':<15} {'Initial SOC':<10} {'Final SOC':<10} {'SOC Change':<10} {'Max Charge':<10} {'Max Discharge':<10}")
        print(f"{'-'*75}")
        
        for algo_name, data in all_traces.items():
            initial_soc = data['ess_soc_trace'][0]
            final_soc = data['ess_soc_trace'][-1]
            soc_change = final_soc - initial_soc
            max_charge = max(data['ess_power_trace'])
            max_discharge = min(data['ess_power_trace'])
            
            print(f"{algo_name:<15} {initial_soc:<10.3f} {final_soc:<10.3f} {soc_change:<+10.3f} "
                  f"{max_charge:<10.2f} {max_discharge:<10.2f}")
        
        print(f"\n💾 Comparison chart saved to: {diagnosis_file}")
        
        return all_traces
    
    def plot_algorithms_ev_soc_comparison(self, results_summary):
        """Plot EV SOC change comparison chart for all algorithms"""
        print(f"\n🔍 All algorithms EV SOC change comparison analysis:")
        
        # List of algorithms to analyze
        algorithms = ['proposed_rl', 'rainbow_dqn', 'td3', 'sac']
        algorithm_names = ['PPO', 'Rainbow DQN', 'TD3', 'SAC']
        colors = ['#4ECDC4', '#45B7D1', '#FECA57', '#9B59B6']
        
        # Store data for all algorithms
        all_traces = {}
        
        # Run one episode for each algorithm
        for i, (algo_key, algo_name) in enumerate(zip(algorithms, algorithm_names)):
            print(f"🧪 Running {algo_name} algorithm EV episode...")
            
            env = HomeEnergyManagementEnv()
            state = env.reset()
            
            # Record data
            time_steps = []
            ev_soc_trace = []
            ev_power_trace = []
            ev_at_home_trace = []
            
            strategy_config = self.strategies[algo_key]
            config = self.configurations['full_optimization']
            
            for step in range(self.episode_length):
                # Get algorithm action
                try:
                    raw_action = self.get_model_action(state, env, algo_key)
                except Exception as e:
                    print(f"❌ {algo_name} model failed: {e}")
                    continue
                
                # Apply device mask and comfort mask
                action = self.apply_device_mask(raw_action, config['mask_type'])
                # Disable comfort mask to allow all algorithms to compete fairly
                # if algo_key != 'proposed_rl':  # Other algorithms need comfort mask
                #     action = self.apply_comfort_mask(action, state, algo_key, env)
                
                # Record current state
                time_steps.append(step)
                ev_soc_trace.append(state['ev_battery_state'] / 24)  # Convert to 0-1
                ev_power_trace.append(action['ev_power'])
                ev_at_home_trace.append(env.is_ev_at_home())
                
                # Execute action
                next_state, reward, done = env.step(state, action)
                state = next_state
            
            # Store data
            all_traces[algo_name] = {
                'time_steps': time_steps,
                'ev_soc_trace': ev_soc_trace,
                'ev_power_trace': ev_power_trace,
                'ev_at_home_trace': ev_at_home_trace,
                'color': colors[i]
            }
        
        # Plot comparison charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. EV SOC change comparison
        for algo_name, data in all_traces.items():
            ax1.plot(data['time_steps'], data['ev_soc_trace'], 
                    color=data['color'], linewidth=2, label=f'{algo_name}')
        
        ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Initial SOC (10%)')
        ax1.set_ylabel('EV SOC', fontsize=12)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_title('All Algorithms EV SOC Change Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. EV power change comparison
        for algo_name, data in all_traces.items():
            ax2.plot(data['time_steps'], data['ev_power_trace'], 
                    color=data['color'], linewidth=2, label=f'{algo_name}')
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('EV Power (kW)', fontsize=12)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_title('All Algorithms EV Power Change Comparison', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. EV at home status comparison
        for algo_name, data in all_traces.items():
            ax3.plot(data['time_steps'], data['ev_at_home_trace'], 
                    color=data['color'], linewidth=2, label=f'{algo_name}')
        
        ax3.set_ylabel('EV at Home Status', fontsize=12)
        ax3.set_xlabel('Time Step', fontsize=12)
        ax3.set_title('EV at Home Status Change', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.1, 1.1)
        
        # 4. EV charge/discharge statistics
        algo_names = list(all_traces.keys())
        charge_counts = []
        discharge_counts = []
        idle_counts = []
        
        for algo_name in algo_names:
            data = all_traces[algo_name]
            charge_count = sum(1 for p in data['ev_power_trace'] if p > 0)
            discharge_count = sum(1 for p in data['ev_power_trace'] if p < 0)
            idle_count = sum(1 for p in data['ev_power_trace'] if p == 0)
            
            charge_counts.append(charge_count)
            discharge_counts.append(discharge_count)
            idle_counts.append(idle_count)
        
        x = np.arange(len(algo_names))
        width = 0.25
        
        ax4.bar(x - width, charge_counts, width, label='Charging Steps', color='green', alpha=0.7)
        ax4.bar(x, discharge_counts, width, label='Discharging Steps', color='red', alpha=0.7)
        ax4.bar(x + width, idle_counts, width, label='Idle Steps', color='gray', alpha=0.7)
        
        ax4.set_ylabel('Steps', fontsize=12)
        ax4.set_xlabel('Algorithm', fontsize=12)
        ax4.set_title('EV Charge/Discharge Behavior Statistics', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(algo_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save image
        diagnosis_path = os.path.join(self.project_root, 'figures', 'diagnosis')
        os.makedirs(diagnosis_path, exist_ok=True)
        diagnosis_file = os.path.join(diagnosis_path, 'algorithms_ev_soc_comparison.png')
        plt.savefig(diagnosis_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print comparison analysis results
        print(f"\n📊 All algorithms EV SOC change comparison results:")
        print(f"{'Algorithm':<15} {'Initial SOC':<10} {'Final SOC':<10} {'SOC Change':<10} {'Max Charge':<10} {'Max Discharge':<10} {'At Home Time':<10}")
        print(f"{'-'*85}")
        
        for algo_name, data in all_traces.items():
            initial_soc = data['ev_soc_trace'][0]
            final_soc = data['ev_soc_trace'][-1]
            soc_change = final_soc - initial_soc
            max_charge = max(data['ev_power_trace'])
            max_discharge = min(data['ev_power_trace'])
            home_time = sum(data['ev_at_home_trace'])
            
            print(f"{algo_name:<15} {initial_soc:<10.3f} {final_soc:<10.3f} {soc_change:<+10.3f} "
                  f"{max_charge:<10.2f} {max_discharge:<10.2f} {home_time:<10.0f}")
        
        print(f"\n💾 EV comparison chart saved to: {diagnosis_file}")
        
        return all_traces
    
    def create_visualization(self, results_summary):
        """Create visualization charts: plot carbon emissions, cost, and satisfaction separately (one chart each)"""
        plt.style.use('default')
        strategies = list(self.strategies.keys())
        # Fixed color mapping (original color style, one-to-one correspondence with model names)
        color_map = {
            'baseline_grid': '#95A5A6', # Grid Baseline
            'proposed_rl': '#4ECDC4', # PPO (Proposed RL)
            'rainbow_dqn': '#45B7D1', # Rainbow DQN
            # 'ddpg': '#96CEB4',        # DDPG
            'td3': '#FECA57',         # TD3
            'sac': '#9B59B6',         # SAC
        }
        default_color = '#34495E'

        # Extract data
        names, carbons, costs, comforts, colors = [], [], [], [], []
        for i, strategy_key in enumerate(strategies):
            group_key = f"{strategy_key}_full_optimization"
            if group_key not in results_summary:
                continue
            summary = results_summary[group_key]
            names.append({
                'baseline_grid': 'Grid Baseline',
                'proposed_rl': 'Proposed RL',
                'rainbow_dqn': 'Rainbow DQN',
                # 'ddpg': 'DDPG',
                'td3': 'TD3',
                'sac': 'SAC',
            }.get(strategy_key, strategy_key.upper()))
            carbons.append(summary.get('avg_carbon', 0.0))
            costs.append(summary.get('avg_cost', 0.0))
            comforts.append(summary.get('avg_comfort', 0.0))
            colors.append(color_map.get(strategy_key, default_color))

        x = np.arange(len(names))

        def draw_bar(values, ylabel, title, filename, ylim=None):
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', width=0.6)
            for bar in bars:
                h = bar.get_height()
                if h >= 0:
                    ax.text(bar.get_x() + bar.get_width()/2., h + max(h * 0.01, 1e-6), f'{h:.2f}', ha='center', va='bottom', fontsize=11)
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
            ax.set_title(title, fontweight='bold', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            if ylim is not None:
                ax.set_ylim(*ylim)
            elif values:
                ax.set_ylim(0, max(values) * 1.1 if max(values) > 0 else 1)
            plt.tight_layout()
            figures_dir = os.path.join(self.project_root, 'figures', 'experiment_results')
            os.makedirs(figures_dir, exist_ok=True)
            path = os.path.join(figures_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Chart saved as: {filename}")

        # Plot carbon emission comparison chart
        draw_bar(carbons, 'Carbon (kg CO2)', 'Carbon Emissions', 'strategy_carbon.png')
        
        # Plot new 1×2 chart: carbon intensity change + algorithm carbon emission comparison
        self.create_carbon_analysis_plot(results_summary)
    
    def create_carbon_analysis_plot(self, results_summary):
        """Create 1×2 chart for carbon intensity analysis and algorithm comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # First subplot: carbon intensity factor over time
        if self.carbon_intensity_data is not None:
            # Get carbon intensity data during experiment period
            start_date = self.experiment_start_date if self.experiment_start_date is not None else self.carbon_intensity_start_date
            
            # Calculate data points during experiment period (one point every 30 minutes, 336 points total, 7 days)
            carbon_intensities = []
            time_steps = []
            
            for step in range(self.episode_length):
                current_time = start_date + pd.Timedelta(minutes=step * 30)
                mask = self.carbon_intensity_data['timestamp_utc'] <= current_time
                available_data = self.carbon_intensity_data[mask]
                
                if len(available_data) > 0:
                    carbon_intensity = available_data.iloc[-1]['carbon_intensity_kgco2_per_kwh']
                    carbon_intensities.append(carbon_intensity)
                    time_steps.append(step)
            
            # Plot carbon intensity change curve
            ax1.plot(time_steps, carbon_intensities, color='#1E3A8A', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Time Steps (30min intervals)', fontsize=12)
            ax1.set_ylabel('Carbon Intensity (kg CO₂/kWh)', fontsize=12)
            # ax1.set_title('Carbon Intensity Over Time', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Set x-axis labels (show only a few time steps)
            step_interval = max(1, len(time_steps) // 8)  # Show about 8 labels
            ax1.set_xticks(time_steps[::step_interval])
            ax1.set_xticklabels([f'{t}' for t in time_steps[::step_interval]])
            
            # Add statistics
            avg_intensity = np.mean(carbon_intensities)
            ax1.axhline(y=avg_intensity, color='red', linestyle='--', alpha=0.7, 
                       label=f'Average: {avg_intensity:.3f} kg CO₂/kWh')
            ax1.legend(fontsize=10)
            
            # Add subplot label (a)
            ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontsize=20, 
                    fontweight='bold', verticalalignment='top', horizontalalignment='left')
        else:
            ax1.text(0.5, 0.5, 'No Carbon Intensity Data Available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            # ax1.set_title('Carbon Intensity Over Time', fontweight='bold', fontsize=14)
        
        # Second subplot: algorithm carbon emission comparison
        strategies = list(self.strategies.keys())
        
        # Blue gradient color scheme
        blue_gradient_colors = [
            '#E3F2FD',  # Lightest blue
            '#BBDEFB',  # Light blue
            '#90CAF9',  # Medium light blue
            '#64B5F6',  # Medium blue
            '#42A5F5',  # Medium dark blue
            '#2196F3',  # Dark blue
        ]
        
        # Extract data
        names, carbons, colors = [], [], []
        for i, strategy_key in enumerate(strategies):
            group_key = f"{strategy_key}_full_optimization"
            if group_key not in results_summary:
                continue
            summary = results_summary[group_key]
            names.append({
                'baseline_grid': 'Grid Baseline',
                'proposed_rl': 'Proposed RL',
                'rainbow_dqn': 'Rainbow DQN',
                'td3': 'TD3',
                'sac': 'SAC',
            }.get(strategy_key, strategy_key.upper()))
            carbons.append(summary.get('avg_carbon', 0.0))
            # Use blue gradient color scheme, cycle through colors
            colors.append(blue_gradient_colors[i % len(blue_gradient_colors)])
        
        x = np.arange(len(names))
        bars = ax2.bar(x, carbons, color=colors, alpha=0.85, edgecolor='black', width=0.6)
        
        # Add value labels
        for bar in bars:
            h = bar.get_height()
            if h >= 0:
                ax2.text(bar.get_x() + bar.get_width()/2., h + max(h * 0.01, 1e-6), 
                        f'{h:.2f}', ha='center', va='bottom', fontsize=11)
        
        ax2.set_ylabel('Carbon Emissions (kg CO₂)', fontsize=12)
        ax2.set_xlabel('Algorithm', fontsize=12)
        # ax2.set_title('Algorithm Carbon Emissions Comparison', fontweight='bold', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        if carbons:
            ax2.set_ylim(0, max(carbons) * 1.1 if max(carbons) > 0 else 1)
        
        # Add subplot label (b)
        ax2.text(-0.05, 1.1, 'b', transform=ax2.transAxes, fontsize=20, 
                fontweight='bold', verticalalignment='top', horizontalalignment='left')
        
        plt.tight_layout()
        
        # Save chart
        figures_dir = os.path.join(self.project_root, 'figures', 'experiment_results')
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, 'carbon_analysis_comparison.png')
        plt.savefig(path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"📊 Carbon analysis comparison chart saved as: carbon_analysis_comparison.png")
    
    def save_results(self, results_summary):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure results directory exists
        results_dir = os.path.join(self.project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed data
        for group_key, results in self.results.items():
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(results_dir, f'{group_key}_strategy_comparison_{timestamp}.csv'), index=False)
        
        # Save comparison analysis results
        summary_df = pd.DataFrame(results_summary).T
        summary_df.to_csv(os.path.join(results_dir, f'strategy_comparison_summary_{timestamp}.csv'))
        
        print(f"\n💾 Experiment results saved to ../results/ directory")

def main():
    """Main function"""
    # Create experiment object
    experiment = StrategyCarbonComparison()
    
    # Optional: set specific experiment time period
    experiment.set_experiment_period('2021-07-01')  # Set experiment start date
    
    # Run all experiments
    experiment.run_all_experiments()
    
    # Analyze results
    results_summary, df_all = experiment.analyze_results()
    
    # Create visualization
    experiment.create_visualization(results_summary)
    
    # Save results
    experiment.save_results(results_summary)

if __name__ == "__main__":
    main()
