"""
策略碳排放对比实验
目标：对比不同策略（随机策略、各种训练模型）在不同配置下的碳排放效果
支持扩展：可以轻松添加更多算法模型进行对比
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

# 添加项目根目录到Python路径（使用相对路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# 添加model目录到路径
sys.path.append(os.path.join(project_root, 'model'))

from environment import HomeEnergyManagementEnv
import random
from datetime import datetime
import warnings
import multiprocessing as mp

# 设置matplotlib后端和警告
warnings.filterwarnings('ignore')  # 忽略警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置字体为支持英文的字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 兼容性垫片：部分序列化文件在保存时依赖 numpy._core（NumPy 2.x 命名），
# 在当前环境（NumPy 1.21.x）中不存在该模块，导致 torch.load 反序列化失败。
# 如果检测不到 numpy._core，则将 numpy.core 代理为 numpy._core。
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
        # 碳强度因子设置
        self.carbon_intensity_mapping = {
            'low_valley': 0.3,    # 低谷期：清洁能源主导  
            'flat': 0.6,          # 平段期：清洁能源与火电混合  kgCO2/kWh
            'peak': 0.9           # 高峰期：火电主导  kgCO2/kWh
        }
        
        # 实验参数
        self.num_episodes = 10
        self.episode_length = 336  # 7天
        
        # 获取项目根目录的相对路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.current_dir)
        model_dir = os.path.join(self.project_root, 'model', 'saved_models')
        
        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 策略配置：支持扩展多种策略
        self.strategies = {
            'random': {
                'name': '随机策略',
                'model_path': None,
                'strategy_func': self.get_random_action
            },
            'proposed_rl': {
                'name': 'PPO算法',
                'model_path': os.path.join(model_dir, 'proposed_rl.pth'),
                'strategy_func': self.get_model_action
            },
            'rainbow_dqn': {
                'name': 'Rainbow DQN',
                'model_path': os.path.join(model_dir, 'rainbow_dqn_model_20250805_022938_norm.pth'),
                'strategy_func': self.get_model_action
            },
            'ddpg': {
                'name': 'DDPG算法',
                'model_path': os.path.join(model_dir, 'ddpg.pth'),
                'strategy_func': self.get_model_action
            },
            'td3': {
                'name': 'TD3算法',
                'model_path': os.path.join(model_dir, 'td3_model_min.pth'),
                'strategy_func': self.get_model_action
            },
            'sac': {
                'name': 'SAC算法',
                'model_path': os.path.join(model_dir, 'sac2_model_min.pth'),
                'strategy_func': self.get_model_action
            }
        }
        
        # 实验配置：统一使用完整配置
        self.configurations = {
            'full_optimization': {
                'name': '完整优化配置',
                'mask_type': 'full'
            }
        }
        
        # 结果存储：{strategy}_{config} 格式
        self.results = {}
        
        # 加载所有模型
        self.loaded_models = {}
        self.load_all_models()
    
    def _safe_torch_load(self, model_path):
        """兼容性增强的torch.load，尽量避免环境差异导致的反序列化失败"""
        # 优先尝试weights_only（若PyTorch版本支持）
        try:
            return torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:
            pass
        except Exception:
            pass
        # 常规加载
        try:
            return torch.load(model_path, map_location='cpu')
        except Exception:
            pass
        # 尝试使用pickle以latin1编码回退
        try:
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e:
            raise e

    def _extract_checkpoint_keys_worker(self, model_path, required_keys, conn):
        """子进程工作函数：仅提取所需键，避免主进程卡死"""
        try:
            import sys as _sys
            import importlib as _importlib
            import types as _types
            import os as _os
            # 限制子进程的线程数，避免BLAS/OMP卡死
            try:
                _os.environ.setdefault('OMP_NUM_THREADS', '1')
                _os.environ.setdefault('MKL_NUM_THREADS', '1')
            except Exception:
                pass
            # 子进程内注入 numpy._core 兼容垫片，避免 structseq 相关反序列化崩溃
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
            # 1) 优先 weights_only（若可用）
            try:
                ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            except TypeError:
                pass
            except Exception:
                pass
            # 2) 常规加载
            if ckpt is None:
                try:
                    ckpt = torch.load(model_path, map_location='cpu')
                except Exception:
                    ckpt = None
            # 3) pickle latin1 回退
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
                # 允许(key -> default)形式
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
        """在子进程中最小化加载checkpoint，仅返回指定键，超时则放弃"""
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
        """加载所有配置的模型"""
        print("🔍 开始加载模型...")
        for strategy_key, strategy_config in self.strategies.items():
            model_path = strategy_config['model_path']
            print(f"📁 检查策略 {strategy_key}: 路径 = {model_path}")
            
            if model_path and os.path.exists(model_path):
                print(f"✅ 文件存在，开始加载 {strategy_config['name']} 模型...")
                model = self.load_model(model_path, strategy_key)
                self.loaded_models[strategy_key] = model
                if model:
                    print(f"✅ {strategy_config['name']} 模型加载成功")
                else:
                    print(f"❌ {strategy_config['name']} 模型加载失败")
            else:
                self.loaded_models[strategy_key] = None
                if model_path:
                    print(f"❌ {strategy_config['name']} 模型文件不存在: {model_path}")
                    # 调试：显示目录内容
                    model_dir = os.path.dirname(model_path)
                    if os.path.exists(model_dir):
                        print(f"   📂 目录内容: {os.listdir(model_dir)}")
                    else:
                        print(f"   📂 目录不存在: {model_dir}")
                else:
                    print(f"ℹ️ {strategy_config['name']} 无需加载模型")
    
    def load_model(self, model_path, strategy_key):
        """加载单个模型（根据策略类型）"""
        try:
            print(f"   ⏳ 正在读取checkpoint({strategy_key}) -> {model_path}")
            # SAC/TD3 使用子进程最小化加载，避免主进程卡死
            if strategy_key in ('sac', 'td3'):
                # 对于已精简的最小文件，直接在主进程安全加载，避免Windows下子进程句柄问题
                if os.path.basename(model_path).endswith('_min.pth'):
                    checkpoint = self._safe_torch_load(model_path)
                    print(f"   ✅ checkpoint已读取({strategy_key})，包含键: {list(checkpoint.keys())}")
                else:
                    required = ['training_config', 'state_keys', 'actor_state_dict']
                    checkpoint, err = self._load_checkpoint_minimal(model_path, required, timeout_sec=180)
                    if checkpoint is None:
                        print(f"   ❌ {strategy_key} 子进程最小化加载失败: {err}")
                        return None
                    print(f"   ✅ checkpoint已读取({strategy_key})，包含键: {list(checkpoint.keys())}")
            else:
                # 使用兼容性增强的安全加载，避免不同环境下的反序列化问题
                checkpoint = self._safe_torch_load(model_path)
                print(f"   ✅ checkpoint已读取({strategy_key})，包含键: {list(checkpoint.keys())[:6]} ...")
            
            if strategy_key == 'proposed_rl':
                return self.load_ppo_model(checkpoint)
            elif strategy_key == 'rainbow_dqn':
                return self.load_rainbow_dqn_model(checkpoint)
            elif strategy_key == 'ddpg':
                return self.load_ddpg_model(checkpoint)
            elif strategy_key == 'td3':
                return self.load_td3_model(checkpoint)
            elif strategy_key == 'sac':
                return self.load_sac_model(checkpoint)
            else:
                print(f"⚠️ 未知的策略类型: {strategy_key}")
                return None
                
        except Exception as e:
            print(f"❌ {strategy_key} 模型加载失败: {e}")
            return None
    
    def load_ppo_model(self, checkpoint):
        """加载PPO模型"""
        try:
            if isinstance(checkpoint, dict) and 'shared_backbone_state_dict' in checkpoint:
                # 确保能找到model模块
                model_path = os.path.join(self.project_root, 'model')
                if model_path not in sys.path:
                    sys.path.append(model_path)
                from PPO_3rd import HomeEnergyPPO
                
                # 创建环境实例
                env = HomeEnergyManagementEnv()
                
                # 从checkpoint获取训练配置
                training_config = checkpoint['training_config']
                
                # 重新构建完整的智能体（使用PPO_3rd的参数）
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
                
                # 加载模型权重
                agent.shared_backbone.load_state_dict(checkpoint['shared_backbone_state_dict'])
                agent.actor_branches.load_state_dict(checkpoint['actor_branches_state_dict'])
                
                # 设置为评估模式
                agent.shared_backbone.eval()
                for branch in agent.actor_branches.values():
                    branch.eval()
                
                # 加载运行统计信息（用于状态标准化）
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
                print("❌ PPO模型格式不支持")
                print(f"   期望的键: shared_backbone_state_dict")
                print(f"   实际的键: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else '非字典类型'}")
                return None
        except Exception as e:
            print(f"❌ PPO模型加载失败: {e}")
            return None
    
    def load_rainbow_dqn_model(self, checkpoint):
        """加载Rainbow DQN模型"""
        try:
            # 确保能找到model模块
            model_path = os.path.join(self.project_root, 'model')
            if model_path not in sys.path:
                sys.path.append(model_path)
            from dqn import RainbowDQN
            
            # 创建环境实例
            env = HomeEnergyManagementEnv()
            
            # 从checkpoint获取配置
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                
                # 重新构建智能体（强制使用CPU）
                import torch
                original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                
                agent = RainbowDQN(
                    state_dim=config['state_dim'],
                    hidden_dim=config['hidden_dim'],
                    action_space_config=config['action_space_config'],
                    lr=1e-4,  # 使用默认值，因为模型文件中没有保存这些训练参数
                    gamma=config.get('gamma', 0.96),
                    tau=config.get('tau', 0.01),
                    buffer_size=100000,  # 使用默认值
                    batch_size=config.get('batch_size', 512),
                    epsilon_start=1.0,  # 使用默认值
                    epsilon_end=0.05,   # 使用默认值
                    epsilon_decay=0.995,  # 使用默认值
                    n_step=config.get('n_step', 3),
                    alpha=config.get('alpha', 0.6),
                    beta=config.get('beta', 0.4)
                )
                
                # 确保所有组件都在CPU上
                agent.device = torch.device('cpu')
                
                # 加载模型权重
                agent.shared_backbone.load_state_dict(checkpoint['shared_backbone_state_dict'])
                agent.q_branches.load_state_dict(checkpoint['q_branches_state_dict'])
                
                # 设置为评估模式并确保在CPU上
                agent.shared_backbone.eval()
                agent.shared_backbone.to('cpu')
                for branch in agent.q_branches.values():
                    branch.eval()
                    branch.to('cpu')
                
                # 加载运行统计信息（如果存在）
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
                print("❌ Rainbow DQN模型格式不支持")
                return None
                
        except Exception as e:
            print(f"❌ Rainbow DQN模型加载失败: {e}")
            return None
    
    def load_ddpg_model(self, checkpoint):
        """加载DDPG模型"""
        try:
            # 确保能找到model模块
            model_path = os.path.join(self.project_root, 'model')
            if model_path not in sys.path:
                sys.path.append(model_path)
            from ddpg import DDPG, Actor, Critic, ActionConverter
            
            # 从checkpoint获取配置
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                
                # 重新构建智能体
                agent = DDPG(config['state_dim'], config['action_dim'], config['action_space_config'])
                
                # 加载模型权重
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                
                # 设置为评估模式
                agent.actor.eval()
                agent.critic.eval()
                
                # 加载运行统计信息（如果存在）
                running_stats = None
                if 'running_stats_mean' in checkpoint:
                    from ddpg import RunningStats
                    running_stats = RunningStats(shape=config['state_dim'])
                    running_stats.mean = checkpoint['running_stats_mean']
                    running_stats.std = checkpoint.get('running_stats_std', checkpoint.get('running_stats_var', 1.0))
                    running_stats.count = checkpoint['running_stats_count']
                
                return {
                    'agent': agent,
                    'running_stats': running_stats,
                    'state_keys': checkpoint.get('state_keys', None)
                }
            else:
                print("❌ DDPG模型格式不支持")
                return None
                
        except Exception as e:
            print(f"❌ DDPG模型加载失败: {e}")
            return None
    
    def load_td3_model(self, checkpoint):
        """加载TD3模型"""
        try:
            # 直接使用轻量级推理封装，避免导入TD3模块引发的卡顿
            # 从checkpoint获取配置
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                
                # 轻量版动作转换器（与TD3一致的映射）
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

                # 轻量版网络结构（与TD3中Actor保持一致）
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

                # 加载权重（仅推理所需）
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.actor.eval()
                
                # 加载运行统计信息（如果存在）
                running_stats = None
                if 'running_stats_mean' in checkpoint:
                    # 轻量版RunningStats
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
                print("❌ TD3模型格式不支持")
                return None
                
        except Exception as e:
            print(f"❌ TD3模型加载失败: {e}")
            return None
    
    def load_sac_model(self, checkpoint):
        """加载SAC模型"""
        try:
            print("   🔧 SAC: 进入轻量加载逻辑")
            # 使用轻量级推理封装，避免导入sac2模块引发的卡顿
            # 从checkpoint获取配置
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                print(f"   🔧 SAC: 读取training_config: state_dim={config.get('state_dim')}, action_dim={config.get('action_dim')}")
                
                # 轻量版SAC的组件（与sac2保持一致的推理接口）
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

                # 从权重中推断隐藏维度，保证结构对齐
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
                        # 与训练端一致：不把 action_scale/bias 作为 state_dict 的一部分
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

                print("   🔧 SAC: 构建轻量Actor")
                agent = _SACAgent(config['state_dim'], config['action_dim'])
                print("   🔧 SAC: 加载actor_state_dict")
                agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
                print("   🔧 SAC: 设置eval模式")
                agent.actor.eval()
                
                return {
                    'agent': agent,
                    'running_stats': None,  # SAC通常不使用运行统计
                    'state_keys': checkpoint.get('state_keys', None)
                }
            else:
                print("❌ SAC模型格式不支持")
                return None
                
        except Exception as e:
            print(f"❌ SAC模型加载失败: {e}")
            return None
    
    def get_carbon_intensity(self, electricity_price):
        """根据电价获取对应的碳强度"""
        if electricity_price <= 0.2:  # 低谷期
            return self.carbon_intensity_mapping['low_valley']
        elif electricity_price >= 0.8:  # 高峰期
            return self.carbon_intensity_mapping['peak']
        else:  # 平段期
            return self.carbon_intensity_mapping['flat']
    
    def get_random_action(self, state, env):
        """生成随机动作（使用环境的动作掩码）"""
        # 获取环境的动作掩码
        action_mask = env.get_action_mask(state)
        
        # 根据掩码随机选择有效动作
        action = {}
        
        # EV功率选择
        ev_valid_indices = [i for i, valid in enumerate(action_mask['ev_power']) if valid]
        if ev_valid_indices:
            ev_idx = random.choice(ev_valid_indices)
            action['ev_power'] = env.action_space['ev_power'][ev_idx]
        else:
            action['ev_power'] = 0
        
        # ESS功率选择
        ess_valid_indices = [i for i, valid in enumerate(action_mask['battery_power']) if valid]
        if ess_valid_indices:
            ess_idx = random.choice(ess_valid_indices)
            action['battery_power'] = env.action_space['battery_power'][ess_idx]
        else:
            action['battery_power'] = 0
        
        # 其他设备随机选择
        action['wash_machine_schedule'] = random.choice(env.action_space['wash_machine_schedule'])
        action['Air_conditioner_set_temp'] = random.choice(env.action_space['Air_conditioner_set_temp'])
        action['Air_conditioner_set_temp2'] = random.choice(env.action_space['Air_conditioner_set_temp2'])
        action['ewh_set_temp'] = random.choice(env.action_space['ewh_set_temp'])
        
        return action
    
    def get_model_action(self, state, env, strategy_key='proposed_rl'):
        """使用训练好的模型获取动作（使用环境的动作掩码）"""
        model = self.loaded_models.get(strategy_key)
        if model is None:
            raise ValueError(f"❌ {strategy_key} 模型未加载，无法执行模型动作")
        
        try:
            if strategy_key == 'proposed_rl':
                action = self.get_ppo_action(state, env, model)
            elif strategy_key == 'rainbow_dqn':
                action = self.get_rainbow_dqn_action(state, env, model)
            elif strategy_key == 'ddpg':
                action = self.get_ddpg_action(state, env, model)
            elif strategy_key == 'td3':
                action = self.get_td3_action(state, env, model)
            elif strategy_key == 'sac':
                action = self.get_sac_action(state, env, model)
            else:
                raise ValueError(f"❌ 未知的策略类型: {strategy_key}")
            
            # 检查动作是否有效
            if action is None:
                raise ValueError(f"❌ {strategy_key} 模型返回None动作，模型推理失败")
            
            return action
                
        except Exception as e:
            print(f"❌ {strategy_key} 模型预测失败: {e}")
            raise  # 重新抛出异常，让调用者知道真正的问题
    
    def get_ppo_action(self, state, env, model):
        """获取PPO模型动作"""
        agent = model['agent']
        
        # 获取环境的动作掩码
        action_mask = env.get_action_mask(state)
        
        # 将状态转换为模型期望的格式
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # 状态标准化
        if model['running_stats'] is not None:
            normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
        else:
            normalized_state = state_tensor
        
        # 确保模型组件在CPU上并设置为float32
        agent.shared_backbone.to('cpu').float()
        for branch in agent.actor_branches.values():
            branch.to('cpu').float()
        normalized_state = normalized_state.float()
        
        # 使用智能体预测动作（传入动作掩码）
        with torch.no_grad():
            action_result = agent.take_action(normalized_state, action_mask)
            action_dict = action_result[0]  # 取actions字典
            
            # 验证动作字典的有效性
            if not isinstance(action_dict, dict):
                raise ValueError(f"❌ PPO模型返回了无效的动作格式: {type(action_dict)}")
            
            # 检查必要的动作键是否存在
            required_keys = ['ev_power', 'battery_power', 'wash_machine_schedule', 
                           'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp']
            for key in required_keys:
                if key not in action_dict:
                    raise ValueError(f"❌ PPO动作字典缺少必要的键: {key}")
        
        return action_dict
    
    def get_rainbow_dqn_action(self, state, env, model):
        """获取Rainbow DQN模型动作"""
        agent = model['agent']
        
        # 获取环境的动作掩码
        action_mask = env.get_action_mask(state)
        
        # 将状态转换为模型期望的格式
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # 状态标准化
        if model['running_stats'] is not None:
            normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
        else:
            normalized_state = state_tensor
        
        # 确保所有模型组件在CPU上并设置为float32
        agent.shared_backbone.to('cpu').float()
        for branch in agent.q_branches.values():
            branch.to('cpu').float()
        normalized_state = normalized_state.float()
        
        # 使用DQN智能体预测动作（传入动作掩码，不探索）
        with torch.no_grad():
            action_dict = agent.select_action(normalized_state.cpu(), action_mask, explore=False)
        
        return action_dict
    
    def get_ddpg_action(self, state, env, model):
        """获取DDPG模型动作"""
        agent = model['agent']
        
        # 将状态转换为模型期望的格式
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # 状态标准化
        if model['running_stats'] is not None:
            normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
        else:
            normalized_state = state_tensor
        
        # 确保所有模型组件在CPU上并设置为float32
        agent.actor.to('cpu').float()
        agent.critic.to('cpu').float()
        normalized_state = normalized_state.float()
        
        # 使用DDPG智能体预测动作（不添加噪声，用于评估）
        with torch.no_grad():
            continuous_action = agent.actor(normalized_state).cpu().numpy().flatten()
            # 使用动作转换器将连续动作转换为离散动作
            action_dict = agent.converter.continuous_to_discrete(continuous_action)
            
            # 验证动作字典的有效性
            if not isinstance(action_dict, dict):
                raise ValueError(f"❌ DDPG动作转换器返回了无效的动作格式: {type(action_dict)}")
            
            # 检查必要的动作键是否存在
            required_keys = ['ev_power', 'battery_power', 'wash_machine_schedule', 
                           'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp']
            for key in required_keys:
                if key not in action_dict:
                    raise ValueError(f"❌ DDPG动作字典缺少必要的键: {key}")
        
        return action_dict
    
    def get_td3_action(self, state, env, model):
        """获取TD3模型动作"""
        agent = model['agent']
        
        # 将状态转换为模型期望的格式
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # 状态标准化
        if model['running_stats'] is not None:
            normalized_state = model['running_stats'].normalize(state_tensor).clamp(-5, 5)
        else:
            normalized_state = state_tensor
        
        # 确保所有模型组件在CPU上并设置为float32
        agent.actor.to('cpu').float()
        normalized_state = normalized_state.float()
        
        # 使用TD3智能体预测动作（不添加噪声，用于评估）
        with torch.no_grad():
            continuous_action = agent.actor(normalized_state).cpu().numpy().flatten()
            # 使用动作转换器将连续动作转换为离散动作
            action_dict = agent.converter.continuous_to_discrete(continuous_action)
        
        return action_dict
    
    def get_sac_action(self, state, env, model):
        """获取SAC模型动作"""
        agent = model['agent']
        
        # 将状态转换为模型期望的格式
        state_vector = self.state_to_vector(state, model['state_keys'])
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to('cpu')
        
        # 确保所有模型组件在CPU上并设置为float32
        agent.actor.to('cpu').float()
        state_tensor = state_tensor.float()
        
        # 使用SAC智能体预测动作（不添加噪声，用于评估）
        with torch.no_grad():
            action_dict = agent.select_action(state_tensor)
        
        return action_dict
    
    def state_to_vector(self, state, state_keys):
        """将状态字典转换为向量"""
        if state_keys:
            state_vector = [state[k] for k in state_keys]
        else:
            # 备用方案：按字母顺序排序
            ordered_keys = sorted(state.keys())
            state_vector = [state[k] for k in ordered_keys]
        return state_vector
    
    def apply_device_mask(self, action, mask_type):
        """应用设备掩码（储能设备的启用/禁用）"""
        masked_action = action.copy()
        
        # 为确保控制变量严格性，基础用电设备使用固定策略
        masked_action['Air_conditioner_set_temp'] = 24     # 固定温度
        masked_action['Air_conditioner_set_temp2'] = 24    # 固定温度
        masked_action['ewh_set_temp'] = 55                 # 固定温度
        
        if mask_type == 'baseline':
            # 对照组：禁用储能功能
            masked_action['battery_power'] = 0
            if masked_action['ev_power'] < 0:
                masked_action['ev_power'] = 0
                
        elif mask_type == 'v2g_only':
            # V2G组：部分储能功能
            masked_action['battery_power'] = 0
            
        elif mask_type == 'full':
            # 完整优化组：允许所有功能
            pass
        
        return masked_action
    
    def calculate_home_total_carbon(self, env, electricity_price):
        """计算家庭总负荷的碳排放"""
        carbon_intensity = self.get_carbon_intensity(electricity_price)
        
        # 获取家庭总负荷
        total_home_load = env.total_load_compute()
        
        # 只有净购电才产生碳排放
        if total_home_load > 0:  # 家庭需要从电网购电
            step_carbon = total_home_load * 0.5 * carbon_intensity
            grid_purchase = total_home_load * 0.5
            grid_sale = 0
        else:  # 家庭向电网售电或平衡
            step_carbon = 0
            grid_purchase = 0
            grid_sale = abs(total_home_load) * 0.5
        
        return step_carbon, grid_purchase, grid_sale, carbon_intensity, total_home_load
    
    def run_single_episode(self, strategy_key, config_key, episode_num):
        """运行单个episode"""
        env = HomeEnergyManagementEnv()
        state = env.reset()
        
        strategy_config = self.strategies[strategy_key]
        config = self.configurations[config_key]
        
        # 初始化episode数据记录
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
            'high_carbon_purchase': 0,  # 高碳时段购电量
            'low_carbon_purchase': 0,   # 低碳时段购电量
            'ev_initial_soc': state['ev_battery_state'] / 24,
            'ess_initial_soc': state['ess_state'] / 24,
            'ev_final_soc': 0,
            'ess_final_soc': 0
        }
        
        for step in range(self.episode_length):
            # 获取策略动作
            if strategy_key == 'random':
                raw_action = self.get_random_action(state, env)
            else:
                try:
                    raw_action = self.get_model_action(state, env, strategy_key)
                except Exception as e:
                    print(f"❌ Episode {episode_num}, Step {step}: {strategy_key} 模型失败: {e}")
                    raise  # 重新抛出异常，停止实验
            
            # 应用设备掩码
            action = self.apply_device_mask(raw_action, config['mask_type'])
            
            # 执行动作，更新环境状态
            next_state, reward, done = env.step(state, action)
            
            # 计算当前步骤的家庭总碳排放
            step_carbon, grid_purchase, grid_sale, carbon_intensity, total_home_load = self.calculate_home_total_carbon(
                env, state['electricity_price']
            )
            
            # 记录碳排放和电网交互数据
            episode_data['total_carbon'] += step_carbon
            episode_data['total_grid_purchase'] += grid_purchase
            episode_data['total_grid_sale'] += grid_sale
            episode_data['carbon_intensity_avg'] += carbon_intensity
            episode_data['home_load_avg'] += abs(total_home_load)

            # 成本（按购电量与电价计价；grid_purchase 已含0.5h步长系数）
            step_cost = grid_purchase * state['electricity_price']
            episode_data['total_cost'] += step_cost
            
            # 记录高碳/低碳时段的购电量
            if carbon_intensity >= 0.9:  # 高碳时段
                episode_data['high_carbon_purchase'] += grid_purchase
            elif carbon_intensity <= 0.3:  # 低碳时段
                episode_data['low_carbon_purchase'] += grid_purchase
            
            # 记录能量数据
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
            
            # 更新状态
            state = next_state

            # 舒适度（基于室温偏好与热水温度范围的简单指标）
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
        
        # 记录最终SOC
        episode_data['ev_final_soc'] = state['ev_battery_state'] / 24
        episode_data['ess_final_soc'] = state['ess_state'] / 24
        
        # 计算平均值
        episode_data['carbon_intensity_avg'] /= self.episode_length
        episode_data['home_load_avg'] /= self.episode_length
        episode_data['comfort_score_avg'] = episode_data['comfort_score_sum'] / self.episode_length
        
        return episode_data
    
    def run_experiment_group(self, strategy_key, config_key):
        """运行一组实验"""
        strategy_name = self.strategies[strategy_key]['name']
        config_name = self.configurations[config_key]['name']
        group_key = f"{strategy_key}_{config_key}"
        
        print(f"🔄 正在运行 {strategy_name} - {config_name}...")
        
        results = []
        for episode in range(self.num_episodes):
            episode_data = self.run_single_episode(strategy_key, config_key, episode)
            results.append(episode_data)
        
        self.results[group_key] = results
        print(f"✅ {strategy_name} - {config_name} 完成")
        
        return results
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("🚀 开始策略碳排放对比实验...")
        
        # 运行所有策略和配置的组合
        for strategy_key in self.strategies.keys():
            for config_key in self.configurations.keys():
                self.run_experiment_group(strategy_key, config_key)
        
        print("\n🎉 所有实验组完成！")
    
    def analyze_results(self):
        """分析实验结果"""
        # 转换为DataFrame
        all_results = []
        for group_key, results in self.results.items():
            for result in results:
                result['group_key'] = group_key
                all_results.append(result)
        
        df_all = pd.DataFrame(all_results)
        
        # 计算各组平均值
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
                'avg_high_carbon_purchase': df['high_carbon_purchase'].mean(),
                'avg_low_carbon_purchase': df['low_carbon_purchase'].mean(),
                'avg_vehicle_storage_actions': df['vehicle_storage_actions'].mean(),
                'avg_battery_storage_actions': df['battery_storage_actions'].mean(),
                'avg_comfort': df['comfort_score_avg'].mean(),
                'strategy': df['strategy'].iloc[0],
                'configuration': df['configuration'].iloc[0]
            }
        
        # 打印结果分析
        print(f"\n{'='*70}")
        print(f"📊 算法策略碳排放对比实验结果分析")
        print(f"{'='*70}")
        
        # 各策略碳排放对比
        print(f"\n🔍 完整配置下各算法策略碳排放对比：")
        print(f"{'算法策略':<20} {'平均碳排放(kg CO2)':<20}")
        print(f"{'-'*45}")
        
        carbon_data = {}
        for strategy_key, strategy in self.strategies.items():
            group_key = f"{strategy_key}_full_optimization"
            if group_key in results_summary:
                summary = results_summary[group_key]
                carbon_data[strategy['name']] = summary['avg_carbon']
                print(f"{strategy['name']:<20} {summary['avg_carbon']:<20.2f}")
        
        # 详细的碳排放对比数据打印
        print(f"\n📊 详细碳排放对比数据:")
        for strategy_name, carbon_value in carbon_data.items():
            print(f"  {strategy_name}: {carbon_value:.4f} kg CO2")
        
        # 如果有随机策略，计算改进幅度
        if 'random' in [k for k in self.strategies.keys()]:
            random_key = "random_full_optimization"
            if random_key in results_summary:
                random_carbon = results_summary[random_key]['avg_carbon']
                print(f"\n📈 相对随机策略的改进:")
                print(f"  随机策略基准: {random_carbon:.4f} kg CO2")
                
                for strategy_key, strategy in self.strategies.items():
                    if strategy_key != 'random':
                        model_key = f"{strategy_key}_full_optimization"
                        if model_key in results_summary:
                            model_carbon = results_summary[model_key]['avg_carbon']
                            improvement = random_carbon - model_carbon
                            improvement_pct = (improvement / random_carbon * 100) if random_carbon > 0 else 0
                            print(f"  {strategy['name']}: 减少 {improvement:.4f} kg CO2 ({improvement_pct:.2f}%)")

        
        return results_summary, df_all
    
    def create_visualization(self, results_summary):
        """创建可视化图表：分别绘制碳排放、成本、满意度（各一张图）"""
        plt.style.use('default')
        strategies = list(self.strategies.keys())
        # 固定配色映射（最初配色风格，且与模型名称一一对应）
        color_map = {
            'random': '#FF6B6B',      # Random
            'proposed_rl': '#4ECDC4', # PPO (Proposed RL)
            'rainbow_dqn': '#45B7D1', # Rainbow DQN
            'ddpg': '#96CEB4',        # DDPG
            'td3': '#FECA57',         # TD3
            'sac': '#9B59B6',         # SAC
        }
        default_color = '#34495E'

        # 抽取数据
        names, carbons, costs, comforts, colors = [], [], [], [], []
        for i, strategy_key in enumerate(strategies):
            group_key = f"{strategy_key}_full_optimization"
            if group_key not in results_summary:
                continue
            summary = results_summary[group_key]
            names.append({
                'random': 'Random',
                'proposed_rl': 'Proposed RL',
                'rainbow_dqn': 'Rainbow DQN',
                'ddpg': 'DDPG',
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
            print(f"📊 图表已保存为: {filename}")

        # 分别绘制三张图
        draw_bar(carbons, 'Carbon (kg CO2)', 'Carbon Emissions', 'strategy_carbon.png')
        draw_bar(costs, 'Cost', 'Energy Cost', 'strategy_cost.png')
        draw_bar(comforts, 'Comfort (0-1)', 'User Comfort', 'strategy_comfort.png', ylim=(0, 1.05))
    
    def save_results(self, results_summary):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保results目录存在
        results_dir = os.path.join(self.project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细数据
        for group_key, results in self.results.items():
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(results_dir, f'{group_key}_strategy_comparison_{timestamp}.csv'), index=False)
        
        # 保存对比分析结果
        summary_df = pd.DataFrame(results_summary).T
        summary_df.to_csv(os.path.join(results_dir, f'strategy_comparison_summary_{timestamp}.csv'))
        
        print(f"\n💾 实验结果已保存到 ../results/ 目录")

def main():
    """主函数"""
    # 创建实验对象
    experiment = StrategyCarbonComparison()
    
    # 运行所有实验
    experiment.run_all_experiments()
    
    # 分析结果
    results_summary, df_all = experiment.analyze_results()
    
    # 创建可视化
    experiment.create_visualization(results_summary)
    
    # 保存结果
    experiment.save_results(results_summary)

if __name__ == "__main__":
    main()
