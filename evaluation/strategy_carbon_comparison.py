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

# 设置matplotlib后端和警告
warnings.filterwarnings('ignore')  # 忽略警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置字体为支持英文的字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
            }
            # 'td3': {
            #     'name': 'TD3算法',
            #     'model_path': os.path.join(model_dir, 'td3_model_20250805_004024.pth'),
            #     'strategy_func': self.get_model_action
            # },
            # 'sac': {
            #     'name': 'SAC算法',
            #     'model_path': os.path.join(model_dir, 'sac2_model_20250805_004024.pth'),
            #     'strategy_func': self.get_model_action
            # }
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
            checkpoint = torch.load(model_path, map_location='cpu')
            
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
            # 确保能找到model模块
            model_path = os.path.join(self.project_root, 'model')
            if model_path not in sys.path:
                sys.path.append(model_path)
            from TD3 import TD3, Actor, Critic, ActionConverter
            
            # 从checkpoint获取配置
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                
                # 重新构建智能体
                agent = TD3(config['state_dim'], config['action_dim'], config['action_space_config'])
                
                # 加载模型权重
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
                agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
                
                # 设置为评估模式
                agent.actor.eval()
                agent.critic1.eval()
                agent.critic2.eval()
                
                # 加载运行统计信息（如果存在）
                running_stats = None
                if 'running_stats_mean' in checkpoint:
                    from TD3 import RunningStats
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
                print("❌ TD3模型格式不支持")
                return None
                
        except Exception as e:
            print(f"❌ TD3模型加载失败: {e}")
            return None
    
    def load_sac_model(self, checkpoint):
        """加载SAC模型"""
        try:
            # 确保能找到model模块
            model_path = os.path.join(self.project_root, 'model')
            if model_path not in sys.path:
                sys.path.append(model_path)
            from sac2 import EnhancedSAC, Actor, Critic, ActionConverter
            
            # 从checkpoint获取配置
            if isinstance(checkpoint, dict) and 'training_config' in checkpoint:
                config = checkpoint['training_config']
                
                # 重新构建智能体
                agent = EnhancedSAC(config['state_dim'], config['action_dim'], device='cpu')
                
                # 加载模型权重
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                
                # 设置为评估模式
                agent.actor.eval()
                agent.critic.eval()
                
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
        agent.critic1.to('cpu').float()
        agent.critic2.to('cpu').float()
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
        agent.critic.to('cpu').float()
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
        
        # 记录最终SOC
        episode_data['ev_final_soc'] = state['ev_battery_state'] / 24
        episode_data['ess_final_soc'] = state['ess_state'] / 24
        
        # 计算平均值
        episode_data['carbon_intensity_avg'] /= self.episode_length
        episode_data['home_load_avg'] /= self.episode_length
        
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
        """创建可视化图表"""
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 算法策略的碳排放对比
        strategies = list(self.strategies.keys())
        strategy_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        carbons = []
        strategy_names = []
        colors = []
        
        for i, strategy_key in enumerate(strategies):
            group_key = f"{strategy_key}_full_optimization"
            if group_key in results_summary:
                carbons.append(results_summary[group_key]['avg_carbon'])
                # 解决中文显示问题：使用简洁的英文名称
                if strategy_key == 'random':
                    strategy_names.append('Random Strategy')
                elif strategy_key == 'proposed_rl':
                    strategy_names.append('PPO Algorithm')
                elif strategy_key == 'rainbow_dqn':
                    strategy_names.append('Rainbow DQN Algorithm')
                elif strategy_key == 'ddpg':
                    strategy_names.append('DDPG Algorithm')
                elif strategy_key == 'td3':
                    strategy_names.append('TD3 Algorithm')
                elif strategy_key == 'sac':
                    strategy_names.append('SAC Algorithm')
                else:
                    # 默认使用策略key的英文形式
                    strategy_names.append(strategy_key.upper() + ' Algorithm')
                colors.append(strategy_colors[i % len(strategy_colors)])
        
        x = np.arange(len(strategy_names))
        bars = ax.bar(x, carbons, color=colors, alpha=0.8, edgecolor='black', width=0.6)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Carbon Emissions (kg CO2)', fontweight='bold', fontsize=12)
        ax.set_title('Carbon Emissions Comparison of Different Algorithms', fontweight='bold', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 设置y轴范围，留出空间显示数值标签
        if carbons:
            max_carbon = max(carbons)
            ax.set_ylim(0, max_carbon * 1.1)
        
        plt.tight_layout()
        # 确保figures目录存在
        figures_dir = os.path.join(self.project_root, 'figures', 'experiment_results')
        os.makedirs(figures_dir, exist_ok=True)
        figures_path = os.path.join(figures_dir, 'strategy_carbon_comparison.png')
        plt.savefig(figures_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        print(f"\n📊 图表已保存为: strategy_carbon_comparison.png")
    
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
