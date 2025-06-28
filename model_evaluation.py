"""
模型评估模块
用于评估训练好的PPO模型在家庭能源管理任务中的性能
"""
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from environment import HomeEnergyManagementEnv
from model.PPO_3rd import HomeEnergyPPO, RunningStats, SharedFeatureExtractor, ActionBranch, ValueNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelEvaluator:
    def __init__(self, model_path, env=None):
        """
        初始化模型评估器
        
        Args:
            model_path: 训练好的模型文件路径
            env: 环境实例，如果为None则创建新环境
        """
        self.model_path = model_path
        self.env = env if env is not None else HomeEnergyManagementEnv()
        
        # 加载模型
        self.agent, self.running_stats, self.state_keys = self._load_model()
        
        # 评估结果存储
        self.evaluation_results = {
            'episode_returns': [],
            'total_costs': [],
            'energy_costs': [],
            'user_satisfaction': [],
            'constraint_violations': [],
            'device_usage': {
                'ev_charging': [],
                'ess_charging': [],
                'air_conditioner': [],
                'wash_machine': [],
                'water_heater': []
            }
        }
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"正在加载模型: {self.model_path}")
        
        # 加载模型文件
        checkpoint = torch.load(self.model_path, map_location=device)
        
        # 重建网络结构
        training_config = checkpoint['training_config']
        agent = HomeEnergyPPO(
            env=self.env,
            state_dim=training_config['state_dim'],
            hidden_dim=training_config['hidden_dim'],
            action_space_config=training_config['action_space_config'],
            gamma=training_config['gamma'],
            lmbda=training_config['lmbda'],
            eps=training_config['eps'],
            epochs=training_config['epochs'],
            ent_coef=training_config['ent_coef'],
            max_grad_norm=training_config['max_grad_norm'],
            device=device
        )
        
        # 加载模型权重
        agent.shared_backbone.load_state_dict(checkpoint['shared_backbone_state_dict'])
        agent.actor_branches.load_state_dict(checkpoint['actor_branches_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # 重建running_stats
        running_stats = RunningStats(shape=training_config['state_dim'])
        running_stats.mean = checkpoint['running_stats_mean']
        running_stats.std = checkpoint['running_stats_std']
        running_stats.count = checkpoint['running_stats_count']
        
        # 获取状态键
        state_keys = checkpoint['state_keys']
        
        print("模型加载完成！")
        return agent, running_stats, state_keys
    
    def evaluate_episode(self, episode_id, render=False):
        """
        评估单个episode
        
        Args:
            episode_id: episode编号
            render: 是否显示详细信息
            
        Returns:
            episode_results: 包含episode评估结果的字典
        """
        state = self.env.reset()
        episode_return = 0
        episode_cost = 0
        step_count = 0
        
        # 记录设备使用情况
        device_usage = {
            'ev_charging': [],
            'ess_charging': [],
            'air_conditioner': [],
            'wash_machine': [],
            'water_heater': []
        }
        
        # 记录约束违反
        constraint_violations = []
        
        while True:
            # 准备状态张量
            state_values = [state[k] for k in self.state_keys]
            state_tensor = torch.FloatTensor(state_values).unsqueeze(0).to(device)
            normalized_state = self.running_stats.normalize(state_tensor).clamp(-5, 5)
            
            # 获取动作掩码
            action_mask = self.env.get_action_mask(state)
            
            # 智能体选择动作（评估模式下不使用探索）
            with torch.no_grad():
                actions, _, _, value = self.agent.take_action(normalized_state, action_mask=action_mask)
            
            # 环境执行动作
            next_state, reward, done = self.env.step(state, actions)
            
            # 记录设备使用情况
            device_usage['ev_charging'].append(actions.get('ev_power', 0))
            device_usage['ess_charging'].append(actions.get('battery_power', 0))
            device_usage['air_conditioner'].append(actions.get('Air_conditioner_set_temp', 22))
            device_usage['wash_machine'].append(actions.get('wash_machine_schedule', 0))
            device_usage['water_heater'].append(actions.get('ewh_set_temp', 50))
            
            # 检查约束违反
            ess_violation = 0
            ev_violation = 0
            if state['ess_state'] < 0.1 * self.env.ess_capacity or state['ess_state'] > 0.9 * self.env.ess_capacity:
                ess_violation = 1
            if state['ev_battery_state'] < 0.1 * self.env.ev_capacity or state['ev_battery_state'] > 0.9 * self.env.ev_capacity:
                ev_violation = 1
            constraint_violations.append(ess_violation + ev_violation)
            
            episode_return += reward
            episode_cost += self.env.current_step_cost
            step_count += 1
            state = next_state
            
            if render:
                print(f"Step {step_count}: Actions={actions}, Reward={reward:.3f}, Cost={self.env.current_step_cost:.3f}")
            
            if done:
                break
        
        # 计算用户满意度（简化计算）
        user_satisfaction = self._calculate_user_satisfaction(device_usage)
        
        episode_results = {
            'episode_id': episode_id,
            'episode_return': episode_return,
            'total_cost': episode_cost,
            'step_count': step_count,
            'user_satisfaction': user_satisfaction,
            'constraint_violation_rate': np.mean(constraint_violations),
            'device_usage': device_usage
        }
        
        return episode_results
    
    def _calculate_user_satisfaction(self, device_usage):
        """计算用户满意度（简化版本）"""
        satisfaction = 1.0
        
        # 空调温度满意度
        ac_temps = device_usage['air_conditioner']
        if ac_temps:
            avg_temp = np.mean(ac_temps)
            # 假设用户偏好22度，温度偏离越大满意度越低
            temp_penalty = abs(avg_temp - 22) / 10
            satisfaction -= temp_penalty * 0.3
        
        # 热水器温度满意度
        wh_temps = device_usage['water_heater']
        if wh_temps:
            avg_wh_temp = np.mean(wh_temps)
            # 假设用户偏好55度
            wh_temp_penalty = abs(avg_wh_temp - 55) / 20
            satisfaction -= wh_temp_penalty * 0.2
        
        return max(0, satisfaction)
    
    def evaluate_model(self, num_episodes=10, render_episodes=None):
        """
        评估模型性能
        
        Args:
            num_episodes: 评估的episode数量
            render_episodes: 需要详细显示的episode列表
        """
        print(f"开始评估模型，共{num_episodes}个episodes...")
        
        if render_episodes is None:
            render_episodes = []
        
        for episode in range(num_episodes):
            render = episode in render_episodes
            if render:
                print(f"\n=== Episode {episode + 1} (详细模式) ===")
            
            episode_results = self.evaluate_episode(episode + 1, render=render)
            
            # 存储结果
            self.evaluation_results['episode_returns'].append(episode_results['episode_return'])
            self.evaluation_results['total_costs'].append(episode_results['total_cost'])
            self.evaluation_results['user_satisfaction'].append(episode_results['user_satisfaction'])
            self.evaluation_results['constraint_violations'].append(episode_results['constraint_violation_rate'])
            
            # 合并设备使用数据
            for device, usage in episode_results['device_usage'].items():
                self.evaluation_results['device_usage'][device].extend(usage)
            
            if not render:
                print(f"Episode {episode + 1}: Return={episode_results['episode_return']:.2f}, "
                      f"Cost={episode_results['total_cost']:.2f}, "
                      f"Satisfaction={episode_results['user_satisfaction']:.3f}")
        
        # 计算统计指标
        self._calculate_statistics()
        
        # 保存评估结果
        self._save_evaluation_results()
        
        # 生成评估报告
        self._generate_evaluation_report()
    
    def _calculate_statistics(self):
        """计算评估统计指标"""
        self.statistics = {
            'mean_return': np.mean(self.evaluation_results['episode_returns']),
            'std_return': np.std(self.evaluation_results['episode_returns']),
            'mean_cost': np.mean(self.evaluation_results['total_costs']),
            'std_cost': np.std(self.evaluation_results['total_costs']),
            'mean_satisfaction': np.mean(self.evaluation_results['user_satisfaction']),
            'mean_constraint_violation': np.mean(self.evaluation_results['constraint_violations']),
            'device_usage_stats': {}
        }
        
        # 计算设备使用统计
        for device, usage in self.evaluation_results['device_usage'].items():
            if usage:
                self.statistics['device_usage_stats'][device] = {
                    'mean': np.mean(usage),
                    'std': np.std(usage),
                    'min': np.min(usage),
                    'max': np.max(usage)
                }
    
    def _save_evaluation_results(self):
        """保存评估结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建评估结果目录
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(eval_dir, f"evaluation_results_{timestamp}.csv")
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Return', 'Total_Cost', 'User_Satisfaction', 'Constraint_Violation_Rate'])
            
            for i in range(len(self.evaluation_results['episode_returns'])):
                writer.writerow([
                    i + 1,
                    self.evaluation_results['episode_returns'][i],
                    self.evaluation_results['total_costs'][i],
                    self.evaluation_results['user_satisfaction'][i],
                    self.evaluation_results['constraint_violations'][i]
                ])
        
        # 保存统计摘要
        summary_file = os.path.join(eval_dir, f"evaluation_summary_{timestamp}.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Mean_Return', self.statistics['mean_return']])
            writer.writerow(['Std_Return', self.statistics['std_return']])
            writer.writerow(['Mean_Cost', self.statistics['mean_cost']])
            writer.writerow(['Std_Cost', self.statistics['std_cost']])
            writer.writerow(['Mean_Satisfaction', self.statistics['mean_satisfaction']])
            writer.writerow(['Mean_Constraint_Violation', self.statistics['mean_constraint_violation']])
        
        print(f"评估结果已保存到: {eval_dir}")
    
    def _generate_evaluation_report(self):
        """生成评估报告"""
        print("\n" + "="*50)
        print("模型评估报告")
        print("="*50)
        print(f"平均回报: {self.statistics['mean_return']:.2f} ± {self.statistics['std_return']:.2f}")
        print(f"平均成本: {self.statistics['mean_cost']:.2f} ± {self.statistics['std_cost']:.2f}")
        print(f"平均用户满意度: {self.statistics['mean_satisfaction']:.3f}")
        print(f"平均约束违反率: {self.statistics['mean_constraint_violation']:.3f}")
        
        print("\n设备使用统计:")
        for device, stats in self.statistics['device_usage_stats'].items():
            print(f"  {device}: {stats['mean']:.2f} ± {stats['std']:.2f} "
                  f"(范围: {stats['min']:.2f} - {stats['max']:.2f})")
        
        print("="*50)
    
    def plot_evaluation_results(self):
        """绘制评估结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 回报分布
        axes[0, 0].hist(self.evaluation_results['episode_returns'], bins=10, alpha=0.7)
        axes[0, 0].axvline(self.statistics['mean_return'], color='red', linestyle='--', 
                          label=f'均值: {self.statistics["mean_return"]:.2f}')
        axes[0, 0].set_title('Episode回报分布')
        axes[0, 0].set_xlabel('回报')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].legend()
        
        # 成本分布
        axes[0, 1].hist(self.evaluation_results['total_costs'], bins=10, alpha=0.7)
        axes[0, 1].axvline(self.statistics['mean_cost'], color='red', linestyle='--',
                          label=f'均值: {self.statistics["mean_cost"]:.2f}')
        axes[0, 1].set_title('总成本分布')
        axes[0, 1].set_xlabel('成本')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        
        # 用户满意度
        axes[1, 0].bar(range(1, len(self.evaluation_results['user_satisfaction'])+1), 
                      self.evaluation_results['user_satisfaction'])
        axes[1, 0].set_title('各Episode用户满意度')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('满意度')
        
        # 设备使用热力图
        device_names = list(self.evaluation_results['device_usage'].keys())
        device_means = [self.statistics['device_usage_stats'][device]['mean'] 
                       for device in device_names]
        
        axes[1, 1].bar(device_names, device_means)
        axes[1, 1].set_title('平均设备使用情况')
        axes[1, 1].set_ylabel('使用量')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        plt.savefig(os.path.join(eval_dir, f"evaluation_plots_{timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数 - 模型评估示例"""
    # 检查是否有保存的模型
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        print("未找到saved_models目录，请先训练模型！")
        return
    
    # 找到最新的模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        print("未找到训练好的模型文件，请先训练模型！")
        return
    
    # 选择最新的模型文件
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    print(f"使用模型: {model_path}")
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path)
    
    # 执行评估
    evaluator.evaluate_model(num_episodes=10, render_episodes=[0])  # 详细显示第一个episode
    
    # 绘制结果
    evaluator.plot_evaluation_results()


if __name__ == "__main__":
    main() 