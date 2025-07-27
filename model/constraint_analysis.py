"""
约束损失不起作用的原因分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_constraint_effectiveness():
    """分析约束损失的有效性"""
    
    # 读取结果文件
    results_dir = "model/results"
    results_files = [
        "returns_ppo_20250629_221142_norm.csv",  # PPO3 with constraint
        "returns_ppo_20250628_233049_no_norm.csv",  # PPO3 without constraint
    ]
    
    data_list = []
    for file in results_files:
        file_path = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(file_path)
            df['config'] = 'with_constraint' if 'norm' in file else 'without_constraint'
            data_list.append(df)
            print(f"成功读取: {file}")
        except FileNotFoundError:
            print(f"文件不存在: {file_path}")
            continue
    
    if not data_list:
        print("没有找到可分析的数据文件")
        return
    
    # 合并数据
    combined_data = pd.concat(data_list, ignore_index=True)
    
    # 分析约束违反情况
    print("\n=== 约束违反分析 ===")
    
    # 1. 检查约束违反率是否真的为0
    ess_violation_rates = combined_data['ESS_Violation_Rate'].dropna()
    ev_violation_rates = combined_data['EV_Violation_Rate'].dropna()
    
    print(f"ESS违反率统计:")
    print(f"  平均值: {ess_violation_rates.mean():.6f}")
    print(f"  最大值: {ess_violation_rates.max():.6f}")
    print(f"  最小值: {ess_violation_rates.min():.6f}")
    print(f"  标准差: {ess_violation_rates.std():.6f}")
    
    print(f"\nEV违反率统计:")
    print(f" 平均值: {ev_violation_rates.mean():.6f}")
    print(f" 最大值: {ev_violation_rates.max():.6f}")
    print(f" 最小值: {ev_violation_rates.min():.6f}")
    print(f" 标准差: {ev_violation_rates.std():.6f}")
    
    # 2. 检查SOC分布
    ess_soc_means = combined_data['ESS_SOC_Mean'].dropna()
    ev_soc_means = combined_data['EV_SOC_Mean'].dropna()
    
    print(f"\n=== SOC分布分析 ===")
    print(f"ESS SOC统计:")
    print(f"  平均值: {ess_soc_means.mean():.3f}")
    print(f"  最大值: {ess_soc_means.max():.3f}")
    print(f"  最小值: {ess_soc_means.min():.3f}")
    print(f"  标准差: {ess_soc_means.std():.3f}")
    
    print(f"\nEV SOC统计:")
    print(f" 平均值: {ev_soc_means.mean():.3f}")
    print(f" 最大值: {ev_soc_means.max():.3f}")
    print(f" 最小值: {ev_soc_means.min():.3f}")
    print(f" 标准差: {ev_soc_means.std():.3f}")
    
    # 3. 检查约束损失值
    constraint_losses = combined_data['Constraint_Loss'].dropna()
    print(f"\n=== 约束损失分析 ===")
    print(f"约束损失统计:")
    print(f"  平均值: {constraint_losses.mean():.6f}")
    print(f"  最大值: {constraint_losses.max():.6f}")
    print(f"  最小值: {constraint_losses.min():.6f}")
    print(f"  标准差: {constraint_losses.std():.6f}")
    
    # 4. 检查总损失中约束损失的占比
    total_losses = combined_data['Total_Loss'].dropna()
    actor_losses = combined_data['Actor_Loss'].dropna()
    critic_losses = combined_data['Critic_Loss'].dropna()
    
    print(f"\n=== 损失比例分析 ===")
    print(f"总损失平均值: {total_losses.mean():.4f}")
    print(f"Actor损失平均值: {actor_losses.mean():.4f}")
    print(f"Critic损失平均值: {critic_losses.mean():.4f}")
    print(f"约束损失平均值: {constraint_losses.mean():.6f}")
    
    if total_losses.mean() > 0:
        constraint_ratio = constraint_losses.mean() / total_losses.mean()
        print(f"约束损失占总损失比例: {constraint_ratio:.6f} ({constraint_ratio*100:.4f}%)")
    
    # 5. 按配置分组分析
    print(f"\n=== 配置对比分析 ===")
    for config in combined_data['config'].unique():
        config_data = combined_data[combined_data['config'] == config]
        print(f"\n{config}配置:")
        print(f"  ESS违反率: {config_data['ESS_Violation_Rate'].mean():.6f}")
        print(f"  EV违反率: {config_data['EV_Violation_Rate'].mean():.6f}")
        print(f"  约束损失: {config_data['Constraint_Loss'].mean():.6f}")
        print(f"  ESS SOC: {config_data['ESS_SOC_Mean'].mean():.3f}")
        print(f"  EV SOC: {config_data['EV_SOC_Mean'].mean():.3f}")
    
    # 6. 可视化分析
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('约束损失有效性分析', fontsize=16)
    
    # SOC分布
    axes[0, 0].hist(ess_soc_means, bins=20, alpha=0.7, label='ESS SOC')
    axes[0, 0].hist(ev_soc_means, bins=20, alpha=0.7, label='EV SOC')
    axes[0, 0].axvline(x=0.2, color='red', linestyle='--', label='Lower Bound')
    axes[0, 0].axvline(x=0.8, color='red', linestyle='--', label='Upper Bound')
    axes[0, 0].set_xlabel('SOC')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('SOC Distribution')
    axes[0, 0].legend()
    
    # 约束违反率
    axes[0, 1].hist(ess_violation_rates, bins=20, alpha=0.7, label='ESS Violation')
    axes[0, 1].hist(ev_violation_rates, bins=20, alpha=0.7, label='EV Violation')
    axes[0, 1].set_xlabel('Violation Rate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Constraint Violation Rate')
    axes[0, 1].legend()
    
    # 约束损失分布
    axes[0, 2].hist(constraint_losses, bins=20, alpha=0.7)
    axes[0, 2].set_xlabel('Constraint Loss')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Constraint Loss Distribution')
    
    # 损失对比
    loss_data = [actor_losses, critic_losses, constraint_losses]
    loss_labels = ['Actor Loss', 'Critic Loss', 'Constraint Loss']
    axes[1, 0].boxplot(loss_data, labels=loss_labels)
    axes[1, 0].set_ylabel('Loss Value')
    axes[1, 0].set_title('Loss Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # SOC vs 违反率散点图
    axes[1, 1].scatter(ess_soc_means, ess_violation_rates, alpha=0.6, label='ESS')
    axes[1, 1].scatter(ev_soc_means, ev_violation_rates, alpha=0.6, label='EV')
    axes[1, 1].set_xlabel('SOC')
    axes[1, 1].set_ylabel('Violation Rate')
    axes[1, 1].set_title('SOC vs Violation Rate')
    axes[1, 1].legend()
    
    # 约束损失 vs 违反率
    axes[1, 2].scatter(ess_violation_rates + ev_violation_rates, constraint_losses, alpha=0.6)
    axes[1, 2].set_xlabel('Total Violation Rate')
    axes[1, 2].set_ylabel('Constraint Loss')
    axes[1, 2].set_title('Violation Rate vs Constraint Loss')
    
    plt.tight_layout()
    plt.savefig('constraint_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. 问题诊断
    print(f"\n=== 问题诊断 ===")
    
    # 检查是否所有SOC都在安全范围内
    ess_in_safe_range = ((ess_soc_means >= 0.2) & (ess_soc_means <= 0.8)).sum()
    ev_in_safe_range = ((ev_soc_means >= 0.2) & (ev_soc_means <= 0.8)).sum()
    
    print(f"ESS SOC在安全范围内的比例: {ess_in_safe_range/len(ess_soc_means)*100:.2f}%")
    print(f"EV SOC在安全范围内的比例: {ev_in_safe_range/len(ev_soc_means)*100:.2f}%")
    
    # 检查约束损失是否真的为0
    zero_constraint_loss = (constraint_losses == 0).sum()
    print(f"约束损失为0的比例: {zero_constraint_loss/len(constraint_losses)*100:.2f}%")
    
    # 检查动态掩码是否已经解决了约束问题
    print(f"\n=== 可能的原因 ===")
    print("1. 动态掩码已经很好地防止了约束违反")
    print("2. 约束损失权重太小，被其他损失淹没")
    print("3. 约束损失计算方式有问题")
    print("4. 环境本身就不容易违反约束")
    print("5. 训练轮数不够，约束效果还没显现")

def suggest_improvements():
    """建议改进方案"""
    print(f"\n=== 改进建议 ===")
    print("1. 增加约束损失权重（从2增加到10-50）")
    print("2. 在奖励函数中添加约束违反惩罚")
    print("3. 使用更严格的约束边界（如0.2-0.8）")
    print("4. 添加约束违反的梯度惩罚")
    print("5. 使用约束优化算法（如PPO-Lagrangian）")
    print("6. 增加训练轮数观察长期效果")
    print("7. 添加约束违反的监控和可视化")

if __name__ == "__main__":
    analyze_constraint_effectiveness()
    suggest_improvements() 