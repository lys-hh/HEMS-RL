'''
该文件绘制消融实验结果的图
'''
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# 文件路径与实验标签
files = {
    "No Dynamic Mask": "model/results/returns_ppo_20250709_121258_norm_unconstrained.csv",
    "Proposed RL": "model/results/returns_ppo_20250709_144004_norm_unconstrained.csv",
    "No State Normalization": "model/results/returns_ppo_20250709_172401_no_norm_unconstrained.csv",
    "No Advantage Normalization": "model/results/returns_ppo_20250709_222349_norm_unconstrained.csv",
    "no shared layer": "model/results/returns_ppo2_20250710_202931_norm.csv"
}

# 读取数据
data = {}
for label, file in files.items():
    data[label] = pd.read_csv(file)

os.makedirs('figures/algorithm_comparison', exist_ok=True)

# 颜色方案
color_map = {
    "No Dynamic Mask": '#a6cee3',
    "Proposed RL": '#1f78b4',
    "No State Normalization": '#FFB347',  # 将浅绿色改为橙色
    "No Advantage Normalization": '#33a02c',
    "no shared layer": '#DDA0DD'  # 将浅红色改为浅紫色
}
dark_color_map = {
    "No Dynamic Mask": '#377eb8',
    "Proposed RL": '#e41a1c',
    "No State Normalization": '#FF8C00',  # 将深绿色改为深橙色
    "No Advantage Normalization": '#984ea3',
    "no shared layer": '#8B5A8B'  # 将橙色改为紫色
}
# 集中管理柱状图配色
bar_colors = ['#F0EEEF','#C6CCDC','#9DACCB','#7789B7','#8985B7']  # 用于Energy Cost、Violation Rate等
ac1_color = '#257D8B'
ac2_color = '#EAA558'
ewh_color = '#68BED9'
total_color = '#ED8D5A'

# 平滑函数
def smooth_curve(y, window=50):
    return np.convolve(y, np.ones(window)/window, mode='valid')

def plot_with_smooth(ax, x, y, label, color, dark_color):
    ax.plot(x, y, color=color, alpha=0.4)  # 不加label
    # 计算每50个episode的均值
    x_avg = x[:len(y)//50*50].reshape(-1, 50).mean(axis=1) if isinstance(x, np.ndarray) else np.arange(0, len(y), 50)
    y_avg = [np.mean(y[i:i+50]) for i in range(0, len(y), 50)]
    x_avg = x[::50][:len(y_avg)]
    ax.plot(x_avg, y_avg, color=color, linewidth=2, label=label+' (avg)')  # 使用相同的color而不是dark_color

# 创建2×2布局的折线图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Returns (左上)
for label, df in data.items():
    if label == 'No Dynamic Mask':
        continue
    if 'Return' in df.columns:
        x = df['Episode'].values
        y = df['Return'].values
        plot_with_smooth(axes[0, 0], x, y, label, color_map[label], dark_color_map[label])
axes[0, 0].set_xlabel('Episode', fontsize=12)
axes[0, 0].set_ylabel('Return', fontsize=12)
axes[0, 0].set_title('Return Comparison', fontsize=14)
axes[0, 0].text(-0.11, 1.1, 'a', transform=axes[0, 0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axes[0, 0].legend(loc='lower right', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(labelsize=11)

# 2. Costs (右上)
for label, df in data.items():
    if label == 'No Dynamic Mask':
        continue
    if 'Energy_Cost' in df.columns:
        x = df['Episode'].values
        y = df['Energy_Cost'].values
        plot_with_smooth(axes[0, 1], x, y, label, color_map[label], dark_color_map[label])
axes[0, 1].set_xlabel('Episode', fontsize=12)
axes[0, 1].set_ylabel('Cost', fontsize=12)
axes[0, 1].set_title('Cost Comparison', fontsize=14)
axes[0, 1].text(-0.08, 1.1, 'b', transform=axes[0, 1].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axes[0, 1].legend(loc='lower right', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(labelsize=11)

# 3. User Satisfaction (左下)
for label, df in data.items():
    if label == 'No Dynamic Mask':
        continue
    if 'User_Satisfaction' in df.columns:
        x = df['Episode'].values
        y = df['User_Satisfaction'].values
        plot_with_smooth(axes[1, 0], x, y, label, color_map[label], dark_color_map[label])
axes[1, 0].set_xlabel('Episode', fontsize=12)
axes[1, 0].set_ylabel('User Satisfaction', fontsize=12)
axes[1, 0].set_title('User Satisfaction Comparison', fontsize=14)
axes[1, 0].text(-0.11, 1.1, 'c', transform=axes[1, 0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axes[1, 0].legend(loc='lower right', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(labelsize=11)

# 4. Violation Rate (右下)
for label, df in data.items():
    if 'Total_Violation_Rate' in df.columns:
        x = df['Episode'].values
        y = df['Total_Violation_Rate'].values
        plot_with_smooth(axes[1, 1], x, y, label, color_map[label], dark_color_map[label])
axes[1, 1].set_xlabel('Episode', fontsize=12)
axes[1, 1].set_ylabel('Violation Rate', fontsize=12)
axes[1, 1].set_title('Violation Rate Comparison', fontsize=14)
axes[1, 1].text(-0.08, 1.1, 'd', transform=axes[1, 1].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axes[1, 1].legend(loc='lower right', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(labelsize=11)

# 保存2×2布局的折线图
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/line_charts_comparison.png', dpi=300)
plt.close()

# Energy Cost柱状图
last_costs = [df['Energy_Cost'].values[-1] for df in data.values()]
model_names = list(data.keys())
# 去掉第一列（No Dynamic Mask）
filtered_costs = [cost for i, cost in enumerate(last_costs) if i != 0]
filtered_names = [name for i, name in enumerate(model_names) if i != 0]
plt.figure(figsize=(8, 5))
plt.bar(filtered_names, filtered_costs, color=bar_colors[1:len(data.keys())])
plt.ylabel('Energy Cost', fontsize=12)
# plt.title('Energy Cost Comparison')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/energy_cost_bar.png', dpi=300)
plt.close()

# User Satisfaction柱状图
last_user_satisfaction = [df['User_Satisfaction'].values[-1] for df in data.values()]
# 去掉第一列（No Dynamic Mask）
filtered_user_satisfaction = [val for i, val in enumerate(last_user_satisfaction) if i != 0]
plt.figure(figsize=(8, 5))
plt.bar(filtered_names, filtered_user_satisfaction, color=bar_colors[1:len(data.keys())])
plt.ylabel('User Satisfaction', fontsize=12)
# plt.title('User Satisfaction Comparison')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/user_satisfaction_bar.png', dpi=300)
plt.close()

# Violation Rate柱状图
last_violation_rates = [df['Total_Violation_Rate'].values[-1] for df in data.values()]
plt.figure(figsize=(8, 5))
plt.bar(list(data.keys()), last_violation_rates, color=bar_colors[:len(data.keys())])
plt.ylabel('Total Violation Rate', fontsize=12)
# plt.title('Violation Rate Comparison')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/violation_rate_bar.png', dpi=300)
# plt.show()

# 合并AC1, AC2, EWH, Total Comfort舒适度柱状图
ac1_col = 'AC1_Temp_Comfort'
ac2_col = 'AC2_Temp_Comfort'
ewh_col = 'EWH_Temp_Comfort'
total_col = 'Temperature_Comfort'
model_names = list(data.keys())
ac1_vals = [df[ac1_col].values[-1] if ac1_col in df.columns else np.nan for df in data.values()]
ac2_vals = [df[ac2_col].values[-1] if ac2_col in df.columns else np.nan for df in data.values()]
ewh_vals = [df[ewh_col].values[-1] if ewh_col in df.columns else np.nan for df in data.values()]
total_vals = [df[total_col].values[-1] if total_col in df.columns else np.nan for df in data.values()]


# 去掉第一列（No Dynamic Mask）
filtered_names = [name for i, name in enumerate(model_names) if i != 0]
filtered_ac1_vals = [val for i, val in enumerate(ac1_vals) if i != 0]
filtered_ac2_vals = [val for i, val in enumerate(ac2_vals) if i != 0]
filtered_ewh_vals = [val for i, val in enumerate(ewh_vals) if i != 0]
filtered_total_vals = [val for i, val in enumerate(total_vals) if i != 0]



bar_width = 0.18
x = np.arange(len(filtered_names))
plt.figure(figsize=(14, 7))
plt.bar(x - 1.5*bar_width, filtered_ac1_vals, width=bar_width, color=ac1_color, label='AC1 Comfort')
plt.bar(x - 0.5*bar_width, filtered_ac2_vals, width=bar_width, color=ac2_color, label='AC2 Comfort')
plt.bar(x + 0.5*bar_width, filtered_ewh_vals, width=bar_width, color=ewh_color, label='EWH Comfort')
plt.bar(x + 1.5*bar_width, filtered_total_vals, width=bar_width, color=total_color, label='Total Comfort')
plt.xticks(x, filtered_names, fontsize=10)
plt.ylabel('Comfort (Last Episode)', fontsize=12)
plt.title('AC1, AC2, EWH, Total Comfort Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/ac1_ac2_ewh_total_comfort_bar.png', dpi=300)
# plt.show()

# 单独的AC1舒适度柱状图
plt.figure(figsize=(8, 5))
plt.bar(filtered_names, filtered_ac1_vals, color=bar_colors[1:len(data.keys())])
plt.ylabel('AC1 Comfort', fontsize=12)
# plt.title('AC1 Comfort Comparison')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/ac1_comfort_bar.png', dpi=300)
# plt.show()

# 单独的AC2舒适度柱状图
plt.figure(figsize=(8, 5))
plt.bar(filtered_names, filtered_ac2_vals, color=bar_colors[1:len(data.keys())])
plt.ylabel('AC2 Comfort', fontsize=12)
# plt.title('AC2 Comfort Comparison')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/ac2_comfort_bar.png', dpi=300)
# plt.show()

# 单独的EWH舒适度柱状图
plt.figure(figsize=(8, 5))
plt.bar(filtered_names, filtered_ewh_vals, color=bar_colors[1:len(data.keys())])
plt.ylabel('EWH Comfort', fontsize=12)
# plt.title('EWH Comfort Comparison')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/ewh_comfort_bar.png', dpi=300)
# plt.show()

# 综合柱状图：将6个指标画在一个2行3列的布局中
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 第一行：AC1, AC2, EWH
axes[0, 0].bar(filtered_names, filtered_ac1_vals, color=bar_colors[1:len(data.keys())])
axes[0, 0].set_ylabel('AC1 Comfort', fontsize=12)
axes[0, 0].set_xticks([])  # 删除x轴标签
axes[0, 0].tick_params(axis='y', labelsize=10)
# 创建新的x轴标签
x_labels = ['Proposed RL'] + [f'D{i}' for i in range(1, len(filtered_names))]

# 第一行：AC1, AC2, EWH
axes[0, 0].set_xticks(range(len(filtered_names)))
axes[0, 0].set_xticklabels(x_labels, fontsize=12)
axes[0, 0].set_ylabel('AC1 Comfort', fontsize=14)
axes[0, 0].tick_params(axis='y', labelsize=12)
axes[0, 0].text(-0.12, 1.1, 'a', transform=axes[0, 0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')

axes[0, 1].bar(filtered_names, filtered_ac2_vals, color=bar_colors[1:len(data.keys())])
axes[0, 1].set_ylabel('AC2 Comfort', fontsize=14)
axes[0, 1].set_xticks(range(len(filtered_names)))
axes[0, 1].set_xticklabels(x_labels, fontsize=12)
axes[0, 1].tick_params(axis='y', labelsize=12)
axes[0, 1].text(-0.12, 1.1, 'b', transform=axes[0, 1].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')

axes[0, 2].bar(filtered_names, filtered_ewh_vals, color=bar_colors[1:len(data.keys())])
axes[0, 2].set_ylabel('EWH Comfort', fontsize=14)
axes[0, 2].set_xticks(range(len(filtered_names)))
axes[0, 2].set_xticklabels(x_labels, fontsize=12)
axes[0, 2].tick_params(axis='y', labelsize=12)
axes[0, 2].text(-0.12, 1.1, 'c', transform=axes[0, 2].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')

# 第二行：User Satisfaction, Energy Cost, Violation Rate
axes[1, 0].bar(filtered_names, filtered_user_satisfaction, color=bar_colors[1:len(data.keys())])
axes[1, 0].set_ylabel('User Satisfaction', fontsize=14)
axes[1, 0].set_xticks(range(len(filtered_names)))
axes[1, 0].set_xticklabels(x_labels, fontsize=12)
axes[1, 0].tick_params(axis='y', labelsize=12)
axes[1, 0].text(-0.12, 1.1, 'd', transform=axes[1, 0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')

axes[1, 1].bar(filtered_names, filtered_costs, color=bar_colors[1:len(data.keys())])
axes[1, 1].set_ylabel('Energy Cost', fontsize=14)
axes[1, 1].set_xticks(range(len(filtered_names)))
axes[1, 1].set_xticklabels(x_labels, fontsize=12)
axes[1, 1].tick_params(axis='y', labelsize=12)
axes[1, 1].text(-0.12, 1.1, 'e', transform=axes[1, 1].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')

# 为最后一个图创建标签（包含所有模型）
all_x_labels = ['D4', 'Proposed RL'] + [f'D{i}' for i in range(1, len(list(data.keys()))-1)]
axes[1, 2].bar(list(data.keys()), last_violation_rates, color=bar_colors[:len(data.keys())])
axes[1, 2].set_ylabel('Total Violation Rate', fontsize=14)
axes[1, 2].set_xticks(range(len(list(data.keys()))))
axes[1, 2].set_xticklabels(all_x_labels, fontsize=12)
axes[1, 2].tick_params(axis='y', labelsize=12)
axes[1, 2].text(-0.12, 1.1, 'f', transform=axes[1, 2].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')

plt.tight_layout()
plt.savefig('figures/algorithm_comparison/comprehensive_comparison.png', dpi=300)
# plt.show()
