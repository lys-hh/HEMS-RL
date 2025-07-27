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
    "Independent Actor": "model/results/returns_ppo2_20250710_202931_norm.csv"
}

# 读取数据
data = {}
for label, file in files.items():
    data[label] = pd.read_csv(file)

os.makedirs('save_pictures', exist_ok=True)

# 颜色方案
color_map = {
    "No Dynamic Mask": '#a6cee3',
    "Proposed RL": '#1f78b4',
    "No State Normalization": '#b2df8a',
    "No Advantage Normalization": '#33a02c',
    "Independent Actor": '#fb9a99'
}
dark_color_map = {
    "No Dynamic Mask": '#377eb8',
    "Proposed RL": '#e41a1c',
    "No State Normalization": '#4daf4a',
    "No Advantage Normalization": '#984ea3',
    "Independent Actor": '#ff7f00'
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

def plot_with_smooth(x, y, label, color, dark_color, ylabel, title, filename):
    plt.plot(x, y, color=color, alpha=0.4)  # 不加label
    # 计算每50个episode的均值
    x_avg = x[:len(y)//50*50].reshape(-1, 50).mean(axis=1) if isinstance(x, np.ndarray) else np.arange(0, len(y), 50)
    y_avg = [np.mean(y[i:i+50]) for i in range(0, len(y), 50)]
    x_avg = x[::50][:len(y_avg)]
    plt.plot(x_avg, y_avg, color=dark_color, linewidth=2, label=label+' (avg)')

# 只保留随episode变化的returns, costs, user_satisfaction, violation_rate四张折线图，其它折线图全部删除。
# 1. Returns
plt.figure(figsize=(10, 5))
for label, df in data.items():
    if label == 'No Dynamic Mask':
        continue
    if 'Return' in df.columns:
        x = df['Episode'].values
        y = df['Return'].values
        plot_with_smooth(x, y, label, color_map[label], dark_color_map[label], 'Return', 'Return Comparison', None)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Return Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('save_pictures/return_comparison.png', dpi=300)
# plt.show()

# 2. Costs
plt.figure(figsize=(10, 5))
for label, df in data.items():
    if label == 'No Dynamic Mask':
        continue
    if 'Energy_Cost' in df.columns:
        x = df['Episode'].values
        y = df['Energy_Cost'].values
        plot_with_smooth(x, y, label, color_map[label], dark_color_map[label], 'Cost', 'Cost Comparison', None)
plt.xlabel('Episode')
plt.ylabel('Cost')
plt.title('Cost Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('save_pictures/energy_cost_comparison.png', dpi=300)
# plt.show()
# Energy Cost柱状图
last_costs = [df['Energy_Cost'].values[-1] for df in data.values()]
plt.figure(figsize=(8, 5))
plt.bar(list(data.keys()), last_costs, color=bar_colors[:len(data.keys())])
plt.ylabel('Energy Cost (Last Episode)')
plt.title('Energy Cost Comparison')
plt.xticks(fontsize=7)
plt.tight_layout()
plt.savefig('save_pictures/energy_cost_bar.png', dpi=300)
# plt.show()

# 3. User Satisfaction
plt.figure(figsize=(10, 5))
for label, df in data.items():
    if label == 'No Dynamic Mask':
        continue
    if 'User_Satisfaction' in df.columns:
        x = df['Episode'].values
        y = df['User_Satisfaction'].values
        plot_with_smooth(x, y, label, color_map[label], dark_color_map[label], 'User Satisfaction', 'User Satisfaction Comparison', None)
plt.xlabel('Episode')
plt.ylabel('User Satisfaction')
plt.title('User Satisfaction Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('save_pictures/user_satisfaction_comparison.png', dpi=300)
# plt.show()

# 4. Violation Rate 折线图
plt.figure(figsize=(10, 5))
for label, df in data.items():
    if 'Total_Violation_Rate' in df.columns:
        x = df['Episode'].values
        y = df['Total_Violation_Rate'].values
        plot_with_smooth(x, y, label, color_map[label], dark_color_map[label], 'Violation Rate', 'Violation Rate Comparison', None)
plt.xlabel('Episode')
plt.ylabel('Violation Rate')
plt.title('Violation Rate Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('save_pictures/violation_rate_comparison.png', dpi=300)
# plt.show()

# Violation Rate柱状图
last_violation_rates = [df['Total_Violation_Rate'].values[-1] for df in data.values()]
plt.figure(figsize=(8, 5))
plt.bar(list(data.keys()), last_violation_rates, color=bar_colors[:len(data.keys())])
plt.ylabel('Total Violation Rate (Last Episode)')
plt.title('Violation Rate Comparison')
plt.xticks(fontsize=7)
plt.tight_layout()
plt.savefig('save_pictures/violation_rate_bar.png', dpi=300)
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
bar_width = 0.18
x = np.arange(len(model_names))
plt.figure(figsize=(12, 6))
plt.bar(x - 1.5*bar_width, ac1_vals, width=bar_width, color=ac1_color, label='AC1 Comfort')
plt.bar(x - 0.5*bar_width, ac2_vals, width=bar_width, color=ac2_color, label='AC2 Comfort')
plt.bar(x + 0.5*bar_width, ewh_vals, width=bar_width, color=ewh_color, label='EWH Comfort')
plt.bar(x + 1.5*bar_width, total_vals, width=bar_width, color=total_color, label='Total Comfort')
plt.xticks(x, model_names, fontsize=8)
plt.ylabel('Comfort (Last Episode)')
plt.title('AC1, AC2, EWH, Total Comfort Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('save_pictures/ac1_ac2_ewh_total_comfort_bar.png', dpi=300)
# plt.show()
