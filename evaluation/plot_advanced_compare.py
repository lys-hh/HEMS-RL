'''
This file plots heatmaps, scatter plots, and radar charts
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D

# File paths and experiment labels
files = {
    "No Dynamic Mask": "model/results/returns_ppo_20250709_121258_norm_unconstrained.csv",
    "Proposed RL": "model/results/returns_ppo_20250709_144004_norm_unconstrained.csv",
    "No State Normalization": "model/results/returns_ppo_20250709_172401_no_norm_unconstrained.csv",
    "No Advantage Normalization": "model/results/returns_ppo_20250709_222349_norm_unconstrained.csv",
    "no shared layer": "model/results/returns_ppo2_20250710_202931_norm.csv"
}

# Read data
data = {}
for label, file in files.items():
    data[label] = pd.read_csv(file)

os.makedirs('save_pictures', exist_ok=True)

# Radar chart (enhanced version)
radar_metrics = [
    ('Return', 'Return'),
    ('Energy_Cost', 'Energy Cost'),
    ('Total_Violation_Rate', 'Violation Rate'),
    ('Temperature_Comfort', 'Total Comfort')
]
radar_data = {}
for label, df in data.items():
    values = []
    for col, _ in radar_metrics:
        if col in df.columns:
            values.append(df[col].values[-1])
        else:
            values.append(np.nan)
    radar_data[label] = values
all_values = np.array(list(radar_data.values()))
min_vals = np.nanmin(all_values, axis=0)
max_vals = np.nanmax(all_values, axis=0)
norm_data = {k: (np.array(v)-min_vals)/(max_vals-min_vals+1e-8) for k, v in radar_data.items()}
labels_radar = [name for _, name in radar_metrics]
num_vars = len(labels_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for label, values in norm_data.items():
    stats = list(values) + [values[0]]
    ax.plot(angles, stats, label=label, linewidth=3)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['']*num_vars)  # Don't use default labels
for i, label_txt in enumerate(labels_radar):
    radius = 1.18
    if i in [0, 2, 4]:  # Leftmost, right, rightmost
        ax.text(angles[i], radius, label_txt, ha='center', va='center', rotation=90, fontsize=14, fontweight='bold', transform=ax.transData)
    else:
        ax.text(angles[i], radius, label_txt, ha='center', va='center', fontsize=14, fontweight='bold', transform=ax.transData)
ax.yaxis.grid(True, linestyle='--', linewidth=1, alpha=0.5)
ax.xaxis.grid(True, linestyle='-', linewidth=1.5)
ax.set_yticklabels([])
plt.title('Performance Radar Chart', fontsize=18, fontweight='bold', pad=30)
plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10)
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/performance_radar.png', dpi=300)
plt.show()

# 3D scatter plot
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
for label, df in data.items():
    if all(col in df.columns for col in ['Total_Violation_Rate', 'Energy_Cost', 'Temperature_Comfort']):
        ax3d.scatter(
            df['Total_Violation_Rate'],
            df['Energy_Cost'],
            df['Temperature_Comfort'],
            alpha=0.5,
            label=label
        )
ax3d.set_xlabel('Total Violation Rate')
ax3d.set_ylabel('Energy Cost (¥)')
ax3d.set_zlabel('Total Comfort')
# ax3d.set_title('3D Scatter: Violation Rate, Energy Cost, Total Comfort', fontsize=15)
ax3d.legend(loc='best')
plt.tight_layout()
plt.savefig('figures/algorithm_comparison/3d_violation_energycost_comfort.png', dpi=600)
plt.show()

# Heatmap: Correlation of main numerical indicators for Proposed RL experiment
proposed_df = data.get('Proposed RL')
if proposed_df is not None:
    corr_cols = ['Return', 'Energy_Cost', 'Total_Violation_Rate', 'User_Satisfaction', 'Temperature_Comfort', 'Sample_Efficiency', 'Training_Stability']
    corr_df = proposed_df[[c for c in corr_cols if c in proposed_df.columns]]
    corr = corr_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap (Proposed RL)')
    plt.tight_layout()
    plt.savefig('figures/algorithm_comparison/proposed_rl_corr_heatmap.png', dpi=300)
    plt.show()
