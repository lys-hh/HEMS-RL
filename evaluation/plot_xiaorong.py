'''
This file plots simulated data and database data (input data)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
import matplotlib.dates as mdates
import os

# Set font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # For displaying labels normally
plt.rcParams['axes.unicode_minus'] = False  # For displaying minus sign normally

# Set matplotlib for highest quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

# 1. Database data
cons = pd.read_csv('data/daily_pivot_cons_2011-2012.csv', index_col=0)
prod = pd.read_csv('data/daily_pivot_prod_2011-2012.csv', index_col=0)

date_range = pd.date_range('2011-07-03', periods=7)
date_strs = [d.strftime('%Y-%m-%d') for d in date_range]
cons_week = cons.loc[date_strs]
prod_week = prod.loc[date_strs]

cons_flat = cons_week.values.flatten()
prod_flat = prod_week.values.flatten()

# Generate time series
time_index = pd.date_range('2011-07-03 00:00', periods=7*48, freq='30min')

# Create 2×2 layout charts
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# (a) Database Data chart
axs[0, 0].plot(time_index, cons_flat, label='Load', linewidth=3)
axs[0, 0].plot(time_index, prod_flat, label='PV', linewidth=3)
axs[0, 0].set_title('Database Data', fontsize=14)
axs[0, 0].text(-0.1, 1.1, 'a', transform=axs[0, 0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axs[0, 0].set_xlabel('Time', fontsize=12)
axs[0, 0].set_ylabel('Energy (kWh)', fontsize=12)
axs[0, 0].legend(fontsize=11)
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].tick_params(labelsize=11)
# Set x-axis format to show only month and date, avoid year overlap
axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
axs[0, 0].xaxis.set_major_locator(mdates.DayLocator(interval=1))
# plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)

# (b) Time-of-use electricity price
def get_electricity_price(idx):
    if idx >= 48:
        idx = idx - 48
    if 0 <= idx < 20 or 44 <= idx < 48:
        return 0.2
    elif 8 <= idx < 34:
        return 0.5
    else:
        return 0.8

price_curve = [get_electricity_price(i) for i in range(48)]
hours = np.arange(0, 24, 0.5)  # 0, 0.5, ..., 23.5
axs[0, 1].step(hours, price_curve, where='mid', linewidth=3)
axs[0, 1].set_title('Time-of-use Electricity Price', fontsize=14)
axs[0, 1].text(-0.1, 1.1, 'b', transform=axs[0, 1].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axs[0, 1].set_xlabel('Hour', fontsize=12)
axs[0, 1].set_ylabel('Price (¥/kWh)', fontsize=12)
axs[0, 1].set_xticks(np.arange(0, 25, 2))
axs[0, 1].tick_params(labelsize=11)
axs[0, 1].grid(True, alpha=0.3)

# (c) EV travel schedule (normal distribution)
t1_mean, t1_std, t1_range = 20, 1, (18, 22)  # Home arrival
t2_mean, t2_std, t2_range = 8, 1, (7, 9)     # Leave home

x = np.linspace(0, 24, 400)
y_t1 = norm.pdf(x, t1_mean, t1_std)
y_t2 = norm.pdf(x, t2_mean, t2_std)
axs[1, 0].plot(x, y_t1, label='Arrival Time', linewidth=3)
axs[1, 0].plot(x, y_t2, label='Departure Time', linewidth=3)
axs[1, 0].set_xlim(0, 24)
axs[1, 0].set_xticks(np.arange(0, 25, 2))
axs[1, 0].set_title('EV Travel Schedule (Normal Distribution)', fontsize=14)
axs[1, 0].text(-0.1, 1.1, 'c', transform=axs[1, 0].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axs[1, 0].set_xlabel('Hour', fontsize=12)
axs[1, 0].set_ylabel('Probability Density', fontsize=12)
axs[1, 0].legend(loc='upper left', fontsize=11)
axs[1, 0].tick_params(labelsize=11)
axs[1, 0].grid(True, alpha=0.3)

# (d) Daily mileage (log-normal distribution, PDF)
avg_daily_km = 50
km_std = 0.2
x_km = np.linspace(10, 120, 300)
shape = km_std
scale = avg_daily_km
y_km = lognorm.pdf(x_km, s=shape, scale=scale)
axs[1, 1].plot(x_km, y_km, label='Daily Mileage Probability Density', linewidth=3)
axs[1, 1].set_title('Daily Mileage Distribution (Log-normal Distribution)', fontsize=14)
axs[1, 1].text(-0.1, 1.1, 'd', transform=axs[1, 1].transAxes, fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
axs[1, 1].set_xlabel('Mileage (km)', fontsize=12)
axs[1, 1].set_ylabel('Probability Density', fontsize=12)
axs[1, 1].tick_params(labelsize=11)
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/algorithm_comparison/ablation_results.png', dpi=600, bbox_inches='tight')
plt.close()
