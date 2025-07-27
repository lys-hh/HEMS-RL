'''
该文件绘制模拟数据和数据库数据（输入数据）的图
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
import matplotlib.dates as mdates
import os

# 1. Database data
cons = pd.read_csv('data/daily_pivot_cons_2011-2012.csv', index_col=0)
prod = pd.read_csv('data/daily_pivot_prod_2011-2012.csv', index_col=0)

date_range = pd.date_range('2011-07-03', periods=7)
date_strs = [d.strftime('%Y-%m-%d') for d in date_range]
cons_week = cons.loc[date_strs]
prod_week = prod.loc[date_strs]

cons_flat = cons_week.values.flatten()
prod_flat = prod_week.values.flatten()

# 生成时间序列
time_index = pd.date_range('2011-07-03 00:00', periods=7*48, freq='30min')

# 保存Database Data图
os.makedirs('save_models', exist_ok=True)
plt.figure(figsize=(14, 4))
plt.plot(time_index, cons_flat, label='Load', linewidth=2)
plt.plot(time_index, prod_flat, label='PV', linewidth=2)
plt.title('Database Data')
plt.xlabel('Time')
plt.ylabel('kWh')
plt.legend()
plt.tight_layout()
plt.savefig('save_pictures/database_data.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Simulated data
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

# 2.1 Time-of-use electricity price
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
axs[0].step(hours, price_curve, where='mid')
axs[0].set_title('Time-of-Use Electricity Price')
axs[0].set_xlabel('Hour')
axs[0].set_ylabel('Price (CNY/kWh)')
axs[0].set_xticks(np.arange(0, 25, 2))
# axs[0].grid(True)

# 2.2 EV travel schedule (normal distribution)
t1_mean, t1_std, t1_range = 20, 1, (18, 22)  # Home arrival
t2_mean, t2_std, t2_range = 8, 1, (7, 9)     # Leave home

x = np.linspace(0, 24, 400)
y_t1 = norm.pdf(x, t1_mean, t1_std)
y_t2 = norm.pdf(x, t2_mean, t2_std)
axs[1].plot(x, y_t1, label='Arrival Time')
axs[1].plot(x, y_t2, label='Departure Time')
axs[1].set_xlim(0, 24)
axs[1].set_xticks(np.arange(0, 25, 2))
axs[1].set_title('EV Travel Schedule (Normal Distribution)')
axs[1].set_xlabel('Hour')
axs[1].set_ylabel('Probability Density')
axs[1].legend(loc='upper left')
# axs[1].grid(True)

# 2.3 Daily mileage (log-normal distribution, PDF)
avg_daily_km = 50
km_std = 0.2
x_km = np.linspace(10, 120, 300)
shape = km_std
scale = avg_daily_km
y_km = lognorm.pdf(x_km, s=shape, scale=scale)
axs[2].plot(x_km, y_km, label='Daily Mileage PDF')
axs[2].set_title('Daily Mileage Distribution (Log-normal)')
axs[2].set_xlabel('Mileage (km)')
axs[2].set_ylabel('Probability Density')
# axs[2].grid(True)

plt.suptitle('Simulated Data')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('save_pictures/simulated_data.png', dpi=300, bbox_inches='tight')
plt.show()
