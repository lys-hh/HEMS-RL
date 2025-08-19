# PPO训练数据保存功能

## 功能概述

在运行PPO训练时，系统会自动保存详细的模拟数据到CSV文件，方便后续分析和绘图。

## 数据保存时机

### 训练过程中
- 只保存最后50个episode的详细数据
- 文件名格式：`simulation_data_episode_XXX.csv`

### 训练结束后
- 保存最终完整数据
- 文件名格式：`final_simulation_data_episode_XXX.csv`
- 自动生成可视化图片

## 数据文件位置

所有CSV文件保存在 `simulation_data/` 文件夹中：
```
simulation_data/
├── simulation_data_episode_951.csv
├── simulation_data_episode_952.csv
├── ...
├── simulation_data_episode_1000.csv
├── final_simulation_data_episode_1000.csv
└── final_simulation_data_episode_1000_rewards.csv
```

## 使用方法

### 1. 运行PPO训练
```bash
python model/PPO_3rd.py
```

训练过程中会：
- 只保存最后50个episode的数据
- 训练结束时保存最终数据
- 自动生成可视化图片

### 2. 从CSV数据生成图片

#### 自动选择最新文件：
```bash
python plot_from_csv.py
```

#### 指定特定文件：
```bash
python plot_from_csv.py simulation_data/final_simulation_data_episode_1000.csv
```

#### 指定输出目录：
```bash
python plot_from_csv.py simulation_data/final_simulation_data_episode_1000.csv save_new
```

## 数据内容

### 主要数据文件包含：
- `timestamp` - 时间戳
- `ev_soc` - EV电池状态
- `ess_soc` - ESS电池状态
- `home_load` - 家庭负载
- `pv_generation` - 光伏发电
- `electricity_price` - 电价
- `air_conditioner_power` - 空调功率
- `wash_machine_state` - 洗衣机状态
- `ewh_temp` - 热水器温度
- `ewh_power` - 热水器功率
- `total_load` - 总负载
- `energy_cost` - 能源成本
- 等等...

### 奖励数据文件包含：
- 各种奖励组件的详细分解
- 总奖励变化
- 约束违反情况

## 生成的图片

运行 `plot_from_csv.py` 会生成以下图片：

1. **EV SOC和电价图** (`ev_soc_and_price.png`)
2. **ESS功率和光伏发电图** (`ess_power_and_pv.png`)
3. **空调功率和温度图** (`ac_power_and_temp.png`)
4. **洗衣机状态图** (`wash_machine_state.png`)
5. **热水器状态图** (`water_heater_status.png`)
6. **家庭总负载图** (`total_load.png`)
7. **能源成本图** (`energy_cost.png`)
8. **奖励组件图** (`reward_components.png`)

## 优势

1. **无需重新训练**：调整图片样式时不需要重新训练模型
2. **数据完整**：保存了所有绘图所需的数据
3. **灵活调整**：可以随时修改图片样式
4. **高质量输出**：所有图片都是高分辨率（300 DPI）
5. **自动化**：训练时自动保存数据

## 注意事项

1. 确保 `simulation_data/` 文件夹存在
2. CSV文件较大，注意磁盘空间
3. 图片生成需要一定时间，请耐心等待
4. 如果修改了环境状态空间，需要重新训练模型

## 故障排除

### 常见问题
1. **找不到CSV文件**：检查 `simulation_data/` 文件夹
2. **图片生成失败**：检查数据列名是否匹配
3. **数据不完整**：确保训练正常完成

### 调试方法
1. 检查控制台输出的错误信息
2. 验证CSV文件的数据完整性
3. 确认所有必需的列都存在 