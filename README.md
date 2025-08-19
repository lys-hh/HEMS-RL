# HEMS (Home Energy Management System) - 强化学习项目

## 项目概述

本项目使用强化学习技术来优化家庭能源管理系统，实现智能化的能源调度和成本优化。支持多种先进的强化学习算法，包括PPO、DQN、Rainbow DQN、DDPG、SAC、TD3等。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.8+
- NumPy, Pandas, Matplotlib
- Gym/Gymnasium

### 安装依赖
```bash
# 使用requirements.txt安装所有依赖
pip install -r requirements.txt

# 或者手动安装主要依赖
pip install torch numpy pandas matplotlib gym
```

### 快速运行
```bash
# 运行Rainbow DQN训练
python model/dqn.py

# 运行PPO训练
python model/PPO_3rd.py

# 生成可视化结果
python evaluation/plot_from_csv.py
```

## 项目结构

```
HEMS_project/
├── model/                    # 强化学习算法实现
│   ├── PPO_1st.py           # 基础PPO实现
│   ├── PPO_2nd.py           # 改进版PPO
│   ├── PPO_3rd.py           # 旗舰版PPO（当前使用）
│   ├── sac.py               # SAC算法实现
│   ├── sac2.py              # SAC改进版
│   ├── TD3.py               # TD3算法实现
│   ├── dqn.py               # Rainbow DQN算法实现
│   ├── double_dqn.py        # Double DQN算法实现
│   ├── ddpg.py              # DDPG连续控制算法实现
│   ├── evaluation.py        # 模型评估工具
│   └── rl_utils.py          # 强化学习工具函数
├── evaluation/              # 评估和可视化模块
│   ├── model_evaluation.py  # 模型性能评估
│   ├── plot_from_csv.py     # CSV数据可视化
│   ├── strategy_carbon_comparison.py  # 策略对比实验
│   └── README.md            # 评估模块说明
├── figures/                 # 图片文件组织
│   ├── algorithm_comparison/    # 算法对比图
│   ├── environment_plots/       # 环境状态图
│   ├── experiment_results/      # 实验结果图
│   └── README.md            # 图片组织说明
├── simulation_data/         # 模拟数据存储
├── environment.py           # 家庭能源管理环境
├── interface.py             # 数据接口
├── README.md               # 项目说明
└── README_data_saving.md   # 数据保存功能说明
```

## 核心功能

### 1. 智能能源调度
- **电动汽车充放电**：根据电价和需求智能调度
- **储能电池管理**：优化充放电时机
- **家电调度**：洗衣机、空调等设备的智能调度
- **热水器控制**：根据用水需求优化加热策略

### 2. 约束优化
- **SOC约束**：确保电池SOC在安全范围内（10%-90%）
- **功率约束**：限制充放电功率
- **温度舒适度**：保证用户温度舒适度
- **成本优化**：最小化能源成本

### 3. 算法特性
- **PPO-Lagrangian**：支持约束强化学习
- **状态归一化**：提高训练稳定性
- **动态掩码**：确保动作有效性
- **自动回滚**：防止训练崩溃

## 支持的算法

### 1. Rainbow DQN算法（dqn.py）
- **Rainbow DQN**：结合了多种DQN改进技术，包括Double DQN、Dueling Networks、Prioritized Experience Replay、Multi-step Learning、Distributional RL和Noisy Networks。
- **改进特性**：
  - **Double DQN**：分离动作选择和动作评估，减少Q值过估计
  - **Dueling Networks**：分离状态价值和优势函数，提高学习效率
  - **Prioritized Experience Replay**：优先采样重要经验，提高样本效率
  - **Multi-step Learning**：多步学习，减少偏差
  - **Noisy Networks**：用噪声网络替代epsilon-greedy，更好的探索
  - **Huber Loss**：使用Huber损失提高训练稳定性

#### 运行方法
```bash
python model/dqn.py
```

#### 主要参数
- `buffer_size`: 经验回放池大小（默认100000）
- `batch_size`: 每次更新的批量大小（默认256）
- `n_step`: 多步学习步数（默认3）
- `alpha`: 优先经验回放指数（默认0.6）
- `beta`: 重要性采样指数（默认0.4）
- `gamma`: 折扣因子（默认0.96）
- `tau`: 目标网络软更新系数（默认0.005）
- `USE_STATE_NORMALIZATION`: 是否启用状态归一化

### 2. Double DQN算法（double_dqn.py）
- **Double DQN**：通过分离动作选择和动作评估来减少Q值过估计问题，提高训练稳定性和性能。
- **改进特性**：
  - 使用主网络选择动作，目标网络评估动作
  - 添加Dropout层防止过拟合
  - 使用Huber损失提高稳定性
  - 添加梯度裁剪和学习率调度
  - 增加L2正则化

#### 运行方法
```bash
python model/double_dqn.py
```

#### 主要参数
- `buffer_size`: 经验回放池大小（默认100000）
- `batch_size`: 每次更新的批量大小（默认256）
- `epsilon_start/epsilon_end/epsilon_decay`: 探索率参数
- `gamma`: 折扣因子（默认0.96）
- `tau`: 目标网络软更新系数（默认0.005）
- `USE_STATE_NORMALIZATION`: 是否启用状态归一化

### 3. DDPG连续控制算法（ddpg.py）
- **DDPG**：适配多设备多动作空间，支持连续动作输出、OU噪声探索、状态归一化、约束统计、经验回放。
- **用途**：为HEMS环境提供基于DDPG的强化学习解决方案，适合连续动作空间。

#### 运行方法
```bash
python model/ddpg.py
```

#### 主要参数
- `buffer_size`: 经验回放池大小（默认100000）
- `batch_size`: 每次更新的批量大小（默认128）
- `lr_actor/lr_critic`: Actor/Critic学习率
- `gamma`: 折扣因子（默认0.96）
- `tau`: 目标网络软更新系数（默认0.005）
- `USE_STATE_NORMALIZATION`: 是否启用状态归一化

### 4. PPO算法系列
- **PPO_1st.py**: 基础PPO实现
- **PPO_2nd.py**: 改进版PPO，增加约束处理
- **PPO_3rd.py**: 旗舰版PPO，支持PPO-Lagrangian约束强化学习

#### 运行方法
```bash
python model/PPO_3rd.py
```

## 使用方法

### 训练模型
```bash
# Rainbow DQN训练
python model/dqn.py

# PPO训练
python model/PPO_3rd.py

# DDPG训练
python model/ddpg.py
```

### 评估模型
```bash
python evaluation/model_evaluation.py
```

### 可视化结果
```bash
# 从CSV数据生成图片
python evaluation/plot_from_csv.py

# 算法对比实验
python evaluation/strategy_carbon_comparison.py
```

## 配置参数

### 约束配置
- `soc_lower`: SOC下界（默认0.2）
- `soc_upper`: SOC上界（默认0.8）
- `lambda_init`: 初始拉格朗日乘子（默认0.5）
- `lambda_max`: 最大拉格朗日乘子（默认15.0）

### 训练配置
- `num_episodes`: 训练轮数（默认5000）
- `learning_rate`: 学习率（默认5e-6）
- `gamma`: 折扣因子（默认0.96）
- `eps`: PPO裁剪参数（默认0.2）

## 监控指标

### 约束指标
- **ESS_Violation_Rate**: ESS违反率
- **EV_Violation_Rate**: EV违反率
- **Lambda_ESS**: ESS拉格朗日乘子
- **Lambda_EV**: EV拉格朗日乘子

### 性能指标
- **Return**: 总回报
- **Energy_Cost**: 能源成本
- **User_Satisfaction**: 用户满意度
- **Peak_Valley_Arbitrage**: 峰谷套利效果
- **Carbon_Emission**: 碳排放量 (kg CO2)
- **Carbon_Reduction**: 碳排放减少效果

### 训练指标
- **Actor_Loss**: 策略损失
- **Critic_Loss**: 价值损失
- **Constraint_Loss**: 约束损失
- **Training_Stability**: 训练稳定性

## 数据保存

项目支持详细的训练数据保存功能，包括：
- 模拟数据CSV文件
- 奖励组件分解
- 自动生成可视化图片

详细说明请参考：[README_data_saving.md](README_data_saving.md)

## 注意事项

1. **约束训练**：当前使用PPO-Lagrangian方法，需要仔细调参
2. **状态归一化**：建议启用以提高训练稳定性
3. **动态权重**：约束权重会随训练进度动态调整
4. **自动回滚**：当回报崩溃时会自动回滚到健康状态
5. **文件路径**：确保所有依赖文件路径正确

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者

## 更新日志

### v2.0.0 (2024-08-05)
- 添加Rainbow DQN算法实现
- 完善数据保存和可视化功能
- 优化项目文档结构

### v1.0.0 (2024-07-30)
- 初始版本发布
- 支持PPO、DQN、DDPG等基础算法
- 实现家庭能源管理环境

