# HEMS (Home Energy Management System) - 强化学习项目

## 项目概述

本项目使用强化学习技术（PPO算法）来优化家庭能源管理系统，实现智能化的能源调度和成本优化。

## 最新改进 (PPO_3rd.py)

### 约束训练问题修复

#### 问题分析
从训练数据发现以下关键问题：
1. **Lambda值固定**：最后lambda值都固定在10.0，说明约束更新机制有问题
2. **违反率过高**：ESS和EV的违反率都在40-60%之间，远高于目标
3. **回报异常**：初期回报极差（-1145），说明约束过于严格

#### 修复措施

##### 1. 移除冲突的Lambda更新机制
- **问题**：存在两套lambda更新逻辑（约束更新 + lambda缓冲机制），导致冲突
- **解决**：移除lambda缓冲机制，只保留约束更新逻辑
- **效果**：确保lambda能正常变化，避免固定值

##### 2. 优化约束目标策略
```python
# 更合理的渐进策略
if episode < 500:
    target_violation_amount = 0.25  # 早期：允许25%违反
elif episode < 1500:
    target_violation_amount = 0.20  # 中期：降低到20%
elif episode < 2500:
    target_violation_amount = 0.15  # 后期：降低到15%
elif episode < 3500:
    target_violation_amount = 0.12  # 后期：降低到12%
else:
    target_violation_amount = 0.10  # 最终：控制在10%
```

##### 3. 调整约束配置参数
```python
constraint_config = {
    'soc_lower': 0.2,  # SOC下界20%
    'soc_upper': 0.8,  # SOC上界80%
    'lambda_init': 0.5,  # 降低初始值，避免过度惩罚
    'lambda_max': 15.0,  # 降低最大值，避免过度惩罚
    'dual_ascent_rate': 0.01,  # 降低更新率，更稳定
    'constraint_weight': 2.0,  # 降低早期权重
    'final_constraint_weight': 0.5  # 降低最终权重
}
```

##### 4. 优化Lambda更新策略
```python
# 更温和的更新策略
adaptive_rate = 0.01 / (1 + 0.001 * episode)  # 降低学习率
kp = 0.05   # 降低比例系数
ki = 0.005  # 降低积分系数
```

##### 5. 调整动态约束权重
```python
# 更温和的权重策略
if episode < 1000:
    constraint_weight = 2.0  # 早期：中等约束权重
elif episode < 2000:
    constraint_weight = 1.5  # 中期：降低约束权重
elif episode < 3000:
    constraint_weight = 1.0  # 后期：进一步降低约束权重
else:
    constraint_weight = 0.5  # 最终：最小约束权重
```

### 预期改进效果

1. **Lambda正常变化**：移除冲突机制后，lambda值应该能正常调整
2. **违反率降低**：更合理的约束目标应该能将违反率控制在10-25%
3. **回报提升**：更温和的约束策略应该能显著提升初期回报
4. **训练稳定性**：降低学习率和权重应该能提高训练稳定性

## 项目结构

```
HEMS_project/
├── model/
│   ├── PPO_1st.py      # 基础PPO实现
│   ├── PPO_2nd.py      # 改进版PPO
│   ├── PPO_3rd.py      # 旗舰版PPO（当前使用）
│   ├── sac.py          # SAC算法实现
│   ├── sac2.py         # SAC改进版
│   └── TD3.py          # TD3算法实现
│   ├── dqn.py          # DQN多分支算法实现
│   ├── ddpg.py         # DDPG连续控制算法实现
├── environment.py       # 家庭能源管理环境
├── interface.py         # 用户界面
├── plt.py              # 绘图工具
├── model_evaluation.py  # 模型评估
└── README.md           # 项目说明
```

## 核心功能

### 1. 智能能源调度
- **电动汽车充放电**：根据电价和需求智能调度
- **储能电池管理**：优化充放电时机
- **家电调度**：洗衣机、空调等设备的智能调度
- **热水器控制**：根据用水需求优化加热策略

### 2. 约束优化
- **SOC约束**：确保电池SOC在安全范围内（20%-80%）
- **功率约束**：限制充放电功率
- **温度舒适度**：保证用户温度舒适度
- **成本优化**：最小化能源成本

### 3. 算法特性
- **PPO-Lagrangian**：支持约束强化学习
- **状态归一化**：提高训练稳定性
- **动态掩码**：确保动作有效性
- **自动回滚**：防止训练崩溃

### 4. DQN多分支算法（dqn.py）
- **多分支DQN**：适配多设备多动作空间，支持状态归一化、约束统计、经验回放、epsilon-greedy探索。
- **用途**：为HEMS环境提供基于DQN的强化学习解决方案，适合离散动作空间。

#### 运行方法
```bash
python model/dqn.py
```

#### 主要参数
- `buffer_size`: 经验回放池大小（默认50000）
- `batch_size`: 每次更新的批量大小（默认128）
- `epsilon_start/epsilon_end/epsilon_decay`: 探索率参数
- `gamma`: 折扣因子（默认0.96）
- `tau`: 目标网络软更新系数（默认0.01）
- `USE_STATE_NORMALIZATION`: 是否启用状态归一化

#### 监控指标
- **ESS_Violation_Rate**: 储能SOC违反率
- **EV_Violation_Rate**: 电动汽车SOC违反率
- **Total_Violation_Rate**: 约束总违反率
- **Energy_Cost**: 能源成本
- **User_Satisfaction**: 用户满意度
- **Temperature_Comfort**: 温度舒适度
- **Loss**: DQN损失

#### 结果输出
- 训练过程数据保存在`results/returns_dqn_时间戳.csv`
- 训练模型保存在`saved_models/dqn_model_时间戳.pth`
- 支持训练过程可视化

### 5. DDPG连续控制算法（ddpg.py）
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

#### 监控指标
- **ESS_Violation_Rate**: 储能SOC违反率
- **EV_Violation_Rate**: 电动汽车SOC违反率
- **Total_Violation_Rate**: 约束总违反率
- **Energy_Cost**: 能源成本
- **User_Satisfaction**: 用户满意度
- **Temperature_Comfort**: 温度舒适度
- **Actor_Loss**: 策略损失
- **Critic_Loss**: 价值损失

#### 结果输出
- 训练过程数据保存在`results/returns_ddpg_时间戳.csv`
- 训练模型保存在`saved_models/ddpg_model_时间戳.pth`
- 支持训练过程可视化

## 使用方法

### 训练模型
```bash
python model/PPO_3rd.py
```

### 评估模型
```bash
python model_evaluation.py
```

### 可视化结果
```bash
python plt.py
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

### 训练指标
- **Actor_Loss**: 策略损失
- **Critic_Loss**: 价值损失
- **Constraint_Loss**: 约束损失
- **Training_Stability**: 训练稳定性

## 注意事项

1. **约束训练**：当前使用PPO-Lagrangian方法，需要仔细调参
2. **状态归一化**：建议启用以提高训练稳定性
3. **动态权重**：约束权重会随训练进度动态调整
4. **自动回滚**：当回报崩溃时会自动回滚到健康状态

## 更新日志

### 最新版本 (PPO_3rd.py)
- ✅ 修复Lambda更新冲突问题
- ✅ 优化约束目标策略
- ✅ 调整约束配置参数
- ✅ 改进动态权重策略
- ✅ 简化训练逻辑

### 待优化项目
- [ ] 进一步优化约束目标
- [ ] 改进Lambda更新策略
- [ ] 增加更多监控指标
- [ ] 优化回滚机制 