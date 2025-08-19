# 图片文件组织结构

本文件夹统一存放项目中生成的所有图片和图表，按功能分类管理。

## 📁 文件夹结构

### `algorithm_comparison/` - 算法对比
存放不同算法之间的性能对比图表：
- `algorithm_comparison.png` - 算法综合对比图
- `return_comparison.png` - 回报对比图
- `ablation_results.png` - 消融实验结果
- `line_charts_comparison.png` - 折线图对比
- `energy_cost_bar.png` - 能源成本柱状图
- `user_satisfaction_bar.png` - 用户满意度柱状图
- `violation_rate_bar.png` - 违规率柱状图
- `comprehensive_comparison.png` - 综合对比图
- `performance_radar.png` - 性能雷达图
- `3d_violation_energycost_comfort.png` - 3D散点图
- `proposed_rl_corr_heatmap.png` - 相关性热力图

### `environment_plots/` - 环境模拟图
存放家庭能源管理环境的各种状态图：
- `ev_soc_and_price.png` - EV SOC和电价图
- `ess_power_and_pv.png` - ESS功率和光伏发电图
- `ac_power_and_temp.png` - 空调功率和温度图
- `ac_power_and_temp2.png` - 第二台空调功率和温度图
- `wash_machine_state.png` - 洗衣机状态图
- `water_heater_status.png` - 热水器状态图
- `total_load.png` - 家庭总负载图
- `energy_cost.png` - 能源成本图
- `reward_components.png` - 奖励组件图
- `plot_summary.png` - 图表汇总
- `p2.png` ~ `p10.png` - 各种详细环境状态图

### `experiment_results/` - 实验结果
存放各种对比实验的结果图表：
- `carbon_emission_analysis.png` - 碳排放分析图
- `strategy_carbon_comparison.png` - 策略碳排放对比图
- `seasonal_comparison.png` - 季节性对比图

### `model_evaluation/` - 模型评估
存放模型性能评估相关的图表：
- `arbitrage_analysis_*.png` - 套利分析图
- `arbitrage_breakdown_*.png` - 套利分解图
- `arbitrage_statistics_*.png` - 套利统计图
- `evaluation_plots_*.png` - 评估结果图
- `violation_rate_*.png` - 违规率图

### `training_progress/` - 训练进度
存放模型训练过程的图表：
- `td3_returns_*.png` - TD3算法回报曲线
- `td3_losses_*.png` - TD3算法损失曲线

## 🔧 使用说明

1. **自动生成**: 所有图片都由相应的Python脚本自动生成并保存到对应文件夹
2. **路径配置**: 各个脚本中的保存路径已统一配置到此结构中
3. **文件命名**: 时间戳格式为 `YYYYMMDD_HHMMSS`，确保文件唯一性
4. **文件管理**: 旧的分散文件夹（`save/`, `save_pictures/`, `evaluation_results/`图片）已整合到此结构中

## 📊 相关脚本

- `environment.py` → `environment_plots/`
- `evaluation/model_evaluation.py` → `model_evaluation/`
- `evaluation/plot_*.py` → `algorithm_comparison/`
- `evaluation/*_comparison*.py` → `experiment_results/`
- `model/TD3.py` → `training_progress/`

这种组织方式使得项目图片管理更加清晰，便于查找和维护。
