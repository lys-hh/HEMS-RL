# HEMS项目评估模块

这个文件夹包含了所有与模型评估、结果分析和可视化相关的文件。

## 文件结构

### 核心评估文件
- `strategy_carbon_comparison.py` - 算法策略碳排放对比实验
- `carbon_comparison_experiment.py` - 碳排放对比实验（设备配置对比）  
- `model_evaluation.py` - 模型性能评估
- `seasonal_evaluation.py` - 季节性评估

### 绘图和可视化
- `plot_ablation_results.py` - 消融研究结果绘图
- `plot_advanced_compare.py` - 高级对比分析绘图
- `plot_from_csv.py` - 从CSV数据绘图
- `plot_xiaorong.py` - 特定分析绘图
- `plt.py` - 通用绘图工具
- `show_environment_plots.py` - 环境状态可视化

## 使用方法

### 运行算法对比实验
```bash
cd evaluation
python strategy_carbon_comparison.py
```

### 运行设备配置对比实验
```bash
cd evaluation  
python carbon_comparison_experiment.py
```

### 运行模型评估
```bash
cd evaluation
python model_evaluation.py
```

## 路径说明

所有文件已经配置了正确的相对路径：
- 模型文件：`../model/saved_models/`
- 环境文件：`../environment.py`
- 结果保存：`../results/`
- 数据文件：`../data/`

## 输出说明

- 图片文件会保存在当前evaluation文件夹中
- CSV结果文件会保存在项目根目录的`results/`文件夹中
- 所有文件都带有时间戳以避免覆盖
