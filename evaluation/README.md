# HEMS Project Evaluation Module

This folder contains all files related to model evaluation, result analysis, and visualization.

## File Structure

### Core Evaluation Files
- `strategy_carbon_comparison.py` - Algorithm strategy carbon emission comparison experiments
- `carbon_comparison_experiment.py` - Carbon emission comparison experiments (device configuration comparison)
- `model_evaluation.py` - Model performance evaluation
- `seasonal_evaluation.py` - Seasonal evaluation

### Plotting and Visualization
- `plot_ablation_results.py` - Ablation study result plotting
- `plot_advanced_compare.py` - Advanced comparison analysis plotting
- `plot_from_csv.py` - Plot from CSV data
- `plot_xiaorong.py` - Specific analysis plotting
- `plt.py` - General plotting utilities
- `show_environment_plots.py` - Environment state visualization

## Usage

### Run Algorithm Comparison Experiments
```bash
cd evaluation
python strategy_carbon_comparison.py
```

### Run Device Configuration Comparison Experiments
```bash
cd evaluation
python carbon_comparison_experiment.py
```

### Run Model Evaluation
```bash
cd evaluation
python model_evaluation.py
```

## Path Configuration

All files have been configured with correct relative paths:
- Model files: `../model/saved_models/`
- Environment file: `../environment.py`
- Result saving: `../results/`
- Data files: `../data/`

## Output Description

- Plot files are saved in the current evaluation folder
- CSV result files are saved in the project root `results/` folder
- All files include timestamps to avoid overwriting
