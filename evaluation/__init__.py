"""
HEMS project evaluation module
Contains all evaluation, plotting, and comparison analysis related functionality
"""

# Main evaluation classes
from .strategy_carbon_comparison import StrategyCarbonComparison
from .model_evaluation import ModelEvaluator

# Plotting functions
from .plt import plot_returns

__all__ = [
    'StrategyCarbonComparison',
    'ModelEvaluator',
    'plot_returns'
]
