'''
Algorithm Comparison Evaluation
Plot return comparison charts for different reinforcement learning algorithms
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory to save images
os.makedirs('save_pictures', exist_ok=True)

# Algorithm data file configuration
algorithm_files = {
    "Proposed RL": "model/results/returns_ppo_20250709_144004_norm_unconstrained.csv",
    "TD3": "model/results/returns_td3_20250729_221518.csv",
    # "SAC2": "model/results/returns_sac_20250730_194256.csv",
    # "SAC3": "model/results/returns_sac_20250724_183720.csv",
    "Rainbow DQN": "model/results/returns_rainbow_dqn_20250805_022938_norm.csv",
    "SAC": "model/results/returns_sac_20250723_215709.csv",
    # "SAC5": "model/results/returns_sac_20250710_235548.csv",
    "DDPG": "model/results/returns_ddpg_20250726_173145_no_norm.csv"
}

# Color scheme
colors = {
    "Proposed RL": '#4ECDC4',
    "TD3": '#FECA57',
    "SAC": '#9B59B6',
    "DDPG": '#96CEB4',
    "Rainbow DQN": '#45B7D1'  
}

def load_and_process_data():
    """Load and process data for all algorithms"""
    data = {}
    
    for algorithm, file_path in algorithm_files.items():
        try:
            df = pd.read_csv(file_path)
            if 'Return' in df.columns:
                data[algorithm] = {
                    'episodes': df['Episode'].values,
                    'returns': df['Return'].values,
                    'costs': df['Energy_Cost'].values if 'Energy_Cost' in df.columns else None,
                    'satisfaction': df['User_Satisfaction'].values if 'User_Satisfaction' in df.columns else None
                }
                print(f"Successfully loaded {algorithm}: {len(df)} episodes")
            else:
                print(f"Warning: 'Return' column not found in {algorithm} file")
        except Exception as e:
            print(f"Error: Failed to load {algorithm} file: {e}")
    
    return data

def smooth_curve(y, window=50):
    """Smooth curve function"""
    return np.convolve(y, np.ones(window)/window, mode='valid')

def plot_algorithm_comparison(data):
    """Plot algorithm comparison chart"""
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Return comparison (top left)
    for algorithm, data_dict in data.items():
        episodes = data_dict['episodes']
        returns = data_dict['returns']
        
        # Plot original data (lower opacity)
        axes[0, 0].plot(episodes, returns, color=colors[algorithm], alpha=0.3, linewidth=1)
        
        # Plot smoothed curve
        if len(returns) >= 50:
            smoothed_returns = smooth_curve(returns)
            smoothed_episodes = episodes[49:len(smoothed_returns)+49]
            axes[0, 0].plot(smoothed_episodes, smoothed_returns, 
                           color=colors[algorithm], linewidth=2, label=algorithm)
        else:
            axes[0, 0].plot(episodes, returns, color=colors[algorithm], 
                           linewidth=2, label=algorithm)
    
    axes[0, 0].set_title('Return Comparison', fontsize=16)
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Return', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(-0.2, 1.1, 'a', transform=axes[0, 0].transAxes, 
                     fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    # 2. Cost comparison (top right)
    for algorithm, data_dict in data.items():
        if data_dict['costs'] is not None:
            episodes = data_dict['episodes']
            costs = data_dict['costs']
            
            # Plot original data
            axes[0, 1].plot(episodes, costs, color=colors[algorithm], alpha=0.3, linewidth=1)
            
            # Plot smoothed curve
            if len(costs) >= 50:
                smoothed_costs = smooth_curve(costs)
                smoothed_episodes = episodes[49:len(smoothed_costs)+49]
                axes[0, 1].plot(smoothed_episodes, smoothed_costs, 
                               color=colors[algorithm], linewidth=2, label=algorithm)
            else:
                axes[0, 1].plot(episodes, costs, color=colors[algorithm], 
                               linewidth=2, label=algorithm)
    
    axes[0, 1].set_title('Energy Cost Comparison', fontsize=16)
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Energy Cost', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(-0.2, 1.1, 'b', transform=axes[0, 1].transAxes, 
                     fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    # 3. User satisfaction comparison (bottom left)
    for algorithm, data_dict in data.items():
        if data_dict['satisfaction'] is not None:
            episodes = data_dict['episodes']
            satisfaction = data_dict['satisfaction']
            
            # Plot original data
            axes[1, 0].plot(episodes, satisfaction, color=colors[algorithm], alpha=0.3, linewidth=1)
            
            # Plot smoothed curve
            if len(satisfaction) >= 50:
                smoothed_satisfaction = smooth_curve(satisfaction)
                smoothed_episodes = episodes[49:len(smoothed_satisfaction)+49]
                axes[1, 0].plot(smoothed_episodes, smoothed_satisfaction, 
                               color=colors[algorithm], linewidth=2, label=algorithm)
            else:
                axes[1, 0].plot(episodes, satisfaction, color=colors[algorithm], 
                               linewidth=2, label=algorithm)
    
    axes[1, 0].set_title('User Satisfaction Comparison', fontsize=16)
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('User Satisfaction', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(-0.2, 1.1, 'c', transform=axes[1, 0].transAxes, 
                     fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    # 4. Final performance comparison bar chart (bottom right)
    final_returns = []
    algorithm_names = []
    
    for algorithm, data_dict in data.items():
        if len(data_dict['returns']) > 0:
            # Take average of last 100 episodes as final performance
            final_100_returns = data_dict['returns'][-100:] if len(data_dict['returns']) >= 100 else data_dict['returns']
            final_returns.append(np.mean(final_100_returns))
            algorithm_names.append(algorithm)
    
    if final_returns:
        bars = axes[1, 1].bar(algorithm_names, final_returns, 
                              color=[colors[name] for name in algorithm_names])
        axes[1, 1].set_title('Final Performance Comparison', fontsize=16)
        axes[1, 1].set_ylabel('Average Return (Last 100 Episodes)', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].text(-0.2, 1.1, 'd', transform=axes[1, 1].transAxes, 
                         fontsize=24, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    plt.savefig('figures/algorithm_comparison/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Algorithm comparison chart saved as: figures/algorithm_comparison/algorithm_comparison.png")

def plot_individual_metrics(data):
    """Plot individual metric comparison charts"""
    
    # Return comparison chart
    plt.figure(figsize=(12, 8))
    for algorithm, data_dict in data.items():
        episodes = data_dict['episodes']
        returns = data_dict['returns']
        
        # Plot raw data
        plt.plot(episodes, returns, color=colors[algorithm], alpha=0.3, linewidth=1)
        
        # Plot smoothed curve
        if len(returns) >= 50:
            smoothed_returns = smooth_curve(returns)
            smoothed_episodes = episodes[49:len(smoothed_returns)+49]
            plt.plot(smoothed_episodes, smoothed_returns, 
                    color=colors[algorithm], linewidth=1, label=algorithm)
        else:
            plt.plot(episodes, returns, color=colors[algorithm], 
                    linewidth=2, label=algorithm)
    
    plt.title('Algorithm Return Comparison', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/algorithm_comparison/return_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Return comparison chart saved as: figures/algorithm_comparison/return_comparison.png")

def main():
    """Main function"""
    print("Starting to load algorithm data...")
    data = load_and_process_data()
    
    if not data:
        print("Error: No data loaded successfully")
        return
    
    print(f"Successfully loaded data for {len(data)} algorithms")
    
    # Plot comprehensive comparison chart
    print("Plotting algorithm comparison chart...")
    plot_algorithm_comparison(data)
    
    # Plot individual return comparison chart
    print("Plotting return comparison chart...")
    plot_individual_metrics(data)
    
    print("All plots generated!")

if __name__ == "__main__":
    main() 