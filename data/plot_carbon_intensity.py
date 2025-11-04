"""
Plot carbon intensity over time
Features:
1. Read CSV format carbon intensity data
2. Convert units from gCO2/kWh to kgCO2/kWh
3. Plot time series (sampling period: one day)
4. Support multi-region data visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import os

# Set matplotlib parameters
plt.rcParams['font.family'] = ['DejaVu Sans']  # Use DejaVu Sans for English text
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Font setup removed - using default English fonts

class CarbonIntensityPlotter:
    """Carbon intensity data visualization class"""
    
    def __init__(self, data_file_path):
        """
        Initialize the plotter
        
        Args:
            data_file_path: CSV data file path
        """
        self.data_file_path = data_file_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load CSV data"""
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file does not exist: {self.data_file_path}")
        
        print(f"Loading data: {self.data_file_path}")
        self.data = pd.read_csv(self.data_file_path)
        
        # Convert timestamp column
        if 'timestamp_utc' in self.data.columns:
            self.data['timestamp_utc'] = pd.to_datetime(self.data['timestamp_utc'])
            print("Timestamp conversion successful")
        
        print(f"Data shape: {self.data.shape}")
        print(f"Data columns: {list(self.data.columns)}")
        print(f"Time range: {self.data['timestamp_utc'].min()} to {self.data['timestamp_utc'].max()}")
    
    def convert_units(self, value_g_co2_per_kwh):
        """
        Unit conversion: from gCO2/kWh to kgCO2/kWh
        
        Args:
            value_g_co2_per_kwh: Carbon intensity value in gCO2/kWh
            
        Returns:
            Carbon intensity value in kgCO2/kWh
        """
        if pd.isna(value_g_co2_per_kwh):
            return np.nan
        return value_g_co2_per_kwh / 1000.0  # Convert grams to kilograms
    
    def plot_carbon_intensity_time_series(self, save_path='data/carbon_intensity_time_series.png'):
        """
        Plot carbon intensity over time (sampling period: one day)
        
        Args:
            save_path: Image save path
        """
        plt.figure(figsize=(20, 8))
        
        # Find carbon intensity columns
        carbon_columns = [col for col in self.data.columns if col.startswith('carbon_intensity_gco2_per_kwh_')]
        
        if not carbon_columns:
            raise ValueError("Carbon intensity data columns not found")
        
        # Get carbon intensity data column (should only have one NSW region)
        carbon_col = carbon_columns[0]  # Take the first (and only) carbon intensity column
        region = carbon_col.replace('carbon_intensity_gco2_per_kwh_', '').upper()
        
        # Data preprocessing: sample by day (take daily average)
        df_region = self.data[['timestamp_utc', carbon_col]].copy()
        df_region['date'] = df_region['timestamp_utc'].dt.date
        
        # Group by day and calculate average
        daily_data = df_region.groupby('date')[carbon_col].mean().reset_index()
        daily_data['timestamp_utc'] = pd.to_datetime(daily_data['date'])
        
        # Get original data (gCO2/kWh) and convert units
        original_values = daily_data[carbon_col].values
        converted_values = [self.convert_units(val) for val in original_values]
        
        # Plot curve
        plt.plot(daily_data['timestamp_utc'], converted_values, 
                label=f'{region} Carbon Intensity (Daily Average)', 
                color='#1f77b4', 
                linewidth=2, 
                alpha=0.8)
        
        # Set chart properties
        plt.xlabel('Time', fontsize=14, fontweight='bold')
        plt.ylabel('Carbon Intensity (kg CO₂/kWh)', fontsize=14, fontweight='bold')
        plt.title(f'Australia {region} Region Carbon Intensity Time Series (Daily Average)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Set time axis format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Set y-axis range
        plt.ylim(0, None)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Carbon intensity time series plot (daily average) saved: {save_path}")
    
    def generate_plot(self, output_dir='data'):
        """Generate carbon intensity time series plot"""
        print("Starting to generate carbon intensity time series plot...")
        
        # Generate time series plot
        self.plot_carbon_intensity_time_series(f'{output_dir}/carbon_intensity_time_series.png')
        
        print("Carbon intensity time series plot generation completed!")

def main():
    """Main function"""
    # Data file path
    data_file_path = "data/nsw_carbon_intensity.csv"
    
    # Check if file exists
    if not os.path.exists(data_file_path):
        print(f"Data file does not exist: {data_file_path}")
        print("Please run data/carbon.py first to fetch carbon intensity data")
        return
    
    try:
        # Create plotter
        plotter = CarbonIntensityPlotter(data_file_path)
        
        # Generate carbon intensity time series plot
        plotter.generate_plot()
        
        print(f"\nCarbon intensity time series plot saved to data folder")
        print(f"Data source: {data_file_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
