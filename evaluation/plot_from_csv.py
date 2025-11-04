"""
Generate plots from CSV data
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta

class CSVPlotter:
    def __init__(self, csv_file_path):
        """Initialize the plotter"""
        self.csv_file_path = csv_file_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load CSV data"""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_file_path}")
        
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Data loaded: {self.csv_file_path}")
        print(f"Data shape: {self.data.shape}")
        print(f"Data columns: {list(self.data.columns)}")
        
        # Convert timestamp
        if 'timestamp' in self.data.columns:
            try:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                print("Timestamp conversion successful")
            except Exception as e:
                print(f"Timestamp conversion failed: {e}")
                # If conversion fails, try other possible column names
                possible_timestamp_cols = ['time', 'date', 'datetime', 'Time', 'Date', 'DateTime']
                for col in possible_timestamp_cols:
                    if col in self.data.columns:
                        try:
                            self.data['timestamp'] = pd.to_datetime(self.data[col])
                            print(f"Using column '{col}' as timestamp")
                            break
                        except:
                            continue
                else:
                    print("Warning: Unable to find valid timestamp column")
            # Create a simple time index
            self.data['timestamp'] = pd.date_range('2011-07-03', periods=len(self.data), freq='30min')
        else:
            print("Warning: CSV file does not contain 'timestamp' column")
            # Create a simple time index
            self.data['timestamp'] = pd.date_range('2011-07-03', periods=len(self.data), freq='30min')
    
    def filter_data_by_date_range(self, start_date=None, end_date=None):
        """Filter data by specified date range
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            end_date (str): End date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        
        Returns:
            pandas.DataFrame: Filtered data, returns None if no data found
        """
        if self.data is None:
            raise ValueError("Data not loaded, please call load_data() first")
        
        filtered_data = self.data.copy()
        
        # Process start date
        if start_date is not None:
            start_datetime = pd.to_datetime(start_date)
            # If no time specified, default to 00:00:00
            if start_datetime.time() == pd.Timestamp('00:00:00').time():
                start_datetime = start_datetime.replace(hour=0, minute=0, second=0)
            filtered_data = filtered_data[filtered_data['timestamp'] >= start_datetime]
        
        # Process end date
        if end_date is not None:
            end_datetime = pd.to_datetime(end_date)
            # If no time specified, default to 23:59:59
            if end_datetime.time() == pd.Timestamp('00:00:00').time():
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            filtered_data = filtered_data[filtered_data['timestamp'] <= end_datetime]
        
        if len(filtered_data) == 0:
            print(f"Warning: No data found for specified date range")
            print(f"Start date: {start_date}, End date: {end_date}")
            return None
        
        print(f"Filtered data for date range: {len(filtered_data)} records")
        if start_date:
            print(f"Start time: {start_date}")
        if end_date:
            print(f"End time: {end_date}")
        
        return filtered_data
    
    def filter_data_by_date(self, target_date='2011-07-03'):
        """Filter data by specified date (maintain backward compatibility)"""
        return self.filter_data_by_date_range(target_date, target_date)
    
    def plot_ev_soc_and_price(self, save_path='figures/environment_plots/ev_soc_and_price.png', start_date=None, end_date=None, target_date=None):
        """Plot EV SOC and electricity price
        
        Args:
            save_path (str): Save path
            start_date (str): Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            end_date (str): End date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            target_date (str): Target date (maintain backward compatibility)
        """
        plt.figure(figsize=(20, 8))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        # Use already converted timestamps
        timestamps = plot_data['timestamp']
        ev_soc = plot_data['ev_soc'].values
        ev_at_home = plot_data['ev_at_home'].values
        electricity_price = plot_data['electricity_price'].values
        
        # Set away periods to NaN
        ev_soc[~ev_at_home.astype(bool)] = np.nan
        
        ax1 = plt.subplot(1, 1, 1)
        mpl_dates = mdates.date2num(timestamps)
        
        # Plot EV SOC curve
        ax1.plot(mpl_dates, ev_soc, color='blue', label='EV SOC', linewidth=4)
        ax1.set_ylabel('EV SOC (kWh)', fontsize=30)
        
        # Plot electricity price curve (right axis) - using step function
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', label='Electricity Price', 
                       where='post', linewidth=4)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=30)
        
        # Merge legends from both axes and place at lower left
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_price.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(0, 0.05))
        # Set legend font size
        for text in legend.get_texts():
            text.set_fontsize(20)
        
        # Set legend border color
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(2)
        
        # Draw filled areas for home periods segment by segment
        home_segments = self._find_contiguous_segments(ev_at_home.astype(bool))
        for start, end in home_segments:
            segment_dates = mdates.date2num(timestamps.iloc[start:end + 1])
            ax1.fill_between(segment_dates, 0, 1, color='lightblue', alpha=0.3,
                           transform=ax1.get_xaxis_transform())
        
        # Set time axis format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax1.tick_params(axis='x', labelsize=25)
        
        # Set x-axis label
        ax1.set_xlabel('Time', fontsize=30)
        
        # Set y-axis number font size
        ax1.tick_params(axis='y', labelsize=25)
        ax1_price.tick_params(axis='y', labelsize=25)
        
        # Add darker black borders
        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        for spine in ax1_price.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_ess_power_and_pv(self, save_path='figures/environment_plots/ess_power_and_pv.png', start_date=None, end_date=None, target_date=None):
        """Plot ESS power and PV generation"""
        plt.figure(figsize=(20, 10))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # Ensure x-axis range is consistent for both subplots
        min_date = mpl_dates[0]
        max_date = mpl_dates[-1]
        
        # Check if ess_actions column exists, if not calculate from ess_soc
        if 'ess_actions' not in plot_data.columns:
            print("Warning: CSV file missing 'ess_actions' column, calculating ESS power from ess_soc difference")
            
            # Calculate ess_actions from ess_soc difference
            ess_soc = plot_data['ess_soc'].values
            ess_actions = []
            
            for i in range(len(ess_soc)):
                if i == 0:
                    # First time step, assume initial power is 0
                    ess_actions.append(0.0)
                else:
                    # Calculate SOC change rate, convert to power
                    # Assume time interval is 0.5 hours, efficiency is 0.95
                    time_interval = 0.5  # hours
                    efficiency = 0.95
                    
                    soc_change = ess_soc[i] - ess_soc[i-1]
                    power = soc_change / (time_interval * efficiency)
                    ess_actions.append(power)
            
            ess_actions = np.array(ess_actions)
        else:
            ess_actions = plot_data['ess_actions'].values
        
        # ESS charging/discharging power
        ess_charge_power = [max(power, 0) for power in ess_actions]
        ess_discharge_power = [min(power, 0) for power in ess_actions]
        
        # PV generation
        pv_generation = plot_data['pv_generation'].values
        electricity_price = plot_data['electricity_price'].values
        
        # First subplot: ESS power and PV generation
        ax1 = plt.subplot(2, 1, 1)
        
        # Plot ESS charging/discharging bar chart
        ax1.bar(mpl_dates, ess_charge_power, width=0.015, color='#05B9E2',
                label='ESS Charging Power', alpha=0.8)
        ax1.bar(mpl_dates, ess_discharge_power, width=0.015, color='#FFBE7A', 
                label='ESS Discharging Power', alpha=0.8)
        
        # Draw horizontal line at 0 tick mark
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax1.set_ylim(-5, 5)
        ax1.set_ylabel('Power (kW)', fontsize=20)
        # ax1.set_title('ESS Charging/Discharging Power, PV Generation and Electricity Price', 
                    #   fontsize=14, fontweight='bold')
        # Set time axis format and range
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax1.set_xlim(min_date, max_date)
        # Set time axis label font size
        ax1.tick_params(axis='x', labelsize=15)
      
        
        # PV generation plot (right axis)
        ax1_pv = ax1.twinx()
        ax1_pv.plot(mpl_dates, pv_generation, color='green', marker='o', 
                    linestyle='-', label='PV Generation', markersize=3)
        ax1_pv.set_ylabel('PV Generation (kW)', fontsize=20)
        ax1_pv.set_ylim(-1, 1)
        
        # Electricity price plot (right of right axis)
        ax1_price = ax1.twinx()
        ax1_price.spines['right'].set_position(('outward', 60))
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=20)
        
        # Merge all legends and place at lower left
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_pv.get_legend_handles_labels()
        lines3, labels3 = ax1_price.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower left', bbox_to_anchor=(0, 0.05), fontsize=14)
        
        # Second subplot: ESS SOC changes
        ax2 = plt.subplot(2, 1, 2)
        ess_soc = plot_data['ess_soc'].values
        
        time_interval = 0.5 / 24
        bar_width = 0.8 * time_interval
        
        bars = ax2.bar(x=mpl_dates, height=ess_soc, width=bar_width,
                       color='#23BAC5', edgecolor='none', align='edge', label='ESS SOC')
        
        ax2.set_xlim(min_date, max_date)
        ax2.set_ylabel('ESS SOC (kWh)', color='#4EC0E9', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='#4EC0E9')
        # ax2.set_title('ESS State of Charge', fontsize=14, fontweight='bold')
        ax2.legend([bars], ['ESS SOC'], loc='upper left', fontsize=14)
        
        # Set time axis format
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        # Set time axis label font size
        ax2.tick_params(axis='x', labelsize=15)
      
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_ess_power_only(self, save_path='figures/environment_plots/ess_power_only.png', start_date=None, end_date=None, target_date=None):
        """Plot ESS power (power only)"""
        plt.figure(figsize=(20, 8))
        
        # Set subplot layout, leave more space for right axis
        plt.subplots_adjust(right=0.8)  # Compress plot area, leave 35% space for right axis
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # Check if ess_actions column exists, if not calculate from ess_soc
        if 'ess_actions' not in plot_data.columns:
            print("Warning: CSV file missing 'ess_actions' column, calculating ESS power from ess_soc difference")
            
            # Calculate ess_actions from ess_soc difference
            ess_soc = plot_data['ess_soc'].values
            ess_actions = []
            
            for i in range(len(ess_soc)):
                if i == 0:
                    # First time step, assume initial power is 0
                    ess_actions.append(0.0)
                else:
                    # Calculate SOC change rate, convert to power
                    # Assume time interval is 0.5 hours, efficiency is 0.95
                    time_interval = 0.5  # hours
                    efficiency = 0.95
                    
                    soc_change = ess_soc[i] - ess_soc[i-1]
                    power = soc_change / (time_interval * efficiency)
                    ess_actions.append(power)
            
            ess_actions = np.array(ess_actions)
        else:
            ess_actions = plot_data['ess_actions'].values
        
        # ESS charging/discharging power
        ess_charge_power = [max(power, 0) for power in ess_actions]
        ess_discharge_power = [min(power, 0) for power in ess_actions]
        
        # PV generation
        pv_generation = plot_data['pv_generation'].values
        electricity_price = plot_data['electricity_price'].values
        
        # Plot ESS charging/discharging bar chart
        ax = plt.subplot(1, 1, 1)
        ax.bar(mpl_dates, ess_charge_power, width=0.015, color='#05B9E2',
                label='ESS Charging Power', alpha=0.8)
        ax.bar(mpl_dates, ess_discharge_power, width=0.015, color='#FFBE7A', 
                label='ESS Discharging Power', alpha=0.8)
        
        # Draw horizontal line at 0 tick mark
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylim(-5, 5)
        ax.set_ylabel('Power (kW)', fontsize=30)
        
        # Add grid (only horizontal lines, parallel to left axis ticks)
        ax.grid(True, axis='y', linestyle='-', alpha=0.8, color='gray', linewidth=0.5)
        # Add more horizontal lines for ESS power plot (double the number)
        ax.set_yticks(ax.get_yticks())  # Get current ticks
        minor_ticks = []
        for i in range(len(ax.get_yticks())-1):
            minor_ticks.append((ax.get_yticks()[i] + ax.get_yticks()[i+1]) / 2)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True, axis='y', which='minor', linestyle='-', alpha=0.4, color='gray', linewidth=0.4)
        
        # PV generation plot (right axis)
        ax_pv = ax.twinx()
        ax_pv.plot(mpl_dates, pv_generation, color='green', marker='o', 
                    linestyle='-', label='PV Generation', markersize=3)
        ax_pv.set_ylabel('PV Generation (kW)', fontsize=25)
        ax_pv.set_ylim(-1, 1)
        # Set PV axis label position
        ax_pv.yaxis.set_label_position('right')
        # Set PV axis label x position to separate from price axis label
        ax_pv.yaxis.label.set_x(1.05)
        
        # Electricity price plot (right of right axis)
        ax_price = ax.twinx()
        ax_price.spines['right'].set_position(('outward', 120))  # Further increase offset distance
        ax_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax_price.set_ylabel('Price ($/kWh)', fontsize=25)
        # Adjust price axis label position to avoid overlap with PV axis label
        ax_price.yaxis.set_label_position('right')
        ax_price.yaxis.label.set_rotation(90)  # Change to 90 degrees, align with PV label direction
        ax_price.yaxis.label.set_horizontalalignment('center')
        ax_price.yaxis.label.set_verticalalignment('top')
        # Set price axis label x position, further shift right
        ax_price.yaxis.label.set_x(1.15)
        
        # Merge all legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_pv.get_legend_handles_labels()
        lines3, labels3 = ax_price.get_legend_handles_labels()
        legend = ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower left', bbox_to_anchor=(0, 0.05))
        # Set legend font size
        for text in legend.get_texts():
            text.set_fontsize(20)
        # Set legend border color darker
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(2)
        
        # Set time axis format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.tick_params(axis='x', labelsize=25)
        
        # Set x-axis label
        ax.set_xlabel('Time', fontsize=30)
        
        # Set y-axis number font size
        ax.tick_params(axis='y', labelsize=25)
        ax_pv.tick_params(axis='y', labelsize=25)
        ax_price.tick_params(axis='y', labelsize=25)
        
        # Add black borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        for spine in ax_pv.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        for spine in ax_price.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_ess_soc_only(self, save_path='figures/environment_plots/ess_soc_only.png', start_date=None, end_date=None, target_date=None):
        """Plot ESS SOC (SOC only)"""
        plt.figure(figsize=(20, 8))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        ess_soc = plot_data['ess_soc'].values
        
        time_interval = 0.5 / 24
        bar_width = 0.8 * time_interval
        
        # Plot ESS SOC bar chart
        ax = plt.subplot(1, 1, 1)
        bars = ax.bar(x=mpl_dates, height=ess_soc, width=bar_width,
                       color='#23BAC5', edgecolor='none', align='edge', label='ESS SOC')
        
        ax.set_ylabel('ESS SOC (kWh)', fontsize=30)
        
        # Add grid (only horizontal lines, parallel to left axis ticks)
        ax.grid(True, axis='y', linestyle='-', alpha=0.8, color='gray', linewidth=0.5)
        
        ax.legend([bars], ['ESS SOC'], loc='upper left', fontsize=20)
        
        # Set time axis format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.tick_params(axis='x', labelsize=25)
        
        # Set x-axis label
        ax.set_xlabel('Time', fontsize=30)
        
        # Set y-axis number font size
        ax.tick_params(axis='y', labelsize=25)
        
        # Add black borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_combined_ac_temp(self, save_path='figures/environment_plots/combined_ac_temp.png', start_date=None, end_date=None, target_date=None):
        """Plot comparison of two ACs' indoor temperatures (outdoor temperature same)"""
        plt.figure(figsize=(28, 14))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # Temperature data
        indoor_temp = plot_data['indoor_temp'].values
        indoor_temp2 = plot_data['indoor_temp2'].values if 'indoor_temp2' in plot_data.columns else None
        outdoor_temp = plot_data['outdoor_temp'].values
        
        # AC power data
        ac_power = plot_data['air_conditioner_power'].values
        ac_power2 = plot_data['air_conditioner_power2'].values if 'air_conditioner_power2' in plot_data.columns else None
        electricity_price = plot_data['electricity_price'].values
        
        # First subplot: AC power comparison (top)
        ax1 = plt.subplot(2, 1, 1)
        
        # Plot first AC power
        ax1.step(mpl_dates, ac_power, color='limegreen', label='AC1 Power', 
                 where='post', linewidth=3)
        
        # Plot second AC power (if exists)
        if ac_power2 is not None:
            ax1.step(mpl_dates, ac_power2, color='orange', label='AC2 Power', 
                     where='post', linewidth=3)
        
        ax1.set_ylabel('AC Power (kW)', fontsize=30)
        # ax1.set_title('Air Conditioner Power and Electricity Price', fontsize=14, fontweight='bold')
        
        # Electricity price curve (right axis)
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=30)
        
        # Set time axis format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax1.tick_params(axis='x', labelsize=25)
        ax1.tick_params(axis='y', labelsize=25)
        ax1_price.tick_params(axis='y', labelsize=25)
        
        # Set x-axis label
        ax1.set_xlabel('Time', fontsize=30)
        
        # Thicken borders
        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        for spine in ax1_price.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        # Second subplot: Temperature comparison (bottom)
        ax2 = plt.subplot(2, 1, 2)
        
        # Plot outdoor temperature (shared by both ACs)
        ax2.plot(mpl_dates, outdoor_temp, color='deepskyblue', label='Outdoor Temperature', linewidth=2)
        
        # Plot first AC's indoor temperature
        ax2.plot(mpl_dates, indoor_temp, color='limegreen', label='AC1 Indoor Temperature', linewidth=4)
        
        # Plot second AC's indoor temperature (if exists)
        if indoor_temp2 is not None:
            ax2.plot(mpl_dates, indoor_temp2, color='orange', label='AC2 Indoor Temperature', linewidth=4)
        
        # Add first AC's comfort temperature range (20-24°C)
        ax2.axhline(20, color='limegreen', linestyle='--', alpha=0.7, label='AC1 Comfort Zone Lower')
        ax2.axhline(24, color='limegreen', linestyle='--', alpha=0.7, label='AC1 Comfort Zone Upper')
        ax2.fill_between(mpl_dates, 20, 24, color='lightgreen', alpha=0.3, label='AC1 Comfort Zone')
        
        # Add second AC's comfort temperature range (16-20°C)
        if indoor_temp2 is not None:
            ax2.axhline(16, color='orange', linestyle='--', alpha=0.7, label='AC2 Comfort Zone Lower')
            ax2.axhline(20, color='orange', linestyle='--', alpha=0.7, label='AC2 Comfort Zone Upper')
            ax2.fill_between(mpl_dates, 16, 20, color='lightcoral', alpha=0.3, label='AC2 Comfort Zone')
        
        ax2.set_ylabel('Temperature (°C)', fontsize=30)
        # ax2.set_title('Indoor and Outdoor Temperature Comparison', fontsize=14, fontweight='bold')
        
        # Set time axis format
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax2.tick_params(axis='x', labelsize=25)
        ax2.tick_params(axis='y', labelsize=25)
        
        # Set x-axis label
        ax2.set_xlabel('Time', fontsize=30)
        
        # Thicken borders
        for spine in ax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        # Adjust layout to leave space for bottom legend
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        
        # Create legend at bottom, aligned with top plot width
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_price.get_legend_handles_labels()
        lines3, labels3 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2 + lines3
        all_labels = labels1 + labels2 + labels3
        
        legend = plt.legend(all_lines, all_labels, loc='lower center', ncol=4, fontsize=24,
                          bbox_to_anchor=(0.5, 0.02), frameon=True,
                          bbox_transform=plt.gcf().transFigure)
        
        # Set legend style to align border with top plot
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(3)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_boxstyle("round,pad=0.02")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_combined_ac_temp_only(self, save_path='figures/environment_plots/combined_ac_temp_only.png', start_date=None, end_date=None, target_date=None):
        """Plot comparison of two ACs' indoor temperatures (temperature only)"""
        plt.figure(figsize=(20, 8))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # Temperature data
        indoor_temp = plot_data['indoor_temp'].values
        indoor_temp2 = plot_data['indoor_temp2'].values if 'indoor_temp2' in plot_data.columns else None
        outdoor_temp = plot_data['outdoor_temp'].values
        
        # Plot outdoor temperature (shared by both ACs)
        plt.plot(mpl_dates, outdoor_temp, color='deepskyblue', label='Outdoor Temperature', linewidth=2)
        
        # Plot first AC's indoor temperature
        plt.plot(mpl_dates, indoor_temp, color='limegreen', label='AC1 Indoor Temperature', linewidth=4)
        
        # Plot second AC's indoor temperature (if exists)
        if indoor_temp2 is not None:
            plt.plot(mpl_dates, indoor_temp2, color='orange', label='AC2 Indoor Temperature', linewidth=4)
        
        # Add first AC's comfort temperature range (20-24°C)
        # plt.axhline(20, color='limegreen', linestyle='--', alpha=0.7, label='AC1 Comfort Zone Lower')
        # plt.axhline(24, color='limegreen', linestyle='--', alpha=0.7, label='AC1 Comfort Zone Upper')
        plt.fill_between(mpl_dates, 20, 24, color='lightgreen', alpha=0.3, label='AC1 Comfort Zone')
        
        # Add second AC's comfort temperature range (16-20°C)
        if indoor_temp2 is not None:
            # plt.axhline(16, color='orange', linestyle='--', alpha=0.7, label='AC2 Comfort Zone Lower')
            # plt.axhline(20, color='orange', linestyle='--', alpha=0.7, label='AC2 Comfort Zone Upper')
            plt.fill_between(mpl_dates, 16, 20, color='lightcoral', alpha=0.3, label='AC2 Comfort Zone')
        
        plt.ylabel('Temperature (°C)', fontsize=30)
        plt.legend(loc='upper left', fontsize=20)
        
        # Set time axis format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.gca().tick_params(axis='x', labelsize=25)
        
        # Set x-axis label
        plt.xlabel('Time', fontsize=30)
        
        # Set y-axis number font size
        plt.gca().tick_params(axis='y', labelsize=25)
        
        # Add black borders
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_ac_power_and_temp(self, save_path='figures/environment_plots/ac_power_and_temp.png', start_date=None, end_date=None, target_date=None):
        """Plot AC power and temperature"""
        # Call new merge function
        self.plot_combined_ac_temp(save_path.replace('ac_power_and_temp.png', 'combined_ac_temp.png'), target_date)
        
        # Keep original separate plotting functionality
        plt.figure(figsize=(20, 10))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # AC power and electricity price
        ac_power = plot_data['air_conditioner_power'].values
        electricity_price = plot_data['electricity_price'].values
        
        ax1 = plt.subplot(2, 1, 1)
        # Use step function to plot AC power changes
        ax1.step(mpl_dates, ac_power, color='#B2DBB9', label='AC Power', 
                 where='post', linewidth=2)
        ax1.set_ylabel('AC Power (kW)', fontsize=12)
        ax1.set_title('AC Power and Electricity Price', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=12)
        ax1_price.legend(loc='upper right', fontsize=10)
        
        # Set time axis format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax1.tick_params(axis='x', labelsize=20)
        
        # Temperature change plot
        ax2 = plt.subplot(2, 1, 2)
        indoor_temp = plot_data['indoor_temp'].values
        outdoor_temp = plot_data['outdoor_temp'].values
        
        ax2.plot(mpl_dates, indoor_temp, color='limegreen', label='Indoor Temperature', linewidth=2)
        ax2.plot(mpl_dates, outdoor_temp, color='deepskyblue', label='Outdoor Temperature', linewidth=2)
        
        # Add comfort temperature range lines and fill area
        ax2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax2.axhline(24, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax2.fill_between(mpl_dates, 20, 24, color='lightblue', alpha=0.3)
        
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_title('Indoor and Outdoor Temperature Trends', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        
        # Set time axis format
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax2.tick_params(axis='x', labelsize=20)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_ac_power_and_temp2(self, save_path='figures/environment_plots/ac_power_and_temp2.png', start_date=None, end_date=None, target_date=None):
        """Plot second AC power and temperature"""
        plt.figure(figsize=(20, 10))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        # Check if second AC data exists
        if 'air_conditioner_power2' not in plot_data.columns:
            print("Warning: CSV file missing 'air_conditioner_power2' column, skipping second AC plot")
            plt.close()
            return
        
        # Second AC power and electricity price
        ac_power2 = plot_data['air_conditioner_power2'].values
        electricity_price = plot_data['electricity_price'].values
        
        ax1 = plt.subplot(2, 1, 1)
        # Use step function to plot AC power changes
        ax1.step(mpl_dates, ac_power2, color='#B2DBB9', label='AC Power', 
                 where='post', linewidth=2)
        ax1.set_ylabel('AC Power (kW)', fontsize=15)
        ax1.set_title('AC Power and Electricity Price', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=13)
        
        ax1_price = ax1.twinx()
        ax1_price.step(mpl_dates, electricity_price, color='red', 
                       label='Electricity Price', where='post', linewidth=2)
        ax1_price.set_ylabel('Price ($/kWh)', fontsize=15)
        ax1_price.legend(loc='upper right', fontsize=13)
        
        # Set time axis format
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax1.tick_params(axis='x', labelsize=15)
        
        # Temperature change plot
        ax2 = plt.subplot(2, 1, 2)
        indoor_temp2 = plot_data['indoor_temp2'].values
        outdoor_temp = plot_data['outdoor_temp'].values
        
        ax2.plot(mpl_dates, indoor_temp2, color='limegreen', label='Indoor Temperature', linewidth=2)
        ax2.plot(mpl_dates, outdoor_temp, color='deepskyblue', label='Outdoor Temperature', linewidth=2)
        
        # Add comfort temperature range lines and fill area (second AC comfort zone is 16-20°C)
        ax2.axhline(16, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Lower')
        ax2.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Comfort Zone Upper')
        ax2.fill_between(mpl_dates, 16, 20, color='lightblue', alpha=0.3)
        
        ax2.set_ylabel('Temperature (°C)', fontsize=15)
        ax2.set_title('Indoor and Outdoor Temperature Trends', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=13)
        
        # Set time axis format
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax2.tick_params(axis='x', labelsize=15)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_wash_machine_state(self, save_path='figures/environment_plots/wash_machine_state.png', start_date=None, end_date=None, target_date=None):
        """Plot washing machine state"""
        plt.figure(figsize=(20, 8))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        wash_machine_state = plot_data['wash_machine_state'].values
        electricity_price = plot_data['electricity_price'].values
        
        ax = plt.subplot(1, 1, 1)
        time_interval = 0.5 / 24
        bar_width = time_interval
        
        # Use align='edge' for left alignment, consistent with environment.py
        ax.bar(mpl_dates, wash_machine_state, width=bar_width, color='#F0A19A',
               label='Washing Machine State', align='edge', alpha=0.8)
        ax.set_ylabel('Washing Machine State', fontsize=30)
        # ax.set_title('Washing Machine State and Electricity Price', fontsize=14, fontweight='bold')
        
        # Use step function to plot step-like electricity price curve
        ax_price = ax.twinx()
        ax_price.step(mpl_dates, electricity_price, color='red', 
                     label='Electricity Price', where='post', linewidth=2)
        ax_price.set_ylabel('Price ($/kWh)', fontsize=30)
    
        # Merge legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_price.get_legend_handles_labels()
        legend = ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        # Set legend font size
        for text in legend.get_texts():
            text.set_fontsize(20)
        # Set legend border color darker
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(2)
        
        # Set time axis format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.tick_params(axis='x', labelsize=25)
        
        # Set x-axis label
        ax.set_xlabel('Time', fontsize=30)
        
        # Set y-axis number font size
        ax.tick_params(axis='y', labelsize=25)
        ax_price.tick_params(axis='y', labelsize=25)
        
        # Add black borders
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        for spine in ax_price.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        
        # Add washing machine preferred time period shadow (assuming preferred time is 8-12)
        # Note: This should be set according to actual preferred time period
        wash_machine_preferred_time = [6, 20]  # Assuming preferred time is 8-12
        
        # Get recorded date range
        start_date = timestamps.iloc[0]
        end_date = timestamps.iloc[-1]
        
        # Iterate through each day
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date_rounded = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        while current_date <= end_date_rounded:
            # Calculate preferred time period for the day
            preferred_start = current_date.replace(hour=wash_machine_preferred_time[0], minute=0)
            preferred_end = current_date.replace(hour=wash_machine_preferred_time[1], minute=0)
            
            # Add vertical shadow area and dashed lines
            ax.axvspan(preferred_start, preferred_end, facecolor='lightblue', alpha=0.3)  # Changed to lightblue color
            ax.axvline(preferred_start, color='lightblue', linestyle='--', linewidth=1)  # Dashed line
            ax.axvline(preferred_end, color='lightblue', linestyle='--', linewidth=1)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Set time axis range to recorded timestamp range
        ax.set_xlim(mpl_dates[0], mpl_dates[-1])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_water_heater_status(self, save_path='figures/environment_plots/water_heater_status.png', start_date=None, end_date=None, target_date=None):
        """Plot water heater status"""
        plt.figure(figsize=(28, 14))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        ewh_power = plot_data['ewh_power'].values
        ewh_temp = plot_data['ewh_temp'].values
        user_flow = plot_data['user_flow'].values
        
        # Create 2 rows 1 column grid layout
        from matplotlib import gridspec
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        
        # Dual-axis display power and water flow (top subplot)
        ax1 = plt.subplot(gs[0])
        ax1.bar(mpl_dates, ewh_power, width=0.015, color='#1F77B4', label='Power', alpha=0.8)
        ax1.set_ylabel('Power (kW)', color='#1F77B4', fontsize=30)
        ax1.tick_params(axis='y', labelcolor='#1F77B4', labelsize=25)
        
        ax1_flow = ax1.twinx()
        ax1_flow.plot(mpl_dates, user_flow, color='#2CA02C', marker='o', 
                      label='Water Flow', markersize=3)
        ax1_flow.set_ylabel('Flow Rate (L/h)', color='#2CA02C', fontsize=30)
        ax1_flow.tick_params(axis='y', labelcolor='#2CA02C', labelsize=25)
        
        # Set x-axis label
        ax1.set_xlabel('Time', fontsize=30)
        
        # Thicken borders
        for spine in ax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        for spine in ax1_flow.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        # Water temperature curve (bottom subplot)
        ax2 = plt.subplot(gs[1])
        ax2.plot(mpl_dates, ewh_temp, color='#FF7F0E', label='Water Temperature', linewidth=3)
        # ax2.axhline(40, color='grey', linestyle='--', label='Target Temp')
        ax2.set_ylabel('Temperature (°C)', fontsize=30)
        # ax2.set_title('Water Heater Status', fontsize=14, fontweight='bold')
        
        # Add comfort temperature range fill area
        peak_low = 50
        peak_high = 60
        non_peak_low = 40
        non_peak_high = 50
        
        # Create unified temperature range for filling
        all_low = [peak_low if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_low 
                   for ts in timestamps]
        all_high = [peak_high if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_high 
                    for ts in timestamps]
        
        # Plot unified fill area
        ax2.fill_between(mpl_dates, all_low, all_high, color='lightgreen', alpha=0.3, 
                        label='Comfort Zone')
        
        # Plot special marker for peak hours
        peak_mask = [(6 <= ts.hour <= 9) or (18 <= ts.hour <= 22) for ts in timestamps]
        ax2.fill_between(mpl_dates, peak_low, peak_high, where=peak_mask, 
                        color='lightcoral', alpha=0.3, label='Peak Comfort Zone')
        
        # Set x-axis label
        ax2.set_xlabel('Time', fontsize=30)
        
        # Unify time axis format
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)
        
        # Thicken borders
        for spine in ax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
        
        # Adjust layout to leave space for bottom legend
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        # Create legend at bottom, aligned with top plot width
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_flow.get_legend_handles_labels()
        lines3, labels3 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2 + lines3
        all_labels = labels1 + labels2 + labels3
        
        legend = plt.legend(all_lines, all_labels, loc='lower center', ncol=len(all_labels), fontsize=24,
                          bbox_to_anchor=(0.5, 0.03 ), frameon=True,
                          bbox_transform=plt.gcf().transFigure)
        
        # Set legend style to align border with top plot
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(3)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_boxstyle("round,pad=0.02")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_water_heater_temp_only(self, save_path='figures/environment_plots/water_heater_temp_only.png', start_date=None, end_date=None, target_date=None):
        """Plot water heater temperature (temperature only)"""
        plt.figure(figsize=(20, 8))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        ewh_temp = plot_data['ewh_temp'].values
        
        # Plot water temperature curve
        plt.plot(mpl_dates, ewh_temp, color='#FF7F0E', label='Water Temperature', linewidth=4)
        plt.ylabel('Temperature (°C)', fontsize=30)
        
        # Add comfort temperature range fill area
        peak_low = 50
        peak_high = 60
        non_peak_low = 40
        non_peak_high = 50
        
        # Create unified temperature range for filling
        all_low = [peak_low if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_low 
                   for ts in timestamps]
        all_high = [peak_high if (6 <= ts.hour <= 9 or 18 <= ts.hour <= 22) else non_peak_high 
                    for ts in timestamps]
        
        # Plot comfort zone for non-peak hours
        non_peak_mask = [not ((6 <= ts.hour <= 9) or (18 <= ts.hour <= 22)) for ts in timestamps]
        comfort_zone = plt.fill_between(mpl_dates, non_peak_low, non_peak_high, where=non_peak_mask, 
                        color='lightgreen', alpha=0.3, label='Comfort Zone')
        
        # Plot special marker for peak hours
        peak_mask = [(6 <= ts.hour <= 9) or (18 <= ts.hour <= 22) for ts in timestamps]
        peak_zone = plt.fill_between(mpl_dates, peak_low, peak_high, where=peak_mask, 
                        color='#FF6B6B', alpha=0.3, label='Peak Comfort Zone')
        
        # Set time axis format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.gca().tick_params(axis='x', labelsize=25)
        
        # Set x-axis label
        plt.xlabel('Time', fontsize=30)
        
        # Set y-axis number font size
        plt.gca().tick_params(axis='y', labelsize=25)
        
        # Add black borders
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)  # Increase border line width
        
        # Set legend, ensure color matches fill area
        legend = plt.legend(loc='upper left', fontsize=16)
        # Manually set fill area colors in legend
        legend_elements = legend.get_patches()
        if len(legend_elements) >= 2:
            legend_elements[0].set_facecolor('lightgreen')  # Comfort Zone color
            legend_elements[0].set_alpha(0.3)
            legend_elements[1].set_facecolor('#FF6B6B')  # Peak Comfort Zone color (red)
            legend_elements[1].set_alpha(0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_total_load(self, save_path='figures/environment_plots/total_load.png', start_date=None, end_date=None, target_date=None):
        """Plot total household load"""
        plt.figure(figsize=(20, 5))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        total_load = plot_data['total_load'].values
        
        ax = plt.subplot(1, 1, 1)
        ax.plot(mpl_dates, total_load, color='purple', label='Total Load', linewidth=2)
        ax.set_ylabel('Load (kW)', fontsize=12)
        ax.set_title('Household Total Load Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # Set time axis format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        # Set grid
        ax.grid(alpha=0.3)
        
        # Fill red area (upper part)
        ax.fill_between(mpl_dates, total_load, 0, where=(total_load > 0),
                       color='red', alpha=0.3)
        # Fill green area (lower part)
        ax.fill_between(mpl_dates, total_load, 0, where=(total_load < 0),
                       color='green', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_energy_cost(self, save_path='figures/environment_plots/energy_cost.png', start_date=None, end_date=None, target_date=None):
        """Plot energy cost"""
        plt.figure(figsize=(20, 5))
        
        # Filter data based on whether date is specified
        if target_date is not None:
            # Backward compatibility: use original single-day filter
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
        elif start_date is not None or end_date is not None:
            # Use new date range filter
            plot_data = self.filter_data_by_date_range(start_date, end_date)
            if plot_data is None:
                return
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        energy_cost = plot_data['energy_cost'].values
        
        ax = plt.subplot(1, 1, 1)
        ax.plot(mpl_dates, energy_cost, color='purple', label='Energy Cost', linewidth=2)
        ax.set_ylabel('Cost ($)', fontsize=12)
        ax.set_title('Energy Cost Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        # Set time axis format
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_reward_components(self, save_path='figures/environment_plots/reward_components.png', start_date=None, end_date=None, target_date=None):
        """Plot reward components"""
        # Try to load reward data
        reward_file = self.csv_file_path.replace('.csv', '_rewards.csv')
        if not os.path.exists(reward_file):
            print(f"Reward data file does not exist: {reward_file}")
            return
        
        reward_data = pd.read_csv(reward_file)
        
        # Filter data based on whether date is specified
        if target_date is not None:
            plot_data = self.filter_data_by_date(target_date)
            if plot_data is None:
                return
            # Filter reward data - reward data has no timestamp, so need to filter by index
            # Get index range for target date in main data
            target_date_dt = pd.to_datetime(target_date).date()
            main_data_filtered = self.data[self.data['timestamp'].dt.date == target_date_dt]
            start_idx = main_data_filtered.index[0]
            end_idx = main_data_filtered.index[-1]
            # Filter reward data by index
            reward_data = reward_data.iloc[start_idx:end_idx+1].copy()
        else:
            plot_data = self.data
        
        timestamps = plot_data['timestamp']
        mpl_dates = mdates.date2num(timestamps)
        
        plt.figure(figsize=(20, 8))
        
        # Create color mapping
        colors = {
            'energy_cost': '#FF6B6B',
            'violation_penalty': '#4ECDC4',
            'ess_reward': '#D95319',
            'ev_reward': '#96CEB4',
            'user_penalty': '#FFEEAD',
            'temp_reward': '#D4A5A5'
        }
        
        # Plot stacked area chart
        components = ['energy_cost', 'violation_penalty', 'ess_reward',
                     'ev_reward', 'user_penalty', 'temp_reward']
        
        # Cumulative values for stacking
        cumulative = np.zeros(len(mpl_dates))
        
        for comp in components:
            if comp in reward_data.columns:
                values = np.array(reward_data[comp])
                plt.fill_between(mpl_dates, cumulative, cumulative + values,
                               label=comp.replace('_', ' ').title(),
                               color=colors[comp], alpha=0.8)
                cumulative += values
        
        # Plot total reward line
        if 'total' in reward_data.columns:
            plt.plot(mpl_dates, reward_data['total'],
                    color='#2C3E50', linewidth=2, label='Total Reward')
        
        # Format settings
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.xticks(rotation=45)
        
        plt.ylabel('Reward Value', fontsize=12)
        plt.title('Reward Component Breakdown', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def _find_contiguous_segments(self, bool_list):
        """Find contiguous True segments"""
        segments = []
        start = None
        
        for i, value in enumerate(bool_list):
            if value and start is None:
                start = i
            elif not value and start is not None:
                segments.append((start, i - 1))
                start = None
        
        if start is not None:
            segments.append((start, len(bool_list) - 1))
        
        return segments
    
    def generate_all_plots(self, output_dir='save', start_date=None, end_date=None, target_date=None):
        """Generate all plots
        
        Args:
            output_dir (str): Output directory
            start_date (str): Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            end_date (str): End date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
            target_date (str): Target date (maintain backward compatibility)
        """
        print("Starting to generate all plots...")
        if target_date:
            print(f"Target date: {target_date}")
        elif start_date or end_date:
            print(f"Date range: {start_date or 'start'} to {end_date or 'end'}")
        else:
            print("Generating plots for all data")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib parameters
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams['savefig.dpi'] = 600
        # plt.rcParams['font.size'] = 10  # Commented out global font setting to avoid affecting legend font size
        
        # Generate all plots
        self.plot_ev_soc_and_price(f'{output_dir}/ev_soc_and_price.png', start_date, end_date, target_date)
        self.plot_ess_power_and_pv(f'{output_dir}/ess_power_and_pv.png', start_date, end_date, target_date)
        # Generate separate ESS power plot (power only)
        self.plot_ess_power_only(f'{output_dir}/ess_power_only.png', start_date, end_date, target_date)
        # Generate separate ESS SOC plot (SOC only)
        self.plot_ess_soc_only(f'{output_dir}/ess_soc_only.png', start_date, end_date, target_date)
        self.plot_ac_power_and_temp(f'{output_dir}/ac_power_and_temp.png', start_date, end_date, target_date)
        self.plot_ac_power_and_temp2(f'{output_dir}/ac_power_and_temp2.png', start_date, end_date, target_date)
        # Generate combined AC temperature plot
        self.plot_combined_ac_temp(f'{output_dir}/combined_ac_temp.png', start_date, end_date, target_date)
        # Generate separate AC temperature plot (temperature only)
        self.plot_combined_ac_temp_only(f'{output_dir}/combined_ac_temp_only.png', start_date, end_date, target_date)
        self.plot_wash_machine_state(f'{output_dir}/wash_machine_state.png', start_date, end_date, target_date)
        self.plot_water_heater_status(f'{output_dir}/water_heater_status.png', start_date, end_date, target_date)
        # Generate separate water heater temperature plot (temperature only)
        self.plot_water_heater_temp_only(f'{output_dir}/water_heater_temp_only.png', start_date, end_date, target_date)
        self.plot_total_load(f'{output_dir}/total_load.png', start_date, end_date, target_date)
        self.plot_energy_cost(f'{output_dir}/energy_cost.png', start_date, end_date, target_date)
        self.plot_reward_components(f'{output_dir}/reward_components.png', start_date, end_date, target_date)
        
        print("All plots generated!")

def main():
    """Main function"""
    # Specify file path directly in code
    csv_file_path = "simulation_data/final_simulation_data_episode_5000.csv"
    output_dir = "figures/algorithm_comparison"
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"File does not exist: {csv_file_path}")
        print("Please check if the file path is correct")
        return
    
    # Check if it's a rewards file
    if '_rewards.csv' in csv_file_path:
        print("Warning: Detected rewards file, please specify main data file!")
        print("Main data file usually does not contain '_rewards' suffix")
        print("Example: simulation_data/simulation_data_episode_5000.csv")
        return
    
    try:
        # Create plotter
        plotter = CSVPlotter(csv_file_path)
        
        # Example 1: Generate single-day plots (maintain backward compatibility)
        # target_date = '2011-07-03'
        # plotter.generate_all_plots(output_dir + "_single_day", target_date=target_date)
        # print(f"Single-day plots saved, showing date: {target_date}")
        
        # Example 2: Generate plots for specified date range
        start_date = '2011-07-07'
        end_date = '2011-07-08'
        plotter.generate_all_plots("figures/plots", start_date=start_date, end_date=end_date)
        print(f"Date range plots saved, range: {start_date} to {end_date}")
        
        # Example 3: Generate plots for specified time period (precise to hour)
        # start_datetime = '2011-07-03 06:00:00'
        # end_datetime = '2011-07-03 18:00:00'
        # plotter.generate_all_plots(output_dir + "_time_range", start_date=start_datetime, end_date=end_datetime)
        # print(f"Time period plots saved, range: {start_datetime} to {end_datetime}")
        
        # Example 4: Only specify start time, until data end
        # start_date = '2011-07-05'
        # plotter.generate_all_plots(output_dir + "_from_start", start_date=start_date)
        # print(f"Plots from specified start date saved, start date: {start_date}")
        
        # Example 5: Only specify end time, from data start
        # end_date = '2011-07-05'
        # plotter.generate_all_plots(output_dir + "_until_end", end_date=end_date)
        # print(f"Plots until specified end date saved, end date: {end_date}")
        
        # Default: Generate plots for all 7 days
        # print("\nGenerating plots for all 7 days...")
        # plotter.generate_all_plots(output_dir + "_all_days")
        
        print("\n📊 Plotting function guide:")
        print("1. Single-day plotting: plotter.generate_all_plots(output_dir, target_date='2011-07-03')")
        print("2. Date range plotting: plotter.generate_all_plots(output_dir, start_date='2011-07-03', end_date='2011-07-05')")
        print("3. Precise time plotting: plotter.generate_all_plots(output_dir, start_date='2011-07-03 06:00:00', end_date='2011-07-03 18:00:00')")
        print("4. From specific date: plotter.generate_all_plots(output_dir, start_date='2011-07-03')")
        print("5. Until specific date: plotter.generate_all_plots(output_dir, end_date='2011-07-05')")
        print("6. All data: plotter.generate_all_plots(output_dir)")
        print("\n🔧 Individual plotting functions:")
        print("7. AC temperature plot: plotter.plot_combined_ac_temp_only('path/to/save.png', start_date='2011-07-03', end_date='2011-07-05')")
        print("8. Water heater temperature plot: plotter.plot_water_heater_temp_only('path/to/save.png', start_date='2011-07-03', end_date='2011-07-05')")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 