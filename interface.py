"""
This file contains interface functions for data processing
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
class DataInterface:
    def __init__(self, cons_file, prod_file):
        # Define file paths
        self.cons_file = cons_file
        self.prod_file = prod_file

        # Read data
        self.cons_data = pd.read_csv(self.cons_file, parse_dates=['date'], index_col='date')
        self.prod_data = pd.read_csv(self.prod_file, parse_dates=['date'], index_col='date')

        # Store daily EV arrival and departure times
        self.ev_schedule = {}
        # Set random seed
        self.np_random = np.random.RandomState(0)
    def get_home_load(self, current_date, current_time_index):
        # Get current household electricity consumption
        return self.cons_data.loc[current_date, self.cons_data.columns[current_time_index]]

    def get_pv_generation(self, current_date, current_time_index):
        # Get current PV system generation
        return self.prod_data.loc[current_date, self.cons_data.columns[current_time_index]]

    def get_electricity_price(self, current_date, current_time_index):
        # Convert current date string to datetime object
        if current_time_index >= 48:
            current_time_index = current_time_index-48

        # Define different time periods in a day and their corresponding electricity prices
        time_based_prices = {
            'valley': 0.2,  # Assume valley price is 0.4 yuan/kWh
            'flat': 0.5,  # Assume flat price is 0.6 yuan/kWh
            'peak': 0.8,  # Assume peak price is 0.9 yuan/kWh
        }

        # Define time periods in a day (in hours)
        # Assume valley period is 00:00-06:00, flat period is 06:00-18:00, peak period is 18:00-24:00
        if 0 <= current_time_index < 20 or 44 <= current_time_index < 48:
            price = time_based_prices['valley']
        elif 8 <= current_time_index < 34:
            price = time_based_prices['flat']
        else:
            price = time_based_prices['peak']

        # Return electricity price
        return price

    def get_date_time(self, current_date, current_time_index):
        return current_date


    def is_ev_at_home(self, current_date, current_time_index):
        """
        Determine if the electric vehicle is at home at the current time.
        Generate arrival and departure times only once per day, regenerate the next day.
        """
        # Check if arrival and departure times have been generated for current date
        if current_date not in self.ev_schedule:
            self.generate_daily_ev_schedule(current_date)

        # If Saturday or Sunday, directly return True indicating EV is at home all day
        current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
        if current_weekday >= 5:  # 5 represents Saturday, 6 represents Sunday
            return True

        t1, t2 = self.ev_schedule[current_date]
        current_hour = current_time_index
        return not t2 <= current_hour < t1

    def generate_daily_ev_schedule(self, current_date):
        """
        Generate EV arrival and departure times for the current date.
        """
        # If Saturday or Sunday, directly set EV to be at home all day
        current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
        if current_weekday >= 5:  # 5 represents Saturday, 6 represents Sunday
            self.ev_schedule[current_date] = (0, 24 * 2)  # At home all day
            return

        # Adjust distribution parameters for arrival and departure times
        t1_mean = 20  # Increase mean arrival time (later return home)
        t1_std = 1
        t1_range = (18, 22)  # Expand arrival time range

        t2_mean = 8  # Decrease mean departure time (earlier departure)
        t2_std = 1
        t2_range = (7, 9)  # Narrow departure time range to ensure t2 < t1

        while True:
            t1 = self.np_random.normal(t1_mean, t1_std)
            t2 = self.np_random.normal(t2_mean, t2_std)
            t1 = max(min(t1, t1_range[1]), t1_range[0])
            t2 = max(min(t2, t2_range[1]), t2_range[0])
            if t2 < t1:
                break  # Ensure departure time is earlier than arrival time

        # Store generated times in units of 0.5 (representing half-hour)
        self.ev_schedule[current_date] = (t1 * 2, t2 * 2)

    def is_ev_departing_soon(self,current_date, current_time_idx):
        """Determine if EV is about to depart (within next 2 hours)"""
        return self.get_hours_until_departure(current_date, current_time_idx) <= 2

    def get_hours_until_departure(self, current_date, current_time_idx):
        """Get remaining time until next departure (in hours)"""
        # current_date = datetime.strptime(current_date, '%Y-%m-%d')
        if current_date not in self.ev_schedule:
            self.generate_daily_ev_schedule(current_date)

        departure_time = self.ev_schedule[current_date][1] / 2  # Convert to hours
        current_hour = current_time_idx / 2
        return departure_time - current_hour if current_hour < departure_time else (
                24 + departure_time - current_hour)

    def get_outdoor_temp(self, current_time, current_time_index):
        """
        Outdoor temperature simulation
        """
        current_datetime = datetime.strptime(current_time, '%Y-%m-%d')
        day_of_year = current_datetime.timetuple().tm_yday

        # ===== Improved Parameter Settings =====
        # Annual base temperature model
        base_temp = 24 + 10 * np.sin(2 * np.pi * (day_of_year - 200) / 365)  # July peak adjustment

        # Daily temperature variation model (increased day-night temperature difference)
        hour = current_time_index // 2
        minute = 30 * (current_time_index % 2)
        time_of_day = hour + minute / 60
        daily_temp_variation = 8 * np.sin(2 * np.pi * (time_of_day - 14.5) / 24)

        # Summer high temperature
        summer_boost = 0
        if 172 <= day_of_year <= 265:  # Period from 6/21 to 9/22
            summer_boost = 4 * np.sin(np.pi * (day_of_year - 172) / 93)  # Additional summer temperature increase

        # Weather uncertainty
        random_noise = np.random.normal(0, 2.5)

        # ===== Final Temperature Calculation =====
        outdoor_temp = base_temp + daily_temp_variation + summer_boost + random_noise

        # Temperature boundary protection
        outdoor_temp = np.clip(outdoor_temp, -5, 45)


        return round(outdoor_temp, 1)