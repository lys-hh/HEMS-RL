# PPO Training Data Saving Functionality

## Overview

When running PPO training, the system automatically saves detailed simulation data to CSV files for subsequent analysis and plotting.

## Data Saving Timing

### During Training
- Only saves detailed data from the last 50 episodes
- File name format: `simulation_data_episode_XXX.csv`

### After Training
- Saves final complete data
- File name format: `final_simulation_data_episode_XXX.csv`
- Automatically generates visualization plots

## Data File Location

All CSV files are saved in the `simulation_data/` folder:
```
simulation_data/
├── simulation_data_episode_951.csv
├── simulation_data_episode_952.csv
├── ...
├── simulation_data_episode_1000.csv
├── final_simulation_data_episode_1000.csv
└── final_simulation_data_episode_1000_rewards.csv
```

## Usage

### 1. Run PPO Training
```bash
python model/PPO_3rd.py
```

During training:
- Only saves data from the last 50 episodes
- Saves final data when training ends
- Automatically generates visualization plots

### 2. Generate Plots from CSV Data

#### Auto-select Latest File:
```bash
python plot_from_csv.py
```

#### Specify a Particular File:
```bash
python plot_from_csv.py simulation_data/final_simulation_data_episode_1000.csv
```

#### Specify Output Directory:
```bash
python plot_from_csv.py simulation_data/final_simulation_data_episode_1000.csv save_new
```

## Data Content

### Main Data File Contains:
- `timestamp` - Timestamp
- `ev_soc` - EV battery state
- `ess_soc` - ESS battery state
- `home_load` - Home load
- `pv_generation` - PV generation
- `electricity_price` - Electricity price
- `air_conditioner_power` - Air conditioner power
- `wash_machine_state` - Washing machine state
- `ewh_temp` - Water heater temperature
- `ewh_power` - Water heater power
- `total_load` - Total load
- `energy_cost` - Energy cost
- etc...

### Reward Data File Contains:
- Detailed breakdown of various reward components
- Total reward changes
- Constraint violation information

## Generated Plots

Running `plot_from_csv.py` generates the following plots:

1. **EV SOC and Price Plot** (`ev_soc_and_price.png`)
2. **ESS Power and PV Generation Plot** (`ess_power_and_pv.png`)
3. **AC Power and Temperature Plot** (`ac_power_and_temp.png`)
4. **Washing Machine State Plot** (`wash_machine_state.png`)
5. **Water Heater Status Plot** (`water_heater_status.png`)
6. **Total Home Load Plot** (`total_load.png`)
7. **Energy Cost Plot** (`energy_cost.png`)
8. **Reward Components Plot** (`reward_components.png`)

## Advantages

1. **No Re-training Required**: Adjust plot styles without retraining the model
2. **Complete Data**: All data needed for plotting is saved
3. **Flexible Adjustments**: Can modify plot styles at any time
4. **High Quality Output**: All plots are high resolution (300 DPI)
5. **Automated**: Data is automatically saved during training

## Notes

1. Ensure the `simulation_data/` folder exists
2. CSV files are large, watch disk space
3. Plot generation takes time, please be patient
4. If environment state space is modified, the model needs to be retrained

## Troubleshooting

### Common Issues
1. **CSV file not found**: Check the `simulation_data/` folder
2. **Plot generation failed**: Check if data column names match
3. **Incomplete data**: Ensure training completed normally

### Debugging Methods
1. Check console output error messages
2. Verify CSV file data integrity
3. Confirm all required columns exist
