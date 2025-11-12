import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from efficiency import get_efficiency_vectorized

#from efficiency import get_efficiency

# --- 1. USER CONFIGURATION ---
# Set your file and column names here

# The exact (case-sensitive) name of your CSV 
csv_filename = 'Data-Viewer/Log__2024_10_13__17_13_04.csv' # UTA fast run

csv_filename = 'Data-Viewer/Log__2025_11_03__10_37_30.csv' # inverter Failure

# (Optional) Define new "math channels" to create from existing columns
# Use a dictionary where:
#   key = 'new_channel_name'
#   value = 'math expression as a string' (must be valid pandas .eval() syntax)
# You can use existing column names in your expressions.
math_channels = {
    'mech_power_kW': '`Actual Torque` * RPM / 9550',
    'Brake Pressure 1': '(`BSE 1 Voltage` - .512) * 3737.5',
    'Throttle Position (%)': '(`APPS Telemetry`) * 100',
    'electrical_power_kW': '(`Voltage Input into DC` * `Current Input into DC`) / 1000',
    'Torque Command / 1e16': '`Torque Command` / 1e16',
    'RPM / 10' : 'RPM / 100'
    #'Efficiency (%)': 'mech_power_kW / electrical_power_kW * 100'
    
}

# (Optional) Define channels to filter (e.g., with a rolling average)
# Use a dictionary where:
#   key = 'channel_name_to_filter' (can be an original column or a math channel)
#   value = window_size (integer, e.g., 10 for a 10-point rolling average)
filter_channels = {
    'Brake Pressure 1': 20,
    #'Power Delta kW': 10,
    'electrical_power_kW': 50
}


# The exact (case-sensitive) column header for the X-axis
x_variable = 'Time'

# A list of one or more exact (case-sensitive) column headers for the Y-axis
# You can include your new 'math_channel' names here.

# Driver Analysis
#y_variables = ['Throttle Position (%)','electrical_power_kW'] #'Brake Pressure 1_filtered',

# Power Limit Analysis
#y_variables = ['mech_est_power_kW','electrical_power_kW_filtered','Torque Limit Nm','Power with Limit kW','Torque Command','Actual Torque','kW Overshoot', 'efficiency']#,'Efficiency (%)']

# Broken Inverter
#y_variables = ['electrical_power_kW_filtered','RPM','Torque Command / 1e16','Inverter Torque Request','Throttle Position (%)','Voltage Input into DC']#,'Efficiency (%)']'Torque Command',

# Current Analysis
y_variables = ['RPM / 10','Current Input into DC','Torque Command / 1e16','Inverter Torque Request', 'Phase A Current','Phase B Current','Phase C Current','AB Voltage','BC Voltage']

# LV?
y_variables = ['RPM / 10','Motor Angle','Resolver Angle']
#Temp
y_variables = ['Inverter Temp','RPM / 10','Current Input into DC']

# (Optional) Set a start and end value for the X-axis to filter data
# Set to None to disable filtering
x_start_value = None#115 # e.g., 1000
x_end_value = None#117   # e.g., 5000


# --- 2. DATA PROCESSING AND PLOTTING ---

try:
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filename)

    # --- Create Math Channels ---
    if math_channels:
        print("Creating math channels...")
        for new_col, expression in math_channels.items():
            try:
                df[new_col] = df.eval(expression)
                print(f"  Successfully created channel: '{new_col}'")
            except Exception as e:
                print(f"  Error creating channel '{new_col}' with expression '{expression}':")
                print(f"  {e}")
                print("  Skipping this channel...")
        print("-" * 20)

    # --- Create Filtered Channels ---
    if filter_channels:
        print("Creating filtered channels...")
        for col_to_filter, window_size in filter_channels.items():
            if col_to_filter in df.columns:
                new_col_name = f"{col_to_filter}_filtered"
                # Apply a rolling average.
                # center=True ensures the filtered plot isn't phase-shifted (lagging)
                # min_periods=1 handles the edges of the dataset gracefully
                df[new_col_name] = df[col_to_filter].rolling(window=window_size, center=True, min_periods=1).mean()
                print(f"  Successfully created channel: '{new_col_name}' with window {window_size}")
            else:
                print(f"  Warning: Column '{col_to_filter}' not found for filtering. Skipping.")
        print("-" * 20)

    # Manually add complicated channels
    
    powerlimit = 50  # kW
    safety_margin_kW = 5  # kW

    df['efficiency'] = np.maximum(get_efficiency_vectorized(df['RPM'], df['Inverter Torque Request']),50)

    df['mech_est_power_kW'] = 100 * df['mech_power_kW'] / df['efficiency'] + safety_margin_kW
    df['Torque Limit Nm'] = np.minimum(9550 * powerlimit / df['RPM'],220)
    df['Power with Limit kW'] = df['RPM'] * df['Torque Limit Nm'] / 9550

    df['kW Overshoot'] = np.minimum(df['mech_est_power_kW'] - df['electrical_power_kW_filtered'],0)

    # Check if all specified plot columns exist (including new math channels)
    all_vars = [x_variable] + y_variables
    missing_cols = [col for col in all_vars if col not in df.columns]

    if missing_cols:
        print(f"Error: The following columns were not found in {csv_filename} or as math channels:")
        for col in missing_cols:
            print(f"- {col}")
        print("\nPlease check your variable names. Available columns are:")
        print(list(df.columns))
    else:
        # Apply x-axis filtering if values are provided
        original_row_count = len(df)
        if x_start_value is not None:
            df = df[df[x_variable] >= x_start_value]
        if x_end_value is not None:
            df = df[df[x_variable] <= x_end_value]
        
        filtered_row_count = len(df)
        if original_row_count != filtered_row_count:
            print(f"Filtered data on '{x_variable}' between {x_start_value} and {x_end_value}.")
            print(f"Original rows: {original_row_count}, Filtered rows: {filtered_row_count}")

        if df.empty:
            print("Warning: No data left after filtering. Plot will be empty.")
        
        # Create a new plot
        plt.figure(figsize=(12, 7))
        
        # Loop through each y_variable and plot it against the x_variable
        for y_var in y_variables:
            plt.plot(df[x_variable], df[y_var], label=y_var)

        # --- 3. CUSTOMIZE PLOT ---
        
        # Create a dynamic title based on the variables
        y_vars_str = ', '.join(y_variables)
        plt.title(f'{y_vars_str} vs. {x_variable}', fontsize=16)
        
        # Add labels
        plt.xlabel(x_variable, fontsize=12)
        plt.ylabel(y_vars_str, fontsize=12)
        
        # Add a legend to identify the different lines
        plt.legend(fontsize=10)
        
        # Add a grid for easier reading
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Improve the layout
        plt.tight_layout()
        
        # Display the plot
        print(f"Successfully plotted {y_vars_str} vs. {x_variable}")
        plt.show()

except FileNotFoundError:
    print(f"Error: File not found.")
    print(f"Please make sure '{csv_filename}' is in the same directory as this script.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

