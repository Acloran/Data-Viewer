import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Union

# 1. Define the updated high-resolution grid points and efficiency data
# Axis points for motor speed (rpm) and torque (Nm)
TORQUE_POINTS = np.array([
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 
    150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250
])
SPEED_POINTS = np.array([
    0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 
    2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000
])

# UPDATED High-Resolution Efficiency Data Grid [%]
# The shape is (len(TORQUE_POINTS), len(SPEED_POINTS)).
EFFICIENCY_DATA = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  #   0 Nm
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  #  10 Nm
    [ 0,  0, 86, 86, 86, 86, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 86, 86,  0],  #  20 Nm
    [ 0,  0, 86, 86, 86, 86, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 86, 86,  0],  #  30 Nm
    [ 0,  0, 86, 86, 90, 90, 90, 90, 94, 94, 94, 94, 94, 94, 94, 94, 90, 90, 90, 90, 86],  #  40 Nm
    [ 0,  0, 86, 86, 90, 90, 90, 90, 94, 94, 94, 94, 94, 94, 94, 94, 90, 90, 90, 90, 86],  #  50 Nm
    [ 0,  0, 86, 86, 90, 90, 90, 90, 94, 94, 94, 94, 94, 94, 94, 94, 90, 90, 90, 90, 86],  #  60 Nm
    [ 0,  0, 86, 86, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 94, 94, 90],  #  70 Nm
    [ 0,  0, 86, 86, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 94, 94, 90],  #  80 Nm
    [ 0,  0, 86, 86, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 95, 95, 90],  #  90 Nm
    [ 0,  0, 86, 86, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 95, 95, 90],  # 100 Nm
    [ 0,  0, 86, 86, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 95, 95, 90],  # 110 Nm
    [ 0,  0, 86, 86, 94, 94, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 95, 95, 94, 94, 86],  # 120 Nm
    [ 0,  0, 86, 86, 94, 94, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 95, 95, 94, 94, 86],  # 130 Nm
    [ 0,  0, 86, 86, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 94, 94, 90, 90, 86],  # 140 Nm
    [ 0,  0, 86, 86, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 94, 94, 90, 90, 86],  # 150 Nm
    [ 0,  0, 86, 86, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 94, 94, 90, 90, 86],  # 160 Nm
    [ 0,  0, 86, 86, 94, 94, 94, 94, 94, 94, 95, 95, 95, 95, 94, 94, 90, 90, 90, 90,  0],  # 170 Nm
    [ 0,  0, 86, 86, 94, 94, 94, 94, 94, 94, 95, 95, 95, 95, 94, 94, 90, 90, 90, 90,  0],  # 180 Nm
    [ 0,  0, 86, 86, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 90, 90,  0],  # 190 Nm
    [ 0,  0, 86, 86, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 90, 90,  0],  # 200 Nm
    [ 0,  0, 86, 86, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 90, 90,  0],  # 210 Nm
    [ 0,  0, 86, 86, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,  0],  # 220 Nm
    [ 0,  0, 86, 86, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,  0],  # 230 Nm
    [ 0,  0,  0,  0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,  0,  0,  0],  # 240 Nm
    [ 0,  0,  0,  0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90,  0,  0,  0],  # 250 Nm
], dtype=int)


# 2. Create the interpolator object (logic is unchanged)
_interpolator = RegularGridInterpolator(
    (TORQUE_POINTS, SPEED_POINTS), EFFICIENCY_DATA,
    bounds_error=False,
    fill_value=0
)


def get_efficiency(speed_rpm: float, torque_nm: float) -> float:
    """Calculates efficiency for a single speed/torque point."""
    point = np.array([torque_nm, speed_rpm])
    return float(_interpolator(point))


def get_efficiency_vectorized(
    speeds_rpm: "Union[np.ndarray, list, 'pd.Series']",
    torques_nm: "Union[np.ndarray, list, 'pd.Series']"
) -> np.ndarray:
    """Calculates motor efficiencies for arrays of speed and torque values."""
    points = np.column_stack((torques_nm, speeds_rpm))
    return _interpolator(points)


# 3. Example usage block
if __name__ == '__main__':
    try:
        import pandas as pd
        print("--- Testing with Pandas DataFrame using new high-resolution map ---")
        
        # Create a sample DataFrame
        df = pd.DataFrame({
            'speed_rpm': [1600, 2600, 4100, 3000, 900],
            'torque_nm': [105, 145, 82, 185, 245]
        })
        
        # Use the vectorized function to add an efficiency column
        df['efficiency_%'] = get_efficiency_vectorized(df['speed_rpm'], df['torque_nm'])
        
        # Print results, formatted for clarity
        print(df.round(2))

    except ImportError:
        print("Pandas is not installed. Skipping DataFrame test.")