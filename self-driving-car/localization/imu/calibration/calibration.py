# IMU calibration code
import numpy as np

def calibrate_imu(data):
    # Implement IMU calibration
    calibrated_data = data - np.mean(data, axis=0)
    return calibrated_data
