# IMU sensor fusion code
import numpy as np

def fuse_imu_data(data):
    # Implement IMU sensor fusion
    fused_data = np.mean(data, axis=0)
    return fused_data
