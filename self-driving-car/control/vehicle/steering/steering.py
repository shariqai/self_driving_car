# Vehicle steering control code
import numpy as np

def control_steering(data):
    # Implement steering control
    steering_angle = np.mean(data)
    return steering_angle
