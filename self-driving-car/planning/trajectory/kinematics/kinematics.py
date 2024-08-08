# Trajectory kinematics code
import numpy as np

def compute_kinematics(data):
    # Implement trajectory kinematics
    kinematics = np.diff(data, axis=0)
    return kinematics
