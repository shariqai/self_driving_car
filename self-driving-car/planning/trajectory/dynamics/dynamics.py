# Trajectory dynamics code
import numpy as np

def compute_dynamics(data):
    # Implement trajectory dynamics
    dynamics = np.diff(data, axis=0)
    return dynamics
