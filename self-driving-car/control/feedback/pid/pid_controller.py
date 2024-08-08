# PID controller code
import numpy as np

def pid_controller(data):
    # Implement PID controller
    pid_output = np.mean(data)
    return pid_output
