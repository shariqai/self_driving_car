# Path optimization code
import numpy as np

def optimize_path(data):
    # Implement path optimization
    optimized_path = []
    for point in data:
        if point[2] > -1.5:
            optimized_path.append(point)
    return optimized_path
