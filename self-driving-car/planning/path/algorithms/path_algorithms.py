# Path planning algorithms
import numpy as np

def plan_path(data):
    # Implement path planning algorithm
    path = []
    for point in data:
        if point[2] > -1.5:
            path.append(point)
    return path
