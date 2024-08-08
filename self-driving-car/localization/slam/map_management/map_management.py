# SLAM map management code
import numpy as np

def manage_map(data):
    # Implement SLAM map management
    map = {}
    for point in data:
        grid_x = int(point[0] / 0.1)
        grid_y = int(point[1] / 0.1)
        map[(grid_x, grid_y)] = point[2]
    return map
