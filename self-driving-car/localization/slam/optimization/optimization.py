# SLAM optimization code
import numpy as np

def optimize_slam(data):
    # Implement SLAM optimization
    optimized_map = {}
    for point in data:
        grid_x = int(point[0] / 0.1)
        grid_y = int(point[1] / 0.1)
        if (grid_x, grid_y) not in optimized_map:
            optimized_map[(grid_x, grid_y)] = point[2]
        else:
            optimized_map[(grid_x, grid_y)] = (optimized_map[(grid_x, grid_y)] + point[2]) / 2
    return optimized_map
