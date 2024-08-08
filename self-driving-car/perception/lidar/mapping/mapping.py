# LiDAR mapping code
import numpy as np

def create_map(point_cloud):
    # Implement SLAM to create a map from point cloud data
    map = {}
    for point in point_cloud:
        grid_x = int(point[0] / 0.1)
        grid_y = int(point[1] / 0.1)
        map[(grid_x, grid_y)] = point[2]
    return map
