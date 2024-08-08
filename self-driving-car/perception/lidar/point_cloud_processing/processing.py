# LiDAR point cloud processing code
import numpy as np

def process_point_cloud(point_cloud):
    # Implement point cloud processing, e.g., downsampling and filtering
    downsampled_cloud = point_cloud[::10]
    filtered_cloud = downsampled_cloud[downsampled_cloud[:,2] > -1.5]
    return filtered_cloud
