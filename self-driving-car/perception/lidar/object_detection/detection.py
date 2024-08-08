# LiDAR object detection code
import numpy as np

def detect_objects(point_cloud):
    # Implement object detection in point cloud data
    objects = []
    for point in point_cloud:
        if point[2] > -1.5:
            objects.append(point)
    return objects
