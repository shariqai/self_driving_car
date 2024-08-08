# Radar tracking code
import numpy as np

def track_objects(signal, objects):
    # Implement object tracking in radar signal data
    tracked_objects = []
    for obj in objects:
        if signal[obj] > np.mean(signal) + 2 * np.std(signal):
            tracked_objects.append(obj)
    return tracked_objects
