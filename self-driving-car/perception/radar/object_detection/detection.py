# Radar object detection code
import numpy as np

def detect_objects(signal):
    # Implement object detection in radar signal data
    objects = []
    threshold = np.mean(signal) + 2 * np.std(signal)
    for i, s in enumerate(signal):
        if s > threshold:
            objects.append(i)
    return objects
