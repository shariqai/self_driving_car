# GPS data processing code
import numpy as np

def process_gps_data(data):
    # Implement GPS data processing
    processed_data = data - np.mean(data, axis=0)
    return processed_data
