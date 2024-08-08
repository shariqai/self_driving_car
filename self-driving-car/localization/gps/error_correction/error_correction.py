# GPS error correction code
import numpy as np

def correct_gps_error(data):
    # Implement GPS error correction
    corrected_data = data + np.random.normal(0, 1, data.shape)
    return corrected_data
