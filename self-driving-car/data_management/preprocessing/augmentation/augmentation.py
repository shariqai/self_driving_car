# Data augmentation code
def augment_data(data):
    # Implement data augmentation
    augmented_data = data.copy()
    augmented_data['augmented'] = data['original'] * 2
    return augmented_data
