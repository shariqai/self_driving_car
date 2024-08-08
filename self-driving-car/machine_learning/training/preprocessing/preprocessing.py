# Data preprocessing code
import tensorflow as tf

def preprocess_data(data):
    # Implement data preprocessing
    data = tf.image.resize(data, (224, 224))
    data = data / 255.0
    return data
