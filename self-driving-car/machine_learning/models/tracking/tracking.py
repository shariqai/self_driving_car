# Tracking model
import tensorflow as tf

def build_tracking_model():
    # Implement tracking model
    model = tf.keras.applications.ResNet50(weights='imagenet')
    return model
