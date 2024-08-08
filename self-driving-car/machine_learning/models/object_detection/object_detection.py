# Object detection model
import tensorflow as tf

def build_object_detection_model():
    # Implement object detection model
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model
