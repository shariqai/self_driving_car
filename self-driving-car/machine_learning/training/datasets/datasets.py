# Dataset management code
import tensorflow as tf

def load_datasets():
    # Implement dataset loading
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)
