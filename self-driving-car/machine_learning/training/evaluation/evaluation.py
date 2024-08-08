# Model evaluation code
import tensorflow as tf

def evaluate_model(model, data, labels):
    # Implement model evaluation
    loss, accuracy = model.evaluate(data, labels)
    return loss, accuracy
