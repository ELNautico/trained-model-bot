import tensorflow as tf

def compute_feature_sensitivity(model, sample, feature_names):
    """
    Uses TensorFlow's GradientTape to compute the sensitivity of the model's output
    with respect to each input feature.

    Args:
        model: Trained Keras model
        sample: 3D input tensor of shape (1, timesteps, features)
        feature_names: List of feature names corresponding to the last axis of sample

    Returns:
        Dictionary mapping feature names to average absolute gradient values.
    """
    sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
    sample_tensor = tf.Variable(sample_tensor)

    with tf.GradientTape() as tape:
        prediction = model(sample_tensor)

    gradients = tape.gradient(prediction, sample_tensor)
    avg_abs_grads = tf.reduce_mean(tf.abs(gradients), axis=[1, 0]).numpy().flatten()
    return dict(zip(feature_names, avg_abs_grads))
