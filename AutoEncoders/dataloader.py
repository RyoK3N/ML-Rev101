import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from visualize import visualize_samples

def load_data():
    """Load MNIST dataset from TensorFlow Datasets."""
    # Properly load the dataset with as_supervised=True to get (image, label) tuples
    (train_data, test_data), info = tfds.load(
        name="mnist",
        split=["train", "test"],
        as_supervised=True,  # Returns tuple (img, label) instead of dict
        with_info=True
    )
    
    return train_data, test_data, info

def preprocess_data(data):
    """Normalize pixel values to [0, 1] range."""
    return data.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

def augment_data(data):
    """Apply data augmentation to training data."""
    # Define augmentations
    augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomFlip(mode="horizontal"),
    ])
    
    # Apply augmentations
    return data.map(lambda x, y: (augmentations(x, training=True), y))

def prepare_mnist_data(batch_size=32, shuffle_buffer_size=10000, augment=True):
    """Complete data pipeline for MNIST dataset."""
    # Load data
    train_data, test_data, info = load_data()
    
    # Preprocess data
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    
    # Shuffle training data
    train_data = train_data.shuffle(shuffle_buffer_size)
    
    # Apply augmentation to training data only
    if augment:
        train_data = augment_data(train_data)
    
    # Batch data and prefetch
    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_data, test_data, info

if __name__ == "__main__":
    # Load and prepare data
    train_data, test_data, info = prepare_mnist_data(batch_size=32)
    
    # Print dataset info
    print(f"Dataset: {info.name}, version: {info.version}")
    print(f"Training samples: {info.splits['train'].num_examples}")
    print(f"Test samples: {info.splits['test'].num_examples}")
    
    # Print shapes
    for images, labels in train_data.take(1):
        print(f"Training batch shape: {images.shape}, labels shape: {labels.shape}")
    
    # Visualize samples from training data - pass the dataset directly
    visualize_samples(train_data)
    
    # Visualize samples from test data - pass the dataset directly
    visualize_samples(test_data)

