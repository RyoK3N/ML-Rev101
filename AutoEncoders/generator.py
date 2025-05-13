import tensorflow as tf
import numpy as np 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, LeakyReLU, BatchNormalization, Dropout, Conv2DTranspose, InputLayer
from visualize import generate_images, visualize_generated_images

class Generator(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.input_shape = (28, 28, 1)
        self.number_of_stacked_layers = 1  # Reduced from 5 to avoid excessive layers
        self.input_dims = len(self.input_shape)
        self.number_of_channels = 1
        self.number_of_conv_layers = 1  # Reduced from 3 to simplify

        # Create sequential model
        self.model = Sequential()
        
        # Input layer
        self.model.add(InputLayer(input_shape=(self.latent_dim,)))
        
        # Initial dense layer to get enough features
        self.model.add(Dense(7 * 7 * 128))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.8))
        
        # Reshape to prepare for convolutions
        self.model.add(Reshape((7, 7, 128)))
        
        # Stacked layers with proper dimensionality
        for i in range(self.number_of_stacked_layers):
            # Add convolutional layers
            for j in range(self.number_of_conv_layers):
                self.model.add(Conv2D(filters=64 // (i + 1), kernel_size=3, padding="same"))
                self.model.add(LeakyReLU(alpha=0.2))
                self.model.add(BatchNormalization(momentum=0.8))
                self.model.add(Dropout(0.25))
            
            # Upsampling with transposed convolution
            if i < self.number_of_stacked_layers - 1:  # Don't upsample on the last layer
                self.model.add(Conv2DTranspose(filters=32 // (i + 1), kernel_size=4, strides=2, padding="same"))
                self.model.add(LeakyReLU(alpha=0.2))
                self.model.add(BatchNormalization(momentum=0.8))
        
        # Final output layer
        self.model.add(Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid"))
    
    def call(self, x):
        return self.model(x)
    
    def build_model(self):
        # Build the model
        self.model.build(input_shape=(None, self.latent_dim))
        self.model.summary()


if __name__ == "__main__":
    # Create and build the generator
    generator = Generator(latent_dim=100)
    generator.build_model()
    
    # Generate and display sample images
    import matplotlib.pyplot as plt
    
    # Generate sample images using the function from visualize.py
    sample_images = generate_images(generator.model, num_images=2, latent_dim=100)
    
    # Display the generated images
    visualize_generated_images(sample_images)
