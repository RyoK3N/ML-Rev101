import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, LeakyReLU, BatchNormalization, Dropout, InputLayer

class Discriminator(tf.keras.models.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Input shape for MNIST images
        self.input_shape = (28, 28, 1)
        self.number_of_stacked_layers = 3
        self.input_dims = len(self.input_shape)
        self.number_of_channels = 1
        self.number_of_conv_layers = 2
        
        # Create sequential model
        self.model = Sequential()
        
        # Input layer
        self.model.add(InputLayer(shape=self.input_shape))
        
        # Stacked layers with proper dimensionality
        for i in range(self.number_of_stacked_layers):
            # Add convolutional layers
            for j in range(self.number_of_conv_layers):
                self.model.add(Conv2D(filters=32 * (2 ** i), kernel_size=3, padding="same"))
                self.model.add(LeakyReLU(negative_slope=0.2))
                self.model.add(BatchNormalization(momentum=0.8))
                self.model.add(Dropout(0.25))
            
            # Downsampling with MaxPooling
            if i < self.number_of_stacked_layers - 1:  # Don't downsample on the last layer
                self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Flatten the output
        self.model.add(Flatten())
        
        # Dense layers for classification
        self.model.add(Dense(128))
        self.model.add(LeakyReLU(negative_slope=0.2))
        self.model.add(BatchNormalization(momentum=0.8))
        self.model.add(Dropout(0.3))
        
        # Output layer
        self.model.add(Dense(1, activation='sigmoid'))
    
    def call(self, x):
        return self.model(x)
    
    def build_model(self):
        # Build the model
        self.model.build(input_shape=(None, *self.input_shape))
        self.model.summary()


if __name__ == "__main__":
    # Create and build the discriminator
    discriminator = Discriminator()
    discriminator.build_model()
