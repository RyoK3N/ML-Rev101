import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator
from visualize import generate_images, visualize_generated_images
from visualize import generate_and_save_images
from visualize import plot_latent_space

class GAN(tf.keras.models.Model):
    def __init__(self, latent_dim=100):
        super(GAN, self).__init__()
        
        # Initialize latent dimension
        self.latent_dim = latent_dim
        
        # Create generator and discriminator
        self.generator = Generator(latent_dim=self.latent_dim)
        self.discriminator = Discriminator()
        
        # Build both models
        self.generator.build_model()
        self.discriminator.build_model()
    
    def compile(self, g_optimizer=None, d_optimizer=None, loss_fn=None):
        """Compile the model with custom optimizers and loss function"""
        super(GAN, self).compile()
        
        # Set default optimizers if not provided
        self.generator_optimizer = g_optimizer or tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = d_optimizer or tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Set default loss function if not provided
        self.loss_fn = loss_fn or tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    def generate_latent_points(self, batch_size):
        """Generate random points in the latent space"""
        return tf.random.normal(shape=(batch_size, self.latent_dim))
    
    def generate_fake_samples(self, batch_size):
        """Generate fake samples using the generator"""
        # Generate latent points
        latent_points = self.generate_latent_points(batch_size)
        # Generate fake images
        fake_images = self.generator.model(latent_points, training=False)
        return fake_images
    
    
    def save_models(self, generator_path="generator.h5", discriminator_path="discriminator.h5"):
        """Save both generator and discriminator models"""
        self.generator.model.save(generator_path)
        self.discriminator.model.save(discriminator_path)
        print(f"Models saved to {generator_path} and {discriminator_path}")
    
    def load_models(self, generator_path="generator.h5", discriminator_path="discriminator.h5"):
        """Load both generator and discriminator models"""
        self.generator.model = tf.keras.models.load_model(generator_path)
        self.discriminator.model = tf.keras.models.load_model(discriminator_path)
        print(f"Models loaded from {generator_path} and {discriminator_path}")
    
if __name__ == "__main__":
    # Create GAN model
    gan = GAN(latent_dim=100)
    
    # Compile the model
    gan.compile()
    
    # Generate and visualize some random samples
    samples = gan.generate_fake_samples(5)
    visualize_generated_images(samples)
