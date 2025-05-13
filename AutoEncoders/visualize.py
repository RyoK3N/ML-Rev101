import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def visualize_samples(dataset, num_samples=5):
    """
    Visualize samples from a TensorFlow dataset.
    
    Args:
        dataset: A TensorFlow dataset that yields (image, label) tuples
        num_samples: Number of samples to visualize
    """
    plt.figure(figsize=(12, 4))
    
    # Get a single batch from the dataset
    for i, (images, labels) in enumerate(dataset.take(1)):
        # Display up to num_samples images from the batch
        for j in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, j + 1)
            
            # Get the image
            img = images[j].numpy()
            
            # Handle different image shapes
            if img.shape[-1] == 1:  # Grayscale image with channel dimension
                img = np.squeeze(img)
            
            plt.imshow(img, cmap='gray')
            plt.title(f"Label: {labels[j].numpy()}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_images(model, num_images=1, latent_dim=100):
    """
    Generate random images using a generator model.
    
    Args:
        model: The generator model
        num_images: Number of images to generate
        latent_dim: Dimension of the latent space
        
    Returns:
        Generated images
    """
    random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))
    generated_images = model(random_latent_vectors)
    return generated_images

def visualize_generated_images(images, num_samples=5):
    """
    Visualize generated images.
    
    Args:
        images: Generated images tensor
        num_samples: Number of images to visualize
    """
    plt.figure(figsize=(12, 4))
    
    # Display up to num_samples images
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        
        # Get the image
        img = images[i].numpy()
        
        # Handle different image shapes
        if img.shape[-1] == 1:  # Grayscale image with channel dimension
            img = np.squeeze(img)
        
        plt.imshow(img, cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def generate_and_save_images(self, epoch, num_examples=16, save_path=None):
    """Generate and save images during training"""
    # Generate images
    latent_points = self.generate_latent_points(num_examples)
    generated_images = self.generator.model(latent_points, training=False)
    
    # Plot images
    plt.figure(figsize=(4, 4))
    for i in range(num_examples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    
    # Save or show the plot
    if save_path:
        plt.savefig(f"{save_path}/image_at_epoch_{epoch:04d}.png")
    else:
        plt.show()
    
    plt.close()


def plot_latent_space(self, num_points=100):
    """Plot points in the latent space"""
    latent_points = self.generate_latent_points(num_points)
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_points[:, 0], latent_points[:, 1], c='blue', alpha=0.5)
    plt.title("Latent Space")
    plt.show()
