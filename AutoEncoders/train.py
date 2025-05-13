import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import argparse
from datetime import datetime

from GAN import GAN
from dataloader import prepare_mnist_data
from visualize import visualize_generated_images, generate_and_save_images

def train_gan(gan, train_dataset, epochs=50, batch_size=128, save_interval=10, 
              output_dir="output", log_dir="logs"):
    """
    Train the GAN model
    
    Args:
        gan: The GAN model
        train_dataset: TensorFlow dataset containing training data
        epochs: Number of epochs to train
        batch_size: Batch size for training
        save_interval: Interval (in epochs) to save model and generate images
        output_dir: Directory to save generated images
        log_dir: Directory to save TensorBoard logs
    """
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Setup TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, current_time)
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        # Initialize metrics for this epoch
        d_losses = []
        g_losses = []
        
        # Progress bar for batches
        progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Train on batches
        for batch_idx, (real_images, _) in enumerate(progress_bar):
            # Train discriminator and generator
            d_loss, g_loss = train_step(gan, real_images)
            
            # Store losses
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                'd_loss': f"{d_loss:.4f}", 
                'g_loss': f"{g_loss:.4f}"
            })
        
        # Calculate average losses for the epoch
        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        
        # Log to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('discriminator_loss', avg_d_loss, step=epoch)
            tf.summary.scalar('generator_loss', avg_g_loss, step=epoch)
            
            # Generate and log images
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                # Generate images
                generated_images = gan.generate_fake_samples(16)
                
                # Convert to displayable format
                display_images = tf.clip_by_value(generated_images, 0, 1)
                display_images = tf.reshape(display_images, [16, 28, 28, 1])
                
                # Log images to TensorBoard
                tf.summary.image("Generated Images", display_images, max_outputs=16, step=epoch)
        
        # Save model and generate images at intervals
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            # Save model
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"gan_epoch_{epoch+1}")
            gan.save_models(
                generator_path=f"{checkpoint_path}_generator.keras",
                discriminator_path=f"{checkpoint_path}_discriminator.keras"
            )
            
            # Generate and save images
            images_path = os.path.join(output_dir, "images")
            generate_and_save_images(gan, epoch + 1, 16, images_path)
        
        # Print epoch summary
        time_taken = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - {time_taken:.2f}s - d_loss: {avg_d_loss:.4f} - g_loss: {avg_g_loss:.4f}")
    
    # Final save
    final_checkpoint_path = os.path.join(output_dir, "checkpoints", "gan_final")
    gan.save_models(
        generator_path=f"{final_checkpoint_path}_generator.keras",
        discriminator_path=f"{final_checkpoint_path}_discriminator.keras"
    )

    print("Training completed!")

@tf.function
def train_step(gan, real_images):
    """
    Perform a single training step (train both discriminator and generator)
    
    Args:
        gan: The GAN model
        real_images: Batch of real images
        
    Returns:
        Tuple of (discriminator_loss, generator_loss)
    """
    batch_size = tf.shape(real_images)[0]
    
    # Generate random noise for the generator
    noise = gan.generate_latent_points(batch_size)
    
    # Train the discriminator
    with tf.GradientTape() as disc_tape:
        # Generate fake images
        fake_images = gan.generator.model(noise, training=True)
        
        # Get discriminator outputs for real and fake images
        real_output = gan.discriminator.model(real_images, training=True)
        fake_output = gan.discriminator.model(fake_images, training=True)
        
        # Create labels with smoothing
        real_labels = tf.ones_like(real_output) * 0.9  # Label smoothing
        fake_labels = tf.zeros_like(fake_output)
        
        # Calculate discriminator loss
        real_loss = gan.loss_fn(real_labels, real_output)
        fake_loss = gan.loss_fn(fake_labels, fake_output)
        disc_loss = real_loss + fake_loss
    
    # Get discriminator gradients and update weights
    disc_gradients = disc_tape.gradient(disc_loss, gan.discriminator.model.trainable_variables)
    gan.discriminator_optimizer.apply_gradients(
        zip(disc_gradients, gan.discriminator.model.trainable_variables)
    )
    
    # Train the generator
    with tf.GradientTape() as gen_tape:
        # Generate fake images
        fake_images = gan.generator.model(noise, training=True)
        
        # Get discriminator output for fake images
        fake_output = gan.discriminator.model(fake_images, training=True)
        
        # We want the discriminator to think these are real
        gen_labels = tf.ones_like(fake_output)
        
        # Calculate generator loss
        gen_loss = gan.loss_fn(gen_labels, fake_output)
    
    # Get generator gradients and update weights
    gen_gradients = gen_tape.gradient(gen_loss, gan.generator.model.trainable_variables)
    gan.generator_optimizer.apply_gradients(
        zip(gen_gradients, gan.generator.model.trainable_variables)
    )
    
    return disc_loss, gen_loss

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of latent space")
    parser.add_argument("--save-interval", type=int, default=5, help="Interval to save model and images")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save TensorBoard logs")
    args = parser.parse_args()
    
    # Load and prepare MNIST data
    train_data, _, _ = prepare_mnist_data(batch_size=args.batch_size)
    
    # Create and compile the GAN model
    gan = GAN(latent_dim=args.latent_dim)
    gan.compile()
    
    # Train the GAN
    train_gan(
        gan=gan,
        train_dataset=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )
    
    # Generate final samples
    print("Generating final samples...")
    samples = gan.generate_fake_samples(25)
    visualize_generated_images(samples, num_samples=25)

if __name__ == "__main__":
    main()


