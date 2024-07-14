import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Generator
def make_generator():
    return models.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

# Discriminator
def make_discriminator():
    return models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])

# Generate synthetic data
def generate_synthetic_data(num_samples=60000):
    synthetic_data = np.random.rand(num_samples, 28, 28, 1).astype('float32')
    synthetic_data = (synthetic_data - 0.5) * 2  # Scale to [-1, 1]
    return synthetic_data

# Load synthetic data
train_images = generate_synthetic_data()
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(256)

# Create models
generator = make_generator()
discriminator = make_discriminator()

# Define loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([256, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    gen_losses = []
    disc_losses = []
    for image_batch in dataset:
        gen_loss, disc_loss = train_step(image_batch)
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Gen Loss: {np.mean(gen_losses):.4f}, Disc Loss: {np.mean(disc_losses):.4f}")
    
    if (epoch + 1) % 10 == 0:
        noise = tf.random.normal([16, 100])
        generated_images = generator(noise, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
            plt.axis('off')
        plt.savefig(f'dcgan_epoch_{epoch+1}.png')
        plt.close()

print("Training completed.")

# Generate final image
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.figure(figsize=(4, 4))
plt.imshow(generated_image[0, :, :, 0] * 0.5 + 0.5, cmap='gray')
plt.axis('off')
plt.savefig('final_generated_image.png')
plt.show()
