import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim):
    """Build a generator model for GANs."""
    model = tf.keras.Sequential([
        layers.Dense(128 * 128 * 3, input_dim=latent_dim),
        layers.Reshape((128, 128, 3)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')
    ])
    return model

def build_discriminator(img_shape):
    """Build a discriminator model for GANs."""
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    """Build and compile the GAN model."""
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
