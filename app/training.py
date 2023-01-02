import tensorflow as tf
from keras_cv.models.stable_diffusion import NoiseScheduler

from app.stable_diffusion import StableDiffusionFineTuner


def get_image_encoder(stable_diffusion):
    # Remove the top layer from the encoder, which cuts off the variance and only returns the mean
    return tf.keras.Model(
        stable_diffusion.image_encoder.input,
        stable_diffusion.image_encoder.layers[-2].output,
    )


def get_noise_scheduler():
    return NoiseScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        train_timesteps=1000,
    )


def get_trainer(
    stable_diffusion,
    training_image_encoder,
    noise_scheduler,
    max_prompt_length,
    placeholder_token,
):
    trainer = StableDiffusionFineTuner(
        stable_diffusion,
        training_image_encoder,
        noise_scheduler,
        max_prompt_length,
        placeholder_token,
        name="trainer",
    )
    return trainer


def setup_trainer(trainer, train_ds, epochs):
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4, decay_steps=train_ds.cardinality() * epochs
    )

    optimizer = tf.keras.optimizers.Adam(
        weight_decay=0.004,
        learning_rate=learning_rate,
        epsilon=1e-8,
        global_clipnorm=10,
    )

    trainer.compile(
        optimizer=optimizer,
        # We are performing reduction manually in our train step, so none is required here.
        loss=tf.keras.losses.MeanSquaredError(reduction="none"),
    )
    return trainer
