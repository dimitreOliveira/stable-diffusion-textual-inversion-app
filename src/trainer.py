import keras_cv
import tensorflow as tf

from src.finetuner import StableDiffusionFineTuner


def get_image_encoder(
    stable_diffusion: keras_cv.models.StableDiffusion,
) -> tf.keras.Model:
    """Takes the image encoder from the StableDiffusion and remove the top layer from the it,
    which cuts off the variance and only returns the mean.

    Args:
        stable_diffusion (keras_cv.models.StableDiffusion): Stable diffusion model

    Returns:
        tf.keras.Model: Image encoder from the stable diffusion model
    """
    image_encoder = tf.keras.Model(
        stable_diffusion.image_encoder.input,
        stable_diffusion.image_encoder.layers[-2].output,
    )
    return image_encoder


def get_noise_scheduler(
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    beta_schedule: str = "scaled_linear",
    train_timesteps: int = 1000,
) -> keras_cv.models.stable_diffusion.NoiseScheduler:
    """Gets the noise scheduler that will be used during the trainig

    Args:
        beta_start (float, optional): beta_start parameter for the noise scheduler. Defaults to 0.00085.
        beta_end (float, optional): beta_end parameter for the noise scheduler. Defaults to 0.012.
        beta_schedule (str, optional): beta_schedule parameter for the noise scheduler. Defaults to "scaled_linear".
        train_timesteps (int, optional): train_timesteps parameter for the noise scheduler. Defaults to 1000.

    Returns:
        keras_cv.models.stable_diffusion.NoiseScheduler: Noise scheduler object
    """
    noiseScheduler = keras_cv.models.stable_diffusion.NoiseScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        train_timesteps=train_timesteps,
    )
    return noiseScheduler


def get_trainer(
    stable_diffusion: keras_cv.models.StableDiffusion,
    training_image_encoder: tf.keras.Model,
    noise_scheduler: keras_cv.models.stable_diffusion.NoiseScheduler,
    max_prompt_length: int,
    placeholder_token: str,
    name: str = "trainer",
) -> StableDiffusionFineTuner:
    """Gets the fine tunner object that will be used during the trainig

    Args:
        stable_diffusion (keras_cv.models.StableDiffusion): Stable diffusion model
        training_image_encoder (tf.keras.Model): Image encoder model
        noise_scheduler (keras_cv.models.stable_diffusion.NoiseScheduler): Noise scheduler
        max_prompt_length (int): Max allowed prompt length
        placeholder_token (str): Token used as the textual inversion target
        name (str, optional): Name for the trainer object. Defaults to "trainer".

    Returns:
        StableDiffusionFineTuner: Stable diffusion finetuner object
    """
    trainer = StableDiffusionFineTuner(
        stable_diffusion,
        training_image_encoder,
        noise_scheduler,
        max_prompt_length,
        placeholder_token,
        name=name,
    )
    return trainer


def setup_trainer(
    trainer: StableDiffusionFineTuner, train_ds: tf.data.Dataset, epochs: int
) -> StableDiffusionFineTuner:
    """Configures the fine tunner object that will be used during the trainig

    Args:
        trainer (StableDiffusionFineTuner): Stable diffusion finetuner object
        train_ds (tf.data.Dataset): Dataset that will be used during the finetuning
        epochs (int): Number of epochs used during the finetuning

    Returns:
        StableDiffusionFineTuner: Stable diffusion finetuner object
    """
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
