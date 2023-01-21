import logging

import keras_cv
import tensorflow as tf
from keras_cv.models.stable_diffusion.constants import (
    _ALPHAS_CUMPROD,
    _UNCONDITIONAL_TOKENS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Exporter")


def text_encoder_exporter(model: tf.keras.Model):
    """Exports the text encoder module from the StableDiffusion model to be served by TFServing."""
    MAX_PROMPT_LENGTH = 77
    POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
    UNCONDITIONAL_TOKENS = tf.convert_to_tensor([_UNCONDITIONAL_TOKENS], dtype=tf.int32)

    @tf.function(
        input_signature=[
            {
                "tokens": tf.TensorSpec(
                    shape=[None, 77], dtype=tf.int32, name="tokens"
                ),
                "batch_size": tf.TensorSpec(
                    shape=[], dtype=tf.int32, name="batch_size"
                ),
            }
        ]
    )
    def serving_fn(inputs):
        batch_size = inputs["batch_size"]

        encoded_text = model([inputs["tokens"], POS_IDS], training=False)
        encoded_text = tf.squeeze(encoded_text)

        if tf.rank(encoded_text) == 2:
            encoded_text = tf.repeat(
                tf.expand_dims(encoded_text, axis=0), batch_size, axis=0
            )

        unconditional_context = model([UNCONDITIONAL_TOKENS, POS_IDS], training=False)
        unconditional_context = tf.repeat(unconditional_context, batch_size, axis=0)
        return {"context": encoded_text, "unconditional_context": unconditional_context}

    return serving_fn


def diffusion_model_exporter(model: tf.keras.Model):
    """Exports the diffusion model module from the StableDiffusion model to be served by TFServing."""
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    _ALPHAS_CUMPROD_tf = tf.constant(_ALPHAS_CUMPROD)
    UNCONDITIONAL_GUIDANCE_SCALE = 7.5

    @tf.function
    def get_timestep_embedding(timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(
            -log_max_preiod * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    @tf.function(
        input_signature=[
            {
                "context": tf.TensorSpec(
                    shape=[None, 77, 768], dtype=tf.float32, name="context"
                ),
                "unconditional_context": tf.TensorSpec(
                    shape=[None, 77, 768],
                    dtype=tf.float32,
                    name="unconditional_context",
                ),
                # "num_steps": tf.TensorSpec(shape=[], dtype=tf.int32, name="num_steps"),
                # "batch_size": tf.TensorSpec(shape=[], dtype=tf.int32, name="batch_size"),
            }
        ]
    )
    def serving_fn(inputs):
        img_height = tf.cast(tf.math.round(IMG_HEIGHT / 128) * 128, tf.int32)
        img_width = tf.cast(tf.math.round(IMG_WIDTH / 128) * 128, tf.int32)

        unconditional_context = inputs["unconditional_context"]
        context = inputs["context"]
        # TODO: add support for thse parameters as inputs
        batch_size = 1  # inputs["batch_size"]
        num_steps = 10  # inputs["num_steps"]

        latent = tf.random.normal((batch_size, img_height // 8, img_width // 8, 4))

        timesteps = tf.range(1, 1000, 1000 // num_steps)
        alphas = tf.map_fn(lambda t: _ALPHAS_CUMPROD_tf[t], timesteps, dtype=tf.float32)
        alphas_prev = tf.concat([[1.0], alphas[:-1]], 0)

        index = num_steps - 1
        latent_prev = None
        for timestep in timesteps[::-1]:
            latent_prev = latent
            t_emb = get_timestep_embedding(timestep, batch_size)
            unconditional_latent = model(
                [latent, t_emb, unconditional_context], training=False
            )
            latent = model([latent, t_emb, context], training=False)
            latent = unconditional_latent + UNCONDITIONAL_GUIDANCE_SCALE * (
                latent - unconditional_latent
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - tf.math.sqrt(1 - a_t) * latent) / tf.math.sqrt(a_t)
            latent = (
                latent * tf.math.sqrt(1.0 - a_prev) + tf.math.sqrt(a_prev) * pred_x0
            )
            index = index - 1

        return {"latent": latent}

    return serving_fn


def decoder_exporter(model: tf.keras.Model):
    """Exports the decoder module from the StableDiffusion model to be served by TFServing."""
    @tf.function(
        input_signature=[
            {
                "latent": tf.TensorSpec(
                    shape=[None, 64, 64, 4], dtype=tf.float32, name="latent"
                ),
            }
        ]
    )
    def serving_fn(inputs):
        latent = inputs["latent"]
        decoded = model(latent, training=False)
        decoded = ((decoded + 1) / 2) * 255
        images = tf.clip_by_value(decoded, 0, 255)
        images = tf.cast(images, tf.uint8)
        return {"generated_images": images}

    return serving_fn


def export_stable_diffusion(
    stable_diffusion: keras_cv.models.StableDiffusion,
    models_dir: str,
    models_version: int = 1,
):
    """Exports the StableDiffusion model's inner mudules to be served by TFServing."""
    text_encoder_output_path = f"{models_dir}/text_encoder/{models_version}/"
    diffusion_model_output_path = f"{models_dir}/diffusion_model/{models_version}/"
    decoder_output_path = f"{models_dir}/decoder/{models_version}/"

    logger.info(f'Exporting text encoder to: "{text_encoder_output_path}"')
    tf.saved_model.save(
        stable_diffusion.text_encoder,
        text_encoder_output_path,
        signatures={
            "serving_default": text_encoder_exporter(stable_diffusion.text_encoder)
        },
    )

    logger.info(f'Exporting diffusion model to: "{diffusion_model_output_path}"')
    tf.saved_model.save(
        stable_diffusion.diffusion_model,
        diffusion_model_output_path,
        signatures={
            "serving_default": diffusion_model_exporter(
                stable_diffusion.diffusion_model
            )
        },
    )

    logger.info(f'Exporting decoder to: "{decoder_output_path}"')
    tf.saved_model.save(
        stable_diffusion.decoder,
        decoder_output_path,
        signatures={"serving_default": decoder_exporter(stable_diffusion.decoder)},
    )
