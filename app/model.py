import keras_cv
import numpy as np
import tensorflow as tf

from app.model_utils import traverse_layers


def add_new_token(
    stable_diffusion: keras_cv.models.StableDiffusion, initializer_token: str
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the weights of the position embedding and token embedding after adding a new token."""
    tokenized_initializer = stable_diffusion.tokenizer.encode(initializer_token)[1]
    new_weights = stable_diffusion.text_encoder.layers[2].token_embedding(
        tf.constant(tokenized_initializer)
    )

    # The embedding layer is the 2nd layer in the text encoder
    old_token_weights = stable_diffusion.text_encoder.layers[
        2
    ].token_embedding.get_weights()
    old_position_weights = stable_diffusion.text_encoder.layers[
        2
    ].position_embedding.get_weights()

    new_weights = np.expand_dims(new_weights, axis=0)
    new_weights = np.concatenate([old_token_weights[0], new_weights], axis=0)
    new_weights = np.expand_dims(new_weights, axis=0)
    return old_position_weights, new_weights


def build_text_encoder(
    stable_diffusion: keras_cv.models.StableDiffusion,
    position_embedding_weights: np.ndarray,
    token_embedding_weights: np.ndarray,
) -> keras_cv.models.StableDiffusion:
    """
    Takes weights of the position embedding and token embedding
    and builds a new text encoder for a StableDiffusion model.
    """
    # Have to set download_weights False so we can init (otherwise tries to load weights)
    new_encoder = keras_cv.models.stable_diffusion.TextEncoder(
        keras_cv.models.stable_diffusion.stable_diffusion.MAX_PROMPT_LENGTH,
        vocab_size=len(stable_diffusion.tokenizer.vocab),
        download_weights=False,
    )

    for index, layer in enumerate(stable_diffusion.text_encoder.layers):
        # Layer 2 is the embedding layer, so we omit it from our weight-copying
        if index == 2:
            continue
        new_encoder.layers[index].set_weights(layer.get_weights())

    new_encoder.layers[2].token_embedding.set_weights(token_embedding_weights)
    new_encoder.layers[2].position_embedding.set_weights(position_embedding_weights)

    stable_diffusion._text_encoder = new_encoder
    stable_diffusion._text_encoder.compile(jit_compile=True)
    return stable_diffusion


def setup_model_for_training(
    stable_diffusion: keras_cv.models.StableDiffusion,
) -> keras_cv.models.StableDiffusion:
    """Takes a StableDiffusion model and configures which layers will be trained or not."""
    stable_diffusion.diffusion_model.trainable = False
    stable_diffusion.decoder.trainable = False
    stable_diffusion.text_encoder.trainable = True
    stable_diffusion.text_encoder.layers[2].trainable = True

    for layer in traverse_layers(stable_diffusion.text_encoder):
        if (
            isinstance(layer, tf.keras.layers.Embedding)
            or "clip_embedding" in layer.name
        ):
            layer.trainable = True
        else:
            layer.trainable = False

    stable_diffusion._text_encoder.layers[2].position_embedding.trainable = False
    return stable_diffusion
