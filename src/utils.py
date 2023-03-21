import math

import tensorflow as tf
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer


def pad_embedding(
    embedding: tf.Tensor, tokenizer: SimpleTokenizer, max_prompt_length: int
) -> tf.Tensor:
    """Pads a tokenized text to match the desired length

    Args:
        embedding (tf.Tensor): Embedding vector from the text encoder
        tokenizer (SimpleTokenizer): Tokenizer used to tokenize the text
        max_prompt_length (int): Max allowed prompt length

    Returns:
        tf.Tensor: Padded embedding vector
    """ """"""
    return embedding + ([tokenizer.end_of_text] * (max_prompt_length - len(embedding)))


def sample_from_encoder_outputs(outputs: tf.Tensor) -> tf.Tensor:
    """Get samples from the image encoder output

    Args:
        outputs (tf.Tensor): Outputs from the image encoder

    Returns:
        tf.Tensor: Sampled outputs from the image encoder
    """ """"""
    mean, logvar = tf.split(outputs, 2, axis=-1)
    logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    std = tf.exp(0.5 * logvar)
    sample = tf.random.normal(tf.shape(mean))
    return mean + std * sample


def get_timestep_embedding(
    timestep: int, dim: int = 320, max_period: int = 10000
) -> tf.Tensor:
    """Get the embedding from a timestep value.

    Args:
        timestep (int): Timestamp value
        dim (int, optional): Embedding dimension. Defaults to 320.
        max_period (int, optional): Max value of the time period. Defaults to 10000.

    Returns:
        tf.Tensor: Embedding from the time step
    """
    half = dim // 2
    freqs = tf.math.exp(
        -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    return embedding


def get_position_ids(max_prompt_length: int) -> tf.Tensor:
    """Get the position ID for a value.

    Args:
        max_prompt_length (int): Max allowed prompt length

    Returns:
        tf.Tensor: Position ID as tensor
    """
    return tf.convert_to_tensor([list(range(max_prompt_length))], dtype=tf.int32)


def traverse_layers(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """Utility to iterate over layers.

    Args:
        layer (tf.keras.layers.Layer): Layers that will be traversed

    Returns:
        tf.keras.layers.Layer: Individual layers

    Yields:
        Iterator[tf.keras.layers.Layer]: Individual layers
    """
    if hasattr(layer, "layers"):
        for layer in layer.layers:
            yield layer
    if hasattr(layer, "token_embedding"):
        yield layer.token_embedding
    if hasattr(layer, "position_embedding"):
        yield layer.position_embedding
