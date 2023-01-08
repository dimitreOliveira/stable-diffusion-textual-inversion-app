import os

import keras_cv
import numpy as np

from src.model import build_text_encoder


def save_finetuned_weights(
    position_embedding: np.ndarray,
    token_embedding: np.ndarray,
    models_dir: str = "./models",
) -> None:
    """Writes the weights for the  "position_embedding" and "token_embedding" as .npy files."""
    finetuned_position_weights = position_embedding.get_weights()
    finetuned_token_weights = token_embedding.get_weights()

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    np.save(f"{models_dir}/position_embedding_weights", finetuned_position_weights)
    np.save(f"{models_dir}/token_embedding_weights", finetuned_token_weights)


def load_finetuned_weights(
    position_embedding_weights_path: str,
    token_embedding_weights_path: str,
    placeholder_token: str,
) -> keras_cv.models.StableDiffusion:
    """
    Reads the .npy files for the weights for the "position_embedding" and "token_embedding"
    then uses them to re-build a StableDiffusion model.
    """
    position_embedding_weights = np.load(position_embedding_weights_path)
    token_embedding_weights = np.load(token_embedding_weights_path)

    stable_diffusion = keras_cv.models.StableDiffusion()
    stable_diffusion.tokenizer.add_tokens(placeholder_token)
    stable_diffusion = build_text_encoder(
        stable_diffusion, position_embedding_weights, token_embedding_weights
    )
    return stable_diffusion
