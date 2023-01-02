import keras_cv
import numpy as np

from app.model import build_text_encoder


def save_finetuned_weights(
    position_embedding,
    token_embedding,
    position_embedding_weights_path="./models/position_embedding_weights",
    token_embedding_weights_path="./models/token_embedding_weights",
):
    finetuned_position_weights = position_embedding.get_weights()
    finetuned_token_weights = token_embedding.get_weights()
    np.save(position_embedding_weights_path, finetuned_position_weights)
    np.save(token_embedding_weights_path, finetuned_token_weights)


def load_finetuned_weights(
    position_embedding_weights_path, token_embedding_weights_path, placeholder_token
):
    position_embedding_weights = np.load(position_embedding_weights_path)
    token_embedding_weights = np.load(token_embedding_weights_path)

    stable_diffusion = keras_cv.models.StableDiffusion()
    stable_diffusion.tokenizer.add_tokens(placeholder_token)
    stable_diffusion = build_text_encoder(
        stable_diffusion, position_embedding_weights, token_embedding_weights
    )
    return stable_diffusion
