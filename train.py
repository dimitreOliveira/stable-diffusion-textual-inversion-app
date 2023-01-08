import os
import keras_cv
import tensorflow as tf

from src.dataset import get_images_from_urls, assemble_dataset
from src.model import add_new_token, build_text_encoder, setup_model_for_training
from src.params import (
    EPOCHS,
    MAX_PROMPT_LENGTH,
    group_datasets_dir,
    group_prompts,
    group_urls,
    initializer_token,
    token,
    single_datasets_dir,
    single_prompts,
    single_urls,
    models_dir,
)
from src.trainer import (
    get_image_encoder,
    get_noise_scheduler,
    get_trainer,
    setup_trainer,
)
from src.utils import save_finetuned_weights

if single_urls:
    get_images_from_urls(single_urls, output_subdir=single_datasets_dir)

if group_urls:
    get_images_from_urls(group_urls, output_subdir=group_datasets_dir)

stable_diffusion = keras_cv.models.StableDiffusion()
stable_diffusion.tokenizer.add_tokens(token)

# Initialize with empty dataset
train_ds = tf.data.Dataset.from_tensor_slices([])

if os.path.exists(single_datasets_dir) and os.listdir(single_datasets_dir):
    train_ds = assemble_dataset(
        single_datasets_dir,
        single_prompts,
        token,
        stable_diffusion.tokenizer,
        MAX_PROMPT_LENGTH,
    )

if os.path.exists(group_datasets_dir) and os.listdir(group_datasets_dir):
    if train_ds.cardinality().numpy() > 0:
        group_ds = assemble_dataset(
            group_datasets_dir,
            group_prompts,
            token,
            stable_diffusion.tokenizer,
            MAX_PROMPT_LENGTH,
        )
        train_ds = train_ds.concatenate(group_ds)
    else:
        train_ds = assemble_dataset(
            group_datasets_dir,
            group_prompts,
            token,
            stable_diffusion.tokenizer,
            MAX_PROMPT_LENGTH,
        )

assert train_ds.cardinality().numpy() > 0, "There is no data for training."
train_ds = train_ds.batch(1).shuffle(
    train_ds.cardinality(), reshuffle_each_iteration=True
)

old_position_weights, new_weights = add_new_token(stable_diffusion, initializer_token)
stable_diffusion = build_text_encoder(
    stable_diffusion, old_position_weights, new_weights
)

stable_diffusion = setup_model_for_training(stable_diffusion)

image_encoder = get_image_encoder(stable_diffusion)

text_encoder_layers = len(stable_diffusion.text_encoder.trainable_weights)
image_encoder_layers = len(stable_diffusion.diffusion_model.trainable_weights)
image_decoder_laeyrs = len(stable_diffusion.decoder.trainable_weights)
assert text_encoder_layers == 1, f"Text encoder should have only 1 trainable layer but is {text_encoder_layers}."
assert image_encoder_layers == 0, f"Image encoder should have no trainable layers but is {image_encoder_layers}."
assert image_decoder_laeyrs == 0, f"Image decoder should have no trainable layers but is {image_decoder_laeyrs}."

noise_scheduler = get_noise_scheduler()

trainer = get_trainer(
    stable_diffusion,
    image_encoder,
    noise_scheduler,
    MAX_PROMPT_LENGTH,
    token,
)

trainer = setup_trainer(trainer, train_ds, EPOCHS)

trainer.fit(train_ds, epochs=EPOCHS)

save_finetuned_weights(
    stable_diffusion.text_encoder.layers[2].position_embedding,
    stable_diffusion.text_encoder.layers[2].token_embedding,
    models_dir,
)
