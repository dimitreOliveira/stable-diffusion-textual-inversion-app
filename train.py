import os

import keras_cv
import tensorflow as tf

from src.dataset import assemble_dataset, get_images_from_urls
from src.model import add_new_token, build_text_encoder, setup_model_for_training
from src.parser import DatasetParams, ModelParams, Params, TrainParams
from src.trainer import (
    get_image_encoder,
    get_noise_scheduler,
    get_trainer,
    setup_trainer,
)
from src.utils import save_finetuned_weights

params = Params("./params.yaml")
modelParams: ModelParams = params.model
trainParams: TrainParams = params.train
datasetParams: DatasetParams = params.dataset

if datasetParams.single_urls:
    get_images_from_urls(
        datasetParams.single_urls, output_subdir=datasetParams.single_datasets_dir
    )

if datasetParams.group_urls:
    get_images_from_urls(
        datasetParams.group_urls, output_subdir=datasetParams.group_datasets_dir
    )

stable_diffusion = keras_cv.models.StableDiffusion()
stable_diffusion.tokenizer.add_tokens(modelParams.token)

# Initialize with empty dataset
train_ds = tf.data.Dataset.from_tensor_slices([])

if os.path.exists(datasetParams.single_datasets_dir) and os.listdir(
    datasetParams.single_datasets_dir
):
    train_ds = assemble_dataset(
        datasetParams.single_datasets_dir,
        datasetParams.single_prompts,
        modelParams.token,
        stable_diffusion.tokenizer,
        modelParams.max_prompt_length,
    )

if os.path.exists(datasetParams.group_datasets_dir) and os.listdir(
    datasetParams.group_datasets_dir
):
    if train_ds.cardinality().numpy() > 0:
        group_ds = assemble_dataset(
            datasetParams.group_datasets_dir,
            datasetParams.group_prompts,
            modelParams.token,
            stable_diffusion.tokenizer,
            modelParams.max_prompt_length,
        )
        train_ds = train_ds.concatenate(group_ds)
    else:
        train_ds = assemble_dataset(
            datasetParams.group_datasets_dir,
            datasetParams.group_prompts,
            modelParams.token,
            stable_diffusion.tokenizer,
            modelParams.max_prompt_length,
        )

assert train_ds.cardinality().numpy() > 0, "There is no data for training."
train_ds = train_ds.batch(1).shuffle(
    train_ds.cardinality(), reshuffle_each_iteration=True
)

old_position_weights, new_weights = add_new_token(
    stable_diffusion, modelParams.initializer_token
)
stable_diffusion = build_text_encoder(
    stable_diffusion, old_position_weights, new_weights
)

stable_diffusion = setup_model_for_training(stable_diffusion)

image_encoder = get_image_encoder(stable_diffusion)

text_encoder_layers = len(stable_diffusion.text_encoder.trainable_weights)
image_encoder_layers = len(stable_diffusion.diffusion_model.trainable_weights)
image_decoder_laeyrs = len(stable_diffusion.decoder.trainable_weights)
assert (
    text_encoder_layers == 1
), f"Text encoder should have only 1 trainable layer but there is {text_encoder_layers}."
assert (
    image_encoder_layers == 0
), f"Image encoder should have no trainable layers but there is {image_encoder_layers}."
assert (
    image_decoder_laeyrs == 0
), f"Image decoder should have no trainable layers but there is {image_decoder_laeyrs}."

noise_scheduler = get_noise_scheduler()

trainer = get_trainer(
    stable_diffusion,
    image_encoder,
    noise_scheduler,
    modelParams.max_prompt_length,
    modelParams.token,
)

trainer = setup_trainer(trainer, train_ds, trainParams.epochs)

trainer.fit(train_ds, epochs=trainParams.epochs)

save_finetuned_weights(
    stable_diffusion.text_encoder.layers[2].position_embedding,
    stable_diffusion.text_encoder.layers[2].token_embedding,
    datasetParams.models_dir,
)
