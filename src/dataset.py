import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv import layers as cv_layers
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer

from src.model_utils import pad_embedding


def get_images_from_urls(
    urls: list[str], output_dir: str = ".", output_subdir: str = "datasets"
) -> None:
    """Downloads all images from the given URLs and save them to a folder."""
    for url in urls:
        tf.keras.utils.get_file(
            origin=url, cache_dir=output_dir, cache_subdir=output_subdir
        )


def load_and_decode_img(file_path):
    """Convert the compressed string to a 3D uint8 tensor."""
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img


def preprocess_img(img):
    """The StableDiffusion image encoder requires images to be normalized to the [-1, 1] pixel value range."""
    return img / 127.5 - 1


def assemble_image_dataset(dataset_dir: str) -> tf.data.Dataset:
    """Loads images and process them to create a TensorFlow dataset."""
    # Resize images
    resize_fn = tf.keras.layers.Resizing(
        height=512, width=512, crop_to_aspect_ratio=True
    )

    # Create the tf.data.Dataset
    image_dataset = (
        tf.data.Dataset.list_files(f"{dataset_dir}*")
        .map(load_and_decode_img, num_parallel_calls=tf.data.AUTOTUNE)
        .map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(50, reshuffle_each_iteration=True)
        # Data augmentation
        .map(
            cv_layers.RandomCropAndResize(
                target_size=(512, 512),
                crop_area_factor=(0.8, 1.0),
                aspect_ratio_factor=(1.0, 1.0),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            cv_layers.RandomFlip(mode="horizontal"),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    )
    return image_dataset


def assemble_text_dataset(
    prompts: list[str], token, tokenizer, max_prompt_length
) -> tf.data.Dataset:
    """Tokenize and pad all prompts and creates a TensorFlow dataset."""
    prompts = [prompt.format(token) for prompt in prompts]
    tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    padded_prompts = [
        np.array(pad_embedding(tokenized_prompt, tokenizer, max_prompt_length))
        for tokenized_prompt in tokenized_prompts
    ]
    text_dataset = tf.data.Dataset.from_tensor_slices(padded_prompts).shuffle(
        100, reshuffle_each_iteration=True
    )
    return text_dataset


def assemble_dataset(
    dataset_dir: str,
    prompts: list[str],
    token: str,
    tokenizer: SimpleTokenizer,
    max_prompt_length: int,
) -> tf.data.Dataset:
    # TODO: maybe it would be more efficient to combine all images with all prompts to have more diversity?
    """Combines the image and text datasets into an unified TensorFlow dataset."""
    image_dataset = assemble_image_dataset(dataset_dir)
    text_dataset = assemble_text_dataset(
        prompts, token, tokenizer, max_prompt_length
    )
    # the image dataset is quite short, so we repeat it to match the length of the text prompt dataset
    image_dataset = image_dataset.repeat()
    # We use the text prompt dataset to determine the length of the dataset.
    # Due to the fact that there are relatively few prompts we repeat the dataset 5 times.
    # we have found that this anecdotally improves results.
    text_dataset = text_dataset.repeat(5)
    return tf.data.Dataset.zip((image_dataset, text_dataset))
