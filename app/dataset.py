import numpy as np
import tensorflow as tf
from keras_cv import layers as cv_layers

from app.model_utils import pad_embedding


def assemble_image_dataset(urls):
    # Fetch all remote files
    files = [tf.keras.utils.get_file(origin=url) for url in urls]

    # Resize images
    resize = tf.keras.layers.Resizing(height=512, width=512, crop_to_aspect_ratio=True)
    images = [tf.keras.utils.load_img(img) for img in files]
    images = [tf.keras.utils.img_to_array(img) for img in images]
    images = np.array([resize(img) for img in images])

    # The StableDiffusion image encoder requires images to be normalized to the
    # [-1, 1] pixel value range
    images = images / 127.5 - 1

    # Create the tf.data.Dataset
    image_dataset = tf.data.Dataset.from_tensor_slices(images)

    # Shuffle and introduce random noise
    image_dataset = image_dataset.shuffle(50, reshuffle_each_iteration=True)
    image_dataset = image_dataset.map(
        cv_layers.RandomCropAndResize(
            target_size=(512, 512),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(1.0, 1.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    image_dataset = image_dataset.map(
        cv_layers.RandomFlip(mode="horizontal"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return image_dataset


def assemble_text_dataset(prompts, placeholder_token, tokenizer, max_prompt_length):
    prompts = [prompt.format(placeholder_token) for prompt in prompts]
    embeddings = [tokenizer.encode(prompt) for prompt in prompts]
    embeddings = [
        np.array(pad_embedding(embedding, tokenizer, max_prompt_length))
        for embedding in embeddings
    ]
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    return text_dataset


def assemble_dataset(urls, prompts, placeholder_token, tokenizer, max_prompt_length):
    image_dataset = assemble_image_dataset(urls)
    text_dataset = assemble_text_dataset(
        prompts, placeholder_token, tokenizer, max_prompt_length
    )
    # the image dataset is quite short, so we repeat it to match the length of the
    # text prompt dataset
    image_dataset = image_dataset.repeat()
    # we use the text prompt dataset to determine the length of the dataset.  Due to
    # the fact that there are relatively few prompts we repeat the dataset 5 times.
    # we have found that this anecdotally improves results.
    text_dataset = text_dataset.repeat(5)
    return tf.data.Dataset.zip((image_dataset, text_dataset))


def get_simple_dataset(urls, prompts, placeholder_token, tokenizer, max_prompt_length):
    ds = assemble_dataset(
        urls=urls,
        prompts=prompts,
        placeholder_token=placeholder_token,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )
    return ds


def get_dataset(single_ds, group_ds):
    concat_ds = single_ds.concatenate(group_ds)
    ds = concat_ds.batch(1).shuffle(
        concat_ds.cardinality(), reshuffle_each_iteration=True
    )
    return ds
