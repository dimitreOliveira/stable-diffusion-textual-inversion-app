import json
import logging
import os
from typing import List

import gradio as gr
import numpy as np
import requests
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

server_port = int(os.environ.get("SERVER_PORT", 7861))
server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
prompt_token = os.environ.get("TOKEN", "<token>")
text_encoder_url = os.environ.get(
    "TEXT_ENCODER_URL", "http://localhost:8501/v1/models/text_encoder:predict"
)
diffusion_model_url = os.environ.get(
    "DIFFUSION_MODEL_URL", "http://localhost:8501/v1/models/diffusion_model:predict"
)
decoder_url = os.environ.get(
    "DECODER_URL", "http://localhost:8501/v1/models/decoder:predict"
)

max_prompt_length = int(os.environ.get("MAX_PROMPT_LENGTH", 77))
padding_token = int(os.environ.get("PADDING_TOKEN", 49407))
batch_size = int(os.environ.get("BATCH_SIZE", 1))
num_steps = int(os.environ.get("NUM_STEPS", 1))

tokenizer = SimpleTokenizer()
tokenizer.add_tokens(prompt_token)

logger.info(f'Inversed token used: "{prompt_token}"')

logger.info(f'text_encoder_url: "{text_encoder_url}"')
logger.info(f'diffusion_model_url: "{diffusion_model_url}"')
logger.info(f'decoder_url: "{decoder_url}"')
logger.info(f'server_port: "{server_port}"')
logger.info(f'server_name: "{server_name}"')


def predict_rest(json_data: str, url: str) -> np.ndarray:
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    rest_outputs = np.array(response["predictions"])
    return rest_outputs


def text_encoder_fn(input_prompt: str) -> np.ndarray:
    tokens = tokenizer.encode(input_prompt)
    tokens = tokens + [padding_token] * (max_prompt_length - len(tokens))

    json_tokens = json.dumps(
        {
            "signature_name": "serving_default",
            "instances": [{"tokens": tokens, "batch_size": batch_size}],
        }
    )
    encoded_text = predict_rest(json_tokens, text_encoder_url)
    return encoded_text


def diffusion_model_fn(encoded_text: List[dict]) -> np.ndarray:
    json_encoded_text = json.dumps(
        {
            "signature_name": "serving_default",
            "instances": [
                {
                    "context": encoded_text[0]["context"],
                    "unconditional_context": encoded_text[0]["unconditional_context"],
                    # "num_steps": num_steps,
                    # "batch_size": batch_size,
                }
            ],
        }
    )
    latents = predict_rest(json_encoded_text, diffusion_model_url)
    return latents


def decoder_fn(latents: List[np.ndarray]) -> np.ndarray:
    json_latents = json.dumps(
        {
            "signature_name": "serving_default",
            "instances": [{"latent": latents[0].tolist()}],
        }
    )
    decoded_images = predict_rest(json_latents, decoder_url)
    return decoded_images


def generate_fn(input_prompt: str) -> np.ndarray:
    logger.info(f'input_prompt: "{input_prompt}"')
    encoded_text = text_encoder_fn(input_prompt)
    logger.info(f'encoded_text: "{encoded_text}"')
    return encoded_text
    # latents = diffusion_model_fn(encoded_text)
    # logger.info(f'encoded_text: "{latents}"')
    # decoded_images = decoder_fn(latents)
    # logger.info(f'encoded_text: "{decoded_images}"')
    # return decoded_images


iface = gr.Interface(
    fn=generate_fn,
    title="Textual Inversion",
    description="Textual Inversion Demo",
    article="Note: Keras-cv uses lazy intialization, so the first use will be slower while the model is initialized.",
    inputs=gr.Textbox(
        label="Prompt",
        show_label=False,
        max_lines=2,
        placeholder="Enter your prompt",
        elem_id="input-prompt",
    ),
    outputs=gr.Image(),
)

if __name__ == "__main__":
    app, local_url, share_url = iface.launch(
        server_port=server_port,
        server_name=server_name,
    )
