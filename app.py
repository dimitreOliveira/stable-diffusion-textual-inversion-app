import logging
import os

import gradio as gr
import keras_cv
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")
server_port = os.environ.get("SERVER_PORT", 7861)
server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
prompt_token = os.environ.get("TOKEN", "<custom-token>")
text_encoder_path = os.environ.get(
    "TEXT_ENCODER", "./models/example/text_encoder/keras"
)

logger.info(f'Inversed token used: "{prompt_token}"')
logger.info(f'Loading text encoder from: "{text_encoder_path}"')

stable_diffusion = keras_cv.models.StableDiffusion()
stable_diffusion.tokenizer.add_tokens("<token>")

loaded_text_encoder_ = tf.keras.models.load_model(text_encoder_path)
stable_diffusion._text_encoder = loaded_text_encoder_
stable_diffusion._text_encoder.compile(jit_compile=True)


def generate_fn(input_prompt: str):
    generated = stable_diffusion.text_to_image(prompt=input_prompt, batch_size=1)
    return generated[0]


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
