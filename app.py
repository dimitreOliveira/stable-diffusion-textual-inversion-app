import os

import gradio as gr

from src.utils import load_finetuned_weights

server_port = os.environ.get("SERVER_PORT", 7861)
server_name = os.environ.get("SERVER_NAME", "0.0.0.0")
prompt_token = os.environ.get("TOKEN", "<custom-token>")
token_embedding_weights_path = os.environ.get(
    "TOKEN_WEIGHTS", "./models/token_embedding_weights.npy"
)
position_embedding_weights_path = os.environ.get(
    "POSITION_WEIGHTS", "./models/position_embedding_weights.npy"
)

print(f'Inversed token used: "{prompt_token}"')
print(f'Loading token embedding weights from: "{token_embedding_weights_path}"')
print(f'Loading position embedding weights from: "{position_embedding_weights_path}"')
stable_diffusion = load_finetuned_weights(
    position_embedding_weights_path, token_embedding_weights_path, prompt_token
)


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
