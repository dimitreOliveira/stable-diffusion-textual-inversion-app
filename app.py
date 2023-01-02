import keras_cv

from app.dataset import get_dataset, get_simple_dataset
from app.model import (add_new_token, build_text_encoder,
                       setup_model_for_training)
from app.params import (EPOCHS, MAX_PROMPT_LENGTH, group_prompts, group_urls,
                        initializer_token, output_folder, placeholder_token,
                        single_prompts, single_urls)
from app.training import (get_image_encoder, get_noise_scheduler, get_trainer,
                          setup_trainer)
from app.utils import save_finetuned_weights

stable_diffusion = keras_cv.models.StableDiffusion()

stable_diffusion.tokenizer.add_tokens(placeholder_token)

single_ds = get_simple_dataset(
    single_urls,
    single_prompts,
    placeholder_token,
    stable_diffusion.tokenizer,
    MAX_PROMPT_LENGTH,
)

group_ds = get_simple_dataset(
    group_urls,
    group_prompts,
    placeholder_token,
    stable_diffusion.tokenizer,
    MAX_PROMPT_LENGTH,
)

train_ds = get_dataset(single_ds, group_ds)

old_position_weights, new_weights = add_new_token(stable_diffusion, initializer_token)
stable_diffusion = build_text_encoder(
    stable_diffusion, old_position_weights, new_weights
)
stable_diffusion = setup_model_for_training(stable_diffusion)

image_encoder = get_image_encoder(stable_diffusion)

noise_scheduler = get_noise_scheduler()

trainer = get_trainer(
    stable_diffusion,
    image_encoder,
    noise_scheduler,
    MAX_PROMPT_LENGTH,
    placeholder_token,
)

trainer = setup_trainer(trainer, train_ds, EPOCHS)

trainer.fit(train_ds, epochs=EPOCHS)

save_finetuned_weights(
    stable_diffusion.text_encoder.layers[2].token_embedding,
    stable_diffusion.text_encoder.layers[2].position_embedding,
    position_embedding_weights_path=f"./models/{output_folder}/position_embedding_weights",
    token_embedding_weights_path=f"./models/{output_folder}/token_embedding_weights",
)
