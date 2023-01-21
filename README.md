# StableDiffusion - Textual-Inversion
![](https://i.imgur.com/KqEeBsM.jpg)

This code was heavily inspired by the [Teach StableDiffusion new concepts via Textual Inversion](https://keras.io/examples/generative/fine_tune_via_textual_inversion/) Keras code example from Luke Wood.

# Usage
This repository has a collection of Makefile commands that covers all the functionalities provided.

```bash
make train
```
Runs the textual inversion training script, you may want to customize the `params.yaml` file.

```bash
make app
```
Starts the Gradio app, this version of the Gradio app also load the model for inference.

```bash
make app_serving
```
Starts the Gradio app, this version of the Gradio app used the TFServing endpoints for inference.

```bash
make serve
```
Starts the TFServing instance to serve the three models from StableDiffusion, you may want to customize the `serving_config.config` file.

```bash
make lint
```
Applies code linting and formating.

```bash
make test
```
Runs unit tests.

```bash
make jupyter
```
Starts the JupyterLab instance.


# References
- Keras-io blog [Teach StableDiffusion new concepts via Textual Inversion](https://keras.io/examples/generative/fine_tune_via_textual_inversion/)
- Various ways of serving Stable Diffusion [keras-sd-serving](https://github.com/deep-diver/keras-sd-serving)
- [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://textual-inversion.github.io/)

TODO list:

- Improve docstrings
- Improve Tests
- Git pipeline hooks
  - Linting
  - Formatter
  - Tests
- Update README
  - Stable Diffusion license disclaimer
  - Sections
    - Description
    - Training
    - Inference (notebook/app)
- TFRecord support
- Docker support
- Docker-compose support
- KubeFlow support
  - Training pipeline
  - TF Serving instances
- Vertex AI support
  - Training pipeline
  - TF Serving instances