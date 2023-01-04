# StableDiffusion - Textual-Inversion

![](https://i.imgur.com/KqEeBsM.jpg)

This code was heavily inpired by the [Teach StableDiffusion new concepts via Textual Inversion](https://keras.io/examples/generative/fine_tune_via_textual_inversion/) Keras code example from Luke Wood.

TODO list:

- Makefile commands
  - Linting
  - Formatter
  - Run app
  - Start Jupyter Lab instance
  - Create images
  - Tests
- Improve docstrings
- Tests
- Images inputs
  - [x] From urls
  - [ ] From images
  - [ ] Single and/or group images
- Git pipeline hooks
  - Linting
  - Formatter
  - Tests
- Update README
  - Initial README release
  - Stable Diffusion license disclaimer
  - Sections
    - Description
    - Training
    - Inference (notebook/app)
- Custom TensorFlow signature to embed the image generation process
- Gradio APP
- StableDiffusionV2 support
- YAML support
  - Parse parameter arguments form the .yaml file
- Docker support
- Docker-compose support
- TF Serving support
- KubeFlow support
  - Training pipeline
  - TF Serving instances
- Vertex AI support
  - Training pipeline
  - TF Serving instances