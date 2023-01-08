# StableDiffusion - Textual-Inversion

![](https://i.imgur.com/KqEeBsM.jpg)

This code was heavily inspired by the [Teach StableDiffusion new concepts via Textual Inversion](https://keras.io/examples/generative/fine_tune_via_textual_inversion/) Keras code example from Luke Wood.

TODO list:

- Improve docstrings
- Improve Tests
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
- TFRecord support
- Replace prints by Loggers
- Docker support
- Docker-compose support
- TF Serving support
- KubeFlow support
  - Training pipeline
  - TF Serving instances
- Vertex AI support
  - Training pipeline
  - TF Serving instances