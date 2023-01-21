docker run -it --rm -p 8501:8501 \
    -p 8500:8500 \
    --mount type=bind,source=$(pwd)/serving_config.config,target=/models/models.config \
    -v "$(pwd)/models/example/:/models/" emacski/tensorflow-serving:latest-linux_arm64 \
    --model_config_file=/models/models.config \
    --model_config_file_poll_wait_seconds=60