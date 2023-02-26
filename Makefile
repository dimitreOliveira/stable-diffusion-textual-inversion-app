lint:
	isort ./
	black ./

test:
	pytest tests

train:
	docker run -it \
	-v $(pwd)/models/:/app/models/ \
	-v $(pwd)/datasets/:/app/datasets/ \
	-v /Users/$(whoami)/.keras/:/root/.keras/ \
	-e TEXT_ENCODER=models/doll_cat/text_encoder/keras \
	-t textual-inversion-app python train.py

app:
	docker run -it -p 7861:7861 \
	-v $(pwd)/models/:/app/models/ \
	-v /Users/$(whoami)/.keras/:/root/.keras/ \
	-e SERVER_PORT=7861 \
	-e SERVER_NAME="0.0.0.0" \
	-e TOKEN="<token>" \
	-e TEXT_ENCODER="models/example/text_encoder/keras" \
	-t app python app.py

app_serving:
	docker run -it -p 7861:7861 \
	-v $(pwd)/models/:/app/models/ \
	-v /Users/$(whoami)/.keras/:/root/.keras/ \
	-e SERVER_PORT=7861 \
	-e SERVER_NAME="0.0.0.0" \
	-e TOKEN="<token>" \
	-e TEXT_ENCODER_URL="http://localhost:8501/v1/models/text_encoder:predict" \
	-e DIFFUSION_MODEL_URL="http://localhost:8501/v1/models/diffusion_model:predict" \
	-e DECODER_URL="http://localhost:8501/v1/models/decoder:predict" \
	-t app python app_serving.py

serve:
	sh scripts/start_tf_serving_macos.sh

jupyter:
	jupyter lab

build:
	docker build -t textual-inversion-app .
	docker build -t app ./gradio_app/.
	