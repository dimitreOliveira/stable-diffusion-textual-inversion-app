lint:
	sudo isort ./
	sudo black ./

test:
	pytest tests

train:
	python train.py

app:
	python app.py

app_serving:
	python app_serving.py

serve:
	sh scripts/start_tf_serving_macos.sh

jupyter:
	jupyter lab