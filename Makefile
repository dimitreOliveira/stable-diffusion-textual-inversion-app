lint:
	sudo isort ./
	sudo black ./

test:
	pytest tests

train:
	python train.py

app:
	python app.py

jupyter:
	jupyter lab