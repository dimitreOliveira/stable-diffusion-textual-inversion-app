lint:
	sudo isort ./
	sudo black ./

train:
	python train.py

app:
	python app.py

jupyter:
	jupyter lab