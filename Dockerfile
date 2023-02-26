FROM python:3.10

WORKDIR /app
COPY ["requirements.txt", "train.py", "params.yaml", "Makefile", "./"]
RUN pip install -r requirements.txt
COPY src/ src/