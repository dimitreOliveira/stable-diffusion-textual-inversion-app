FROM python:3.10

ENV PYTHONUNBUFFER ED True
ENV APP_HOME /app

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR $APP_HOME
COPY ["train.py", "app.py", "app_serving.py", "params.yaml", "Makefile", "./"]
COPY scripts/ scripts/
COPY src/ src/