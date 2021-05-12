# Dockerfile GPU
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y apt-utils wget

RUN pip install --upgrade pip
RUN pip install transformers \
    flask \
    Flask-Cors \
    parlai \
    waitress

WORKDIR /app
COPY . .

EXPOSE 8000

RUN python model_download.py blender
RUN python model_download.py dialogpt
RUN python model_download.py gptneo

CMD ["python", "tests/run_openchat.py"]
