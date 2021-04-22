# Dockerfile CPU
FROM python

RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y apt-utils wget

RUN pip install --upgrade pip
RUN pip install transformers \
    flask \
    Flask-Cors \
    waitress \
    torch \
    parlai

WORKDIR /app
COPY . .

EXPOSE 8080

CMD ["python", "tests/run_openchat.py"]
