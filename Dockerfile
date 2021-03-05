FROM python3

RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y apt-utils wget

RUN pip install --upgrade pip
RUN pip install transformers && \
    pip install flask && \
    pip install waitress && \
    pip install Flask-Cors && \
    pip install openchat

WORKDIR /app
COPY . .

RUN cd demo

EXPOSE 80

CMD ["python3", "start_demo.py"]
