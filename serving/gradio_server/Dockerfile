FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install awscli

COPY . /app

ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

RUN aws s3 cp s3://mlp-data-2024/rtdetr_model/ ./rtdetr_model --recursive


EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python3", "gradio_ui.py", "--model_path", "./rtdetr_model"]