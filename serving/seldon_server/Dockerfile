# Use a base image with Python and the required libraries
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /app

RUN pip install markupsafe==2.0.1 awscli

ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
RUN pip install ray[serve]

RUN aws s3 cp s3://mlp-data-2024/rtdetr_model/ ./rtdetr_model --recursive
ENTRYPOINT ["seldon-core-microservice", "ObjectDetectionModel", "--service-type", "MODEL", "--persistence", "0"]
