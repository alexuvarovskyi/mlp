FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

# Install required packages
RUN pip install -r requirements.txt
RUN pip install awscli
RUN pip install python-multipart
RUN pip install ray[serve]  # Install Ray and Ray Serve

COPY . /app

EXPOSE 8000

ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

RUN aws s3 cp s3://mlp-data-2024/rtdetr_model/ ./rtdetr_model --recursive

CMD ["python", "fastapi_server.py"]
