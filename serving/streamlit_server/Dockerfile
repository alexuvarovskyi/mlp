FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip install awscli

EXPOSE 8501

COPY . /app
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

RUN aws s3 cp s3://mlp-data-2024/rtdetr_model/ ./rtdetr_model --recursive

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--", "--model_path", "./rtdetr_model"]