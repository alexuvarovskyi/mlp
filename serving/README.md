# Streamlit


## Local Deployment
```bash
streamlit run streamlit_ui.py -- --model_path path/to/model
```


## Build Container
```bash
docker build \
 --build-arg AWS_ACCESS_KEY_ID=key  \
 --build-arg AWS_SECRET_ACCESS_KEY="secret_key" \
 -t streamlit_app:latest .
```

Run:
```bash
docker run -it --rm -p 8501:8501  streamlit_app:latest
```


# Gradio

## Local Deployment
```bash
python gradio_ui.py --model_path path/to/model
```

## Build Container
```bash
docker build \
    --build-arg AWS_ACCESS_KEY_ID=key \
    --build-arg AWS_SECRET_ACCESS_KEY="secret_key" \
     -t gradio_app:latest .
```

Run:
```bash
docker run -it --rm -p 7860:7860 gradio_app:latest
```


# FastAPI
## Local Deployment
```bash
pip install -r requirements.txt
python fastapi_server.py
```

## Build Container
```bash
docker build \
    --build-arg AWS_ACCESS_KEY_ID=key \
    --build-arg AWS_SECRET_ACCESS_KEY="secret_key" \
    -t fastapi_app:latest .
```

Run:
```bash
docker run -it --rm -p 8000:8000 fastapi_app:latest
```

How to make a request:
```bash
curl -X POST "http://localhost:8000/predict/" -F "image=@/path/to/image.jpg" -F "threshold=0.5"
```


## Kubernetes Deployment
```bash
kubectl apply -f k8s/deployment.yaml 
kubectl apply -f k8s/service.yaml
kubectl port-forward <pod_name> 8000:8000
```