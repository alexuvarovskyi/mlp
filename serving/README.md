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