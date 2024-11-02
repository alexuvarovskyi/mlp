## Triton serving

```bash
pip install nvidia-pyindex
pip install tritonclient[all]
```


Run to convert model to onnx
```bash
python convert_model.py
```


Pull triton server image
```bash
docker pull nvcr.io/nvidia/tritonserver:23.01-py3
```
```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ./redetr_onnx:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models
```



Test the server
```bash
curl -X POST "http://127.0.0.1:8000/predict/?threshold=0.5" \
    -H "accept: application/json" \
    -F "image=@/path/to/your/image.jpg"
```

