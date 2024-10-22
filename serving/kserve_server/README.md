## Kserve Server


```bash
kubectl apply -f inference-service.yaml
```

Check the Status
```bash
kubectl get inferenceservice
```

Get IP from the output

Port forward the service
```bash
kubectl port-forward svc/fastapi-object-detection-predictor 8000:80
```

Test the service
```bash
curl -X POST "http:/localhost:8000/predict/" -F "image=@path/to/your/image.jpg" -F "threshold=0.5"
```

The output should be like this:
```json
{
    "predictions": [
        {
            "class": "person",
            "score": 0.98,
            "box": [100.0, 150.0, 200.0, 300.0]
        },
        ...
    ]
}
```