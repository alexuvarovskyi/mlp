
# ray inference server
docker build
docker run

build container
```bash
docker build \
 --build-arg AWS_ACCESS_KEY_ID=key  \
 --build-arg AWS_SECRET_ACCESS_KEY="secret_key" \
 -t alexuvarovskii/object_detection_rayserve:latest .
```

```bash
docker run -p 8000:8000 fastapi-rayserve
```

run
```bash
curl -X POST "http://localhost:8000/predict/" -F "image=@path_to_your_image.jpg" -F "threshold=0.5"
```