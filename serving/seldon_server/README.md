# Seldon Installation

```bash
git clone https://github.com/SeldonIO/seldon-core --branch=v2


docker build \
 --build-arg AWS_ACCESS_KEY_ID=key  \
 --build-arg AWS_SECRET_ACCESS_KEY="secret_key" \
 -t usr/seldon_app:latest .
docker run -p 5001:5000 usr/seldon_app:latest REST
```

Than run the following command to install the seldon-core:

```bash
python make_request.py --image_path /path/to/image.jpg
```


As output there will be a dict with predicitons:
