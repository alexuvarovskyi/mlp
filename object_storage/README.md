## Minio Setup

### Local installation
Be sure hou have homebrew installed 
For MAC arm64 (M1) chip, use the following command to install minio:

```bash
brew install minio/stable/minio
```

To run minio server, use the following command:

```bash
mkdir -p /minio/data # here data will be stored
export MINIO_CONFIG_ENV_FILE=/etc/default/minio
minio server --console-address :9001 /minio/data
```

After this, in your browser, go to `http://localhost:9001` to access the minio dashboard.

The api endpoint will be `http://localhost:9000`

### Docker installation
To run minio server in docker, use the following command:

```bash
mkdir -p /minio/data

export MINIO_ROOT_USER=minio
export MINIO_ROOT_PASSWORD=minio123

docker run \
   -p 9000:9000 \
   -p 9001:9001 \
   --name minio \
   -v ~/minio/data:/data \
   -e "MINIO_ROOT_USER=$MINIO_ROOT_USER" \
   -e "MINIO_ROOT_PASSWORD=$MINIO_ROOT_PASSWORD" \
   quay.io/minio/minio server /data --console-address ":9001"
```


### K8s installation
```bash
mikikube start

mkdir -p /minio/data

kubectl apply -f ./minio_resources/minio-pod.yaml

kubectl port-forward minio 9000 9000
kubectl port-forward minio 9001 9001
```

## S3 Client for Minio
To use minio client, install reqirements
   
```bash
pip install -r requirements.txt
```

To use the client, use the following code:

```python
from s3_client import S3Client
access_key = 'minio'
secret_key = 'minio123'
bucket_name = 'mlp-test'
file_path = 'README.md'

client = S3Client(access_key, secret_key)

client.create_bucket(bucket_name)
client.upload_file(file_path, bucket_name)

content = client.read_file(bucket_name, file_path)
print(content.decode('utf-8'))

client.delete_file(bucket_name, file_path)
client.delete_bucket(bucket_name)
```

To run tests, use the following command:

```bash
pytest -ss s3_client_tests.py```
