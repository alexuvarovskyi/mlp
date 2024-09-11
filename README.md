# mlp
ML in Production 2024


## Simple App
### How to run 

docker pull alexuvarovskii/flask-app
docker run -p 5050:5050 alexuvarovskii/flask-app

## DVC
Install DVC
```
pip install dvc
```


Init DVC
```
mkdir data
cd data
dvc init --subdir
```

Export AWS credentials
```
export AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
```

Create bucket on S3, you can do thies via AWS console or using AWS CLI
```
aws s3api create-bucket --bucket <YOUR_BUCKET_NAME>
```

Configure remote storage
```
dvc remote add -d storage s3://mlp-data-2024/data
```

```
git commit -m "Initialize DVC"
```

Add data
```

cp -r /path/to/data data/
dvc add data

git add data.dvc data/.gitignore

git commit -m "Add data"
git push
```

Pull data
```
git clone https://github.com/alexuvarovskyi/mlp
cd data
dvc pull
```
