# Setup
## KubeFlow Pipelines Installation
Make sure you have installed kind and kubectl.

After this run

```bash
kind create cluster 
```

After cluster is created, run installation of `Kubeflow Pipelines`

```bash
export PIPELINE_VERSION=2.2.0

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

After installation is done, you can access Kubeflow Pipelines dashboard by running port forwarding. preferablu tun in `tmux` or `screen` session.

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

In another terminal run

```bash
export WANDB_API_KEY=your_wandb_api_key
export AWS_ACCESS_KEY_ID=your_aws_access_key_id
export AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

Then go to http://127.0.0.1:8080/


To get access from python run the following command

```bash
pip install requirements.txt
```

Run training pipeline
```bash
python kfp_training_pipeline.py
```

Run inference pipeline
```bash
python kfp_inference_pipeline.py
```