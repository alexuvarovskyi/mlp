# Setup
Make sure you have installed kind and kubectl.

After this run

```bash
kind create cluster 
```

After cluster is created, run installation of `Kubeflow Pipelines`

```bash
export PIPELINE_VERSION=2.3.0

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"
```

After installation is done, you can access Kubeflow Pipelines dashboard by running port forwarding. preferablu tun in `tmux` or `screen` session.

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```


Then go to http://127.0.0.1:8080/


To get access from python run the following command

```bash
pip install kfp
```