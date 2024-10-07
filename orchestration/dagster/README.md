# Dagster

## Installation

```bash
pip install -r requirements.txt

mkdir dagster_home
export DAGSTER_HOME=$PWD/dagster_home
```

Make sure to update configs with rewuired values and keys:
infer_config.yaml
train_config.yaml

Run UI
```bash
dagster dev -f ../../training/dagster_train_pipeline.py
```


In another terminal run:
```bash
dagster job execute --job training_pipeline -c ./train_config.yaml -f ../../training/dagster_train_pipeline.py
dagster job execute --job inference_pipeline -c ./infer_config.yaml -f ../../training/dagster_infer_pipeline.py
```
