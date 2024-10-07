import os
from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


IMAGE = "alexuvarovskii/training_mlp:latest"
WANDB_PROJECT = "huggingface"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# STORAGE = "training-storage"
STORAGE = "airflow-pipeline-pvc"

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

volume = k8s.V1Volume(
    name=STORAGE,
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name=STORAGE
    ),
)

volume_mount = k8s.V1VolumeMount(name=STORAGE, mount_path="/tmp", sub_path=None)

with DAG(
    start_date=datetime(2023, 10, 1),
    catchup=False,
    schedule=None,
    dag_id="train_dag",
) as dag:

    pull_data_operator = KubernetesPodOperator(
        name='pull_data_from_s3',
        image=IMAGE,
        cmds=[
            "python",
            "utils/pull_data.py",
            "--access_key", AWS_ACCESS_KEY_ID,
            "--secret_key", AWS_SECRET_ACCESS_KEY,
            "--s3_bucket_name", "mlp-data-2024",
            "--s3_dir_name", "data_mlp/train_50",
            "--local_dir_name", "/tmp/data",
        ],
        task_id='pull_data_from_s3',
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default2",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    train_model_operator = KubernetesPodOperator(
        name='train_model',
        image=IMAGE,
        cmds=[
            "python",
            "src/train_model_cli.py",
            "--config", "src/config.yaml",
            "--output_dir", "/tmp/output",
            "--train_data_path", "/tmp/data/data",
            "--train_labels_path", "/tmp/data/labels.json",
            "--val_data_path", "/tmp/data/data",
            "--val_labels_path", "/tmp/data/labels.json",
        ],
        env_vars={"WANDB_PROJECT": WANDB_PROJECT, "WANDB_API_KEY": WANDB_API_KEY},
        task_id='train_model',
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default2",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    upload_model_operator = KubernetesPodOperator(
        name='upload_model_to_s3',
        image=IMAGE,
        cmds=[
            "python",
            "utils/upload_data.py",
            "--access_key", AWS_ACCESS_KEY_ID,
            "--secret_key", AWS_SECRET_ACCESS_KEY,
            "--s3_bucket_name", "mlp-data-2024",
            "--s3_dir_name", "MODEL_TEST",
            "--local_dir_name", "/tmp/output",
        ],
        task_id='upload_model_to_s3',
        in_cluster=False,
        is_delete_operator_pod=False,
        namespace="default2",
        startup_timeout_seconds=600,
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )


    pull_data_operator >> train_model_operator >> upload_model_operator
