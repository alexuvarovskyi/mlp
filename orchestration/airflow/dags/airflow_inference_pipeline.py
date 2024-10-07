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
    dag_id="inference_dag",
) as dag:

    download_data_operator = KubernetesPodOperator(
        name='download_data_from_s3',
        image=IMAGE,
        cmds=[
            "python",
            "utils/pull_data.py",
            "--access_key", AWS_ACCESS_KEY_ID,
            "--secret_key", AWS_SECRET_ACCESS_KEY,
            "--s3_bucket_name", "mlp-data-2024",
            "--s3_dir_name", "data_mlp/val_50",
            "--local_dir_name", "/tmp/data",
        ],
        task_id='download_data_from_s3',
        in_cluster=False,
        is_delete_operator_pod=False,
        startup_timeout_seconds=600,
        namespace="default2",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    download_model_operator = KubernetesPodOperator(
        name='download_model_from_s3',
        image=IMAGE,
        cmds=[
            "python",
            "utils/pull_data.py",
            "--access_key", AWS_ACCESS_KEY_ID,
            "--secret_key", AWS_SECRET_ACCESS_KEY,
            "--s3_bucket_name", "mlp-data-2024",
            "--s3_dir_name", "rtdetr_test",
            "--local_dir_name", "/tmp/model",
        ],
        task_id='download_model_from_s3',
        volumes=[volume],
        volume_mounts=[volume_mount],
        in_cluster=False,
        is_delete_operator_pod=False,
        startup_timeout_seconds=600,
        namespace="default2",
    )

    inference_operator = KubernetesPodOperator(
        name='inference_model',
        image=IMAGE,
        cmds=[
            "python",
            "src/infer_model_cli.py",
            "--model_path", "/tmp/model",
            "--data_path", "/tmp/data",
            "--ann_save_path", "/tmp/ann",
        ],
        task_id='inference_model',
        in_cluster=False,
        is_delete_operator_pod=False,
        startup_timeout_seconds=600,
        namespace="default2",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    upload_results_operator = KubernetesPodOperator(
        name='upload_results_to_s3',
        image=IMAGE,
        cmds=[
            "python",
            "utils/upload_data.py",
            "--access_key", AWS_ACCESS_KEY_ID,
            "--secret_key", AWS_SECRET_ACCESS_KEY,
            "--s3_bucket_name", "mlp-data-2024",
            "--s3_dir_name", "results_airflow",
            "--local_dir_name", "/tmp/ann",
        ],
        task_id='upload_results_to_s3',
        volumes=[volume],
        volume_mounts=[volume_mount],
        in_cluster=False,
        is_delete_operator_pod=False,
        startup_timeout_seconds=600,
        namespace="default2",
    )

    download_data_operator >> download_model_operator >> inference_operator >> upload_results_operator


