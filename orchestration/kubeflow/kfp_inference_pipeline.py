import kfp
import os
import kubernetes as k8s
# from kfp import kubernetes as k8s

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Model, Output, OutputPath, InputPath, PipelineTask

from dagster import asset
from kfp import kubernetes

IMAGE = "alexuvarovskii/training_mlp:latest"


# load data
# run inference
# save results


@dsl.component(base_image=IMAGE)
def load_data_from_s3(s3_access_key: str, s3_secret_key: str, s3_bucket: str, s3_folder: str, local_dir: OutputPath()):
    from utils.s3_client import S3Client
    import os

    os.makedirs(local_dir, exist_ok=True)

    client = S3Client(s3_access_key, s3_secret_key)
    client.download_directory(bucket_name=s3_bucket, s3_folder=s3_folder, local_dir=local_dir)

    print(f"Data from {s3_bucket}/{s3_folder} downloaded to {local_dir}")


@dsl.component(base_image=IMAGE)
def infer_model(confidence: float, model_dir: InputPath(), data_dir: InputPath(),  ann_save_path: OutputPath()):
    from src.inference import inference_model
    from pathlib import Path
    import os

    print(os.listdir(str(model_dir)))

    inference_model(str(model_dir), str(data_dir), str(ann_save_path), confidence)


@dsl.component(base_image=IMAGE)
def upload_annotations(s3_access_key: str, s3_secret_key: str, s3_bucket: str, s3_folder: str, ann_save_path: InputPath()):
    from utils.s3_client import S3Client
    import os

    client = S3Client(s3_access_key, s3_secret_key)
    client.upload_dir(ann_save_path, s3_bucket, s3_folder)


g
@dsl.pipeline(
    name="Model Inference Pipeline",
    description="A pipeline to run inference on a dataset using a pre-trained model",
)
def inferece_pipeline(
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket: str,
    s3_model_dir: str,
    s3_data_dir: str,
    confidence: float,
    ann_save_path: str,
):
    load_data_step = load_data_from_s3(
        s3_access_key=s3_access_key, 
        s3_secret_key=s3_secret_key, 
        s3_bucket=s3_bucket, 
        s3_folder=s3_data_dir
    )
    load_model_step = load_data_from_s3(
        s3_access_key=s3_access_key, 
        s3_secret_key=s3_secret_key, 
        s3_bucket=s3_bucket, 
        s3_folder=s3_model_dir
    )
    inference_task = infer_model(
        confidence=confidence, 
        model_dir=load_model_step.outputs['local_dir'], 
        data_dir=load_data_step.outputs['local_dir'], 
    )
    upload_annotations_task = upload_annotations(
        s3_access_key=s3_access_key, 
        s3_secret_key=s3_secret_key, 
        s3_bucket=s3_bucket, 
        s3_folder=ann_save_path, 
        ann_save_path=inference_task.outputs['ann_save_path']
    )

    # load_model_step.after(load_data_step)
    inference_task.after(load_data_step).after(load_model_step)
    upload_annotations_task.after(inference_task)


if __name__ == "__main__":
    s3_access_key = "AKIASI5UH4GGQ2MFVT4G"
    s3_secret_key = "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"
    s3_bucket = "mlp-data-2024"
    model_dir = "data_mlp/test_model/"
    data_dir = "data_mlp/val_50/"

    ann_save_path = "data_mlp/ann_save_path/"

    confidence = 0.2

    kfp.compiler.Compiler().compile(inferece_pipeline, "inference_pipeline.yaml")
    client = kfp.Client()

    client.create_run_from_pipeline_func(
        inferece_pipeline, 
        arguments={
            "s3_access_key": s3_access_key,
            "s3_secret_key": s3_secret_key,
            "s3_bucket": s3_bucket,
            "s3_model_dir": model_dir,
            "s3_data_dir": data_dir,
            "confidence": confidence,
            "ann_save_path": ann_save_path,
        }
    )
