import os
from dagster import asset, resource, Definitions, job, Config, Field, Shape
from utils.s3_client import S3Client
from src.inference import inference_model  # Ensure this module exists for inference


@resource(config_schema={"aws_access_key_id": str, "aws_secret_access_key": str})
def s3_client_resource(init_context):
    access_key = init_context.resource_config["aws_access_key_id"]
    secret_key = init_context.resource_config["aws_secret_access_key"]
    return S3Client(access_key, secret_key)


@asset(
    config_schema=Shape({
        "bucket_name": Field(str, description="Name of the S3 bucket"),
        "s3_folder": Field(str, description="Path to the S3 folder"),
        "local_dir": Field(str, description="Local directory to download files to")
    }),
    required_resource_keys={"s3_client"}
)
def load_data_from_S3(context) -> str:
    """Asset to download data from S3."""
    bucket_name = context.op_config["bucket_name"]
    s3_folder = context.op_config["s3_folder"]
    local_dir = context.op_config["local_dir"]

    context.log.info(f"Downloading data from S3: {bucket_name}/{s3_folder} to {local_dir}")

    try:
        context.resources.s3_client.download_directory(bucket_name, s3_folder, local_dir)
        context.log.info(f"Data successfully downloaded to {local_dir}")
    except Exception as e:
        context.log.error(f"Failed to download data: {str(e)}")
        raise

    return local_dir


@asset(
    config_schema=Shape({
        "confidence": Field(float, description="Confidence threshold for inference"),
        "model_dir": Field(str, description="Directory containing the trained model"),
        "data_dir": Field(str, description="Directory containing the data for inference"),
        "ann_save_path": Field(str, description="Path to save the inference results")
    })
)
def run_inference(context, load_data_from_S3: str) -> str:
    """Asset to run inference on the data."""
    confidence = context.op_config["confidence"]
    model_dir = context.op_config["model_dir"]
    data_dir = load_data_from_S3  # Path to downloaded data
    ann_save_path = context.op_config["ann_save_path"]

    context.log.info("Running inference...")
    
    inference_model(model_dir, data_dir, ann_save_path, confidence)

    context.log.info(f"Inference results saved to {ann_save_path}")
    return ann_save_path


@asset(
    required_resource_keys={"s3_client"},
    config_schema=Shape({
        "bucket_name": Field(str, description="S3 Bucket Name"),
        "s3_folder": Field(str, description="S3 Directory to save results"),
        "ann_save_path": Field(str, description="Path to saved inference results"),
    })
)
def upload_inference_results(context, run_inference: str) -> None:
    """Asset to upload inference results to S3."""
    bucket_name = context.op_config["bucket_name"]
    s3_folder = context.op_config["s3_folder"]
    ann_save_path = run_inference  

    context.log.info(f"Uploading inference results from {ann_save_path} to S3 bucket {bucket_name}")

    client: S3Client = context.resources.s3_client

    try:
        client.upload_dir(ann_save_path, bucket_name, s3_folder)
        context.log.info(f"Inference results successfully uploaded to {s3_folder} in {bucket_name}")
    except Exception as e:
        context.log.error(f"Failed to upload results: {str(e)}")
        raise


@job(
    resource_defs={
        "s3_client": s3_client_resource,
    }
)
def inference_pipeline():
    local_dir = load_data_from_S3()
    ann_save_path = run_inference(local_dir)
    upload_inference_results(ann_save_path)

