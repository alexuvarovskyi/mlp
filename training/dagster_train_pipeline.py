import os
from dagster import asset, resource, Definitions, Config, Field, Shape, job
from utils.s3_client import S3Client
import yaml
from src.train import train

def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

# Define a resource for the S3Client to manage its configuration
@resource(config_schema={"aws_access_key_id": str, "aws_secret_access_key": str})
def s3_client_resource(init_context):
    access_key = init_context.resource_config["aws_access_key_id"]
    secret_key = init_context.resource_config["aws_secret_access_key"]
    return S3Client(access_key, secret_key)

# Define a config schema for the asset
class S3Config(Config):
    bucket_name: str
    s3_folder: str
    local_dir: str

# Define an asset to download the data from S3
@asset(
    config_schema=Shape({
        "bucket_name": Field(str, description="Name of the S3 bucket"),
        "s3_folder": Field(str, description="Path to the S3 folder"),
        "local_dir": Field(str, description="Local directory to download files to")
    }),
    required_resource_keys={"s3_client"}
)
def download_data_from_S3(context) -> str:
    """Asset to download data from S3 using values from config.yaml."""
    # Accessing config from Dagster context
    bucket_name = context.op_config["bucket_name"]
    s3_folder = context.op_config["s3_folder"]
    local_dir = context.op_config["local_dir"]

    # Logging to ensure paths are being processed correctly
    context.log.info(f"Downloading data from S3 bucket: {bucket_name}, folder: {s3_folder}")
    context.log.info(f"Local directory: {local_dir}")

    # Download data from S3
    try:
        context.resources.s3_client.download_directory(bucket_name, s3_folder, local_dir)
        context.log.info(f"Data successfully downloaded to {local_dir}")
    except Exception as e:
        context.log.error(f"Failed to download data: {str(e)}")
        raise

    # Ensure the local_dir is returned
    if local_dir:
        return local_dir
    else:
        context.log.error("Download failed, local_dir is None")
        raise ValueError("Local directory for downloaded data is None")

@asset(
    config_schema=Shape({
        "config_file": Field(str, description="Path to the config file"),
        "model_output_dir": Field(str, description="Directory to save the trained model")
    })
)
def train_model(context, download_data_from_S3: str) -> str:
    """Asset to train a model using a config file and downloaded data."""
    # Accessing config from Dagster context
    config_file = context.op_config["config_file"]
    model_output_dir = context.op_config["model_output_dir"]

    # Load the YAML config
    config = load_yaml(config_file)

    # Update the data paths in the config
    config['data']['train']['data_path'] = os.path.join(download_data_from_S3, 'data')
    config['data']['train']['ann_path'] = os.path.join(download_data_from_S3, 'labels.json')
    config['data']['val']['data_path'] = os.path.join(download_data_from_S3, 'data')
    config['data']['val']['ann_path'] = os.path.join(download_data_from_S3, 'labels.json')

    # Optionally, you can customize other config parameters if needed
    config['data']['size'] = 640
    config['model']['output_dir'] = model_output_dir
    config['model']['hyps']['output_dir'] = model_output_dir
    
    # Start the training process
    train(config)

    # Log the path to the trained model
    context.log.info(f"Model trained and saved to {model_output_dir}")

    # Return the model output directory as the asset output
    return model_output_dir

@asset(
    required_resource_keys={"s3_client"},
    config_schema=Shape({
        "bucket_name": Field(str, description="S3 Bucket Name"),
        "s3_folder": Field(str, description="S3 Directory to save the model"),
        "model_name": Field(str, description="Name of the model file"),
    })
)
def upload_model(context, train_model: str) -> None:
    """Asset to upload a trained model to S3 using values from config.yaml."""
    # Access config from Dagster context
    bucket_name = context.op_config["bucket_name"]
    s3_folder = context.op_config["s3_folder"]
    model_name = context.op_config["model_name"]

    model_dir = train_model  # Directory where the trained model was saved

    context.log.info(f"Uploading model from {model_dir} to S3 bucket {bucket_name}")

    # Use the s3_client resource
    client: S3Client = context.resources.s3_client

    # Upload model to S3
    try:
        s3_upload_dir = os.path.join(s3_folder, model_name)
        client.upload_dir(model_dir, bucket_name, s3_upload_dir)
        context.log.info(f"Model successfully uploaded to {s3_upload_dir}/{model_name}")
    except Exception as e:
        context.log.error(f"Failed to upload model: {str(e)}")
        raise



@job(
    resource_defs={
        "s3_client": s3_client_resource,
    }
)
def training_pipeline():
    local_dir = download_data_from_S3()
    model_dir = train_model(local_dir)
    upload_model(model_dir)


defs = Definitions(
    assets=[download_data_from_S3, train_model, upload_model],  # Add the assets to your repository
    jobs=[training_pipeline],  # Add the job to your repository
    resources={"s3_client": s3_client_resource},  # Define your resources
)