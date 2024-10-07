import kfp
import os
import kubernetes as k8s

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Model, Output, OutputPath, InputPath, PipelineTask

from dagster import asset
from kfp import kubernetes


# pull data
# train model
# save model


IMAGE = "alexuvarovskii/training_mlp:latest"
WANDB_PROJECT = "huggingface"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


@dsl.component(base_image=IMAGE)
def load_data_from_s3(s3_access_key: str, s3_secret_key: str, s3_bucket: str, s3_folder: str, local_dir: OutputPath()):
    from utils.s3_client import S3Client
    import os
    
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Initialize S3 client and download data
    client = S3Client(s3_access_key, s3_secret_key)
    client.download_directory(bucket_name=s3_bucket, s3_folder=s3_folder, local_dir=local_dir)
    
    print(f"Data from {s3_bucket}/{s3_folder} downloaded to {local_dir}")


@dsl.component(base_image=IMAGE)
def train_model(config_file: str, data_dir: InputPath(), model_output: OutputPath()):
    from src.train import load_yaml, train
    import torch
    # Load configuration file
    config = load_yaml(config_file)
    
    # Update data paths in config
    config['data']['train']['data_path'] = f"{data_dir}/data"
    config['data']['train']['ann_path'] = f"{data_dir}/labels.json"
    config['data']['val']['data_path'] = f"{data_dir}/data"
    config['data']['val']['ann_path'] = f"{data_dir}/labels.json"
    config['data']['size'] = 640
    config['model']['output_dir'] = model_output
    print(config['model']['hyps']['dataloader_num_workers'])

    train(config)

@dsl.component(base_image=IMAGE)
def save_model(s3_access_key: str, s3_secret_key: str, s3_bucket: str, s3_dir: str, model_name: str, model_dir: InputPath()):
    from utils.s3_client import S3Client
    import os
    
    # Initialize S3 client
    client = S3Client(s3_access_key, s3_secret_key)
    model_dir = str(model_dir)
    s3_dir = os.path.join(s3_dir, model_name)
    
    # Upload model to S3
    client.upload_dir(model_dir, s3_bucket, s3_dir)
    
    print(f"Model saved to {s3_dir}/{model_name}")


@dsl.pipeline(
    name="Model Training Pipeline",
    description="Pipeline for training a model using data from S3 and saving the trained model back to S3"
)
def training_pipeline(s3_access_key: str, s3_secret_key: str, s3_bucket: str, s3_folder: str, config_file: str, model_name: str):

    # Step 1: Load data from S3
    load_data_step = load_data_from_s3(
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        s3_folder=s3_folder,
    )

    # Step 2: Train the model
    train_model_step = train_model(
        config_file=config_file,
        data_dir=load_data_step.outputs['local_dir']
    )

    train_model_step.set_env_variable(
        name="WANDB_API_KEY", value=WANDB_API_KEY
    )
    train_model_step.set_env_variable(
        name="WANDB_PROJECT", value=WANDB_PROJECT
    )


    # # Step 3: Save the trained model
    save_model_step = save_model(
        model_dir=train_model_step.outputs['model_output'],
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        s3_dir='data_mlp',
        model_name=model_name
    )

    # # Define dependencies
    train_model_step.after(load_data_step)
    save_model_step.after(train_model_step)

if __name__ == "__main__":
    s3_access_key = AWS_ACCESS_KEY_ID
    s3_secret_key = AWS_SECRET_ACCESS_KEY
    s3_bucket = "mlp-data-2024"
    s3_folder = "data_mlp/train_50"
    config_file = "/app/src/config.yaml"
    model_name = "rtdetr_test"

    kfp.compiler.Compiler().compile(training_pipeline, 'training_pipeline.yaml')

    client = kfp.Client()

    client.create_run_from_pipeline_func(
        training_pipeline,
        arguments={
            's3_access_key': s3_access_key,
            's3_secret_key': s3_secret_key,
            's3_bucket': s3_bucket,
            's3_folder': s3_folder,
            'config_file': config_file,
            'model_name': model_name
        }
    )
