import wandb
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Upload model to wandb')
    parser.add_argument('--project', type=str, required=True, help='wandb project name')
    parser.add_argument('--entity', type=str, required=True, help='wandb entity name')
    parser.add_argument('--name', type=str, required=True, help='wandb run name')
    parser.add_argument('--model_dir', type=str, required=True, help='model directory')
    parser.add_argument('--artifact_name', type=str, required=True, help='artifact name')
    parser.add_argument('--artifact_type', type=str, default='model', help='artifact type')
    return parser.parse_args()

def upload_model_to_wandb(
    project: str,
    entity: str,
    name: str,
    model_dir: str,
    artifact_name: str,
    artifact_type: str = "model"
):
    wandb.init(project=project, entity=entity, name=name)

    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    for root, _, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            artifact.add_file(file_path, name=os.path.relpath(file_path, model_dir))

    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    upload_model_to_wandb(args.project, args.entity, args.name, args.model_dir, args.artifact_name, args.artifact_type)