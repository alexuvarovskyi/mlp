import yaml
import torch
import argparse
import supervision as sv

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForObjectDetection, AutoImageProcessor

from dataloader import PyTorchDetectionDataset, collate_fn, get_augmentations
from evaluation import MAPEvaluator
from transformers.trainer_utils import default_compute_objective


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

def wandb_hp_space(trial):
    sweep_configuration = {
        "name": "sweep_try",
        "method": "random",
        "metric": {"goal": "minimize", "name": "validation_loss"},
        "parameters": {
            "learning_rate": {"min": 5e-6, "max": 5e-3},
            "batch_size": {"values": [4, 8, 16]},
            "epochs": {"values": [5, 10, 15]},
            "optimizer": {"values": ["adam", "sgd"]},
        },
    }
    return sweep_configuration


def train(config: dict[str, any]):
    init_model = config['model']['name']
    device = torch.device(
        config['model']['device'] if torch.cuda.is_available() else "cpu"
    )

    processor = AutoImageProcessor.from_pretrained(
        init_model,
        do_resize=True,
        size={"width": config['data']['size'], "height": config['data']['size']},
    )

    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=config['data']['train']['data_path'],
        annotations_path=config['data']['train']['ann_path'],
    )

    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=config['data']['val']['data_path'],
        annotations_path=config['data']['val']['ann_path'],
    )

    pytorch_dataset_train = PyTorchDetectionDataset(
        ds_train, processor, transform=get_augmentations(target="train")
    )


    pytorch_dataset_valid = PyTorchDetectionDataset(
        ds_valid, processor, transform=get_augmentations(target="val")
    )

    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for id, label in enumerate(ds_train.classes)}

    eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.01, id2label=id2label)
    
    def model_init(trial):
        return AutoModelForObjectDetection.from_pretrained(
            init_model,
            id2label=id2label,
            label2id=label2id,
            anchor_image_size=None,
            ignore_mismatched_sizes=True,
        ).to(device)

    model = AutoModelForObjectDetection.from_pretrained(
        init_model,
        id2label=id2label,
        label2id=label2id,
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    ).to(device)

    training_args = TrainingArguments(
        metric_for_best_model="eval_map",
        max_grad_norm=0.1,
        greater_is_better=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="wandb",
        **config['model']['hyps']
    )

    trainer = Trainer(
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
        model_init=model_init
    )

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=10,
        compute_objective=default_compute_objective
    )

    print(best_trial)


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml(args.config)
    train(config)