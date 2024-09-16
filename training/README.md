
# Training Models

We are using transformers library that uses wandb logger by default. To log your training runs, you need to have a wandb account and be logged in. You can do this by running the following command:

```bash
wandb login
```
Then we can export our project name to the environment variable `WANDB_PROJECT`:

```bash
export WANDB_PROJECT=test_test_test
```

Also login to huggingface-cli:

```bash
huggingface-cli login
```

Before running training we need to modify the config file `config.yaml` in the training folder. You need to specify the model name, dataset name, and other hyperparameters if you want. Also pass the train and val data paths in the config file.

To run training, you need to run the following command:

```bash
python src/train.py --config <path/to/config.yaml>
```

To evaluate the model, you need to run the following command:

```bash
python src/eval.py --checkpoint <path_to_checkpoint> --data_path <path_to_data> --ann_path <path_to_ann>
```

## Hyperparameter tuning
We already rpedefined some hyperparameter arguments in train file
To run tuning run the following command:

```bash
python src/hyp_search.py --config <path/to/config.yaml>
```

# Model Card
---
library_name: transformers

language:
- en

base_model: PekingU/rtdetr_r18vd

datasets:
- COCO10k

model-index:
- name: rtdetr_r18vd_train_pascal_480

experiment-logs:
- wandb: https://wandb.ai/uvarovskii/huggingface/runs/wdn54gu7?nw=nwuseruvarovskii
---



## Model description

Model trained for three classes: `person`, `car`, `pet`. Used a truncated version of the COCO10k dataset, which contains 10k images with bounding boxes for 3 classes. The model was trained for 3 epochs with a learning rate of 5e-05 and a batch size of 2. The model was fine-tuned from the `rt-detr` model.


## Training and evaluation data

Data could be found here: [COCO10k] s3://mlp-data-2024/data_mlp/

## Training procedure

For training we have to specify training config and just run the following command:

```bash
python src/train.py --config <path/to/config.yaml>
```


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-04
- train_batch_size: 24
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 24
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 300
- num_epochs: 50

### Framework versions

- Transformers 4.44.2
- Pytorch 2.4.0+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1

## Evaluation results
| Model    |mAP |
| -------- | ------- |
| Our  | 89*    |
| Yolov5 | 52.2     |
| RtDETR    | 54.8    |


* mAP was calculated on the COCO10k dataset, and this value is estimated on full dataset

# Tests
Testing out module contains in three parts:
- Testing the code
- Testing the model
- Testing the dataset