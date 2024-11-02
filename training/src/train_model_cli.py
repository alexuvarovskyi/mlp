from train import train
import argparse
import yaml

def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(path, data):
    with open(path, "w") as file:
        yaml.dump(data, file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--train_labels_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--val_labels_path", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


if __name__ == "__main__":
    print("STARTING TRAINING")
    args = parse_args()
    config = load_yaml(args.config)
    config['output_dir'] = args.output_dir
    config['data']['train']['data_path'] = args.train_data_path
    config['data']['train']['ann_path'] = args.train_labels_path
    config['data']['val']['data_path'] = args.val_data_path
    config['data']['val']['ann_path'] = args.val_labels_path
    config['data']['size'] = args.imgsz
    train(config)
    print("TRAINING COMPLETE")
