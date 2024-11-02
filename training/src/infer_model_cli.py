from inference  import inference_model
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ann_save_path", type=str, required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.2)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference_model(args.model_path, args.data_path, args.ann_save_path)