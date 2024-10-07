from s3_client import S3Client
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Pull data from S3")
    parser.add_argument("--access_key", type=str, help="AWS access key")
    parser.add_argument("--secret_key", type=str, help="AWS secret key")
    parser.add_argument("--s3_bucket_name", type=str, help="S3 bucket name")
    parser.add_argument("--s3_dir_name", type=str, help="S3 object name")
    parser.add_argument("--local_dir_name", type=str, help="Local directory name")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    s3_client = S3Client(args.access_key, args.secret_key)
    s3_client.download_directory(args.s3_bucket_name, args.s3_dir_name, args.local_dir_name)
    print("Data has been downloaded to {}".format(args.local_dir_name))
