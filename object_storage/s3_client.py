import boto3
import os

class S3Client:
    def __init__(self, access_key, secret_key, endpoint_url='http://localhost:9000'):
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint_url
            )
            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint_url
            )
        except Exception as e:
            print("Credentials not available or incomplete.")

    def create_bucket(self, bucket_name):
        try:
            response = self.s3.create_bucket(Bucket=bucket_name)
            print(f'Bucket {bucket_name} created successfully.')
            return response
        except Exception as e:
            print(f'Failed to create bucket {bucket_name}: {e}')

    def upload_file(self, file_name, bucket_name, object_name=None):
        if object_name is None:
            object_name = file_name
        try:
            self.s3.upload_file(file_name, bucket_name, object_name)
            print(f'File {file_name} uploaded to {bucket_name}/{object_name}')
        except Exception as e:
            print(f'Failed to upload file {file_name} to {bucket_name}: {e}')

    def read_file(self, bucket_name, object_name):
        try:
            response = self.s3.get_object(Bucket=bucket_name, Key=object_name)
            print(f'File {object_name} read from bucket {bucket_name}.')
            return response['Body'].read()
        except Exception as e:
            print(f'Failed to read file {object_name} from bucket {bucket_name}: {e}')

    def delete_file(self, bucket_name, object_name):
        try:
            self.s3.delete_object(Bucket=bucket_name, Key=object_name)
            print(f'File {object_name} deleted from bucket {bucket_name}.')
        except Exception as e:
            print(f'Failed to delete file {object_name} from bucket {bucket_name}: {e}')

    def delete_bucket(self, bucket_name):
        try:
            self.s3.delete_bucket(Bucket=bucket_name)
            print(f'Bucket {bucket_name} deleted successfully.')
        except Exception as e:
            print(f'Failed to delete bucket {bucket_name}: {e}')

    def download_file(self, bucket_name, object_name, file_name):
        try:
            self.s3.download_file(bucket_name, object_name, file_name)
            print(f'File {object_name} downloaded from {bucket_name} to {file_name}.')
        except Exception as e:
            print(f'Failed to download file {object_name} from bucket {bucket_name}: {e}')


    def download_directory(self, bucket_name, s3_folder, local_dir=None):
        """
        Download the contents of a folder directory
        Args:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        bucket = self.s3_resource.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=s3_folder):
            print(f"Downloading {obj.key}")
            target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target)


if __name__ == "__main__":


    access_key = 'minio'
    secret_key = 'minio123'
    bucket_name = 'mlp-test'
    file_path = 'object_storage/README.md'

    client = S3Client(access_key, secret_key)

    # Create a bucket
    client.create_bucket(bucket_name)
    client.upload_file(file_path, bucket_name)

    content = client.read_file(bucket_name, file_path)
    print(content.decode('utf-8'))

    client.delete_file(bucket_name, file_path)
    client.delete_bucket(bucket_name)
