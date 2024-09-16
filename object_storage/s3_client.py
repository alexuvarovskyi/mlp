import boto3


class S3Client:
    def __init__(self, access_key, secret_key, endpoint_url='http://localhost:9000'):
        try:
            self.s3 = boto3.client(
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
