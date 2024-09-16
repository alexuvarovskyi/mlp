import pytest
from s3_client import S3Client


@pytest.fixture
def s3_client() -> S3Client:

    access_key = 'minio'
    secret_key = 'minio123'
    client = S3Client(access_key, secret_key)
    
    client.create_bucket('test-bucket')
    
    yield client

def test_create_bucket(s3_client):
    response = s3_client.create_bucket('new-test-bucket')
    assert response['ResponseMetadata']['HTTPStatusCode'] == 200

def test_upload_file(s3_client):
    file_content = b"Hello, world!"
    file_name = "test_file.txt"
    bucket_name = "test-bucket"
    
    with open(file_name, 'wb') as f:
        f.write(file_content)
    
    s3_client.upload_file(file_name, bucket_name)
    
    s3 = s3_client.s3
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    assert response['Body'].read() == file_content

def test_read_file(s3_client):
    file_content = b"Hello, world!"
    file_name = "test_file.txt"
    bucket_name = "test-bucket"
    
    s3 = s3_client.s3
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=file_content)
    
    content = s3_client.read_file(bucket_name, file_name)
    assert content == file_content

def test_delete_file(s3_client):
    file_content = b"Hello, world!"
    file_name = "test_file.txt"
    bucket_name = "test-bucket"
    
    s3 = s3_client.s3
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=file_content)
    
    s3_client.delete_file(bucket_name, file_name)
    
    with pytest.raises(s3.exceptions.NoSuchKey):
        s3.get_object(Bucket=bucket_name, Key=file_name)

def test_delete_bucket(s3_client):
    bucket_name = "test-bucket"
    
    s3_client.delete_bucket(bucket_name)
    
    s3 = s3_client.s3
    with pytest.raises(s3.exceptions.ClientError):
        s3.head_bucket(Bucket=bucket_name)


if __name__ == "__main__":
    pytest.main(['-v', __file__])
