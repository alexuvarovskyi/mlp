import os
import json
import pytest
from dotenv import load_dotenv
from object_storage.s3_client import S3Client
from botocore.exceptions import NoCredentialsError
load_dotenv()


def read_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


@pytest.fixture
def data_dir(tmpdir):
    bucket_name = 'mlp-data-2024'
    s3_key = 'data_mlp/train_50/'  # S3 object key (file path)

    s3_client = S3Client(
        access_key=os.getenv('AWS_ACCESS_KEY_ID'),
        secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=None
    )

    download_dir = tmpdir.mkdir("s3_data")
    print('download dir: ' ,download_dir)

    try:
        s3_client.download_directory(
            bucket_name=bucket_name,
            s3_folder=s3_key,
            local_dir=download_dir
        )
        yield download_dir

    except NoCredentialsError:
        pytest.fail("S3 credentials not found.")


def test_data_download(data_dir):
    print(os.listdir(data_dir))
    assert os.path.exists(data_dir)
    assert os.path.isdir(data_dir)
    assert len(os.listdir(data_dir)) > 0


def test_annoation(data_dir):
    ann_path = os.path.join(data_dir, 'labels.json')
    ann = read_json(ann_path)
    assert isinstance(ann, dict)
    assert ann.get('annotations') is not None
    assert ann.get('categories') is not None
    assert len(ann.get('annotations')) > 0
    assert os.path.exists(ann_path)
    assert ann.get('images') is not None


if __name__ == "__main__":
    pytest.main()
