## Annotation Tool Deploy

We will use [Label Studio](https://labelstud.io/)

Installation
Be sure you have [Docker](https://docs.docker.com/engine/install/) installed on your machine.

Following the official installation guide, you can run the following command to start Label Studio:

```bash
export LABEL_STUDIO_DATA=/path/to/dir/label/studio/will/store/data/

docker run -it -p 8080:8080 -v $LABEL_STUDIO_DATA:/label-studio/data heartexlabs/label-studio:latest
```
This will pull and run Docker container with Label Studio on your machine. You can access Label Studio at http://localhost:8080.

Then you have to sign up and get an API key
Go to http://localhost:8080/user/account
And copy Access API token

To upload an annotaion project you can use `add_data_label_studio.py` script

For this you have to install LabelStudio SDK
```bash
pip install label-studio==1.13.0 label-studio-sdk==1.0.4
```
Or
```bash
pip install -r requirements.txt
```

Then you can run the script
```bash
python add_data_label_studio.py \
--api_url http://localhost:8080 \
--api_key <YOUR_API_KEY> \
--project_name 'Init Object Detection Project' \
--local_dir /path/to/images/data/
```