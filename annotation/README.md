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

## Check Data
We used a Cleanlab as a tool for check dataset correctness
install 
```bash
pip install -r requirements.txt
```
Then we need to create predictions for the dataset.


It can be done by running `inference_yolov8.py` For this you need to pass in the script the data path and the path to the model weights
```bash
python inference_yolov8.py \
--image_dir /path/to/images/dir \
--model_path /path/to/model/or/weights \
--output_path /path/to/save/coco_predictions/
```


Then run the `check_data.py` script the path to the data and the path to the predictions and run it
```bash
python check_data.py \
--coco_ann_path /path/to/gt/coco/annotations.json/ \
--preds_path /path/to/predictions.json
--classes_mapping 0=person 1=car \ # according to the dataset
--data_dir /path/to/images/
```
And as the output it will show as visualizations of the data with mistakes, and the avarage image quality annotation score

For our self labelled dataset we got the `Average quality score 0.8439354990876237`, that is not very bad, but we can improve it by adding more data checks.<br>
Most of the mistakes are in the localization of the objects in the crowded scenes, and the quality of the images is good.
