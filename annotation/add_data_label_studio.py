import os
from label_studio_sdk import Client
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Create a new Label Studio project and upload data from a local directory')
    parser.add_argument('--api_url', type=str, required=True, help='Label Studio API URL')
    parser.add_argument('--api_key', type=str, required=True, help='Label Studio API key')
    parser.add_argument('--project_name', type=str, required=True, help='Name of the new project')
    parser.add_argument('--local_dir', type=str, required=True, help='Local directory with images to upload')
    return parser.parse_args()


def create_annotation_project(api_url, api_key, project_name, local_dir):
    # Connect to Label Studio API
    ls = Client(url=api_url, api_key=api_key)
    
    # Create a new project
    project = ls.start_project(
        title=project_name,
        label_config='''
        <View>
            <Image name="image" value="$image"/>
            <RectangleLabels name="label" toName="image">
                <Label value="person" background="blue"/>
                <Label value="car" background="red"/>
            </RectangleLabels>
        </View>
        '''
    )

    print(f"Created project: {project_name} (ID: {project.id})")

    # Upload data from the local directory
    for filename in os.listdir(local_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(local_dir, filename)
            project.import_tasks([{'image': f'data:image/png;base64,{_read_image_as_base64(image_path)}'}])

    print("Data uploaded successfully!")

def _read_image_as_base64(image_path):
    import base64
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')



if __name__ == "__main__":
    args = parse_args()

    create_annotation_project(
        api_url=args.api_url,
        api_key=args.api_key,
        project_name=args.project_name,
        local_dir=args.local_dir
    )
