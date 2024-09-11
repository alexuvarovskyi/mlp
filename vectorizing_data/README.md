## Data 


For vectorization of our dataset we will use [Milvus DB](https://milvus.io/), 'cause I already have experience with it and CLIP model by OpenAI for embedding data.

So we will predefine all necessary steps and variables in the script, and all we need to do is to run it and pass the path to images or image

### Milvus DB Installation

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

Install Milvus client and requirements:

```bash
pip install -r requirements.txt

# verify installation
python -c "from pymilvus import Collection"
```

Next we need to run embedder script. It will create the DB, collection and load embeddings to it.

```bash
python setup_milvus.py --images_path /path/to/images
```

To make search in DB, we need to run search script where we pass the path to image we want to search for, and it will output us the top 8 similar images by cosine similarity.

```bash
python query_milvus.py --img_path /path/to/image
```