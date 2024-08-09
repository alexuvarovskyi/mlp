FROM python:3.12-alpine
COPY app/ /app
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["flask", "run", "--host=0.0.0.0", "--port=5050"]