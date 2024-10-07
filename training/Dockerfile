FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . . 

CMD [ "bash" ]