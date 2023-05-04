FROM python:3.10

RUN apt-get update
RUN apt-get install -y libjpeg-dev 
RUN apt-get install -y libpng-dev 
RUN apt-get install -y libwebp-dev 
RUN apt-get install -y python3-pip 
RUN pip3 install --upgrade pip


RUN pip3 install torch torchvision

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY save_checkpoints/multi_resnet50_kd ./save_checkpoints

COPY main.py .
COPY models.py .
COPY update.py .

CMD ["python3", "main.py"]