FROM python:3.10

RUN apt-get update
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y libwebp-dev
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip


RUN pip3 install torch torchvision

COPY save_checkpoints/multi_resnet50_kd ./save_checkpoints

COPY dataset dataset
COPY models models

COPY train.py .

CMD ["python", "train.py"]