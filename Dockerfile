FROM  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libgl1-mesa-glx libglib2.0-0 libxext6 build-essential -y


COPY ./requirements.txt /install/requirements.txt
RUN pip install --upgrade --force-reinstall -r /install/requirements.txt

WORKDIR /code

# jupyter lab --no_browser true --ip 0.0.0.0 --allow-root

# docker run -it -rm --gpus "device=all" --ipc=host -v $(pwd):/code -v <dir>/:/<dir>  swinir