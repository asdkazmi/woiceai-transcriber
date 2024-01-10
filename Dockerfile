FROM nvidia/cuda:11.7.1-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y wget build-essential libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.11 python3-pip

RUN apt-get update
RUN apt-get install -y ffmpeg && apt-get install git-lfs -y
RUN git lfs install

RUN apt-get update && apt-get install -y python3-dev build-essential libicu-dev libbz2-dev

WORKDIR /

RUN git clone https://huggingface.co/nvidia/parakeet-rnnt-1.1b nvidia/parakeet-rnnt-1.1b
ARG HUGGINGFACE_TOKEN
RUN git clone https://${HUGGINGFACE_TOKEN}:${HUGGINGFACE_TOKEN}@huggingface.co/pyannote/speaker-diarization-3.1 pyannote/speaker-diarization-3.1

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install Cython
RUN pip install nemo_toolkit['all']

ENV HF_DATASETS_CACHE="/runpod-volume/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/hub"
ENV TORCH_CACHE="/runpod-volume/torch"

COPY src/ .

CMD [ "python3", "-u", "/app.py" ]