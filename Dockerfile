FROM nvidia/cuda:12.2.0-base-ubuntu20.04
# Install Python 3.11 from the official Python image
FROM python:3.11-slim
# Use the official CUDA 12.2 base image
# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive
# Install necessary packages and CUDA toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Add CUDA repository key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# Add CUDA repository to apt sources list
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda.list
# Install CUDA Toolkit 12.2
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-2 \
    && rm -rf /var/lib/apt/lists/*
# Set the CUDA environment variables
ENV PATH /usr/local/cuda-12.2/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
# Verify the CUDA installation
RUN apt-get update && apt-get install git-lfs -y && \
    apt-get install -y libsndfile1 ffmpeg &&  \
    apt-get clean && rm -rf /var/lib/apt/lists/* &&  \
    git lfs install
RUN nvcc --version

WORKDIR /
RUN git clone https://huggingface.co/nvidia/parakeet-rnnt-0.6b nvidia/parakeet-rnnt-0.6b
RUN git clone https://huggingface.co/nvidia/tts_en_fastpitch nvidia/tts_en_fastpitch
RUN git clone https://huggingface.co/nvidia/tts_hifigan nvidia/tts_hifigan
RUN pip3 install torch==2.4.0
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
RUN pip3 install Cython==3.0.11
RUN pip3 install packaging==24.1
RUN pip3 install nemo_toolkit['all']==1.23.0

ARG HUGGINGFACE_TOKEN
RUN git clone https://${HUGGINGFACE_TOKEN}:${HUGGINGFACE_TOKEN}@huggingface.co/pyannote/speaker-diarization-3.1 pyannote/speaker-diarization-3.1

COPY requirements.txt .
RUN pip install -r requirements.txt

#RUN pip install Cython
#RUN pip install nemo_toolkit['all']

RUN git clone https://${HUGGINGFACE_TOKEN}:${HUGGINGFACE_TOKEN}@huggingface.co/pyannote/segmentation-3.0 /models/segmentation-3.0
RUN git clone https://${HUGGINGFACE_TOKEN}:${HUGGINGFACE_TOKEN}@huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM /models/wespeaker-voxceleb-resnet34-LM
RUN mv /models/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin /models/wespeaker-voxceleb-resnet34-LM/pyannote_model.bin
# Modify the configuration file to replace only the first occurrence
RUN sed -i 's|segmentation: pyannote/segmentation-3.0|segmentation: /models/segmentation-3.0/pytorch_model.bin|g' /pyannote/speaker-diarization-3.1/config.yaml
RUN sed -i 's|embedding: pyannote/wespeaker-voxceleb-resnet34-LM|embedding: /models/wespeaker-voxceleb-resnet34-LM/pyannote_model.bin|g' /pyannote/speaker-diarization-3.1/config.yaml



ENV HF_DATASETS_CACHE="/runpod-volume/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/hub"
ENV TORCH_CACHE="/runpod-volume/torch"

COPY src/ .


CMD [ "python3", "-u", "/app.py" ]