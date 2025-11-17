# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

CMD ["python", "main.py", "--config-name", "conf_random8_AAAI___"]
