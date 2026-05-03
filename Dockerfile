FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    diffusers>=0.30.0 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    "gradio==5.25.0" \
    sentencepiece>=0.2.0 \
    protobuf>=3.20.0

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
