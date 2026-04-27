FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY blur_detector/requirements.txt /app/blur_detector/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r /app/blur_detector/requirements.txt

COPY blur_detector /app/blur_detector

RUN mkdir -p /app/weights /app/samples

ENTRYPOINT []
CMD ["python", "blur_detector/scripts/predict.py", "--help"]
