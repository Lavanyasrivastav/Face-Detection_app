# Production image for batch/image-mode face mesh processing.
# NOTE: webcam mode needs a physical camera device passed through to the
# container (e.g. `--device=/dev/video0`) and an X11/display forward on
# Linux hosts; it is not practical to run headless in most container setups.
FROM python:3.11-slim

# OpenCV/MediaPipe need these system libs even in "headless" use.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY main.py .

RUN mkdir -p /app/data/input /app/data/output

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
