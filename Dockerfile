# AdLens - container for Hugging Face Spaces or any Docker host.
# Full guide: deploy/DEPLOY.md
#
#   docker build -t adlens .
#   docker run -p 8000:7860 --env-file .env adlens
#
# Build options (docker build --build-arg NAME=value):
#   PRELOAD_MODELS=1      download the models during the build, so the first visitor doesn't wait
#   LITE_MODE=0           1 = no Stable Diffusion + small flan-t5 rewriter (~3-4 GB RAM)
#   DISABLE_GENERATIVE=1  Stable Diffusion off by default: on a CPU it takes minutes per image
#   TORCH_INDEX           CPU wheels by default; a GPU host uses https://download.pytorch.org/whl/cu121
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libGL / glib are needed by OpenCV (used by EasyOCR and rembg)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# PyTorch first, from the CPU index (~200 MB instead of several GB of CUDA libraries)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --user --index-url ${TORCH_INDEX} torch torchvision
COPY --chown=user Requirements.txt .
RUN pip install --user -r Requirements.txt

COPY --chown=user . .

ARG PRELOAD_MODELS=1
ARG LITE_MODE=0
ARG DISABLE_GENERATIVE=1
ENV LITE_MODE=${LITE_MODE} DISABLE_GENERATIVE=${DISABLE_GENERATIVE} \
    HOST=0.0.0.0 PORT=7860 SIGNUP_MODE=invite COPY_SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python deploy/download_models.py; fi

EXPOSE 7860
CMD ["python", "03_web_app.py"]
