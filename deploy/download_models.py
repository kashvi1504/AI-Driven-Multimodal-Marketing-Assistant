"""
deploy/download_models.py
-------------------------
Downloads every model the web app needs, so a server doesn't download
several GB while its first visitor waits. Run during the Docker build
(see Dockerfile) or once on a server:

    python deploy/download_models.py

Respects LITE_MODE and DISABLE_GENERATIVE: in lite mode it skips Qwen and
Stable Diffusion; with generative editing off it skips Stable Diffusion.
"""
import os
import sys

from huggingface_hub import HfApi, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import COPY_SENTIMENT_MODEL, DEPLOY  # noqa: E402

CONFIG_FILES = ["*.json", "*.txt", "*.model", "merges.txt", "vocab.*", "tokenizer*", "spiece.model"]


def fetch(repo, patterns=None):
    """Config + tokenizer files and ONE weight format: safetensors when the
    repo has them, else pytorch_model.bin (several of these models ship only .bin)."""
    print(f"-> {repo}", flush=True)
    if patterns is None:
        files = [s.rfilename for s in HfApi().model_info(repo).siblings]
        weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
        patterns = CONFIG_FILES + weights
    snapshot_download(repo, allow_patterns=patterns)


def main():
    if COPY_SENTIMENT_MODEL != "fine-tuned":
        fetch(COPY_SENTIMENT_MODEL)
    fetch("openai/clip-vit-base-patch32")
    fetch("umm-maybe/AI-image-detector")
    fetch("google/flan-t5-base")  # small rewriter: lite mode, and the fallback
    if not DEPLOY["lite_mode"]:
        fetch("Qwen/Qwen2.5-1.5B-Instruct")
    if DEPLOY["generative_enabled"]:
        fetch("stable-diffusion-v1-5/stable-diffusion-v1-5",
              ["model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
               "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
               "text_encoder/model.safetensors"])  # full-precision files only, not the duplicate fp16 copies

    print("-> EasyOCR models", flush=True)
    import easyocr
    easyocr.Reader(["en"], gpu=False, verbose=False)

    print("-> rembg (U2-Net) model", flush=True)
    from rembg import new_session
    new_session("u2net")
    print("All models downloaded.")


if __name__ == "__main__":
    main()
