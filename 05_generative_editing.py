"""
05_generative_editing.py
--------------------------
Real generative image editing - restyles an image using a text prompt
via Stable Diffusion (img2img), instead of the pixel-level tweaks in
04_image_analysis.py.

IMPORTANT - read this before running:
  - First run downloads the model (~4-5 GB) from Hugging Face. One-time.
  - On a CPU-only laptop, generating ONE image can take several minutes
    (5-15+ depending on steps/hardware). This is NOT instant like the
    text model. On a CUDA GPU (college lab), it's seconds.
  - `strength` controls how much the image changes: low (~0.2-0.4) keeps
    it close to the original; high (~0.6-0.9) lets the model change a
    lot more, including background/composition.

Usage:
    editor = StableDiffusionEditor()
    result = editor.generate(image, prompt="on a warm sunset beach", strength=0.55)
"""

import os

from PIL import Image


class StableDiffusionEditor:
    # "runwayml/stable-diffusion-v1-5" was moved on Hugging Face in 2024 and
    # now just redirects here - pointing at the canonical repo directly.
    MODEL_NAME = os.environ.get("SD_MODEL_NAME", "stable-diffusion-v1-5/stable-diffusion-v1-5")

    def __init__(self):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 halves memory/speeds things up, but only works on GPU -
        # CPU must stay in float32 or it'll error/produce garbage.
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=dtype,
            safety_checker=None,  # avoids an extra ~1GB download; keep prompts appropriate
        )
        self.pipe = self.pipe.to(self.device)

        if self.device == "cpu":
            # trades a bit of speed for lower peak memory - helpful on laptops
            self.pipe.enable_attention_slicing()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, extra limbs, watermark",
        strength: float = 0.55,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: int | None = None,
    ) -> Image.Image:
        import torch

        rgb = image.convert("RGB")
        # SD works on multiples of 8; resize to keep things fast + valid
        w, h = rgb.size
        max_side = 512
        scale = max_side / max(w, h)
        if scale < 1:
            rgb = rgb.resize((int(w * scale) // 8 * 8, int(h * scale) // 8 * 8))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=rgb,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return result.images[0]


def estimate_time_warning(device: str, num_inference_steps: int) -> str:
    if device == "cuda":
        return f"~{max(2, num_inference_steps // 8)}-{num_inference_steps // 4} seconds on this GPU."
    minutes_low = round(num_inference_steps * 0.15)
    minutes_high = round(num_inference_steps * 0.35)
    return f"~{minutes_low}-{minutes_high} minutes on CPU - this will feel slow, that's expected."