"""
Copy FLUX.1-dev VAE weights from diffusers format into models/vae_candle.safetensors.

Candle's AutoEncoderKL uses the original diffusers key names verbatim
(down_blocks, resnets, mid_block, up_blocks, conv_norm_out, etc.), so
no renaming is needed — this is a straight copy.
"""

from safetensors import safe_open
from safetensors.torch import save_file
import os

VAE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev"
    "/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    "/vae/diffusion_pytorch_model.safetensors"
)
OUT = "models/vae_candle.safetensors"

print("Loading VAE weights...")
tensors = {}
with safe_open(VAE_PATH, framework="pt", device="cpu") as sf:
    for k in sf.keys():
        tensors[k] = sf.get_tensor(k)
print(f"  {len(tensors)} tensors loaded")
for k in list(tensors.keys())[:20]:
    print(f"  {k}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(f"\nSaving to {OUT} ...")
save_file(tensors, OUT)
print("Done!")
