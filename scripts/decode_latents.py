"""
Step 3-only FLUX decoder.

Loads latents from JSON and runs:
1) optional unpack (if input is packed [B, patches, channels])
2) VAE latent transform
3) VAE decode + save PNG
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from safetensors import safe_open


def unpack_latents(latents: torch.Tensor, height: int, width: int, scale_factor: int) -> torch.Tensor:
    # Matches FLUX pipeline unpack logic: [B, patches, C] -> [B, C/4, H_lat, W_lat]
    batch, _num_patches, channels = latents.shape
    h = 2 * (height // (scale_factor * 2))
    w = 2 * (width // (scale_factor * 2))
    return (
        latents.view(batch, h // 2, w // 2, channels // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, h, w)
    )


def load_latents(path: str) -> torch.Tensor:
    p = Path(path)

    if p.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No tensors found in {path}")
            key = "latents" if "latents" in keys else keys[0]
            return f.get_tensor(key).to(dtype=torch.float32)

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # Supports both {"shape": [...], "data": [...]} and raw nested arrays.
    if isinstance(d, dict) and "shape" in d and "data" in d:
        return torch.tensor(d["data"], dtype=torch.float32).reshape(d["shape"])
    return torch.tensor(d, dtype=torch.float32)


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    temp_dir = root_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Decode FLUX latents JSON with Step 3 only")
    parser.add_argument("--latents", default=str(temp_dir / "latents.safetensors"), help="Path to latents (.safetensors or .json)")
    parser.add_argument("--out", default=str(temp_dir / "output_from_rust.png"), help="Output PNG path")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev", help="Diffusers model id")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--height", type=int, default=512, help="Image height (used for packed latents)")
    parser.add_argument("--width", type=int, default=512, help="Image width (used for packed latents)")
    parser.add_argument(
        "--skip-latent-transform",
        action="store_true",
        help="Skip (latents / scaling_factor) + shift_factor. Useful for debugging pre-scaled latents.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading latents: {args.latents}")
    latents = load_latents(args.latents)
    print(f"  Raw latent shape: {tuple(latents.shape)}")
    print(f"  Raw latent stats: min={latents.min().item():.4f} max={latents.max().item():.4f} mean={latents.mean().item():.4f}")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device)
    vae.enable_slicing()
    vae.enable_tiling()

    scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=scale_factor)

    latents = latents.to(device, dtype=torch.float32)
    if latents.ndim == 3:
        print("Detected packed latents [B, patches, C], applying unpack...")
        latents = unpack_latents(latents, args.height, args.width, scale_factor)
    elif latents.ndim == 4:
        print("Detected unpacked latents [B, C, H, W], skipping unpack.")
    else:
        raise ValueError(f"Unexpected latent rank {latents.ndim}, expected 3 or 4")

    print(f"  Decode latent shape: {tuple(latents.shape)}")

    if not args.skip_latent_transform:
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    else:
        print("Skipping latent scaling/shift transform as requested.")

    with torch.no_grad():
        images = vae.tiled_decode(latents, return_dict=False)[0]

    images = image_processor.postprocess(images, output_type="pil")
    images[0].save(args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
