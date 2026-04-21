"""
FLUX Step-2 only: denoise to packed latents.

Inputs:
- prompt embeddings safetensors (t5_emb, clip_emb)

Output:
- packed latents safetensors (key: latents)
"""

from pathlib import Path

import torch
from diffusers import FluxPipeline
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16
STEPS = 12
CFG = 3.5
WIDTH = 1024
HEIGHT = 1024

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = ROOT_DIR / "temp"
PROMPT_EMBEDS_IN = TEMP_DIR / "prompt_embeds.safetensors"
LATENTS_OUT = TEMP_DIR / "latents.safetensors"
INIT_NOISE = TEMP_DIR / "init_noise.safetensors"


def load_prompt_embeds(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
	with safe_open(str(path), framework="pt", device="cpu") as f:
		keys = set(f.keys())
		if "t5_emb" not in keys or "clip_emb" not in keys:
			raise ValueError(f"Missing required keys in {path}. Found keys: {sorted(keys)}")
		t5_emb = f.get_tensor("t5_emb")
		clip_emb = f.get_tensor("clip_emb")
	return t5_emb, clip_emb


def main() -> None:
	TEMP_DIR.mkdir(parents=True, exist_ok=True)

	if not PROMPT_EMBEDS_IN.exists():
		raise FileNotFoundError(
			f"Prompt embeds not found: {PROMPT_EMBEDS_IN}. Run scripts/flux_ref.py first."
		)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	print(f"Loading prompt embeds: {PROMPT_EMBEDS_IN}")
	prompt_embeds, pooled_prompt_embeds = load_prompt_embeds(PROMPT_EMBEDS_IN)
	print(f"  T5 shape: {tuple(prompt_embeds.shape)}")
	print(f"  CLIP shape: {tuple(pooled_prompt_embeds.shape)}")

	if INIT_NOISE.exists():
		with safe_open(str(INIT_NOISE), framework="pt", device="cpu") as f:
			key = "latents" if "latents" in f.keys() else list(f.keys())[0]
			init_latents = f.get_tensor(key).to(dtype=torch.float32)
		print(f"Loaded shared init noise: {INIT_NOISE} shape={tuple(init_latents.shape)}")
	else:
		g = torch.Generator(device="cpu")
		g.manual_seed(42)
		init_latents = torch.randn((1, 16, HEIGHT // 8, WIDTH // 8), generator=g, dtype=torch.float32)
		save_file({"latents": init_latents}, str(INIT_NOISE))
		print(f"Created shared init noise: {INIT_NOISE} shape={tuple(init_latents.shape)}")

	print("Loading FLUX transformer-only pipeline...")
	pipe = FluxPipeline.from_pretrained(
		MODEL_ID,
		text_encoder=None,
		text_encoder_2=None,
		tokenizer=None,
		tokenizer_2=None,
		vae=None,
		torch_dtype=DTYPE,
		device_map="balanced",
	)
	pipe.enable_attention_slicing()

	# FluxPipeline expects pre-generated `latents` in packed format [B, patches, C].
	# Keep shared init noise on disk in 4D [B, 16, H, W] for Rust compatibility,
	# then pack it here for Diffusers.
	if init_latents.ndim == 4:
		latent_h = 2 * (HEIGHT // (pipe.vae_scale_factor * 2))
		latent_w = 2 * (WIDTH // (pipe.vae_scale_factor * 2))
		num_channels_latents = pipe.transformer.config.in_channels // 4
		init_latents = pipe._pack_latents(
			init_latents,
			batch_size=init_latents.shape[0],
			num_channels_latents=num_channels_latents,
			height=latent_h,
			width=latent_w,
		)
		print(f"Packed init noise for Diffusers: shape={tuple(init_latents.shape)}")

	with torch.no_grad():
		result = pipe(
			prompt_embeds=prompt_embeds.to(dtype=DTYPE, device=device),
			pooled_prompt_embeds=pooled_prompt_embeds.to(dtype=DTYPE, device=device),
			latents=init_latents.to(device=device, dtype=torch.float32),
			width=WIDTH,
			height=HEIGHT,
			num_inference_steps=STEPS,
			guidance_scale=CFG,
			output_type="latent",
		)

	latents = result.images
	print(f"Packed latents shape: {tuple(latents.shape)}")

	save_file({"latents": latents.detach().cpu()}, str(LATENTS_OUT))
	print(f"Saved -> {LATENTS_OUT}")


if __name__ == "__main__":
	main()
