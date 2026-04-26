"""
FLUX reference inference following flux-vram-optimized pattern.
Step 1: encode prompt  (T5 + CLIP)
Step 2: generate latents via FluxPipeline with device_map
Step 3: unpack + VAE decode
"""
import gc
from pathlib import Path

import torch
from diffusers import FluxPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from safetensors.torch import save_file

MODEL_ID = "black-forest-labs/FLUX.1-dev"
PROMPT   = "Naked woman sitting on a bench in a park, photorealistic, 4k, detailed, cinematic lighting"
DEVICE   = "cuda:0"
DTYPE    = torch.bfloat16
STEPS    = 8
CFG      = 2.5
WIDTH    = 1024
HEIGHT   = 1024
ROOT_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = ROOT_DIR / "temp"
OUT      = TEMP_DIR / "output_ref.png"
PROMPT_EMBEDS_OUT = TEMP_DIR / "prompt_embeds_from_flux_ref.safetensors"
LATENTS_PACKED_ST_OUT = TEMP_DIR / "latents_ref_packed.safetensors"

TEMP_DIR.mkdir(parents=True, exist_ok=True)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Step 1 — Encode prompt (load text encoders only, unload after)
# ---------------------------------------------------------------------------
print("Step 1: Encoding prompt...")
pipe_enc = FluxPipeline.from_pretrained(
    MODEL_ID,
    transformer=None,
    vae=None,
    torch_dtype=DTYPE,
)
pipe_enc.text_encoder   = pipe_enc.text_encoder.to(DEVICE)
pipe_enc.text_encoder_2 = pipe_enc.text_encoder_2.to(DEVICE)

with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, _ = pipe_enc.encode_prompt(
        prompt=PROMPT,
        prompt_2=PROMPT,
        device=DEVICE,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

print(f"  T5:   {prompt_embeds.shape}")
print(f"  CLIP: {pooled_prompt_embeds.shape}")


### SAVE PROMPT EMBEDS FOR RUST DEBUGGING ###
save_file(
    {
        "t5_emb": prompt_embeds.detach().cpu(),
        "clip_emb": pooled_prompt_embeds.detach().cpu(),
    },
    str(PROMPT_EMBEDS_OUT),
)
print(f"Saved -> {PROMPT_EMBEDS_OUT}")

del pipe_enc
flush()

# ---------------------------------------------------------------------------
# Step 2 — Generate latents (transformer via device_map, no VAE, no text enc)
# ---------------------------------------------------------------------------
print("Step 2: Generating latents...")
pipe_gen = FluxPipeline.from_pretrained(
    MODEL_ID,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    vae=None,
    torch_dtype=DTYPE,
    device_map="balanced",
)
pipe_gen.enable_attention_slicing()

with torch.no_grad():
    result = pipe_gen(
        prompt_embeds=prompt_embeds.to(DTYPE),
        pooled_prompt_embeds=pooled_prompt_embeds.to(DTYPE),
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        output_type="latent",
    )

latents = result.images  # packed latents [1, patches, channels]
print(f"Latents shape: {latents.shape}")

### SAVE PACKED LATENTS FOR RUST DEBUGGING ###
save_file({"latents": latents.detach().cpu()}, str(LATENTS_PACKED_ST_OUT))
print(f"Saved -> {LATENTS_PACKED_ST_OUT}")

del pipe_gen
flush()

# ---------------------------------------------------------------------------
# Step 3 — Decode (unpack latents + VAE)
# ---------------------------------------------------------------------------
print("Step 3: Decoding...")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
vae = vae.to(DEVICE)
vae.enable_slicing()
vae.enable_tiling()

scale_factor    = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=scale_factor)

# Unpack: [batch, patches, channels] → [batch, channels, H, W]
def unpack_latents(latents, height, width, scale_factor):
    batch, num_patches, channels = latents.shape
    h = 2 * (height // (scale_factor * 2))
    w = 2 * (width  // (scale_factor * 2))
    return (latents
        .view(batch, h // 2, w // 2, channels // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, h, w))

latents = latents.to(DEVICE, dtype=torch.float32)
latents = unpack_latents(latents, HEIGHT, WIDTH, scale_factor)

latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

with torch.no_grad():
    images = vae.tiled_decode(latents, return_dict=False)[0]

images = image_processor.postprocess(images, output_type="pil")
images[0].save(str(OUT))
print(f"Saved → {OUT}")
