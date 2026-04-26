# StreamForge — FLUX.1-dev in Pure Rust

Full BF16 [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) inference pipeline written in **pure Rust** using [candle](https://github.com/huggingface/candle).  
Streams the 24GB transformer **block-by-block** from SSD through CPU RAM to GPU — peak VRAM ~4GB regardless of model size.

Tested on RTX 3060 12GB: **~8.5s/step** at 768×1024, GPU utilization ~94%.

---

## Requirements

- CUDA GPU (tested on RTX 3060 12GB)
- CUDA Toolkit installed
- Rust stable toolchain
- FLUX.1-dev accepted on HuggingFace (gated model)

---

## Build

```bash
cargo build --release --features cuda
```

| Feature | Effect |
|---------|--------|
| `cuda`  | CUDA GPU inference (required for `flux_gpu`) |
| `cudnn` | Enable cuDNN kernels (requires system cuDNN) |
| `metal` | Apple Metal backend |

---

## Model setup

### Transformer + VAE (local)

Convert from diffusers format using the provided scripts:

```bash
# Convert FLUX transformer (output: models/flux_candle.safetensors, ~24GB)
python scripts/convert_flux.py

# Convert VAE (output: models/vae_candle.safetensors)
python scripts/convert_vae.py
```

### Text encoders (HuggingFace cache)

CLIP-L and T5-XXL are loaded directly from the HF cache. Run a one-time download:

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev \
    --include "text_encoder/**" "text_encoder_2/**" "tokenizer/**" "tokenizer_2/**"
```

Or place `tokenizer.json` at `models/tokenizer/tokenizer.json`.

---

## Pipeline

### 1. Encode prompt → embeddings

```bash
./target/release/text_encoder "your prompt here"
# Output: temp/prompt_embeds.safetensors

# Run text encoders on GPU (CLIP on CUDA, T5 always CPU):
./target/release/text_encoder "your prompt" --device cuda

# With LoRA applied to CLIP:
./target/release/text_encoder "your prompt" \
    --lora loras/my.safetensors --lora-scale 1.0
```

### 2. Denoise → latents

```bash
./target/release/flux_gpu --width 1024 --height 1024 --steps 20
# Output: temp/latents.safetensors

# With seed for reproducibility:
./target/release/flux_gpu --width 768 --height 1024 --steps 20 \
    --guidance 3.5 --seed 42

# With LoRA applied to transformer:
./target/release/flux_gpu --width 768 --height 1024 --steps 20 \
    --lora loras/my.safetensors --lora-scale 1.0
```

### 3. Decode latents → PNG

```bash
./target/release/vae_decode
# Width/height read automatically from the latents file
# Output: temp/output_rust.png
```

---

## Full example (with LoRA)

```bash
./target/release/text_encoder "anime2, masterpiece, 1girl, white hair, red eyes" \
    --device cuda \
    --lora loras/curse-anime2.safetensors --lora-scale 1.0

./target/release/flux_gpu \
    --width 768 --height 1024 --steps 20 --guidance 3.5 --seed 42 \
    --lora loras/curse-anime2.safetensors --lora-scale 1.0

./target/release/vae_decode
```

---

## LoRA support

Kohya-style safetensors. Both CLIP and transformer LoRAs work.

| Key prefix | Applied by |
|------------|-----------|
| `lora_te1_*` | `text_encoder --lora` |
| `lora_unet_single_blocks_N_*` | `flux_gpu --lora` |
| `lora_unet_double_blocks_N_*` | `flux_gpu --lora` |

Pass `--lora` to both binaries when the file targets both.  
Scale: `1.0` = full, `0.5` = half, `0.0` = disabled.

---

## Image variations (Redux)

Use `scripts/reduxflux.py` to encode a reference image instead of a text prompt.  
Requires `black-forest-labs/FLUX.1-Redux-dev` (gated — accept license first).

```bash
# Edit image path in scripts/reduxflux.py, then:
python scripts/reduxflux.py
# Outputs the exact flux_gpu command with correct W/H

./target/release/flux_gpu --width 800 --height 1280 --steps 20 --guidance 2.0
./target/release/vae_decode
```

Redux works best with lower guidance (`1.5`–`2.5`) compared to text prompts (`3.5`).

---

## Model paths

| Path | Description |
|------|-------------|
| `models/flux_candle.safetensors` | FLUX transformer (24GB BF16) |
| `models/vae_candle.safetensors` | FLUX VAE |
| `models/tokenizer/tokenizer.json` | CLIP tokenizer |
| `loras/*.safetensors` | LoRA weights (kohya format) |
| `temp/prompt_embeds.safetensors` | Encoded embeddings (pipeline intermediate) |
| `temp/latents.safetensors` | Denoised latents (pipeline intermediate) |
| `temp/output_rust.png` | Final output image |
| HF cache | CLIP-L, T5-XXL, Redux weights (auto-loaded) |

---

## Notes

- Width and height must be multiples of 16 (`flux_gpu` auto-rounds).
- `flux_gpu` writes width/height into the latents file — `vae_decode` reads them automatically.
- If `temp/init_noise.safetensors` exists but is the wrong shape, fresh noise is used.

---

## Kaggle

See [`kaggle_flux_streamforge.ipynb`](kaggle_flux_streamforge.ipynb) to run the full pipeline on a Kaggle T4/P100 GPU.
