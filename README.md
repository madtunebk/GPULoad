# FLUX Rust Pipeline — StreamForge

Full BF16 FLUX.1-dev inference in pure Rust. Streams the 24GB transformer block-by-block
from SSD through CPU RAM to GPU — peak VRAM ~4GB regardless of model size.

---

## Requirements

- CUDA GPU (tested on RTX 3060 12GB)
- CUDA Toolkit installed
- Rust stable toolchain
- FLUX.1-dev weights in HF cache (or locally)

---

## Build

```bash
cargo build --release --features cuda
```

---

## Pipeline

### 1. Encode prompt → embeddings

```bash
./target/release/text_encoder "your prompt here"
# Output: temp/prompt_embeds.safetensors

# With LoRA applied to CLIP:
./target/release/text_encoder "your prompt" --lora loras/my.safetensors --lora-scale 0.8
```

### 2. Denoise → latents

```bash
./target/release/flux_gpu --width 1024 --height 1024 --steps 20
# Output: temp/latents.safetensors  (width/height saved automatically)

# With seed for reproducibility:
./target/release/flux_gpu --width 1024 --height 1024 --steps 20 --seed 42

# With LoRA applied to transformer single blocks:
./target/release/flux_gpu --width 1024 --height 1024 --steps 20 \
    --lora loras/my.safetensors --lora-scale 0.8
```

### 3. Decode latents → PNG

```bash
./target/release/vae_decode
# Width/height read automatically from latents file — no flags needed
# Output: temp/output_rust.png
```

---

## LoRA support

LoRA files must be kohya-style safetensors. The pipeline supports:

| LoRA keys            | Applied in        |
|----------------------|-------------------|
| `lora_unet_single_blocks_N_*` | `flux_gpu` (--lora flag) |
| `lora_te1_*`         | `text_encoder` (--lora flag) |

**Both binaries need `--lora` when a LoRA targets both text encoder and transformer.**

```bash
./target/release/text_encoder "anime girl" \
    --lora loras/curse-anime2.safetensors --lora-scale 1.0

./target/release/flux_gpu --width 1024 --height 1024 --steps 20 \
    --lora loras/curse-anime2.safetensors --lora-scale 1.0

./target/release/vae_decode
```

LoRA scale: `1.0` = full strength, `0.5` = half, `0.0` = disabled (same as no LoRA).

---

## Image variations (Redux)

Use `scripts/reduxflux.py` to encode an input image instead of a text prompt.
Requires `black-forest-labs/FLUX.1-Redux-dev` (gated — accept license on HF first).

```bash
# Edit the image URL/path in scripts/reduxflux.py, then:
python scripts/reduxflux.py
# Prints the exact flux_gpu + vae_decode commands with correct W/H

./target/release/flux_gpu --embeddings temp/prompt_embeds.safetensors \
    --width 800 --height 1280 --steps 20 --guidance 2.0
./target/release/vae_decode
```

---

## Notes

- Width and height must be multiples of 16. `flux_gpu` auto-rounds if not.
- `flux_gpu` saves width/height into the latents file — `vae_decode` reads them automatically.
- If `temp/init_noise.safetensors` exists but is the wrong size, fresh random noise is used.
- Redux works best with lower guidance (`--guidance 1.5`–`2.5`) vs text (`3.5`).

---

## Model paths

| File | Description |
|------|-------------|
| `models/flux_candle.safetensors` | FLUX transformer (24GB, converted from diffusers) |
| `models/vae_candle.safetensors` | FLUX VAE (original diffusers keys) |
| `models/tokenizer/tokenizer.json` | CLIP tokenizer |
| `loras/*.safetensors` | LoRA weights (kohya format) |
| HF cache | CLIP-L, T5-XXL, Redux weights loaded directly |


This repository contains a minimal Stable Diffusion / candle-based runner and small helper utilities used during debugging and model compatibility work. Below is a compact, step-by-step record of what was done, how to reproduce it, and how to continue.

**Repository layout**
- `src/main.rs:1` — main sd-minimal binary (loads tokenizer, text encoder, UNet, VAE)
- `src/dump_shapes.rs:1` — helper binary that lists tensor names & shapes in a safetensors file
- `src/remap_safetensors.rs:1` — tool that remaps safetensors key names and writes a remapped file to `models/converted/`
- `models/tokenizer/tokenizer.json:1` — tokenizer JSON (downloaded)
- `models/converted/:1` — output directory for remapped safetensors

What I did (summary)
- Fixed and enhanced `src/main.rs` to:
  - Add a `--device` flag (`auto | cuda | metal | cpu`) using `clap`.
  - Add a `--skip-tokenizer` flag to allow bypassing tokenizer during debug runs.
  - Improve error messages that name which safetensors file failed to load.
  - Temporarily pointed the CLIP text encoder path to `models/converted/...` for testing.
- Added `src/dump_shapes.rs` to inspect safetensors contents (tensor names & shapes).
- Added `src/remap_safetensors.rs` which:
  - Reads a safetensors file, remaps keys (e.g. `model.*` -> `text_model.*`, `model.embed_tokens.weight` -> `text_model.embeddings.token_embedding.weight`, `model.norm.weight` -> `text_model.final_layer_norm.weight`).
  - Writes a remapped safetensors file to `models/converted/<original>.remapped.safetensors`.
- Ran the remapper on the local text encoder file and produced `models/converted/zImage_textEncoder.remapped.safetensors`.
- Observed a remaining shape mismatch (the remapped tensor shapes did not match the expected shapes for the SD v1.5 CLIP loader).

Why the run failed
- Tokenizer JSON (`models/tokenizer/tokenizer.json:1`) failed deserialization using the `tokenizers` crate even though the file is valid JSON — the format/schema is incompatible with the crate's expected tokenizer object.
- The CLIP text-encoder safetensors used the key namespace `model.*` and large embedding dims (`[151936, 2560]`) but candle's SD v1.5 loader expects keys like `text_model.embeddings.token_embedding.weight` with shapes compatible with the StableDiffusion text encoder (e.g. `[49408, 768]`).
- Key renaming fixed the name lookup but could not fix the shape mismatch — that requires a compatible model (correct architecture) or a projection/adapter.

Files / tools you can use
- Inspect safetensors keys & shapes:
  - `cargo run --release --bin dump_shapes` (edit `src/dump_shapes.rs` to change the hard-coded path or recompile)
  - Example: `cargo run --release --bin dump_shapes` prints `name shape` lines.
- Remap safetensors keys (non-destructive):
  - `target/release/remap_safetensors <input.safetensors>`
  - Example: `target/release/remap_safetensors /path/to/zImage_textEncoder.safetensors`
  - Default output: `models/converted/<filename>.remapped.safetensors`
- Run the sd-minimal binary (device selection):
  - Build: `cargo build --release` or with CUDA: `cargo build --release --features cuda` (cuDNN requires system cuDNN libs)
  - Run: `cargo run --release --bin sd-minimal -- --device cuda` (or `--device auto`)
  - Debug bypass tokenizer: `cargo run --release --bin sd-minimal -- --skip-tokenizer --device cuda`

Notes about Cargo features
- To enable CUDA support: `--features cuda` (links against `candle-core/cuda` feature)
- To enable cuDNN (if you have system cuDNN installed): `--features "cuda cudnn"`
- To enable Metal: `--features metal`

Recommended next steps (pick one)
1. Use a matching text encoder + tokenizer (best outcome)
   - Find a text-encoder safetensors and tokenizer.json that match the SD model you intend to run (embedding dim & vocab must match the SD loader expectations). I can list candidate Hugging Face model files and download them.
2. Implement a projection adapter (experimental)
   - Load the existing text encoder (2560-dim), add a learned or deterministic projection to 768-dim, and continue. This is likely to reduce quality unless you have a proper projection matrix.
3. Create a model-specific StableDiffusionConfig
   - If the zImage model has its own config (non-standard dims), adapt the builder to match that config — may require code changes in candle-transformers usage.

Repro steps (exact commands used)
- Build all bins: `cargo build --release`
- Build remapper only: `cargo build --release --bin remap_safetensors`
- Run remapper: `target/release/remap_safetensors /path/to/zImage_textEncoder.safetensors`
- Inspect converted file: `ls -lh models/converted` and `cargo run --release --bin dump_shapes` (edit `src/dump_shapes.rs` to point to the converted file if needed)
- Run sd-minimal (skip tokenizer for debug): `cargo run --release --bin sd-minimal -- --skip-tokenizer --device cuda`

Caveats / disk usage
- Safetensors files and the remapped output are large (several GB). Ensure you have enough disk space in `models/converted/` (we wrote a 7.5 GB remapped file during debugging).

If you want me to update the README further
- I can:
  - Add exact Hugging Face download URLs for a compatible tokenizer + text-encoder and optionally download them.
  - Revert the temporary change in `src/main.rs` (the remapped text-encoder path) and instead add a CLI option `--clip-weights <path>` so paths are configurable.
  - Add small usage examples in the README showing expected outputs and common errors.

Tell me which of the above additions you want and I will update the README accordingly (or commit the current README for you now).