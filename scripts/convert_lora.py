"""
Convert a kohya-format LoRA safetensors file into a pre-renamed BF16 factored file.

The output is still factored (up [out,rank], down [rank,in]) — NOT expanded.
Expanding up @ down into full matrices would be GB-scale for FLUX.

What changes vs the original kohya file:
  - Keys renamed to candle format  (lora_unet_single_blocks_N_linear1 → single_blocks.N.linear1)
  - Suffix: .lora_up  /  .lora_down  (no .weight)
  - dtype: BF16
  - scale * alpha/rank baked into the `up` matrix  → runtime delta = up @ down, no extra mul

Input keys:
  lora_unet_single_blocks_N_linear1.lora_down.weight   [rank, in]
  lora_unet_single_blocks_N_linear1.lora_up.weight     [out, rank]
  lora_unet_single_blocks_N_linear1.alpha              scalar
  lora_te1_text_model_encoder_layers_N_mlp_fc1.*       CLIP layers

Output keys:
  single_blocks.N.linear1.lora_up    [out, rank]  BF16  (scale baked in)
  single_blocks.N.linear1.lora_down  [rank, in]   BF16
  text_model.encoder.layers.N.mlp.fc1.lora_up     BF16
  text_model.encoder.layers.N.mlp.fc1.lora_down   BF16

Usage:
  python scripts/convert_lora.py input.safetensors output.safetensors [--scale 1.0]
"""

import argparse
import re
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os


def kohya_unet_single_to_candle(base: str) -> str | None:
    m = re.fullmatch(r"lora_unet_single_blocks_(\d+)_(.*)", base)
    if not m:
        return None
    idx, suffix = m.group(1), m.group(2).replace("_", ".")
    return f"single_blocks.{idx}.{suffix}"


def kohya_te1_to_candle(base: str) -> str | None:
    rest = base.removeprefix("lora_te1_")
    if rest == base:
        return None
    return rest.replace("_", ".")


def main():
    ap = argparse.ArgumentParser(description="Pre-rename LoRA to BF16 factored format")
    ap.add_argument("input",  help="Input kohya LoRA safetensors")
    ap.add_argument("output", help="Output pre-converted BF16 safetensors")
    ap.add_argument("--scale", type=float, default=1.0, help="LoRA merge strength (default 1.0)")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    raw: dict[str, torch.Tensor] = {}
    with safe_open(args.input, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            raw[k] = sf.get_tensor(k).float()  # work in F32

    base_names: set[str] = set()
    for key in raw:
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if key.endswith(suffix):
                base_names.add(key[: -len(suffix)])
                break

    out: dict[str, torch.Tensor] = {}
    loaded = skipped = 0

    for base in sorted(base_names):
        down = raw.get(f"{base}.lora_down.weight")
        up   = raw.get(f"{base}.lora_up.weight")
        if down is None or up is None:
            print(f"  SKIP (missing down/up): {base}")
            skipped += 1
            continue

        alpha     = float(raw[f"{base}.alpha"].item()) if f"{base}.alpha" in raw else 1.0
        rank      = down.shape[0]
        s         = args.scale * alpha / rank

        # Bake scale into up — runtime: delta = up_scaled @ down  (no extra mul)
        up_scaled = (s * up).to(torch.bfloat16)
        down_bf16 = down.to(torch.bfloat16)

        if base.startswith("lora_unet_single_blocks_"):
            candle_key = kohya_unet_single_to_candle(base)
            if candle_key is None:
                print(f"  SKIP (key parse fail): {base}")
                skipped += 1
                continue
            out[f"{candle_key}.lora_up"]   = up_scaled
            out[f"{candle_key}.lora_down"] = down_bf16
            loaded += 1

        elif base.startswith("lora_te1_"):
            candle_key = kohya_te1_to_candle(base)
            if candle_key is None:
                print(f"  SKIP (key parse fail): {base}")
                skipped += 1
                continue
            out[f"{candle_key}.lora_up"]   = up_scaled
            out[f"{candle_key}.lora_down"] = down_bf16
            loaded += 1

        else:
            skipped += 1

    print(f"  Converted {loaded} layers, skipped {skipped}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(out, args.output)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved {args.output}  ({size_mb:.1f} MB, {len(out)} tensors, scale={args.scale} baked in)")
    print(f"\nUsage:  --lora {args.output}  (--lora-scale is ignored for pre-converted files)")


if __name__ == "__main__":
    main()