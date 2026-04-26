"""
Convert FLUX.1-dev weights from diffusers format → candle format.
Handles key renames + QKV tensor merges (q+k+v → qkv).
"""

import argparse
import re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SNAP = ("/home/nobus/.cache/huggingface/hub/"
        "models--black-forest-labs--FLUX.1-dev/snapshots/"
        "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer")

SHARDS = [f"{SNAP}/diffusion_pytorch_model-0000{i}-of-00003.safetensors"
          for i in range(1, 4)]

OUT = "/home/nobus/SSD/SSD_migration/Workbench/GPULOAD/models/flux_candle.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        choices=["keep", "bf16", "f16", "f32"],
        default="keep",
        help="Output dtype for floating tensors. Default keeps the source dtype.",
    )
    parser.add_argument(
        "--out",
        default=OUT,
        help="Output safetensors path.",
    )
    return parser.parse_args()


ARGS = parse_args()


def cast_tensor(t: torch.Tensor) -> torch.Tensor:
    if not t.is_floating_point() or ARGS.dtype == "keep":
        return t

    dtype_map = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }
    return t.to(dtype_map[ARGS.dtype])

# ---------------------------------------------------------------------------
# Simple key renames (no tensor shape change)
# ---------------------------------------------------------------------------
SIMPLE = {
    "x_embedder.weight":                              "img_in.weight",
    "x_embedder.bias":                                "img_in.bias",
    "context_embedder.weight":                        "txt_in.weight",
    "context_embedder.bias":                          "txt_in.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "time_text_embed.timestep_embedder.linear_1.bias":   "time_in.in_layer.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "time_text_embed.timestep_embedder.linear_2.bias":   "time_in.out_layer.bias",
    "time_text_embed.text_embedder.linear_1.weight":  "vector_in.in_layer.weight",
    "time_text_embed.text_embedder.linear_1.bias":    "vector_in.in_layer.bias",
    "time_text_embed.text_embedder.linear_2.weight":  "vector_in.out_layer.weight",
    "time_text_embed.text_embedder.linear_2.bias":    "vector_in.out_layer.bias",
    "time_text_embed.guidance_embedder.linear_1.weight": "guidance_in.in_layer.weight",
    "time_text_embed.guidance_embedder.linear_1.bias":   "guidance_in.in_layer.bias",
    "time_text_embed.guidance_embedder.linear_2.weight": "guidance_in.out_layer.weight",
    "time_text_embed.guidance_embedder.linear_2.bias":   "guidance_in.out_layer.bias",
    "norm_out.linear.weight":  "final_layer.adaLN_modulation.1.weight",
    "norm_out.linear.bias":    "final_layer.adaLN_modulation.1.bias",
    "proj_out.weight":         "final_layer.linear.weight",
    "proj_out.bias":           "final_layer.linear.bias",
}


def remap_double(i: int, suffix: str) -> str | None:
    p = f"transformer_blocks.{i}."
    c = f"double_blocks.{i}."
    M = {
        "norm1.linear.weight":            c + "img_mod.lin.weight",
        "norm1.linear.bias":              c + "img_mod.lin.bias",
        "norm1_context.linear.weight":    c + "txt_mod.lin.weight",
        "norm1_context.linear.bias":      c + "txt_mod.lin.bias",
        "attn.norm_q.weight":             c + "img_attn.norm.query_norm.scale",
        "attn.norm_k.weight":             c + "img_attn.norm.key_norm.scale",
        "attn.norm_added_q.weight":       c + "txt_attn.norm.query_norm.scale",
        "attn.norm_added_k.weight":       c + "txt_attn.norm.key_norm.scale",
        "attn.to_out.0.weight":           c + "img_attn.proj.weight",
        "attn.to_out.0.bias":             c + "img_attn.proj.bias",
        "attn.to_add_out.weight":         c + "txt_attn.proj.weight",
        "attn.to_add_out.bias":           c + "txt_attn.proj.bias",
        "norm2.weight":                   c + "img_norm2.weight",
        "norm2.bias":                     c + "img_norm2.bias",
        "norm2_context.weight":           c + "txt_norm2.weight",
        "norm2_context.bias":             c + "txt_norm2.bias",
        "ff.net.0.proj.weight":           c + "img_mlp.0.weight",
        "ff.net.0.proj.bias":             c + "img_mlp.0.bias",
        "ff.net.2.weight":                c + "img_mlp.2.weight",
        "ff.net.2.bias":                  c + "img_mlp.2.bias",
        "ff_context.net.0.proj.weight":   c + "txt_mlp.0.weight",
        "ff_context.net.0.proj.bias":     c + "txt_mlp.0.bias",
        "ff_context.net.2.weight":        c + "txt_mlp.2.weight",
        "ff_context.net.2.bias":          c + "txt_mlp.2.bias",
    }
    return M.get(suffix)


def remap_single(i: int, suffix: str) -> str | None:
    c = f"single_blocks.{i}."
    M = {
        "norm.linear.weight":   c + "modulation.lin.weight",
        "norm.linear.bias":     c + "modulation.lin.bias",
        "attn.norm_q.weight":   c + "norm.query_norm.scale",
        "attn.norm_k.weight":   c + "norm.key_norm.scale",
        "proj_out.weight":      c + "linear2.weight",
        "proj_out.bias":        c + "linear2.bias",
    }
    return M.get(suffix)


# ---------------------------------------------------------------------------
# Load all shards
# ---------------------------------------------------------------------------
print("Loading all shards into CPU RAM...")
tensors: dict[str, torch.Tensor] = {}
for shard in SHARDS:
    print(f"  {shard.split('/')[-1]}")
    with safe_open(shard, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)
print(f"  {len(tensors)} tensors loaded")

# ---------------------------------------------------------------------------
# Build output dict
# ---------------------------------------------------------------------------
out: dict[str, torch.Tensor] = {}
skipped: list[str] = []

for src_key, t in tensors.items():
    # 1. Simple top-level renames
    if src_key in SIMPLE:
        out[SIMPLE[src_key]] = cast_tensor(t)
        continue

    # 2. Double blocks — per-block renames
    m = re.match(r"transformer_blocks\.(\d+)\.(.*)", src_key)
    if m:
        i, suffix = int(m.group(1)), m.group(2)
        # Skip QKV parts — handled below
        if suffix in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                      "attn.to_q.bias",   "attn.to_k.bias",   "attn.to_v.bias",
                      "attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight",
                      "attn.add_q_proj.bias",   "attn.add_k_proj.bias",   "attn.add_v_proj.bias"):
            skipped.append(src_key)
            continue
        dst = remap_double(i, suffix)
        if dst:
            out[dst] = cast_tensor(t)
        else:
            print(f"  UNMAPPED double: {src_key}")
        continue

    # 3. Single blocks — per-block renames
    m = re.match(r"single_transformer_blocks\.(\d+)\.(.*)", src_key)
    if m:
        i, suffix = int(m.group(1)), m.group(2)
        # Skip QKV+mlp parts — handled below
        if suffix in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                      "attn.to_q.bias",   "attn.to_k.bias",   "attn.to_v.bias",
                      "proj_mlp.weight",  "proj_mlp.bias"):
            skipped.append(src_key)
            continue
        dst = remap_single(i, suffix)
        if dst:
            out[dst] = cast_tensor(t)
        else:
            print(f"  UNMAPPED single: {src_key}")
        continue

    print(f"  UNMAPPED top-level: {src_key}")

# ---------------------------------------------------------------------------
# QKV merges — double blocks (img + txt attention)
# ---------------------------------------------------------------------------
print("Merging QKV tensors...")
num_double = max(int(re.match(r"transformer_blocks\.(\d+)\.", k).group(1))
                 for k in tensors if k.startswith("transformer_blocks.")) + 1
num_single = max(int(re.match(r"single_transformer_blocks\.(\d+)\.", k).group(1))
                 for k in tensors if k.startswith("single_transformer_blocks.")) + 1

for i in range(num_double):
    p = f"transformer_blocks.{i}"
    c = f"double_blocks.{i}"
    # img QKV
    out[f"{c}.img_attn.qkv.weight"] = torch.cat([
        tensors[f"{p}.attn.to_q.weight"],
        tensors[f"{p}.attn.to_k.weight"],
        tensors[f"{p}.attn.to_v.weight"],
    ], dim=0)
    out[f"{c}.img_attn.qkv.bias"] = torch.cat([
        tensors[f"{p}.attn.to_q.bias"],
        tensors[f"{p}.attn.to_k.bias"],
        tensors[f"{p}.attn.to_v.bias"],
    ], dim=0)
    # txt QKV
    out[f"{c}.txt_attn.qkv.weight"] = torch.cat([
        tensors[f"{p}.attn.add_q_proj.weight"],
        tensors[f"{p}.attn.add_k_proj.weight"],
        tensors[f"{p}.attn.add_v_proj.weight"],
    ], dim=0)
    out[f"{c}.txt_attn.qkv.bias"] = torch.cat([
        tensors[f"{p}.attn.add_q_proj.bias"],
        tensors[f"{p}.attn.add_k_proj.bias"],
        tensors[f"{p}.attn.add_v_proj.bias"],
    ], dim=0)

    out[f"{c}.img_attn.qkv.weight"] = cast_tensor(out[f"{c}.img_attn.qkv.weight"])
    out[f"{c}.img_attn.qkv.bias"] = cast_tensor(out[f"{c}.img_attn.qkv.bias"])
    out[f"{c}.txt_attn.qkv.weight"] = cast_tensor(out[f"{c}.txt_attn.qkv.weight"])
    out[f"{c}.txt_attn.qkv.bias"] = cast_tensor(out[f"{c}.txt_attn.qkv.bias"])

# Single blocks: Q+K+V+proj_mlp → linear1
for i in range(num_single):
    p = f"single_transformer_blocks.{i}"
    c = f"single_blocks.{i}"
    out[f"{c}.linear1.weight"] = torch.cat([
        tensors[f"{p}.attn.to_q.weight"],
        tensors[f"{p}.attn.to_k.weight"],
        tensors[f"{p}.attn.to_v.weight"],
        tensors[f"{p}.proj_mlp.weight"],
    ], dim=0)
    out[f"{c}.linear1.bias"] = torch.cat([
        tensors[f"{p}.attn.to_q.bias"],
        tensors[f"{p}.attn.to_k.bias"],
        tensors[f"{p}.attn.to_v.bias"],
        tensors[f"{p}.proj_mlp.bias"],
    ], dim=0)

    out[f"{c}.linear1.weight"] = cast_tensor(out[f"{c}.linear1.weight"])
    out[f"{c}.linear1.bias"] = cast_tensor(out[f"{c}.linear1.bias"])

print(f"Output: {len(out)} tensors (from {len(tensors)} source, {len(skipped)} merged away)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
import os
os.makedirs(os.path.dirname(ARGS.out), exist_ok=True)
print(f"Saving to {ARGS.out} ...")
save_file(out, ARGS.out)
print("Done!")
