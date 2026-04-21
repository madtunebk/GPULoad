//! LoRA weight loader and merger for FLUX / CLIP.
//!
//! Key design: deltas are stored in factored form  (up [out,rank], down [rank,in])
//! on GPU/CPU — never expanded to full [out,in] matrices.
//! Full expansion would be GB-scale; the factored LoRA is only ~18 MB.
//!
//! At inference, per block:  delta = up @ down  (tiny GPU matmul, rank << in/out)
//!                           w_merged = w + delta
//!
//! Supported input formats:
//!   1. kohya safetensors  — keys like lora_unet_single_blocks_N_linear1.lora_{down,up}.weight
//!   2. pre-converted BF16 — output of scripts/convert_lora.py
//!      keys: single_blocks.N.linear1.lora_up   [out, rank]  BF16  (scale baked into up)
//!            single_blocks.N.linear1.lora_down  [rank, in]   BF16

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::collections::HashMap;

/// A single LoRA layer stored in factored form.
/// delta() = up @ down  — tiny GPU matmul (rank << in/out).
/// Scale is baked into `up` at load time.
struct LoraLayer {
    up:   Tensor,   // [out, rank]
    down: Tensor,   // [rank, in]
}

impl LoraLayer {
    fn delta(&self) -> candle_core::Result<Tensor> {
        self.up.matmul(&self.down)
    }
}

/// LoRA weights in factored form.  unet_single lives on GPU; clip on CPU.
#[allow(dead_code)]
pub struct LoraWeights {
    unet_single: HashMap<String, LoraLayer>,
    clip:        HashMap<String, LoraLayer>,
}

#[allow(dead_code)]
impl LoraWeights {
    /// Auto-detect format (kohya vs pre-converted) and load.
    pub fn load_auto(path: &str, scale: f32, unet_device: &Device, unet_dtype: DType) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("LoRA open {path}: {e}"))?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let st = SafeTensors::deserialize(&mmap[..])
            .map_err(|e| anyhow::anyhow!("LoRA parse {path}: {e}"))?;

        let first = st.names().into_iter().next().map(|s| s.to_string()).unwrap_or_default();
        if first.starts_with("single_blocks.") || first.starts_with("text_model.") {
            println!("  Detected pre-converted LoRA format");
            Self::load_precomputed(&st, unet_device, unet_dtype)
        } else {
            println!("  Detected kohya LoRA format");
            Self::load_kohya(&st, scale, unet_device, unet_dtype)
        }
    }

    /// Load kohya-format: bake scale into up, upload factored (up, down) to GPU as unet_dtype.
    fn load_kohya(st: &SafeTensors<'_>, scale: f32, unet_device: &Device, unet_dtype: DType) -> Result<Self> {
        let mut raw: HashMap<String, Tensor> = HashMap::new();
        for (name, view) in st.tensors() {
            let src = match view.dtype() {
                safetensors::Dtype::F32  => DType::F32,
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F16  => DType::F16,
                d => anyhow::bail!("LoRA unsupported dtype {:?} for {}", d, name),
            };
            let t = Tensor::from_raw_buffer(view.data(), src, view.shape(), &Device::Cpu)?
                .to_dtype(DType::F32)?;
            raw.insert(name.to_string(), t);
        }

        let mut bases: std::collections::HashSet<String> = std::collections::HashSet::new();
        for key in raw.keys() {
            for suf in [".lora_down.weight", ".lora_up.weight", ".alpha"] {
                if let Some(b) = key.strip_suffix(suf) { bases.insert(b.to_string()); break; }
            }
        }

        let mut unet_single: HashMap<String, LoraLayer> = HashMap::new();
        let mut clip:        HashMap<String, LoraLayer> = HashMap::new();
        let (mut loaded, mut skipped) = (0usize, 0usize);

        for base in &bases {
            let (Some(down_f32), Some(up_f32)) = (
                raw.get(&format!("{base}.lora_down.weight")),
                raw.get(&format!("{base}.lora_up.weight")),
            ) else { skipped += 1; continue };

            let alpha = raw.get(&format!("{base}.alpha"))
                .and_then(|t| t.to_vec0::<f32>().ok()).unwrap_or(1.0);
            let rank = down_f32.dim(0)? as f32;
            let s = (scale * alpha / rank) as f64;

            // Bake scale into up — runtime delta = up_scaled @ down, no extra mul needed
            let up_scaled = match up_f32 * s {
                Ok(u) => u,
                Err(e) => { eprintln!("  LoRA scale error ({base}): {e}"); skipped += 1; continue; }
            };

            if base.starts_with("lora_unet_single_blocks_") {
                let Some(key) = kohya_unet_single_to_candle(base) else { skipped += 1; continue };
                match (
                    up_scaled.to_dtype(unet_dtype).and_then(|t| t.to_device(unet_device)),
                    down_f32.to_dtype(unet_dtype).and_then(|t| t.to_device(unet_device)),
                ) {
                    (Ok(up), Ok(down)) => { unet_single.insert(key, LoraLayer { up, down }); loaded += 1; }
                    (Err(e), _) | (_, Err(e)) => { eprintln!("  LoRA GPU upload ({base}): {e}"); skipped += 1; }
                }
            } else if base.starts_with("lora_te1_") {
                let Some(key) = kohya_te1_to_candle(base) else { skipped += 1; continue };
                match (up_scaled.to_dtype(DType::BF16), down_f32.to_dtype(DType::BF16)) {
                    (Ok(up), Ok(down)) => { clip.insert(key, LoraLayer { up, down }); loaded += 1; }
                    (Err(e), _) | (_, Err(e)) => { eprintln!("  LoRA CLIP cast ({base}): {e}"); skipped += 1; }
                }
            } else {
                skipped += 1;
            }
        }

        println!("  LoRA kohya: loaded {loaded}, skipped {skipped}");
        Ok(Self { unet_single, clip })
    }

    /// Load pre-converted format (scripts/convert_lora.py output).
    /// Keys: "single_blocks.N.suffix.lora_up" / ".lora_down" — scale already baked into up.
    fn load_precomputed(st: &SafeTensors<'_>, unet_device: &Device, unet_dtype: DType) -> Result<Self> {
        let mut raw: HashMap<String, Tensor> = HashMap::new();
        for (name, view) in st.tensors() {
            let src = match view.dtype() {
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F32  => DType::F32,
                safetensors::Dtype::F16  => DType::F16,
                d => anyhow::bail!("LoRA precomputed: unsupported dtype {:?} for {}", d, name),
            };
            let t = Tensor::from_raw_buffer(view.data(), src, view.shape(), &Device::Cpu)?
                .to_dtype(DType::BF16)?;
            raw.insert(name.to_string(), t);
        }

        let mut bases: std::collections::HashSet<String> = std::collections::HashSet::new();
        for key in raw.keys() {
            for suf in [".lora_up", ".lora_down"] {
                if let Some(b) = key.strip_suffix(suf) { bases.insert(b.to_string()); break; }
            }
        }

        let mut unet_single: HashMap<String, LoraLayer> = HashMap::new();
        let mut clip:        HashMap<String, LoraLayer> = HashMap::new();
        let (mut loaded, mut skipped) = (0usize, 0usize);

        for base in &bases {
            let (Some(up_bf16), Some(down_bf16)) = (
                raw.get(&format!("{base}.lora_up")),
                raw.get(&format!("{base}.lora_down")),
            ) else { skipped += 1; continue };

            if base.starts_with("single_blocks.") {
                match (
                    up_bf16.to_dtype(unet_dtype).and_then(|t| t.to_device(unet_device)),
                    down_bf16.to_dtype(unet_dtype).and_then(|t| t.to_device(unet_device)),
                ) {
                    (Ok(up), Ok(down)) => { unet_single.insert(base.clone(), LoraLayer { up, down }); loaded += 1; }
                    (Err(e), _) | (_, Err(e)) => { eprintln!("  LoRA GPU upload ({base}): {e}"); skipped += 1; }
                }
            } else if base.starts_with("text_model.") {
                clip.insert(base.clone(), LoraLayer { up: up_bf16.clone(), down: down_bf16.clone() });
                loaded += 1;
            } else {
                skipped += 1;
            }
        }

        println!("  LoRA precomputed: loaded {loaded}, skipped {skipped}");
        Ok(Self { unet_single, clip })
    }

    /// Apply LoRA for single block `block_idx` into a GPU tensor map.
    /// Computes `up @ down` on GPU (tiny matmul) then adds to the weight.
    pub fn apply_gpu_deltas(&self, gpu_map: &mut HashMap<String, Tensor>, block_idx: usize) {
        for suffix in ["linear1", "linear2", "modulation.lin"] {
            let lora_key   = format!("single_blocks.{block_idx}.{suffix}");
            let weight_key = format!("{suffix}.weight");
            let (Some(layer), Some(w)) = (self.unet_single.get(&lora_key), gpu_map.get(&weight_key))
                else { continue };
            let result = layer.delta()
                .and_then(|d| w + d)
                .map_err(anyhow::Error::from);
            match result {
                Ok(merged) => { gpu_map.insert(weight_key, merged); }
                Err(e) => eprintln!("  LoRA delta warning ({lora_key}): {e}"),
            }
        }
    }

    /// Merge CLIP LoRA deltas into a CPU tensor map (called once at text_encoder startup).
    pub fn merge_clip(&self, clip_map: &mut HashMap<String, Tensor>) {
        for (key, layer) in &self.clip {
            let weight_key = format!("{key}.weight");
            let Some(w) = clip_map.get(&weight_key) else { continue };
            let result = layer.delta()
                .and_then(|d| d.to_dtype(w.dtype()))
                .and_then(|d| w + d)
                .map_err(anyhow::Error::from);
            match result {
                Ok(merged) => { clip_map.insert(weight_key, merged); }
                Err(e) => eprintln!("  LoRA CLIP warning ({key}): {e}"),
            }
        }
    }

    pub fn unet_single_count(&self) -> usize { self.unet_single.len() }
    pub fn clip_count(&self)         -> usize { self.clip.len() }
}

// ---------------------------------------------------------------------------
// Key mappers  (kohya → candle)
// ---------------------------------------------------------------------------

fn kohya_unet_single_to_candle(base: &str) -> Option<String> {
    let rest = base.strip_prefix("lora_unet_single_blocks_")?;
    let (idx_str, suffix) = rest.split_once('_')?;
    let idx: usize = idx_str.parse().ok()?;
    Some(format!("single_blocks.{idx}.{}", suffix.replace('_', ".")))
}

fn kohya_te1_to_candle(base: &str) -> Option<String> {
    Some(base.strip_prefix("lora_te1_")?.replace('_', "."))
}
