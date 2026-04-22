// text_encoder.rs — FLUX text encoding: CLIP-L + T5-XXL → prompt_embeds.safetensors
//
// Usage:
//   cargo run --bin text_encoder --release -- "your prompt here"
//   cargo run --bin text_encoder --release -- "your prompt" --lora loras/my.safetensors
//
// Output: temp/prompt_embeds.safetensors  { t5_emb: [1,512,4096], clip_emb: [1,768] }

mod lora;

mod device_config;
mod path_config;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::clip::text_model::{Activation as ClipActivation, ClipTextConfig, ClipTextTransformer};
use candle_transformers::models::t5::{ActivationWithOptionalGating, Config as T5Config, T5EncoderModel};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Resolve a path inside the HF snapshot for FLUX.1-dev.
/// Looks for any snapshot dir under $HOME/.cache/huggingface/hub/…/snapshots/
/// and returns the first one that contains `rel_path`.
fn hf_path(rel_path: &str) -> Result<std::path::PathBuf> {
    let snaps_dir = path_config::hf_repo_dir().join("snapshots");
    let entries = std::fs::read_dir(&snaps_dir)
        .with_context(|| format!("read_dir {}", snaps_dir.display()))?;
    for entry in entries.flatten() {
        let candidate = entry.path().join(rel_path);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "Could not find '{}' in any snapshot under {}",
        rel_path,
        snaps_dir.display()
    )
}

const CLIP_MAX_LEN: usize = 77;
const T5_MAX_LEN: usize = 512;

// ---------------------------------------------------------------------------
// CLIP-L config (openai/clip-vit-large-patch14)
// ---------------------------------------------------------------------------
fn clip_l_config() -> ClipTextConfig {
    ClipTextConfig {
        vocab_size: 49408,
        embed_dim: 768,
        activation: ClipActivation::QuickGelu,
        intermediate_size: 3072,
        max_position_embeddings: CLIP_MAX_LEN,
        pad_with: Some("!".to_string()),
        num_hidden_layers: 12,
        num_attention_heads: 12,
        projection_dim: 768,
    }
}

// ---------------------------------------------------------------------------
// T5-XXL config (google/t5-v1_1-xxl)
// ---------------------------------------------------------------------------
fn t5_xxl_config() -> T5Config {
    T5Config {
        vocab_size: 32128,
        d_model: 4096,
        d_kv: 64,
        d_ff: 10240,
        num_layers: 24,
        num_decoder_layers: Some(24),
        num_heads: 64,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        dropout_rate: 0.1,
        layer_norm_epsilon: 1e-6,
        initializer_factor: 1.0,
        feed_forward_proj: ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::NewGelu,
        },
        tie_word_embeddings: false,
        is_decoder: false,
        is_encoder_decoder: true,
        use_cache: false,
        pad_token_id: 0,
        eos_token_id: 1,
        decoder_start_token_id: Some(0),
    }
}

// ---------------------------------------------------------------------------
// Helpers: mmap a file and deserialize SafeTensors
// ---------------------------------------------------------------------------
fn mmap_file(path: &str) -> Result<(std::fs::File, memmap2::Mmap)> {
    let f = std::fs::File::open(path)
        .with_context(|| format!("open {path}"))?;
    let mmap = unsafe { MmapOptions::new().map(&f)? };
    Ok((f, mmap))
}

// ---------------------------------------------------------------------------
// CLIP encoding
// ---------------------------------------------------------------------------
fn encode_clip(prompt: &str, device: &Device, dtype: DType, lora_path: Option<&str>, lora_scale: f32) -> Result<Tensor> {
    println!("  Loading CLIP tokenizer...");
    let tokenizer = Tokenizer::from_file(path_config::clip_tokenizer_path())
        .map_err(|e| anyhow::anyhow!("CLIP tokenizer: {e}"))?;

    // Tokenize; CLIP uses BPE, no special padding needed beyond truncation to 77.
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("CLIP encode: {e}"))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    // Truncate or pad to CLIP_MAX_LEN
    ids.truncate(CLIP_MAX_LEN);
    while ids.len() < CLIP_MAX_LEN {
        ids.push(0);
    }

    let input_ids = Tensor::from_vec(ids, (1, CLIP_MAX_LEN), device)?;

    println!("  Loading CLIP weights (mmap)...");
    let t0 = Instant::now();
    let clip_weights = hf_path("text_encoder/model.safetensors")?;
    let (_f, mmap) = mmap_file(clip_weights.to_str().unwrap())?;
    let st = SafeTensors::deserialize(&mmap[..]).context("CLIP safetensors")?;

    // Build tensors map — only 246 MB, fine to load eagerly
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for (name, view) in st.tensors() {
        let src_dtype = match view.dtype() {
            safetensors::Dtype::F32  => DType::F32,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::F16  => DType::F16,
            d => anyhow::bail!("unsupported CLIP dtype {:?} for {}", d, name),
        };
        let shape: Vec<usize> = view.shape().to_vec();
        let t = Tensor::from_raw_buffer(view.data(), src_dtype, &shape, device)?
            .to_dtype(dtype)?;
        map.insert(name.to_string(), t);
    }
    // Optionally merge CLIP LoRA deltas into the weight map
    if let Some(path) = lora_path {
        println!("  Merging CLIP LoRA from {path}...");
        let lora = lora::LoraWeights::load_auto(path, lora_scale, device, DType::BF16)?;
        println!("  LoRA clip={} layers", lora.clip_count());
        lora.merge_clip(&mut map);
    }

    let vb = VarBuilder::from_tensors(map, dtype, device);
    // CLIP HF file has keys prefixed with "text_model." — descend into it:
    let model = ClipTextTransformer::new(vb.pp("text_model"), &clip_l_config())?;
    println!("  CLIP built in {:.1}s", t0.elapsed().as_secs_f32());

    // Module::forward returns pooled output [1, 768] (EOS token embedding)
    let t1 = Instant::now();
    let out = model.forward(&input_ids)?;
    println!("  CLIP encoded in {:.2}s  shape: {:?}", t1.elapsed().as_secs_f32(), out.dims());
    Ok(out)
}

// ---------------------------------------------------------------------------
// T5-XXL encoding
// ---------------------------------------------------------------------------
fn encode_t5(prompt: &str, device: &Device, dtype: DType) -> Result<Tensor> {
    println!("  Loading T5 tokenizer...");
    let t5_tok_path = hf_path("tokenizer_2/tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&t5_tok_path)
        .map_err(|e| anyhow::anyhow!("T5 tokenizer: {e}"))?;

    // Tokenize. T5 tokenizer adds EOS (1) automatically; truncate to T5_MAX_LEN.
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("T5 encode: {e}"))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    // Truncate (keep EOS if present)
    if ids.len() > T5_MAX_LEN {
        ids.truncate(T5_MAX_LEN - 1);
        ids.push(1); // EOS
    }
    // Pad to T5_MAX_LEN with pad token (0)
    while ids.len() < T5_MAX_LEN {
        ids.push(0);
    }

    let input_ids = Tensor::from_vec(ids, (1, T5_MAX_LEN), device)?;

    println!("  Memory-mapping T5-XXL weights ({:.1} GB) from SSD...", 9.52);
    let t0 = Instant::now();
    let t5_shard1 = hf_path("text_encoder_2/model-00001-of-00002.safetensors")?;
    let t5_shard2 = hf_path("text_encoder_2/model-00002-of-00002.safetensors")?;
    let (_f1, mmap1) = mmap_file(t5_shard1.to_str().unwrap())?;
    let (_f2, mmap2) = mmap_file(t5_shard2.to_str().unwrap())?;

    // Build tensors map from both shards. T5-XXL is 9.52 GB — this triggers
    // OS paging as we touch each weight during forward pass.
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for slice in [&mmap1[..], &mmap2[..]] {
        let st = SafeTensors::deserialize(slice).context("T5 safetensors")?;
        for (name, view) in st.tensors() {
            let src_dtype = match view.dtype() {
                safetensors::Dtype::F32  => DType::F32,
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F16  => DType::F16,
                d => anyhow::bail!("unsupported T5 dtype {:?} for {}", d, name),
            };
            let shape: Vec<usize> = view.shape().to_vec();
            // Keep as BF16 on CPU to avoid doubling memory footprint
            let t = Tensor::from_raw_buffer(view.data(), src_dtype, &shape, device)?
                .to_dtype(dtype)?;
            map.insert(name.to_string(), t);
        }
    }
    println!("  T5 weights mapped in {:.1}s", t0.elapsed().as_secs_f32());

    let vb = VarBuilder::from_tensors(map, dtype, device);
    let cfg = t5_xxl_config();
    let mut model = T5EncoderModel::load(vb, &cfg)?;
    println!("  T5 model built in {:.1}s", t0.elapsed().as_secs_f32());

    let t1 = Instant::now();
    // Use forward_dt to stay in BF16 during computation
    let out = model.forward(&input_ids)?;
    println!("  T5 encoded in {:.1}s  shape: {:?}", t1.elapsed().as_secs_f32(), out.dims());
    Ok(out)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
fn main() -> Result<()> {
    // Simple arg parsing: first positional = prompt, then --lora / --lora-scale
    let mut args = std::env::args().skip(1);
    let mut prompt = String::from("A cat sitting on a bench");
    let mut lora_path: Option<String> = None;
    let mut lora_scale: f32 = 1.0;
    let mut device_str = String::from("cpu");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--lora"       => { lora_path   = args.next(); }
            "--lora-scale" => { lora_scale  = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0); }
            "--device"     => { device_str  = args.next().unwrap_or_else(|| "cpu".to_string()); }
            _              => { prompt = arg; }
        }
    }
    println!("Prompt: {:?}", prompt);
    if let Some(ref p) = lora_path {
        println!("LoRA:   {p}  scale={lora_scale:.2}");
    }

    // CLIP is ~250 MB and runs on the requested device.
    // T5-XXL is 9.5 GB — always stays on CPU (no streaming impl; candle's T5Block is private).
    let (clip_device, clip_dtype) = match device_str.as_str() {
        "cuda" => {
            device_config::ensure_cuda_feature_enabled()?;
            let dtype = device_config::auto_cuda_dtype(0)?;
            println!(
                "Device: CUDA  (CLIP on GPU, T5 on CPU)  dtype: {}",
                device_config::dtype_label(dtype)
            );
            (Device::new_cuda(0)?, dtype)
        }
        _ => {
            println!("Device: CPU");
            (Device::Cpu, DType::F32)
        }
    };

    println!("\n=== CLIP-L ===");
    let clip_emb = encode_clip(&prompt, &clip_device, clip_dtype, lora_path.as_deref(), lora_scale)
        .context("CLIP encoding failed")?;

    println!("\n=== T5-XXL ===");
    // T5 always on CPU — candle's T5Block is private, no block-streaming impl available.
    let t5_emb = encode_t5(&prompt, &Device::Cpu, DType::F32)
        .context("T5 encoding failed")?;

    let save_dtype = if clip_device.is_cuda() { clip_dtype } else { DType::BF16 };

    // Store embeddings in the active GPU dtype so pre-Ampere cards avoid BF16 casts.
    // CPU mode still writes BF16 to keep file sizes small.
    let mut out: HashMap<String, Tensor> = HashMap::new();
    out.insert("clip_emb".to_string(), clip_emb.to_dtype(save_dtype)?);
    out.insert("t5_emb".to_string(), t5_emb.to_dtype(save_dtype)?);

    std::fs::create_dir_all("temp")?;
    candle_core::safetensors::save(&out, "temp/prompt_embeds.safetensors")?;
    println!("\nSaved -> temp/prompt_embeds.safetensors");

    Ok(())
}
