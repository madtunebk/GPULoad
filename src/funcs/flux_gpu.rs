// funcs/flux_gpu.rs — FLUX streaming denoising (in-memory interface)
//
// Call `denoise(...)` to run the Euler denoising loop and get back packed latents.
// Call `run(...)` to do the same and save to a .safetensors file.

use crate::device_config;
use crate::flux_blocks;
use crate::lora;
use crate::path_config;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{self as nn, LayerNorm, Module, VarBuilder};
use candle_transformers::models::flux;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;

const BAR_WIDTH: usize = 30;
type NoiseKey = (u64, usize, usize);
static SEEDED_NOISE_CACHE: OnceLock<Mutex<HashMap<NoiseKey, Vec<f32>>>> = OnceLock::new();

fn print_bar(step: usize, steps: usize, done: usize, total: usize, elapsed: f32, finished: bool) {
    let filled = (done * BAR_WIDTH).div_ceil(total.max(1)).min(BAR_WIDTH);
    let bar: String = (0..BAR_WIDTH)
        .map(|i| if i < filled { '█' } else { '░' })
        .collect();
    if finished {
        println!("\x1b[2K\rStep {step:>2}/{steps} [{bar}] {done:>3}/{total} blks  {elapsed:.1}s");
    } else {
        print!("\x1b[2K\rStep {step:>2}/{steps} [{bar}] {done:>3}/{total} blks  {elapsed:.1}s");
        let _ = std::io::stdout().flush();
    }
}

fn load_cpu_block_from_st(
    st: &SafeTensors<'_>,
    prefix: &str,
) -> Result<HashMap<String, Tensor>> {
    let mut m = HashMap::new();
    for (name, view) in st.tensors() {
        if let Some(short) = name.strip_prefix(prefix) {
            let src_dtype = match view.dtype() {
                safetensors::Dtype::F32  => DType::F32,
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F16  => DType::F16,
                safetensors::Dtype::F64  => DType::F64,
                d => anyhow::bail!("unsupported dtype {:?} for tensor {}", d, name),
            };
            let shape: Vec<usize> = view.shape().to_vec();
            let t = Tensor::from_raw_buffer(view.data(), src_dtype, &shape, &Device::Cpu)?;
            m.insert(short.to_string(), t);
        }
    }
    anyhow::ensure!(!m.is_empty(), "no tensors for prefix '{}'", prefix);
    Ok(m)
}

fn cpu_to_gpu_map(
    cpu_map: HashMap<String, Tensor>,
    gpu: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut m = HashMap::new();
    for (k, v) in cpu_map {
        m.insert(k, v.to_dtype(dtype)?.to_device(gpu)?);
    }
    Ok(m)
}

fn cpu_to_gpu_vb(
    cpu_map: HashMap<String, Tensor>,
    gpu: &Device,
    dtype: DType,
) -> Result<VarBuilder<'static>> {
    let m = cpu_to_gpu_map(cpu_map, gpu, dtype)?;
    Ok(VarBuilder::from_tensors(m, dtype, gpu))
}

fn gpu_vb_from_st(
    prefix: &str,
    st: &SafeTensors<'_>,
    gpu: &Device,
    dtype: DType,
) -> Result<VarBuilder<'static>> {
    let mut m: HashMap<String, Tensor> = HashMap::new();
    for (name, view) in st.tensors() {
        if let Some(short) = name.strip_prefix(prefix) {
            let src_dtype = match view.dtype() {
                safetensors::Dtype::F32  => DType::F32,
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F16  => DType::F16,
                safetensors::Dtype::F64  => DType::F64,
                d => anyhow::bail!("unsupported dtype {:?} for tensor {}", d, name),
            };
            let shape: Vec<usize> = view.shape().to_vec();
            let t = Tensor::from_raw_buffer(view.data(), src_dtype, &shape, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(gpu)?;
            m.insert(short.to_string(), t);
        }
    }
    anyhow::ensure!(!m.is_empty(), "no tensors found for prefix '{}'", prefix);
    Ok(VarBuilder::from_tensors(m, dtype, gpu))
}

fn seeded_randn(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let next = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 + 0.5) / (1u64 << 31) as f32
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next(&mut s).max(1e-10);
        let u2 = next(&mut s);
        let r = (-2.0 * u1.ln()).sqrt();
        let a = 2.0 * std::f32::consts::PI * u2;
        out.push(r * a.cos());
        if out.len() < n {
            out.push(r * a.sin());
        }
    }
    out
}

fn seeded_noise_cached(seed: u64, h_lat: usize, w_lat: usize) -> Result<(Vec<f32>, bool)> {
    let key: NoiseKey = (seed, h_lat, w_lat);
    let cache = SEEDED_NOISE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("seeded noise cache lock poisoned"))?;

    if let Some(v) = guard.get(&key) {
        return Ok((v.clone(), true));
    }

    let n = h_lat * w_lat * 16;
    let data = seeded_randn(seed, n);
    guard.insert(key, data.clone());
    Ok((data, false))
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Parameters for the denoising run.
pub struct DenoiseParams<'a> {
    pub clip_emb: &'a Tensor,
    pub t5_emb: &'a Tensor,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance: f32,
    pub seed: Option<u64>,
    pub lora: Option<&'a str>,
    pub lora_scale: f32,
    /// Path to `flux_candle.safetensors`. None = auto-resolve via path_config.
    pub model: Option<&'a str>,
}

/// Result of a denoising run — packed latents + image dimensions.
pub struct Latents {
    /// Packed latent tensor `[1, patches, 64]` in F32 on CPU.
    pub packed: Tensor,
    pub width: usize,
    pub height: usize,
}

// ---------------------------------------------------------------------------
// Public in-memory entry point
// ---------------------------------------------------------------------------
/// Run the FLUX Euler denoising loop and return packed latents in memory.
pub fn denoise(params: DenoiseParams<'_>) -> Result<Latents> {
    device_config::ensure_cuda_feature_enabled()?;
    let gpu = Device::cuda_if_available(0)?;
    let dtype = if gpu.is_cuda() {
        device_config::auto_cuda_dtype(0)?
    } else {
        DType::F32
    };
    println!("Device: {:?}  dtype: {}", gpu, device_config::dtype_label(dtype));

    let cfg = flux::model::Config::dev();
    let h = cfg.hidden_size;

    let t5_emb = params.t5_emb.to_device(&gpu)?.to_dtype(dtype)?;
    let clip_emb = params.clip_emb.to_device(&gpu)?.to_dtype(dtype)?;
    println!("T5: {:?}  CLIP: {:?}", t5_emb.dims(), clip_emb.dims());

    // Round to multiple of 16
    let width  = (params.width  / 16) * 16;
    let height = (params.height / 16) * 16;
    if width != params.width || height != params.height {
        println!(
            "Rounded {}x{} → {}x{} (must be multiple of 16)",
            params.width, params.height, width, height
        );
    }
    let h_lat = height / 8;
    let w_lat = width  / 8;

    // Initial latent noise
    let lat_img = if let Some(seed) = params.seed {
        let (data, from_cache) = seeded_noise_cached(seed, h_lat, w_lat)?;
        if from_cache {
            println!("Seed {} → init noise loaded from in-memory cache", seed);
        } else {
            println!("Seed {} → init noise generated and cached in memory", seed);
        }
        Tensor::from_vec(data, (1usize, 16usize, h_lat, w_lat), &Device::Cpu)?
            .to_device(&gpu)?
            .to_dtype(dtype)?
    } else {
        println!("Sampling random noise (use seed for reproducibility and in-memory reuse)");
        Tensor::randn(0f32, 1.0, (1usize, 16usize, h_lat, w_lat), &gpu)?.to_dtype(dtype)?
    };

    // Memory-map transformer weights
    println!("Memory-mapping transformer weights from SSD...");
    let t0 = Instant::now();
    let model_path = params
        .model
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| path_config::model_path("flux_candle.safetensors"));
    println!("  model: {}", model_path.display());
    let model_file = std::fs::File::open(&model_path)
        .with_context(|| format!("opening transformer model {}", model_path.display()))?;
    let model_mmap = Arc::new(unsafe { MmapOptions::new().map(&model_file) }
        .with_context(|| format!("memory-mapping transformer model {}", model_path.display()))?);
    let st = SafeTensors::deserialize(&model_mmap[..])
        .with_context(|| format!("reading safetensors from {}", model_path.display()))?;
    println!("  mmap ready in {:.1}s", t0.elapsed().as_secs_f32());

    // Build permanent GPU modules
    let img_in = nn::linear(cfg.in_channels, h, gpu_vb_from_st("img_in.", &st, &gpu, dtype)?)?;
    let txt_in = nn::linear(cfg.context_in_dim, h, gpu_vb_from_st("txt_in.", &st, &gpu, dtype)?)?;
    let ti_l1 = nn::linear(256, h, gpu_vb_from_st("time_in.in_layer.", &st, &gpu, dtype)?)?;
    let ti_l2 = nn::linear(h, h, gpu_vb_from_st("time_in.out_layer.", &st, &gpu, dtype)?)?;
    let gi_l1 = nn::linear(256, h, gpu_vb_from_st("guidance_in.in_layer.", &st, &gpu, dtype)?)?;
    let gi_l2 = nn::linear(h, h, gpu_vb_from_st("guidance_in.out_layer.", &st, &gpu, dtype)?)?;
    let vi_l1 = nn::linear(cfg.vec_in_dim, h, gpu_vb_from_st("vector_in.in_layer.", &st, &gpu, dtype)?)?;
    let vi_l2 = nn::linear(h, h, gpu_vb_from_st("vector_in.out_layer.", &st, &gpu, dtype)?)?;
    let fl_mod = nn::linear(h, 2 * h, gpu_vb_from_st("final_layer.adaLN_modulation.1.", &st, &gpu, dtype)?)?;
    let fl_lin = nn::linear(h, cfg.in_channels, gpu_vb_from_st("final_layer.linear.", &st, &gpu, dtype)?)?;
    let fl_norm = LayerNorm::new_no_bias(Tensor::ones(h, dtype, &gpu)?, 1e-6);
    let pe_embedder = flux::model::EmbedNd::new(
        cfg.hidden_size / cfg.num_heads,
        cfg.theta,
        cfg.axes_dim.clone(),
    );
    println!("Permanent GPU modules ready.");
    drop(st);

    // Optional LoRA
    let lora_weights = params.lora.map(|path| {
        println!("Loading LoRA: {path} (scale={:.2})", params.lora_scale);
        lora::LoraWeights::load_auto(path, params.lora_scale, &gpu, dtype)
            .with_context(|| format!("loading LoRA weights from {path}"))
    }).transpose()?;
    if let Some(ref l) = lora_weights {
        println!("  LoRA unet_single={} layers", l.unet_single_count());
    }

    // Setup state
    let state = flux::sampling::State::new(&t5_emb, &clip_emb, &lat_img)?;
    let txt_len = state.txt.dim(1)?;
    let ids = Tensor::cat(&[&state.txt_ids, &state.img_ids], 1)?;
    let pe = pe_embedder.forward(&ids)?;

    // Denoising loop
    let steps = params.steps;
    let image_seq_len = state.img.dim(1)?;
    let timesteps = flux::sampling::get_schedule(steps, Some((image_seq_len, 0.5, 1.15)));
    let guidance_val = params.guidance;
    let mut packed_lat = state.img.clone();
    let state_vec = state.vec.to_dtype(dtype)?;

    // Prefetch thread
    let (pfx_tx, pfx_rx) = mpsc::sync_channel::<Result<HashMap<String, Tensor>>>(1);
    {
        let mmap = Arc::clone(&model_mmap);
        let depth = cfg.depth;
        let depth_single = cfg.depth_single_blocks;
        std::thread::spawn(move || {
            let st = match SafeTensors::deserialize(&mmap[..]) {
                Ok(s) => s,
                Err(e) => { let _ = pfx_tx.send(Err(anyhow::anyhow!("ST: {e}"))); return; }
            };
            for _ in 0..steps {
                for i in 0..depth {
                    let r = load_cpu_block_from_st(&st, &format!("double_blocks.{}.", i));
                    if pfx_tx.send(r).is_err() { return; }
                }
                for i in 0..depth_single {
                    let r = load_cpu_block_from_st(&st, &format!("single_blocks.{}.", i));
                    if pfx_tx.send(r).is_err() { return; }
                }
            }
        });
    }

    for (step, window) in timesteps.windows(2).enumerate() {
        let (t, t_next) = match window {
            [a, b] => (*a as f32, *b as f32),
            _ => continue,
        };
        let dt = t_next - t;
        let t0_step = Instant::now();

        let t_batch = Tensor::new(&[t], &gpu)?.to_dtype(dtype)?;
        let g_batch = Tensor::new(&[guidance_val], &gpu)?.to_dtype(dtype)?;
        let t_emb = flux_blocks::timestep_embedding(&t_batch, 256, dtype)?;
        let g_emb = flux_blocks::timestep_embedding(&g_batch, 256, dtype)?;
        let t_vec = nn::ops::silu(&ti_l1.forward(&t_emb)?)?.apply(&ti_l2)?;
        let g_vec = nn::ops::silu(&gi_l1.forward(&g_emb)?)?.apply(&gi_l2)?;
        let y_vec = nn::ops::silu(&vi_l1.forward(&state_vec)?)?.apply(&vi_l2)?;
        let vec_ = (t_vec + g_vec + y_vec)?;

        let total_blocks = cfg.depth + cfg.depth_single_blocks;
        let mut img = img_in.forward(&packed_lat)?;
        let mut txt = txt_in.forward(&state.txt)?;

        for i in 0..cfg.depth {
            print_bar(step + 1, steps, i, total_blocks, t0_step.elapsed().as_secs_f32(), false);
            let cpu_map = pfx_rx.recv()??;
            let vb = cpu_to_gpu_vb(cpu_map, &gpu, dtype)?;
            let block = flux_blocks::DoubleStreamBlock::new(&cfg, vb)?;
            let (new_img, new_txt) = block.forward(&img, &txt, &vec_, &pe)?;
            img = new_img;
            txt = new_txt;
        }

        let mut merged = Tensor::cat(&[&txt, &img], 1)?;
        for i in 0..cfg.depth_single_blocks {
            print_bar(step + 1, steps, cfg.depth + i, total_blocks, t0_step.elapsed().as_secs_f32(), false);
            let cpu_map = pfx_rx.recv()??;
            let mut gpu_map = cpu_to_gpu_map(cpu_map, &gpu, dtype)?;
            if let Some(ref l) = lora_weights {
                l.apply_gpu_deltas(&mut gpu_map, i);
            }
            let vb = VarBuilder::from_tensors(gpu_map, dtype, &gpu);
            let block = flux_blocks::SingleStreamBlock::new(&cfg, vb)?;
            merged = block.forward(&merged, &vec_, &pe)?;
        }

        img = merged.i((.., txt_len.., ..))?;

        let normed = img.apply(&fl_norm)?;
        let chunks = nn::ops::silu(&vec_)?.apply(&fl_mod)?.chunk(2, 1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]);
        let v_pred = normed
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?
            .apply(&fl_lin)?;

        let dt_t = Tensor::new(&[dt], &gpu)?.to_dtype(dtype)?;
        packed_lat = (packed_lat + v_pred.broadcast_mul(&dt_t.reshape((1, 1, 1))?)?)?;

        print_bar(step + 1, steps, total_blocks, total_blocks, t0_step.elapsed().as_secs_f32(), true);
    }

    println!("Packed latent shape: {:?}", packed_lat.dims());
    let packed = packed_lat.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

    Ok(Latents { packed, width, height })
}

// ---------------------------------------------------------------------------
// Public file-based entry point (used by standalone flux_gpu binary)
// ---------------------------------------------------------------------------
/// Run denoising and save packed latents to `out_path`.
/// Calls [`denoise`] internally — use that if you want to keep data in memory.
pub fn run(params: DenoiseParams<'_>, out_path: &str) -> Result<()> {
    let latents = denoise(params)?;

    let mut out = HashMap::new();
    out.insert("latents", latents.packed);
    out.insert("width",  Tensor::new(&[latents.width  as u32], &Device::Cpu)?);
    out.insert("height", Tensor::new(&[latents.height as u32], &Device::Cpu)?);

    if let Some(parent) = Path::new(out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory {}", parent.display()))?;
        }
    }
    candle_core::safetensors::save(&out, out_path)
        .with_context(|| format!("saving latents to {out_path}"))?;
    println!("Saved packed latents -> {out_path} ({}x{})", latents.width, latents.height);

    Ok(())
}
