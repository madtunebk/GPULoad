// StreamForge Rust — FLUX streaming inference
// Loads all weights to CPU RAM, streams one block at a time to GPU.
// Peak VRAM: ~4GB regardless of model size.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{self as nn, LayerNorm, Module, VarBuilder};
use candle_transformers::models::flux;
use clap::Parser;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::io::Write;
use std::sync::{mpsc, Arc};
use std::time::Instant;

use sd_minimal::{device_config, flux_blocks, lora, path_config};

const BAR_WIDTH: usize = 30;

/// Tqdm-style single-line progress bar.  No flush() — avoids stalling CUDA.
/// Uses \x1b[2K\r to clear+rewrite the line in place.
fn print_bar(step: usize, steps: usize, done: usize, total: usize, elapsed: f32, finished: bool) {
    let filled = (done * BAR_WIDTH).div_ceil(total.max(1)).min(BAR_WIDTH);
    let bar: String = (0..BAR_WIDTH).map(|i| if i < filled { '█' } else { '░' }).collect();
    if finished {
        println!("\x1b[2K\rStep {step:>2}/{steps} [{bar}] {done:>3}/{total} blks  {elapsed:.1}s");
    } else {
        print!("\x1b[2K\rStep {step:>2}/{steps} [{bar}] {done:>3}/{total} blks  {elapsed:.1}s");
        let _ = std::io::stdout().flush();
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "512")] width: usize,
    #[arg(long, default_value = "512")] height: usize,
    #[arg(long, default_value = "8")]   steps: usize,
    #[arg(long, default_value = "3.5")] guidance: f32,
    #[arg(long, default_value = "temp/prompt_embeds.safetensors")] embeddings: String,
    #[arg(long, default_value = "temp/latents.safetensors")]        out_latents: String,
    #[arg(long)] model: Option<String>,
    /// Optional LoRA weights file (kohya safetensors format)
    #[arg(long)] lora: Option<String>,
    /// LoRA merge strength (default 1.0)
    #[arg(long, default_value = "1.0")] lora_scale: f32,
    /// Fixed seed for reproducible noise. Saves noise to temp/init_noise.safetensors.
    /// Omit to use existing temp/init_noise.safetensors, or random if that file is absent.
    #[arg(long)] seed: Option<u64>,
}

/// Load tensors matching `prefix` (e.g. "double_blocks.0.") from a mmap'd
/// safetensors file, move them to `gpu`, and return a VarBuilder backed by
/// those GPU tensors.  Only the matched tensors touch CPU RAM — they are
/// immediately moved to GPU and the CPU copy is dropped.
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

/// Load tensors matching `prefix` from a mmap'd SafeTensors view onto CPU only.
/// Safe to call from a background thread — no GPU involvement.
fn load_cpu_block_from_st(
    st: &SafeTensors<'_>,
    prefix: &str,
) -> anyhow::Result<HashMap<String, Tensor>> {
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

/// Upload a CPU tensor map to GPU with dtype cast — returns the raw map for optional in-place edits.
fn cpu_to_gpu_map(
    cpu_map: HashMap<String, Tensor>,
    gpu: &Device,
    dtype: DType,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut m = HashMap::new();
    for (k, v) in cpu_map {
        m.insert(k, v.to_dtype(dtype)?.to_device(gpu)?);
    }
    Ok(m)
}

/// Upload a CPU tensor map to GPU with dtype cast and wrap in a VarBuilder.
fn cpu_to_gpu_vb(
    cpu_map: HashMap<String, Tensor>,
    gpu: &Device,
    dtype: DType,
) -> anyhow::Result<VarBuilder<'static>> {
    let m = cpu_to_gpu_map(cpu_map, gpu, dtype)?;
    Ok(VarBuilder::from_tensors(m, dtype, gpu))
}

/// Seeded Gaussian noise via Box-Muller + LCG (no extra deps).
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
        if out.len() < n { out.push(r * a.sin()); }
    }
    out
}

fn load_required_embedding<'a>(
    emb: &'a HashMap<String, Tensor>,
    key: &str,
    file_path: &str,
) -> anyhow::Result<&'a Tensor> {
    emb.get(key).ok_or_else(|| {
        let mut keys: Vec<&str> = emb.keys().map(String::as_str).collect();
        keys.sort_unstable();
        anyhow::anyhow!(
            "Missing key '{}' in {}. Available keys: {}",
            key,
            file_path,
            if keys.is_empty() {
                "<none>".to_string()
            } else {
                keys.join(", ")
            }
        )
    })
}

fn run() -> Result<()> {
    let args = Args::parse();
    let gpu = Device::cuda_if_available(0)?;
    let dtype = if gpu.is_cuda() {
        device_config::auto_cuda_dtype(0)?
    } else {
        DType::F32
    };
    println!("Device: {:?}  dtype: {}", gpu, device_config::dtype_label(dtype));
    let cfg = flux::model::Config::dev();
    let h = cfg.hidden_size; // 3072

    // --- Load prompt embeddings ---
    let emb: HashMap<String, Tensor> =
        candle_core::safetensors::load(&args.embeddings, &gpu)
            .with_context(|| format!("loading prompt embeddings from {}", args.embeddings))?;
    let t5_emb = load_required_embedding(&emb, "t5_emb", &args.embeddings)?.to_dtype(dtype)?;
    let clip_emb = load_required_embedding(&emb, "clip_emb", &args.embeddings)?.to_dtype(dtype)?;
    println!("T5: {:?}  CLIP: {:?}", t5_emb.dims(), clip_emb.dims());

    // --- Initial latent noise ---
    let width  = (args.width  / 16) * 16;
    let height = (args.height / 16) * 16;
    if width != args.width || height != args.height {
        println!("Rounded {}x{} → {}x{} (must be multiple of 16)", args.width, args.height, width, height);
    }
    let h_lat = height / 8;
    let w_lat = width  / 8;
    let noise_path = "temp/init_noise.safetensors";
    let lat_img = if let Some(seed) = args.seed {
        let n = h_lat * w_lat * 16;
        let data = seeded_randn(seed, n);
        let noise = Tensor::from_vec(data, (1usize, 16usize, h_lat, w_lat), &Device::Cpu)?
            .to_device(&gpu)?.to_dtype(dtype)?;
        std::fs::create_dir_all("temp").context("creating temp directory for init noise")?;
        let mut m = HashMap::new();
        m.insert("latents", noise.to_dtype(DType::F32)?);
        candle_core::safetensors::save(&m, noise_path)
            .with_context(|| format!("saving init noise to {noise_path}"))?;
        println!("Seed {seed} → noise saved to {noise_path}");
        noise
    } else if Path::new(noise_path).exists() {
        let noise_map: HashMap<String, Tensor> =
            candle_core::safetensors::load(noise_path, &gpu)
                .with_context(|| format!("loading init noise from {noise_path}"))?;
        let noise = noise_map.get("latents")
            .ok_or_else(|| anyhow::anyhow!("Missing key 'latents' in {noise_path}"))?;
        let expected = [1usize, 16, h_lat, w_lat];
        if noise.dims() == expected {
            println!("Loaded noise from {noise_path}  shape: {:?}", noise.dims());
            noise.to_dtype(dtype)?
        } else {
            println!("Noise shape mismatch: file={:?} expected={:?} → sampling fresh random noise", noise.dims(), expected);
            Tensor::randn(0f32, 1.0, (1usize, 16usize, h_lat, w_lat), &gpu)?.to_dtype(dtype)?
        }
    } else {
        println!("Sampling random noise (use --seed N for reproducibility)");
        Tensor::randn(0f32, 1.0, (1usize, 16usize, h_lat, w_lat), &gpu)?.to_dtype(dtype)?
    };

    // --- Memory-map transformer weights (OS pages from SSD on demand) ---
    println!("Memory-mapping transformer weights from SSD...");
    let t0 = Instant::now();
    let model_path = args
        .model
        .as_deref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| path_config::model_path("flux_candle.safetensors"));
    println!("  model: {}", model_path.display());
    let model_file = std::fs::File::open(&model_path)
        .with_context(|| format!("opening transformer model {}", model_path.display()))?;
    let model_mmap = Arc::new(unsafe { MmapOptions::new().map(&model_file) }
        .with_context(|| format!("memory-mapping transformer model {}", model_path.display()))?);
    let st = SafeTensors::deserialize(&model_mmap[..])
        .with_context(|| format!("reading safetensors from transformer model {}", model_path.display()))?;
    println!("  mmap ready in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Build permanent GPU modules (small, always resident on GPU) ---
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
    drop(st); // release mmap borrow — background thread will create its own view

    // --- Load optional LoRA ---
    let lora = args.lora.as_ref().map(|path| {
        println!("Loading LoRA: {path} (scale={:.2})", args.lora_scale);
        lora::LoraWeights::load_auto(path, args.lora_scale, &gpu, dtype)
            .with_context(|| format!("loading LoRA weights from {path}"))
    }).transpose()?;
    if let Some(ref l) = lora {
        println!("  LoRA unet_single={} layers", l.unet_single_count());
    }

    // --- Setup state ---
    let state = flux::sampling::State::new(&t5_emb, &clip_emb, &lat_img)?;
    let txt_len = state.txt.dim(1)?;

    // --- Compute PE (rotary embeddings, constant across steps) ---
    let ids = Tensor::cat(&[&state.txt_ids, &state.img_ids], 1)?;
    let pe = pe_embedder.forward(&ids)?;

    // --- Denoising loop ---
    let steps = args.steps;
    let image_seq_len = state.img.dim(1)?;
    let timesteps = flux::sampling::get_schedule(steps, Some((image_seq_len, 0.5, 1.15)));
    let guidance_val = args.guidance;

    // packed_lat: latent we update each Euler step (before img_in projection)
    let mut packed_lat = state.img.clone();
    // Pre-cast state.vec once — it's constant across all steps
    let state_vec = state.vec.to_dtype(dtype)?;

    // --- Prefetch thread: loads next block CPU tensors while GPU runs current block ---
    // Overlap: background reads mmap→CPU while main thread runs GPU forward + PCIe upload.
    let (pfx_tx, pfx_rx) = mpsc::sync_channel::<anyhow::Result<HashMap<String, Tensor>>>(1);
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

        // Build vec_ (time + guidance + clip vector) — matches Flux::forward exactly
        let t_batch = Tensor::new(&[t], &gpu)?.to_dtype(dtype)?;
        let g_batch = Tensor::new(&[guidance_val], &gpu)?.to_dtype(dtype)?;

        let t_emb = flux_blocks::timestep_embedding(&t_batch, 256, dtype)?;
        let g_emb = flux_blocks::timestep_embedding(&g_batch, 256, dtype)?;

        // MLP: silu(l1(x)) → l2
        let t_vec = nn::ops::silu(&ti_l1.forward(&t_emb)?)?.apply(&ti_l2)?;
        let g_vec = nn::ops::silu(&gi_l1.forward(&g_emb)?)?.apply(&gi_l2)?;
        let y_vec = nn::ops::silu(&vi_l1.forward(&state_vec)?)?  
            .apply(&vi_l2)?;
        let vec_ = (t_vec + g_vec + y_vec)?;

        let total_blocks = cfg.depth + cfg.depth_single_blocks;

        // Project packed latent and text for this step
        let mut img = img_in.forward(&packed_lat)?;
        let mut txt = txt_in.forward(&state.txt)?;

        // --- Stream double blocks (mmap → CPU → GPU → run → drop) ---
        for i in 0..cfg.depth {
            print_bar(step + 1, steps, i, total_blocks, t0_step.elapsed().as_secs_f32(), false);
            let cpu_map = pfx_rx.recv()??;
            let vb = cpu_to_gpu_vb(cpu_map, &gpu, dtype)?;
            let block = flux_blocks::DoubleStreamBlock::new(&cfg, vb)?;
            let (new_img, new_txt) = block.forward(&img, &txt, &vec_, &pe)?;
            img = new_img;
            txt = new_txt;
        }

        // --- Stream single blocks (mmap → CPU → GPU → [+LoRA delta on GPU] → run → drop) ---
        let mut merged = Tensor::cat(&[&txt, &img], 1)?;
        for i in 0..cfg.depth_single_blocks {
            print_bar(step + 1, steps, cfg.depth + i, total_blocks, t0_step.elapsed().as_secs_f32(), false);
            let cpu_map = pfx_rx.recv()??;
            let mut gpu_map = cpu_to_gpu_map(cpu_map, &gpu, dtype)?;
            if let Some(ref l) = lora {
                l.apply_gpu_deltas(&mut gpu_map, i);
            }
            let vb = VarBuilder::from_tensors(gpu_map, dtype, &gpu);
            let block = flux_blocks::SingleStreamBlock::new(&cfg, vb)?;
            merged = block.forward(&merged, &vec_, &pe)?;
        }

        // Remove txt tokens → [1, img_patches, h]
        img = merged.i((.., txt_len.., ..))?;

        // Final layer — matches candle's LastLayer::forward:
        // 1. norm, 2. scale/shift, 3. linear
        let normed = img.apply(&fl_norm)?;
        let chunks = nn::ops::silu(&vec_)?.apply(&fl_mod)?.chunk(2, 1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]); // diffusers: [scale, shift] (scale=first)
        let v_pred = normed
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?
            .apply(&fl_lin)?;

        // Euler step on packed latent: x_new = x + dt * v_pred
        let dt_t = Tensor::new(&[dt], &gpu)?.to_dtype(dtype)?;
        packed_lat = (packed_lat + v_pred.broadcast_mul(&dt_t.reshape((1, 1, 1))?)?)?;

        print_bar(step + 1, steps, total_blocks, total_blocks, t0_step.elapsed().as_secs_f32(), true);
    }

    // Save packed latents [B, patches, channels].
    // Unpack manually on the Python side when needed.
    println!("Packed latent shape: {:?}", packed_lat.dims());
    let lat_f32 = packed_lat.to_dtype(DType::F32)?;
    let mut out = HashMap::new();
    out.insert("latents", lat_f32);
    // Store image dimensions so vae_decode doesn't need --width/--height
    out.insert("width",  Tensor::new(&[width  as u32], &Device::Cpu)?);
    out.insert("height", Tensor::new(&[height as u32], &Device::Cpu)?);
    if let Some(parent) = Path::new(&args.out_latents).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory {}", parent.display()))?;
        }
    }
    candle_core::safetensors::save(&out, &args.out_latents)
        .with_context(|| format!("saving latents to {}", args.out_latents))?;
    println!("Saved packed latents -> {} ({}x{})", args.out_latents, width, height);

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        for cause in err.chain().skip(1) {
            eprintln!("Caused by: {cause}");
        }
        std::process::exit(1);
    }
}
