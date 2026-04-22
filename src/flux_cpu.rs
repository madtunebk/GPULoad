// flux_cpu.rs — FLUX inference entirely on CPU, F32, no streaming.
// Purpose: isolate whether purple output is caused by GPU/BF16/streaming.
// If this gives clean output, the bug is in the GPU path.

mod flux_blocks;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{self as nn, LayerNorm, Module, VarBuilder};
use candle_transformers::models::flux;
use std::collections::HashMap;
use std::time::Instant;


fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let dtype = DType::F32;
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .ok();
    println!("Device: CPU  dtype: F32  threads: {}", rayon::current_num_threads());

    let cfg = flux::model::Config::dev();
    let h = cfg.hidden_size; // 3072

    // --- Load prompt embeddings ---
    let emb: HashMap<String, Tensor> =
        candle_core::safetensors::load("temp/prompt_embeds_from_flux_ref.safetensors", &cpu)?;
    let t5_emb = emb["t5_emb"].to_dtype(dtype)?;
    let clip_emb = emb["clip_emb"].to_dtype(dtype)?;
    println!("T5: {:?}  CLIP: {:?}", t5_emb.dims(), clip_emb.dims());

    // --- Initial latent noise ---
    let init_noise_path = std::path::Path::new("temp/init_noise.safetensors");
    let lat_img = if init_noise_path.exists() {
        println!("Loading shared init noise from temp/init_noise.safetensors");
        let noise_map: HashMap<String, Tensor> =
            candle_core::safetensors::load("temp/init_noise.safetensors", &cpu)?;
        let noise = noise_map
            .get("latents")
            .ok_or_else(|| anyhow::anyhow!("Missing key 'latents'"))?;
        noise.to_dtype(dtype)?
    } else {
        println!("No shared init noise found. Sampling random noise.");
        Tensor::randn(0f32, 1.0, (1usize, 16usize, 64usize, 64usize), &cpu)?
    };

    // --- Memory-map transformer weights (OS pages from SSD on demand, no upfront RAM copy) ---
    println!("Memory-mapping transformer weights from SSD...");
    let t0 = Instant::now();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&["models/flux_candle.safetensors"], dtype, &cpu)?
    };
    println!("  mmap ready in {:.1}s", t0.elapsed().as_secs_f32());

    // --- Build all modules (weights paged from SSD as accessed) ---
    let img_in = nn::linear(cfg.in_channels, h, vb.pp("img_in"))?;
    let txt_in = nn::linear(cfg.context_in_dim, h, vb.pp("txt_in"))?;

    let ti_l1 = nn::linear(256, h, vb.pp("time_in").pp("in_layer"))?;
    let ti_l2 = nn::linear(h, h, vb.pp("time_in").pp("out_layer"))?;
    let gi_l1 = nn::linear(256, h, vb.pp("guidance_in").pp("in_layer"))?;
    let gi_l2 = nn::linear(h, h, vb.pp("guidance_in").pp("out_layer"))?;
    let vi_l1 = nn::linear(cfg.vec_in_dim, h, vb.pp("vector_in").pp("in_layer"))?;
    let vi_l2 = nn::linear(h, h, vb.pp("vector_in").pp("out_layer"))?;

    let fl_mod = nn::linear(h, 2 * h, vb.pp("final_layer").pp("adaLN_modulation").pp("1"))?;
    let fl_lin = nn::linear(h, cfg.in_channels, vb.pp("final_layer").pp("linear"))?;
    let fl_norm = LayerNorm::new_no_bias(Tensor::ones(h, dtype, &cpu)?, 1e-6);

    let pe_embedder = flux::model::EmbedNd::new(
        cfg.hidden_size / cfg.num_heads,
        cfg.theta,
        cfg.axes_dim.clone(),
    );

    // No blocks pre-built — we stream one block at a time to keep RAM low.

    // --- Setup state ---
    let state = flux::sampling::State::new(&t5_emb, &clip_emb, &lat_img)?;
    let txt_len = state.txt.dim(1)?;

    let ids = Tensor::cat(&[&state.txt_ids, &state.img_ids], 1)?;
    let pe = pe_embedder.forward(&ids)?;

    // --- Denoising loop ---
    let steps = 8usize;
    let image_seq_len = state.img.dim(1)?;
    let timesteps = flux::sampling::get_schedule(steps, Some((image_seq_len, 0.5, 1.15)));
    let guidance_val = 3.5f32;

    let mut packed_lat = state.img.clone();

    for (step, window) in timesteps.windows(2).enumerate() {
        let (t, t_next) = match window {
            [a, b] => (*a as f32, *b as f32),
            _ => continue,
        };
        let dt = t_next - t;
        let t0_step = Instant::now();

        let t_batch = Tensor::new(&[t], &cpu)?.to_dtype(dtype)?;
        let g_batch = Tensor::new(&[guidance_val], &cpu)?.to_dtype(dtype)?;

        let t_emb = flux_blocks::timestep_embedding(&t_batch, 256, dtype)?;
        let g_emb = flux_blocks::timestep_embedding(&g_batch, 256, dtype)?;

        let t_vec = nn::ops::silu(&ti_l1.forward(&t_emb)?)?.apply(&ti_l2)?;
        let g_vec = nn::ops::silu(&gi_l1.forward(&g_emb)?)?.apply(&gi_l2)?;
        let y_vec = nn::ops::silu(&vi_l1.forward(&state.vec.to_dtype(dtype)?)?)?
            .apply(&vi_l2)?;
        let vec_ = (t_vec + g_vec + y_vec)?;

        println!("Step {}/{} t={:.4}", step + 1, steps, t);

        let mut img = img_in.forward(&packed_lat)?;
        let mut txt = txt_in.forward(&state.txt)?;

        // Build each double block, run it, then drop it immediately.
        for i in 0..cfg.depth {
            print!("  double {}/{}\r", i + 1, cfg.depth);
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let block = flux_blocks::DoubleStreamBlock::new(&cfg, vb.pp(format!("double_blocks.{}", i)))?;
            let (new_img, new_txt) = block.forward(&img, &txt, &vec_, &pe)?;
            img = new_img;
            txt = new_txt;
        }
        println!("  double {0}/{0} done", cfg.depth);

        let mut merged = Tensor::cat(&[&txt, &img], 1)?;
        // Build each single block, run it, then drop it immediately.
        for i in 0..cfg.depth_single_blocks {
            print!("  single {}/{}\r", i + 1, cfg.depth_single_blocks);
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let block = flux_blocks::SingleStreamBlock::new(&cfg, vb.pp(format!("single_blocks.{}", i)))?;
            merged = block.forward(&merged, &vec_, &pe)?;
        }
        println!("  single {0}/{0} done", cfg.depth_single_blocks);

        img = merged.i((.., txt_len.., ..))?;

        let normed = img.apply(&fl_norm)?;
        let chunks = nn::ops::silu(&vec_)?.apply(&fl_mod)?.chunk(2, 1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]); // diffusers: [scale, shift] (scale=first)
        let v_pred = normed
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?
            .apply(&fl_lin)?;

        let dt_t = Tensor::new(&[dt], &cpu)?.to_dtype(dtype)?;
        packed_lat = (packed_lat + v_pred.broadcast_mul(&dt_t.reshape((1, 1, 1))?)?)?;

        println!("  step done in {:.1}s", t0_step.elapsed().as_secs_f32());
    }

    println!("Packed latent shape: {:?}", packed_lat.dims());
    let mut out = HashMap::new();
    out.insert("latents", packed_lat);
    candle_core::safetensors::save(&out, "temp/latents_cpu.safetensors")?;
    println!("Saved -> temp/latents_cpu.safetensors");
    println!("Decode: python scripts/decode_latents.py --latents temp/latents_cpu.safetensors --out temp/output_cpu.png");

    Ok(())
}
