use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::flux::sampling::unpack;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};
//use clap::Parser;
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::path::Path;

const BAR_WIDTH: usize = 30;
const SCALING_FACTOR: f64 = 0.3611;
const SHIFT_FACTOR: f64 = 0.1159;
const VAE_SCALE: usize = 8;

//use crate::path_config;
use crate::funcs::flux_gpu::Latents;

fn print_bar(done: usize, total: usize, elapsed: f32, finished: bool) {
    let filled = (done * BAR_WIDTH).div_ceil(total.max(1)).min(BAR_WIDTH);
    let bar: String = (0..BAR_WIDTH).map(|i| if i < filled { '█' } else { '░' }).collect();
    let end = if finished { '\n' } else { '\r' };
    print!("  Tiles [{bar}] {done:>3}/{total}  {elapsed:.1}s{end}");
    let _ = std::io::stdout().flush();
}

fn flux_vae_config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: 16,
        norm_num_groups: 32,
        use_quant_conv: false,
        use_post_quant_conv: false,
    }
}

fn edge_weight(pos: usize, size: usize, blend: usize) -> f32 {
    let from_edge = pos.min(size - 1 - pos);
    if blend == 0 || from_edge >= blend {
        1.0
    } else {
        (from_edge as f32 + 0.5) / blend as f32
    }
}

fn load_vae(vae_model: &std::path::Path, device: &Device) -> Result<AutoEncoderKL> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[vae_model], DType::F32, device)
            .with_context(|| format!("loading VAE weights from {}", vae_model.display()))?
    };
    Ok(AutoEncoderKL::new(vb, 3, 3, flux_vae_config())?)
}

fn tile_starts(dim: usize, tile_size: usize, tile_stride: usize) -> Vec<usize> {
    let mut starts = vec![];
    let mut s = 0usize;
    loop {
        starts.push(s);
        if s + tile_size >= dim {
            break;
        }
        s = (s + tile_stride).min(dim - tile_size);
    }
    starts
}

fn tiled_decode(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    out_h: usize,
    out_w: usize,
    tile_size: usize,
    tile_overlap: usize,
) -> Result<Tensor> {
    let (_, _, lh, lw) = latents.dims4()?;
    let tile_stride = tile_size - 2 * tile_overlap;
    let img_blend = tile_overlap * VAE_SCALE;

    let ys = tile_starts(lh, tile_size, tile_stride);
    let xs = tile_starts(lw, tile_size, tile_stride);
    let total = ys.len() * xs.len();
    let t0 = Instant::now();

    let mut acc: Vec<f32> = vec![0.0; out_h * out_w * 3];
    let mut wts: Vec<f32> = vec![0.0; out_h * out_w * 3];
    let mut done = 0usize;

    for &ty in &ys {
        let th = tile_size.min(lh - ty);
        for &tx in &xs {
            let tw = tile_size.min(lw - tx);
            print_bar(done, total, t0.elapsed().as_secs_f32(), false);

            let tile = latents.narrow(2, ty, th)?.narrow(3, tx, tw)?;
            let decoded = vae
                .decode(&tile)?
                .clamp(-1f64, 1f64)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?;

            blend_tile(&decoded, &mut acc, &mut wts, ty, tx, th, tw, out_h, out_w, img_blend)?;
            done += 1;
        }
    }
    print_bar(total, total, t0.elapsed().as_secs_f32(), true);

    let pixels: Vec<f32> = acc.iter().zip(&wts).map(|(a, w)| a / w).collect();
    Ok(Tensor::from_vec(pixels, (1usize, 3, out_h, out_w), &Device::Cpu)?)
}

fn tiled_decode_multi_gpu(
    vaes: &[AutoEncoderKL],
    latents_cpu: &Tensor,
    devices: &[Device],
    out_h: usize,
    out_w: usize,
    tile_size: usize,
    tile_overlap: usize,
) -> Result<Tensor> {
    let (_, _, lh, lw) = latents_cpu.dims4()?;
    let tile_stride = tile_size - 2 * tile_overlap;
    let img_blend = tile_overlap * VAE_SCALE;
    let num_gpus = vaes.len();

    let ys = tile_starts(lh, tile_size, tile_stride);
    let xs = tile_starts(lw, tile_size, tile_stride);
    let xs_len = xs.len();
    let tiles: Vec<(usize, usize, usize, usize, usize)> = ys
        .iter()
        .enumerate()
        .flat_map(|(yi, &ty)| {
            let th = tile_size.min(lh - ty);
            xs.iter()
                .enumerate()
                .map(move |(xi, &tx)| {
                    let tw = tile_size.min(lw - tx);
                    let tile_idx = yi * xs_len + xi;
                    (ty, tx, th, tw, tile_idx)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let total = tiles.len();
    let t0 = Instant::now();
    let done_count = Arc::new(AtomicUsize::new(0));

    let results: Vec<Result<(usize, usize, usize, usize, Vec<f32>)>> = tiles
        .par_iter()
        .map(|&(ty, tx, th, tw, tile_idx)| {
            let gpu_idx = tile_idx % num_gpus;
            let dev = &devices[gpu_idx];
            let vae = &vaes[gpu_idx];

            let tile_cpu = latents_cpu.narrow(2, ty, th)?.narrow(3, tx, tw)?;
            let tile = tile_cpu.to_device(dev)?;

            let decoded = vae
                .decode(&tile)?
                .clamp(-1f64, 1f64)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?;

            let data = decoded.flatten_all()?.to_vec1::<f32>()?;

            let n = done_count.fetch_add(1, Ordering::Relaxed) + 1;
            print_bar(n, total, t0.elapsed().as_secs_f32(), n == total);

            Ok((ty, tx, th, tw, data))
        })
        .collect();

    let mut acc: Vec<f32> = vec![0.0; out_h * out_w * 3];
    let mut wts: Vec<f32> = vec![0.0; out_h * out_w * 3];

    for res in results {
        let (ty, tx, th, tw, data) = res?;
        let pix_h = th * VAE_SCALE;
        let pix_w = tw * VAE_SCALE;
        let iy = ty * VAE_SCALE;
        let ix = tx * VAE_SCALE;

        for c in 0..3usize {
            for y in 0..pix_h {
                let wy = edge_weight(y, pix_h, img_blend);
                for x in 0..pix_w {
                    let wx = edge_weight(x, pix_w, img_blend);
                    let w = wy * wx;
                    let src = c * pix_h * pix_w + y * pix_w + x;
                    let dst = c * out_h * out_w + (iy + y) * out_w + (ix + x);
                    acc[dst] += data[src] * w;
                    wts[dst] += w;
                }
            }
        }
    }

    let pixels: Vec<f32> = acc.iter().zip(&wts).map(|(a, w)| a / w).collect();
    Ok(Tensor::from_vec(pixels, (1usize, 3, out_h, out_w), &Device::Cpu)?)
}

fn blend_tile(
    decoded: &Tensor,
    acc: &mut [f32],
    wts: &mut [f32],
    ty: usize,
    tx: usize,
    th: usize,
    tw: usize,
    out_h: usize,
    out_w: usize,
    img_blend: usize,
) -> Result<()> {
    let pix_h = th * VAE_SCALE;
    let pix_w = tw * VAE_SCALE;
    let iy = ty * VAE_SCALE;
    let ix = tx * VAE_SCALE;
    let data = decoded.flatten_all()?.to_vec1::<f32>()?;

    for c in 0..3usize {
        for y in 0..pix_h {
            let wy = edge_weight(y, pix_h, img_blend);
            for x in 0..pix_w {
                let wx = edge_weight(x, pix_w, img_blend);
                let w = wy * wx;
                let src = c * pix_h * pix_w + y * pix_w + x;
                let dst = c * out_h * out_w + (iy + y) * out_w + (ix + x);
                acc[dst] += data[src] * w;
                wts[dst] += w;
            }
        }
    }
    Ok(())
}

pub fn run(latents: &Latents, output: &str, tile_size: usize, tile_overlap: usize, vae_model: &Path, multi_gpu: bool) -> Result<()> {
    println!("Running VAE decode... {}", multi_gpu.then(|| "(multi-gpu)").unwrap_or("(single-gpu)"));

    #[cfg(feature = "cuda")]
    let (devices, use_multi) = {
        if multi_gpu {
            let mut devs = vec![];
            for i in 0..16 {
                match Device::new_cuda(i) {
                    Ok(d) => devs.push(d),
                    Err(_) => break,
                }
            }
            if devs.is_empty() {
                devs.push(Device::Cpu);
                (devs, false)
            } else {
                let n = devs.len();
                (devs, n > 1)
            }
        } else {
            (vec![Device::new_cuda(0)?], false)
        }
    };
    #[cfg(not(feature = "cuda"))]
    let (devices, use_multi) = (vec![Device::Cpu], false);

    for (i, d) in devices.iter().enumerate() {
        println!("Device[{}]: {:?}", i, d);
    }

    let width = latents.width;
    let height = latents.height;
    println!("  image size: {}x{}", width, height);

    let packed = latents.packed.to_dtype(DType::F32)?;
    println!("  packed shape: {:?}", packed.dims());

    let latents_cpu = {
        let raw = unpack(&packed, height, width)?;
        ((raw / SCALING_FACTOR)? + SHIFT_FACTOR)?
    };
    println!("  unpacked shape: {:?}", latents_cpu.dims());

    println!("Loading VAE weights...");
    let t0 = Instant::now();

    let vaes: Vec<AutoEncoderKL> = devices
        .iter()
        .enumerate()
        .map(|(i, dev)| {
            println!("  loading VAE on device[{}]...", i);
            load_vae(vae_model, dev)
        })
        .collect::<Result<_>>()?;

    println!("  VAE(s) ready in {:.1}s", t0.elapsed().as_secs_f32());

    println!(
        "Decoding (tile_size={}, overlap={}, gpus={})...",
        tile_size,
        tile_overlap,
        vaes.len()
    );
    let t1 = Instant::now();

    let decoded = if tile_size == 0 {
        let lat = latents_cpu.to_device(&devices[0])?;
        vaes[0]
            .decode(&lat)?
            .clamp(-1f64, 1f64)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
    } else if use_multi && vaes.len() > 1 {
        tiled_decode_multi_gpu(&vaes, &latents_cpu, &devices, height, width, tile_size, tile_overlap)?
    } else {
        let lat = latents_cpu.to_device(&devices[0])?;
        tiled_decode(&vaes[0], &lat, height, width, tile_size, tile_overlap)?
    };

    println!(
        "  decoded in {:.1}s  shape: {:?}",
        t1.elapsed().as_secs_f32(),
        decoded.dims()
    );

    let img_t = ((decoded + 1.0)? * 127.5)?
        .clamp(0f64, 255f64)?
        .to_dtype(DType::U8)?;

    let (_, _, h, w) = img_t.dims4()?;
    let r = img_t.i((0, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let g = img_t.i((0, 1))?.flatten_all()?.to_vec1::<u8>()?;
    let b = img_t.i((0, 2))?.flatten_all()?.to_vec1::<u8>()?;

    let mut imgbuf: ImageBuffer<Rgb<u8>, _> = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            imgbuf.put_pixel(x as u32, y as u32, Rgb([r[y * w + x], g[y * w + x], b[y * w + x]]));
        }
    }

    if let Some(parent) = std::path::Path::new(output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    imgbuf
        .save(output)
        .with_context(|| format!("saving PNG to {}", output))?;
    println!("Saved \u{2192} {}", output);

    Ok(())
}    