// vae_decode.rs — Decode FLUX packed latents to PNG using candle VAE.
// Input:  temp/latents.safetensors  (packed [1, patches, 64])
// Output: temp/output_rust.png

mod path_config;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use candle_transformers::models::flux::sampling::unpack;
use clap::Parser;
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

const BAR_WIDTH: usize = 30;

fn print_bar(done: usize, total: usize, elapsed: f32, finished: bool) {
    let filled = (done * BAR_WIDTH).div_ceil(total.max(1)).min(BAR_WIDTH);
    let bar: String = (0..BAR_WIDTH).map(|i| if i < filled { '\u{2588}' } else { '\u{2591}' }).collect();
    let end = if finished { '\n' } else { '\r' };
    print!("  Tiles [{bar}] {done:>3}/{total}  {elapsed:.1}s{end}");
    let _ = std::io::stdout().flush();
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "temp/latents.safetensors")] latents: String,
    #[arg(long, default_value = "temp/output_rust.png")]
    #[arg(alias = "output")] out: String,
    /// If omitted, read from the latents file (saved by flux_gpu)
    #[arg(long)] width: Option<usize>,
    /// If omitted, read from the latents file (saved by flux_gpu)
    #[arg(long)] height: Option<usize>,
    /// Latent tile size for tiled decode (0 = decode whole latent at once)
    #[arg(long, default_value = "64")] tile_size: usize,
    /// Overlap between tiles in latent pixels
    #[arg(long, default_value = "8")]  tile_overlap: usize,
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

const SCALING_FACTOR: f64 = 0.3611;
const SHIFT_FACTOR: f64   = 0.1159;
const VAE_SCALE: usize    = 8; // latent → pixel scale factor

/// Linear blend weight: ramps up over `blend` pixels from each edge.
fn edge_weight(pos: usize, size: usize, blend: usize) -> f32 {
    let from_edge = pos.min(size - 1 - pos);
    if blend == 0 || from_edge >= blend { 1.0 }
    else { (from_edge as f32 + 0.5) / blend as f32 }
}

/// Decode latents in overlapping tiles, blend seams on CPU.
fn tiled_decode(
    vae: &AutoEncoderKL,
    latents: &Tensor,  // [1, 16, lh, lw]
    out_h: usize,
    out_w: usize,
    tile_size: usize,
    tile_overlap: usize,
    _device: &Device,
) -> Result<Tensor> {
    let (_, _, lh, lw) = latents.dims4()?;
    let tile_stride = tile_size - 2 * tile_overlap;
    let img_blend = tile_overlap * VAE_SCALE;

    // Tile start positions (always include a tile anchored at the far edge)
    let tile_starts = |dim: usize| -> Vec<usize> {
        let mut starts = vec![];
        let mut s = 0usize;
        loop {
            starts.push(s);
            if s + tile_size >= dim { break; }
            s = (s + tile_stride).min(dim - tile_size);
        }
        starts
    };

    let ys = tile_starts(lh);
    let xs = tile_starts(lw);
    let total_tiles = ys.len() * xs.len();
    let t0 = Instant::now();

    let mut acc: Vec<f32> = vec![0.0; out_h * out_w * 3];
    let mut wts: Vec<f32> = vec![0.0; out_h * out_w * 3];
    let mut done = 0usize;

    for ty in &ys {
        let ty = *ty;
        let th = tile_size.min(lh - ty);
        for tx in &xs {
            let tx = *tx;
            let tw = tile_size.min(lw - tx);
            print_bar(done, total_tiles, t0.elapsed().as_secs_f32(), false);
            let tile = latents.narrow(2, ty, th)?.narrow(3, tx, tw)?;
            let decoded = vae.decode(&tile)?
                .clamp(-1f64, 1f64)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?;

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
            done += 1;
        }
    }
    print_bar(total_tiles, total_tiles, t0.elapsed().as_secs_f32(), true);

    let pixels: Vec<f32> = acc.iter().zip(&wts).map(|(a, w)| a / w).collect();
    Ok(Tensor::from_vec(pixels, (1usize, 3, out_h, out_w), &Device::Cpu)?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);

    println!("Loading latents: {}", args.latents);
    let lat_map: HashMap<String, Tensor> =
        candle_core::safetensors::load(&args.latents, &Device::Cpu)?;
    let key = if lat_map.contains_key("latents") { "latents" }
              else { lat_map.keys().find(|k| *k != "width" && *k != "height").context("no latents tensor found")? };
    let packed = lat_map[key].to_dtype(DType::F32)?.to_device(&device)?;
    println!("  packed shape: {:?}", packed.dims());

    // Resolve width/height: CLI arg → latents file → error
    let width = match args.width {
        Some(w) => w,
        None => lat_map.get("width")
            .context("--width not given and not found in latents file")?
            .to_vec1::<u32>()?[0] as usize,
    };
    let height = match args.height {
        Some(h) => h,
        None => lat_map.get("height")
            .context("--height not given and not found in latents file")?
            .to_vec1::<u32>()?[0] as usize,
    };
    println!("  image size: {}x{}", width, height);

    let latents = unpack(&packed, height, width)?;
    println!("  unpacked shape: {:?}", latents.dims());

    let latents = ((latents / SCALING_FACTOR)? + SHIFT_FACTOR)?;

    println!("Loading VAE weights (mmap)...");
    let t0 = Instant::now();
    let vb = unsafe {
        let vae_path = path_config::model_path("vae_candle.safetensors");
        VarBuilder::from_mmaped_safetensors(&[vae_path], DType::F32, &device)?
    };
    let vae = AutoEncoderKL::new(vb, 3, 3, flux_vae_config())?;
    println!("  VAE ready in {:.1}s", t0.elapsed().as_secs_f32());

    println!("Decoding (tile_size={}, overlap={})...", args.tile_size, args.tile_overlap);
    let t1 = Instant::now();
    let decoded = if args.tile_size == 0 {
        vae.decode(&latents)?
            .clamp(-1f64, 1f64)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
    } else {
        tiled_decode(&vae, &latents, height, width,
                     args.tile_size, args.tile_overlap, &device)?
    };
    println!("  decoded in {:.1}s  shape: {:?}", t1.elapsed().as_secs_f32(), decoded.dims());

    // [-1,1] → [0,255]
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
            imgbuf.put_pixel(x as u32, y as u32, Rgb([r[y*w+x], g[y*w+x], b[y*w+x]]));
        }
    }
    imgbuf.save(&args.out)?;
    println!("Saved -> {}", args.out);

    Ok(())
}
