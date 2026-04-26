use clap::Parser;
use std::path::Path;

use sd_minimal::funcs::text_encoder;
use sd_minimal::funcs::flux_gpu::{self, DenoiseParams};
use sd_minimal::funcs::vae_decode;
use sd_minimal::path_config;

pub fn path_exists(path: &Option<String>) -> bool {
    if let Some(p) = path {
        Path::new(p).is_file()
    } else {
        false
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Prompt for generation
    #[arg(short, long, default_value = "Donald Trump in a 'borat mankini' beach costume very confused by a orange tree, behind him a big banner with fallow text 'naked and afraid', full glorious body on Iran missile on the Sahara desert")]  prompt: String,
    #[arg(long, default_value_t = 1080)]  width: usize,
    #[arg(long, default_value_t = 1920)]  height: usize,
    #[arg(long, default_value_t = 8)] steps: usize,
    #[arg(long, default_value_t = 1)] seed: u64,
    #[arg(long, default_value_t = 3.5)] guidance: f32,

    /// Optional LoRA path and LoRA scale
    #[arg(short, long)]  lora: Option<String>,
    #[arg(long, default_value_t = 1.0)]  lora_scale: f32,

    //model path
    #[arg(short, long)] model: Option<String>,
    #[arg(short, long)] vae_model: Option<String>,

    /// Output path
    // Whether to save the latent representation as a .safetensors file, only to debug the generation process. The output file will be named "output_latents.safetensors".
    #[arg(short, long, default_value_t = false)] savelatens: bool,   
    // Whether to save the generated image as a .png file. The output file will be named "output.png".
    #[arg(short, long, default_value = "output.png")] output: String,

    /// Device (cpu / cuda) and dtype (f32 / bf16)
    #[arg(short, long, default_value = "cpu")] device: String,
    #[arg(short, long, default_value = "bf16")] dtype: String,
}

#[warn(unused_variables)]
fn run() -> anyhow::Result<()> {
    let args = Args::parse();
  
    if args.lora.is_some() && !path_exists(&args.lora) {
        anyhow::bail!("LoRA file not found: {}", args.lora.as_deref().unwrap_or("<unknown>"));
    }
    let model = args.model
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| path_config::model_path("flux_candle.safetensors"));
    if !model.is_file() {
        anyhow::bail!("Model file not found: {}", model.display());
    }
    let vae_model = args.vae_model
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| path_config::model_path("vae_candle.safetensors"));
    if !vae_model.is_file() {
        anyhow::bail!("VAE model file not found: {}", vae_model.display());
    }
    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            anyhow::bail!(
                "Output directory does not exist: {}",
                parent.display()
            );
        }
    }
    if args.savelatens && !Path::new("temp").is_dir() {
        anyhow::bail!("Latents output directory does not exist: temp");
    }

    let lora = args.lora.as_deref();
    if let Some(path) = lora {
        println!("LoRA {}", path);
    }

    // Stage 1: text encoding
    let (clip_emb, t5_emb) = text_encoder::encode(&args.prompt, &args.device, lora, args.lora_scale)?;

    // Stage 2: denoising
    let latents = flux_gpu::denoise(DenoiseParams {
        clip_emb: &clip_emb,
        t5_emb: &t5_emb,
        width: args.width,
        height: args.height,
        steps: args.steps,
        guidance: args.guidance,
        seed: Some(args.seed),
        lora,
        lora_scale: args.lora_scale,
        model: Some(model.to_str().unwrap_or_default()),
    })?;

    // Optionally save latents for debugging
    if args.savelatens {
        use std::collections::HashMap;
        use candle_core::Tensor;
        let out_path = "temp/output_latents.safetensors";
        let mut out = HashMap::new();
        out.insert("latents", latents.packed.clone());
        out.insert("width",  Tensor::new(&[latents.width  as u32], &candle_core::Device::Cpu)?);
        out.insert("height", Tensor::new(&[latents.height as u32], &candle_core::Device::Cpu)?);
        std::fs::create_dir_all("temp")?;
        candle_core::safetensors::save(&out, out_path)?;
        println!("Latents saved -> {out_path}");
    }

    // Stage 3: VAE decode
    vae_decode::run(&latents, &args.output, 32, 4, &vae_model, true)?;
    //vae_decode::run()?;
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