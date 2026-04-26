#![allow(dead_code)]

use anyhow::{Context, Result};
use candle_core::DType;
use std::process::Command;

pub fn ensure_cuda_feature_enabled() -> Result<()> {
    if cfg!(feature = "cuda") {
        Ok(())
    } else {
        anyhow::bail!(
            "This binary was built without CUDA support. Rebuild with: cargo build --release --features cuda"
        )
    }
}

pub fn parse_dtype_override(value: &str) -> Option<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f16" | "float16" | "half" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        "f32" | "float32" => Some(DType::F32),
        _ => None,
    }
}

fn query_compute_capability(gpu_index: usize) -> Result<Option<(u32, u32)>> {
    let output = Command::new("nvidia-smi")
        .args([
            &format!("--id={gpu_index}"),
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(output) if output.status.success() => output,
        Ok(_) => return Ok(None),
        Err(_) => return Ok(None),
    };

    let value = String::from_utf8(output.stdout)
        .context("nvidia-smi returned non-utf8 compute capability output")?;
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }

    let mut parts = value.split('.');
    let major = parts.next().and_then(|part| part.parse::<u32>().ok());
    let minor = parts.next().and_then(|part| part.parse::<u32>().ok());
    match (major, minor) {
        (Some(major), Some(minor)) => Ok(Some((major, minor))),
        _ => Ok(None),
    }
}

pub fn auto_cuda_dtype(gpu_index: usize) -> Result<DType> {
    if let Ok(value) = std::env::var("STREAMFORGE_DTYPE") {
        if let Some(dtype) = parse_dtype_override(&value) {
            return Ok(dtype);
        }
        anyhow::bail!(
            "Invalid STREAMFORGE_DTYPE='{value}'. Use one of: f16, bf16, f32"
        );
    }

    // SM8+ (Ampere+): BF16 tensor cores, matches training dtype.
    // SM<8 (T4, P100, …): cuBLAS BF16 GEMM is not supported; F16 overflows
    // in deep transformer chains → NaN. F32 is the only stable option.
    let capability = query_compute_capability(gpu_index)?;
    let dtype = match capability {
        Some((major, _)) if major >= 8 => DType::BF16,
        _ => DType::F32,
    };
    Ok(dtype)
}

pub fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::F32 => "F32",
        _ => "other",
    }
}