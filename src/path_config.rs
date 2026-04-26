#![allow(dead_code)]

use std::path::PathBuf;

pub fn model_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("STREAMFORGE_MODEL_DIR") {
        return PathBuf::from(path);
    }

    // Check relative to CWD first (running from project root).
    let cwd_models = PathBuf::from("models");
    if cwd_models.exists() {
        return cwd_models;
    }

    // Walk up from the executable to find a directory containing `models/`
    // (handles running from target/release/ on both Windows and Linux).
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        while let Some(d) = dir {
            let candidate = d.join("models");
            if candidate.exists() {
                return candidate;
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }
    }

    PathBuf::from("models")
}

pub fn model_path(file_name: &str) -> PathBuf {
    model_dir().join(file_name)
}

pub fn clip_tokenizer_path() -> PathBuf {
    model_dir().join("tokenizer").join("tokenizer.json")
}

pub fn hf_repo_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("STREAMFORGE_HF_REPO_DIR") {
        return PathBuf::from(path);
    }

    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join("models--black-forest-labs--FLUX.1-dev")
}