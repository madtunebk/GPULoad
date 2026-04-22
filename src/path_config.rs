#![allow(dead_code)]

use std::path::PathBuf;

pub fn model_dir() -> PathBuf {
    std::env::var_os("STREAMFORGE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
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

    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join("models--black-forest-labs--FLUX.1-dev")
}