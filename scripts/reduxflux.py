import torch
from diffusers import FluxPriorReduxPipeline
from diffusers.utils import load_image

pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16).to("cuda")


image = load_image("https://cdni.pornpics.com/1280/7/491/48938277/48938277_011_a676.jpg")
pipe_prior_output = pipe_prior_redux(image)

# Save in the same format as text_encoder: t5_emb + clip_emb
from safetensors.torch import save_file
save_file(
    {
        "t5_emb":   pipe_prior_output.prompt_embeds[0].unsqueeze(0).to(torch.bfloat16).cpu(),
        "clip_emb": pipe_prior_output.pooled_prompt_embeds[0].unsqueeze(0).to(torch.bfloat16).cpu(),
    },
    "temp/prompt_embeds.safetensors",
)

print("t5_emb :", pipe_prior_output.prompt_embeds.shape)
print("clip_emb:", pipe_prior_output.pooled_prompt_embeds.shape)
print("Image   :", image.size)
print("Saved -> temp/prompt_embeds.safetensors")

# Round to nearest multiple of 16 (FLUX requirement)
w = (image.width  // 16) * 16
h = (image.height // 16) * 16

print(f"\nRun:")
print(f"  ./target/release/flux_gpu --embeddings temp/prompt_embeds.safetensors --width {w} --height {h}")
print(f"  ./target/release/vae_decode --latents temp/latents.safetensors --width {w} --height {h} --out temp/vae_output.png")