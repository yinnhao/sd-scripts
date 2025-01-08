from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import torch
torch.manual_seed(0)

repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
print(model.config)

noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
print(noisy_sample.shape)

with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample
    
# print(noisy_residual.shape)
print(type(noisy_residual))

# scheduler = DDPMScheduler.from_pretrained(repo_id)
# print(scheduler)
