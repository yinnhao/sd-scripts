from diffusers import UNet2DModel
from diffusers import DDPMScheduler
import torch
import tqdm
import PIL.Image
import numpy as np


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    # display(f"Image at step {i}")
    # display(image_pil)

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

# 定义DDPMScheduler
scheduler = DDPMScheduler.from_pretrained(repo_id)
print(scheduler)

# 执行一步去噪
less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample

# 将输入噪声和模型移动到现存上
model.to("cuda")
noisy_sample = noisy_sample.to("cuda")

sample = noisy_sample
for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        display_sample(sample, i + 1)