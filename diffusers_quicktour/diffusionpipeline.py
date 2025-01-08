from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipeline.to("cuda:0")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline("An image of a old man").images[0]
image.save("./result/image_of_squirrel_painting_Euler_2.png")


