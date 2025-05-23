from diffusers import StableDiffusionPipeline
import torch

# Base model
base_model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16
)

# Move to GPU
pipe = pipe.to("cuda")

# Load LoRA weights
lora_path = "C:/Users/matte/Desktop/sd-finetuned-durer"
pipe.load_lora_weights(lora_path)

# Prompt and inference a
prompt = "a lion laying in a cave, detailed lines and hatching, by Durer"
image = pipe(prompt).images[0]

image.show()
image.save("C:/Users/matte/Desktop/lione_in_a_cave.png")  # ← Save to Desktop
