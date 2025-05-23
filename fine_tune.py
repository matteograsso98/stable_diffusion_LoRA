"""
This notebook guides you through:
1. Environment setup (install dependencies, authenticate).
3. Data augmentation of your 9 National Gallery images.
4. Cloning kohya-ss scripts for training.
5. Fine-tuning SD v1.5 with kohya-LoRA.
6. Generating new artworks in your learned style.

Before running, set:
- `HF_TOKEN`: Your Hugging Face token.
- `DATA_PATH`: Path to your data in Drive (e.g. `/content/drive/MyDrive/art_images_durer`).
- `DEVICE`: `'gpu'` or `'tpu'`.
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import glob
import numpy as np
from PIL import Image
from huggingface_hub import login

# 1. Set up paths and environment
HF_TOKEN = "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"  
# Get token from environment
hf_token = os.getenv("HF_TOKEN")
DATA_PATH = r"C:\Users\matte\Desktop"  # <- Your local dataset path
LORA_TRAIN_DATA_DIR = os.path.join(DATA_PATH, 'kohya_train_data')
OUTPUT_DIR = r"C:\Users\matte\Desktop\sd-finetuned-durer"
KOHYA_DIR = r"C:\kohya-sd-scripts"  # <- Where you clone the repo
os.makedirs(LORA_TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Log in to HuggingFace
login(token=HF_TOKEN)
"""
# 3. Data augmentation
import albumentations as A

aug = A.Compose([
    A.RandomRotate90(p=0.5), A.Flip(p=0.5), A.Transpose(p=0.5),
    A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
])

CONCEPT_FOLDER_NAME = "10_durerart"
CONCEPT_IMAGES_SUBDIR = os.path.join(LORA_TRAIN_DATA_DIR, CONCEPT_FOLDER_NAME)
os.makedirs(CONCEPT_IMAGES_SUBDIR, exist_ok=True)

source_image_files = glob.glob(os.path.join(DATA_PATH, '*.[pP][nN][gG]')) + \
                     glob.glob(os.path.join(DATA_PATH, '*.[jJ][pP][gG]')) + \
                     glob.glob(os.path.join(DATA_PATH, '*.[jJ][pP][eE][gG]'))

print(f"Found {len(source_image_files)} source images.")

for img_path in source_image_files:
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for i in range(20):
            out = aug(image=img_array)['image']
            Image.fromarray(out).save(os.path.join(CONCEPT_IMAGES_SUBDIR, f"{base_name}_aug{i:02d}.png"))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print("✅ Data augmentation complete.")
"""
# 4. Run training via subprocess
import subprocess
train_script = os.path.join(KOHYA_DIR, "train_network.py")

cmd = [
    "accelerate", "launch",
    "--num_processes", "1",
    "--num_machines", "1",
    "--num_cpu_threads_per_process", "2",
    "--mixed_precision", "fp16",
    "--dynamo_backend", "no",
    train_script,
    "--pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5",
    "--train_data_dir", LORA_TRAIN_DATA_DIR,
    "--output_dir", OUTPUT_DIR,
    "--resolution", "512,512",
    "--enable_bucket",
    "--min_bucket_reso", "256",
    "--max_bucket_reso", "768",
    "--network_module", "networks.lora",
    "--network_alpha", "128",
    "--network_dim", "128",
    "--learning_rate", "1e-4",
    "--max_train_steps", "1000",
    "--train_batch_size", "1",
    "--max_data_loader_n_workers", "1",
    "--save_model_as", "safetensors",
    "--use_8bit_adam",
    "--gradient_checkpointing",
    "--lowram",
    "--mem_eff_attn",
    f"--logging_dir={os.path.join(OUTPUT_DIR, 'logs')}",
    "--save_every_n_epochs=1",
    "--save_state",
    "--lr_scheduler=cosine_with_restarts",
    "--lr_warmup_steps=100",
    "--persistent_data_loader_workers",
    "--output_name", "durer_lora_checkpoint"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 Launching training...")
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, encoding='utf-8', errors='replace')

for line in process.stdout:
    print(line, end='')

process.wait()
print("✅ Training complete." if process.returncode == 0 else "❌ Training failed.")
