# run_once_download_model.py
# Run this script once to download a pretrained PlantVillage model.

import os
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

url = "https://huggingface.co/spaces/etahamad/plant-disease-detection/resolve/main/model.h5"
save_path = os.path.join(MODELS_DIR, "plant_disease_model.h5")

print("Downloading model...")
r = requests.get(url, stream=True)
r.raise_for_status()

with open(save_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"✅ Downloaded to {save_path}")