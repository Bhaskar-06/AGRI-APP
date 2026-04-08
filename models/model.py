# run_once_download_model.py
# Run this script once to download a pretrained model from HuggingFace

import requests, os

os.makedirs("models", exist_ok=True)

# Pre-trained PlantVillage model (3MB, 38 classes)
url = "https://huggingface.co/spaces/etahamad/plant-disease-detection/resolve/main/model.h5"
print("Downloading model...")
r = requests.get(url, stream=True)
with open("models/plant_disease_model.h5", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
print("✅ Downloaded to models/plant_disease_model.h5")