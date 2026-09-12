"""
merge_datasets.py
Copies additional crop-disease datasets into data/plantvillage/ as new
class folders, so train_plant_model.py picks them up automatically —
no changes to the training script needed, since flow_from_directory()
just reads whatever class folders exist.
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(BASE_DIR, "data", "plantvillage")
SOURCE_DIR = os.path.join(BASE_DIR, "data", "extra_crops")

os.makedirs(TARGET_DIR, exist_ok=True)

for crop_folder in os.listdir(SOURCE_DIR):
    crop_path = os.path.join(SOURCE_DIR, crop_folder)
    if not os.path.isdir(crop_path):
        continue
    for class_folder in os.listdir(crop_path):
        class_path = os.path.join(crop_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        # Name the merged class consistently with PlantVillage style: Crop___ClassName
        clean_class_name = class_folder.replace(" ", "_")
        new_class_name = f"{crop_folder.capitalize()}___{clean_class_name}"
        dest_path = os.path.join(TARGET_DIR, new_class_name)
        os.makedirs(dest_path, exist_ok=True)

        for fname in os.listdir(class_path):
            src_file = os.path.join(class_path, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(dest_path, fname))

        print(f"✅ Merged {class_path} → {dest_path}")

print("\nDone. Now run: python train_plant_model.py")