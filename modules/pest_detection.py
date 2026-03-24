import streamlit as st
import numpy as np
from PIL import Image
import json, os, requests
from database.db import add_pest_log, get_all_farmers

MODEL_PATH = "models/plant_disease_model.h5"
INDEX_PATH = "models/class_indices.json"
MODEL_URL  = "https://huggingface.co/spaces/etahamad/plant-disease-detection/resolve/main/model.h5"

CLASS_INDICES = {
    "0": "Apple___Apple_scab", "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust", "3": "Apple___healthy",
    "4": "Blueberry___healthy", "5": "Cherry___Powdery_mildew",
    "6": "Cherry___healthy", "7": "Corn___Cercospora_leaf_spot",
    "8": "Corn___Common_rust", "9": "Corn___Northern_Leaf_Blight",
    "10": "Corn___healthy", "11": "Grape___Black_rot",
    "12": "Grape___Esca", "13": "Grape___Leaf_blight",
    "14": "Grape___healthy", "15": "Orange___Haunglongbing",
    "16": "Peach___Bacterial_spot", "17": "Peach___healthy",
    "18": "Pepper___Bacterial_spot", "19": "Pepper___healthy",
    "20": "Potato___Early_blight", "21": "Potato___Late_blight",
    "22": "Potato___healthy", "23": "Raspberry___healthy",
    "24": "Soybean___healthy", "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch", "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot", "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight", "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot", "33": "Tomato___Spider_mites",
    "34": "Tomato___Target_Spot", "35": "Tomato___Yellow_Leaf_Curl_Virus",
    "36": "Tomato___mosaic_virus", "37": "Tomato___healthy"
}

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Captan (2g/L) or Mancozeb (2.5g/L) every 7-10 days during wet weather.",
        "organic": "Spray neem oil (5ml/L) or baking soda solution (5g/L) on affected leaves.",
        "cultural": "Rake and destroy fallen leaves. Prune crowded branches for airflow.",
        "prevention": "Plant scab-resistant apple varieties. Apply dormant oil spray before bud break."
    },
    "Apple___Black_rot": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Thiophanate-methyl or Captan fungicide at petal fall stage.",
        "organic": "Copper-based fungicide (Bordeaux mixture). Remove mummified fruits immediately.",
        "cultural": "Prune dead/infected wood. Sanitize pruning tools with 70% alcohol.",
        "prevention": "Remove all infected fruit from tree and ground. Avoid wounding fruit."
    },
    "Apple___Cedar_apple_rust": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Myclobutanil or Propiconazole fungicide from pink bud through early summer.",
        "organic": "Sulfur-based fungicide spray every 7 days during infection period.",
        "cultural": "Remove nearby cedar/juniper trees. Prune galls from cedar in winter.",
        "prevention": "Plant rust-resistant apple varieties like Liberty or Freedom."
    },
    "Apple___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Continue regular neem oil preventive sprays monthly.",
        "cultural": "Maintain good orchard hygiene. Regular pruning for air circulation.",
        "prevention": "Monitor regularly. Apply balanced NPK fertilizer for strong immunity."
    },
    "Blueberry___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Apply compost mulch around base to retain moisture.",
        "cultural": "Maintain soil pH between 4.5-5.5 for optimal blueberry health.",
        "prevention": "Regular monitoring. Ensure adequate irrigation during fruiting."
    },
    "Cherry___Powdery_mildew": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Trifloxystrobin or Myclobutanil fungicide every 10-14 days.",
        "organic": "Potassium bicarbonate (10g/L) or diluted milk spray (1:9 ratio) weekly.",
        "cultural": "Improve air circulation by thinning canopy. Avoid excess nitrogen.",
        "prevention": "Plant resistant cherry varieties. Avoid overhead watering."
    },
    "Cherry___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Preventive neem oil spray every 2-3 weeks.",
        "cultural": "Regular pruning. Balanced fertilization.",
        "prevention": "Monitor for early signs of powdery mildew in humid weather."
    },
    "Corn___Cercospora_leaf_spot": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Azoxystrobin + Propiconazole at VT/R1 stage.",
        "organic": "Copper hydroxide spray. Trichoderma-based biocontrol agents.",
        "cultural": "Crop rotation with non-host crops. Bury infected crop residues.",
        "prevention": "Use resistant hybrids. Avoid continuous corn cropping in same field."
    },
    "Corn___Common_rust": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Propiconazole or Trifloxystrobin at early rust appearance.",
        "organic": "Sulfur dust application. Neem oil spray (3ml/L) weekly.",
        "cultural": "Plant early-maturing varieties to escape peak infection periods.",
        "prevention": "Use rust-resistant corn hybrids. Scout fields regularly from V6 stage."
    },
    "Corn___Northern_Leaf_Blight": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Azoxystrobin or Pyraclostrobin at first sign of lesions.",
        "organic": "Bacillus subtilis-based products (Serenade). Copper sulfate spray.",
        "cultural": "Rotate crops. Till infected residues. Plant after soil reaches 15 degrees C.",
        "prevention": "Use resistant hybrids. Avoid overhead irrigation."
    },
    "Corn___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Apply compost tea as foliar spray for immunity boost.",
        "cultural": "Maintain proper plant spacing (60-75cm row spacing).",
        "prevention": "Monitor weekly. Scout for aphids and armyworms."
    },
    "Grape___Black_rot": {
        "type": "Fungal", "severity": "High",
        "chemical": "Apply Myclobutanil (Rally) or Mancozeb every 7-14 days from bud break.",
        "organic": "Bordeaux mixture (1%) at bud break and bloom stages.",
        "cultural": "Remove all mummified berries from vine and ground. Prune for airflow.",
        "prevention": "Train vines to improve air circulation. Avoid leaf wetness."
    },
    "Grape___Esca": {
        "type": "Fungal (Wood disease)", "severity": "Very High",
        "chemical": "No curative fungicide available. Protect pruning wounds immediately.",
        "organic": "Trichoderma harzianum wound protectant on pruning cuts.",
        "cultural": "Remove and destroy infected wood. Disinfect pruning tools.",
        "prevention": "Protect pruning wounds with sealant. Prune during dry weather."
    },
    "Grape___Leaf_blight": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Iprodione or Fludioxonil fungicide. Mancozeb as protectant.",
        "organic": "Copper oxychloride spray (3g/L). Increase frequency in humid weather.",
        "cultural": "Remove infected leaves promptly. Improve canopy ventilation.",
        "prevention": "Avoid water stress. Maintain balanced nutrition (avoid excess N)."
    },
    "Grape___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Preventive Bordeaux mixture spray before monsoon.",
        "cultural": "Regular canopy management. Proper trellis training.",
        "prevention": "Monitor for downy/powdery mildew weekly."
    },
    "Orange___Haunglongbing": {
        "type": "Bacterial (Citrus Greening)", "severity": "CRITICAL - No Cure",
        "chemical": "No cure exists. Remove and destroy infected trees immediately.",
        "organic": "Neem oil spray to control Asian Citrus Psyllid (vector insect).",
        "cultural": "Quarantine infected trees. Use certified disease-free nursery plants only.",
        "prevention": "Control psyllid population with Imidacloprid. Plant in psyllid-free areas."
    },
    "Peach___Bacterial_spot": {
        "type": "Bacterial", "severity": "High",
        "chemical": "Apply copper hydroxide or oxytetracycline at bud swell. Repeat weekly.",
        "organic": "Copper sulfate spray (Bordeaux mixture) from dormant season.",
        "cultural": "Avoid overhead irrigation. Remove infected twigs during dormancy.",
        "prevention": "Plant resistant peach varieties. Windbreaks reduce spread."
    },
    "Peach___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Dormant copper spray as preventive measure.",
        "cultural": "Annual pruning for canopy airflow. Thin fruit for better size.",
        "prevention": "Monitor for peach leaf curl and brown rot."
    },
    "Pepper___Bacterial_spot": {
        "type": "Bacterial", "severity": "High",
        "chemical": "Apply copper hydroxide (Kocide) + Mancozeb weekly from transplanting.",
        "organic": "Copper-based Bordeaux mixture. Biocontrol with Bacillus amyloliquefaciens.",
        "cultural": "Avoid overhead irrigation. Rotate away from solanaceous crops 2 years.",
        "prevention": "Use certified disease-free seed. Hot water seed treatment (50C for 25 min)."
    },
    "Pepper___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Foliar spray with compost tea for nutrient boost.",
        "cultural": "Stake plants. Ensure good drainage. Proper spacing (45cm).",
        "prevention": "Monitor for aphids and thrips which spread viruses."
    },
    "Potato___Early_blight": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Chlorothalonil (Bravo) or Mancozeb every 7-10 days.",
        "organic": "Neem oil (5ml/L) + copper soap spray. Bacillus subtilis products.",
        "cultural": "Crop rotation (3-year cycle). Remove infected leaves. Avoid excess nitrogen.",
        "prevention": "Plant certified disease-free seed potatoes. Hill soil around stems."
    },
    "Potato___Late_blight": {
        "type": "Fungal (Oomycete)", "severity": "CRITICAL",
        "chemical": "Apply Metalaxyl + Mancozeb (Ridomil Gold) immediately. Repeat every 5-7 days.",
        "organic": "Copper hydroxide spray (Bordeaux mixture 1%). Limited effectiveness.",
        "cultural": "Destroy all infected plant material by burning. Do not compost.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation. Monitor daily."
    },
    "Potato___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Preventive copper spray before monsoon season.",
        "cultural": "Hill potatoes regularly. Ensure proper drainage.",
        "prevention": "Scout weekly. Watch for late blight during cool/wet weather."
    },
    "Raspberry___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Apply compost mulch. Neem oil preventive spray.",
        "cultural": "Remove old fruiting canes after harvest.",
        "prevention": "Monitor for cane blight and botrytis fruit rot."
    },
    "Soybean___healthy": {
        "type": "None", "severity": "None",
        "chemical": "No treatment needed.",
        "organic": "Rhizobium inoculant at planting for nitrogen fixation.",
        "cultural": "Maintain 45cm row spacing. Scout for soybean rust.",
        "prevention": "Monitor for sudden death syndrome and white mold."
    },
    "Squash___Powdery_mildew": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Trifloxystrobin or Tebuconazole at first sign.",
        "organic": "Baking soda solution (5g/L + 2ml dish soap). Neem oil weekly.",
        "cultural": "Improve air circulation. Remove infected leaves. Avoid evening watering.",
        "prevention": "Plant resistant varieties. Maintain dry foliage. Space plants adequately."
    },
    "Strawberry___Leaf_scorch": {
        "type": "Fungal", "severity": "Medium",
        "chemical": "Apply Captan or Myclobutanil fungicide. Start in early spring.",
        "organic": "Copper hydroxide spray. Remove infected leaves immediately.",
        "cultural": "Avoid overhead irrigation. Renovate beds after harvest.",
        "prevention": "Plant certified disease-free runners. Avoid poorly