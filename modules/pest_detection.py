import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Load model once (cached)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'plant_disease_model.h5')
    return tf.keras.models.load_model(model_path)

# Load class names
@st.cache_resource
def load_class_names():
    json_path = os.path.join(os.path.dirname(__file__), '..', 'class_indices.json')
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    # Reverse: {0: 'Apple___scab', ...}
    return {v: k for k, v in class_indices.items()}

# Disease treatment advice
DISEASE_ADVICE = {
    "healthy": "✅ Your plant is healthy! No action needed.",
    "Apple_scab": "🍎 Apply fungicides like Captan or Mancozeb. Remove infected leaves.",
    "Black_rot": "🍇 Prune infected parts. Apply copper-based fungicide.",
    "Early_blight": "🍅 Use Chlorothalonil fungicide. Avoid overhead watering.",
    "Late_blight": "🥔 Apply Metalaxyl fungicide immediately. Remove infected plants.",
    "Powdery_mildew": "🍒 Apply sulfur-based fungicide. Improve air circulation.",
    "Bacterial_spot": "🫑 Use copper bactericide spray. Avoid working with wet plants.",
    "Leaf_Mold": "🍅 Improve ventilation. Apply fungicide like Mancozeb.",
    "Septoria_leaf_spot": "🍅 Remove lower leaves. Apply Chlorothalonil.",
    "Yellow_Leaf_Curl_Virus": "🍅 Control whiteflies (use insecticide). Remove infected plants.",
}

def get_advice(class_name):
    for key, advice in DISEASE_ADVICE.items():
        if key.lower() in class_name.lower():
            return advice
    return "⚠️ Consult a local agricultural expert for treatment advice."

def pest_detection_page():
    st.title("🔍 Plant Disease Detection")
    st.markdown("Upload a **leaf image** to detect diseases using AI (98.71% accuracy)")

    model = load_model()
    class_names = load_class_names()

    uploaded_file = st.file_uploader(
        "📷 Upload Leaf Image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, use_column_width=True)

        with col2:
            st.subheader("🤖 AI Diagnosis")

            # Preprocess
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            with st.spinner("Analyzing leaf..."):
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                predicted_class = class_names[predicted_idx]

            # Display result
            parts = predicted_class.split('___')
            plant = parts[0].replace('_', ' ')
            disease = parts[1].replace('_', ' ') if len(parts) > 1 else "Unknown"

            st.markdown(f"### 🌿 Plant: `{plant}`")
            st.markdown(f"### 🦠 Condition: `{disease}`")

            if confidence >= 90:
                st.success(f"✅ Confidence: **{confidence:.2f}%** (High)")
            elif confidence >= 70:
                st.warning(f"⚠️ Confidence: **{confidence:.2f}%** (Medium)")
            else:
                st.error(f"❌ Confidence: **{confidence:.2f}%** (Low — try clearer image)")

            # Treatment advice
            st.markdown("---")
            st.subheader("💊 Treatment Advice")
            advice = get_advice(predicted_class)
            st.info(advice)

            # Top 3 predictions
            st.markdown("---")
            st.subheader("📊 Top 3 Predictions")
            top3_idx = np.argsort(predictions[0])[::-1][:3]
            for i, idx in enumerate(top3_idx):
                name = class_names[idx].replace('___', ' → ').replace('_', ' ')
                prob = predictions[0][idx] * 100
                st.progress(int(prob), text=f"{i+1}. {name}: {prob:.1f}%")