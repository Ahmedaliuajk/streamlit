import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os

st.set_page_config(page_title="Melanoma Detection - Random Forest", layout="centered")
st.title("Melanoma Detection (Random Forest Model)")

MODEL_FILE = "ph_Model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("Model file ph_Model.pkl not found in repository.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

st.success("Random Forest model loaded successfully")

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(image):
    image = image.resize((256, 256))
    img = np.array(image)

    gray = np.mean(img, axis=2).astype(np.uint8)
    lesion_mask = gray > 10

    if lesion_mask.sum() < 100:
        return None

    features = {
        "Mean_Intensity": gray[lesion_mask].mean(),
        "Std_Intensity": gray[lesion_mask].std(),
        "Area": lesion_mask.sum(),
        "R_Mean": img[:, :, 0][lesion_mask].mean(),
        "G_Mean": img[:, :, 1][lesion_mask].mean(),
        "B_Mean": img[:, :, 2][lesion_mask].mean(),
    }

    return pd.DataFrame([features])

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Dermoscopic Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    X = extract_features(image)

    if X is None:
        st.warning("Lesion area not detected clearly.")
    else:
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        if pred == 1:
            st.error(f"Melanoma Detected (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"No Melanoma Detected (Confidence: {prob[0]*100:.2f}%)")

        st.subheader("Extracted Features")
        st.dataframe(X)
