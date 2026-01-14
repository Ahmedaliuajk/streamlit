import streamlit as st
import numpy as np

import joblib
from PIL import Image
import os

st.set_page_config(page_title="Melanoma Detection (K-Fold)", layout="centered")
st.title("Melanoma Detection – K-Fold Model")

MODEL_FILE = "KFold_Model.joblib"

if not os.path.exists(MODEL_FILE):
    st.error("Model file not found")
    st.stop()

model = joblib.load(MODEL_FILE)

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pixels = gray[gray > 0]

    if len(pixels) < 100:
        return None

    features = [
        np.mean(pixels),
        np.std(pixels),
        len(pixels),
        np.mean(img[:,:,0][gray > 0]),
        np.mean(img[:,:,1][gray > 0]),
        np.mean(img[:,:,2][gray > 0])
    ]
    return np.array(features).reshape(1, -1)

uploaded = st.file_uploader("Upload Dermoscopic Image", type=["jpg","png","bmp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    feats = extract_features(img)

    if feats is None:
        st.warning("Lesion area too small")
    else:
        pred = model.predict(feats)[0]

        if pred == 1:
            st.error("🛑 Melanoma Detected")
        else:
            st.success("✅ No Melanoma Detected")

