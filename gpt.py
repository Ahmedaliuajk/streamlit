import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Melanoma Detection",
    layout="centered"
)

st.title("🧬 Melanoma Detection (PH2 Dataset)")
st.caption("Educational ML-based skin lesion classifier")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_artifacts(model_path, scaler_path=None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
    return model, scaler

MODEL_FILE = "ph_Model.joblib"
SCALER_FILE = "scaler.joblib"   # optional

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found.")
    st.stop()

model, scaler = load_artifacts(MODEL_FILE, SCALER_FILE)

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img):
    # Resize for consistency
    img = cv2.resize(img, (256, 256))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Otsu segmentation
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask = mask > 0

    pixels = gray[mask]
    if len(pixels) < 500:
        return None, None

    # Feature extraction
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    area_ratio = len(pixels) / (256 * 256)

    r_mean = np.mean(img[:, :, 0][mask])
    g_mean = np.mean(img[:, :, 1][mask])
    b_mean = np.mean(img[:, :, 2][mask])

    features = np.array([
        mean_intensity,
        std_intensity,
        area_ratio,
        r_mean,
        g_mean,
        b_mean
    ]).reshape(1, -1)

    return features, mask

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Dermoscopic Image",
    type=["jpg", "png", "bmp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    # ---------------- ANALYZE ----------------
    if st.button("🔍 Analyze Lesion"):
        with st.spinner("Processing..."):
            feats, mask = extract_features(img)

            if feats is None:
                st.warning("⚠️ Lesion not detected. Try a clearer image.")
                st.stop()

            # Apply scaler if available
            if scaler:
                feats = scaler.transform(feats)

            # Debug toggle
            with st.expander("🔧 Debug: Extracted Features"):
                st.write(feats)

            # Prediction
            pred = model.predict(feats)[0]

            # Probability
            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feats)[0]
                confidence = np.max(probs) * 100

            # Show segmentation
            with col2:
                overlay = img.copy()
                overlay[mask] = [255, 0, 0]
                st.image(
                    overlay,
                    caption="Lesion Segmentation",
                    use_container_width=True
                )

            st.markdown("---")

            # Result
            if pred == 1:
                st.error("🛑 **Melanoma Detected**")
            else:
                st.success("✅ **Benign Lesion Detected**")

            if confidence:
                st.metric("Model Confidence", f"{confidence:.2f}%")

            st.info("⚠️ This tool is for educational use only. Consult a dermatologist for diagnosis.")
