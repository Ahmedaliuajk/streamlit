import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Melanoma Detection", layout="centered")
st.title("Melanoma Detection – PH2 Dataset")

# --- MODEL LOADING WITH CACHING ---
# Using @st.cache_resource ensures the model is loaded only once, speeding up the app
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# You can toggle between your two models here
# Ensure these filenames MATCH EXACTLY what you uploaded to GitHub/Streamlit
MODEL_OPTIONS = {
    "K-Fold Model": "KFold_Model.joblib",
    "PH2 Model": "ph_Model.joblib"
}

selected_model_name = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
model_file = MODEL_OPTIONS[selected_model_name]
model = load_model(model_file)

if model is None:
    st.error(f"❌ Error: The file '{model_file}' was not found. Please ensure it is uploaded to the root of your GitHub repository.")
    st.stop()

# --- FEATURE EXTRACTION ---
def extract_features(img):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Binary thresholding to isolate the lesion (dataset dependent logic)
        # We assume dark lesion on light skin; simple thresholding or gray>0
        # If your training data used a specific segmentation mask, you must replicate that here.
        # For now, we use the logic from your snippet: pixels > 0 (assuming preprocessing or simple mask)
        mask = gray > 0
        pixels = gray[mask]

        if len(pixels) < 100:
            return None

        # Extract stats
        mean_intensity = np.mean(pixels)
        std_intensity = np.std(pixels)
        area = len(pixels)
        
        # Extract color means using the mask
        r_mean = np.mean(img[:,:,0][mask])
        g_mean = np.mean(img[:,:,1][mask])
        b_mean = np.mean(img[:,:,2][mask])

        features = [mean_intensity, std_intensity, area, r_mean, g_mean, b_mean]
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- MAIN APP LOGIC ---
uploaded = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "png", "bmp"])

if uploaded:
    # Display Image
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Analyze Lesion"):
        with st.spinner("Processing..."):
            feats = extract_features(img)

            if feats is None:
                st.warning("⚠️ Could not extract features. The lesion area might be too small or low contrast.")
            else:
                try:
                    pred = model.predict(feats)[0]
                    
                    # Probability (if supported by model)
                    try:
                        probs = model.predict_proba(feats)[0]
                        confidence = np.max(probs) * 100
                        st.write(f"**Confidence:** {confidence:.2f}%")
                    except:
                        pass # Model might not support predict_proba

                    if pred == 1:
                        st.error("🛑 **Prediction: Melanoma Detected**")
                        st.write("Please consult a dermatologist for further examination.")
                    else:
                        st.success("✅ **Prediction: No Melanoma Detected**")
                        st.write("The lesion appears benign based on this model's analysis.")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")