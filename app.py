# app.py
import os
import json
import tempfile
import streamlit as st
from PIL import Image
import base64

# Streamlit page config must be the first Streamlit command
st.set_page_config(
    page_title="OMNI-MAD | Morph Detection",
    page_icon="🛡️",
    layout="wide"
)

# Import the predictor after page config
from inference import OMNIMADPredictor
from config import cfg

# --- Styling & CSS ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #4CAF50; color: white;}
    .report-box {padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Loads the model once and caches it to prevent reloading on every interaction."""
    checkpoint_path = os.path.join(cfg.SAVE_DIR, cfg.RUN_NAME, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        return None
    return OMNIMADPredictor(checkpoint_path=checkpoint_path)

def main():
    st.title("🛡️ OMNI-MAD: Omni-Domain Morphing Attack Detection")
    st.markdown("""
    **Upload a facial image to verify its authenticity.** 
    This system analyzes RGB spatial features, frequency spectrums, micro-textures, and structural edges to detect high-quality facial morphing attacks.
    """)

    # Load Model
    with st.spinner("Loading OMNI-MAD Model..."):
        predictor = load_predictor()

    if predictor is None:
        st.error(f"Model weights not found! Please train the model first or place `best_model.pth` in `{cfg.SAVE_DIR}/{cfg.RUN_NAME}/`")
        st.stop()

    # Sidebar
    st.sidebar.header("Control Panel")
    st.sidebar.info(
        "OMNI-MAD uses a multi-branch neural network designed to catch artifacts left by StyleGAN, OpenCV, and Diffusion-based face morphing."
    )
    confidence_threshold = st.sidebar.slider("Morph Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    # File Uploader
    uploaded_file = st.file_uploader("Drop an image here...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Save uploaded file to a temporary directory for inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Display Original Image
        with col1:
            st.subheader("Uploaded Image")
            original_img = Image.open(uploaded_file)
            st.image(original_img, use_column_width=True)

        # Prediction Action
        if st.button("Run OMNI-MAD Analysis", type="primary"):
            with st.spinner("Extracting Multi-Domain Features & Analyzing..."):
                heatmap_path = tmp_path.replace(".jpg", "_heatmap.jpg")
                
                # Run Inference
                results = predictor.predict(tmp_path, save_heatmap=True, out_path=heatmap_path)
                
                # Apply custom threshold
                if results["morph_probability"] > confidence_threshold:
                    final_prediction = "Morph (Attack) 🚨"
                    box_color = "#ffcccc"
                else:
                    final_prediction = "Bona Fide (Real) ✅"
                    box_color = "#ccffcc"

            with col2:
                st.subheader("Analysis Results")
                
                # Result Metrics Box
                st.markdown(f"""
                <div style="background-color: {box_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin: 0; text-align: center;">{final_prediction}</h3>
                    <p style="text-align: center; margin-top: 10px; font-size: 18px;">
                        Model Confidence: <b>{results['confidence']}</b>
                    </p>
                    <p style="text-align: center; margin: 0; font-size: 14px; color: #555;">
                        Inference Time: {results['inference_time_ms']} ms
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Heatmap
                st.subheader("Artifact Localization Map")
                st.markdown("Highlights structural and frequency anomalies detected by the Transformer Context Encoder.")
                if os.path.exists(heatmap_path):
                    st.image(heatmap_path, use_column_width=True)
                
                # Generate Downloadable Report
                report_dict = {
                    "filename": uploaded_file.name,
                    "prediction": final_prediction,
                    "morph_probability": results["morph_probability"],
                    "confidence": results["confidence"],
                    "inference_time_ms": results["inference_time_ms"],
                    "threshold_used": confidence_threshold
                }
                
                report_json = json.dumps(report_dict, indent=4)
                
                st.download_button(
                    label="📥 Download JSON Report",
                    data=report_json,
                    file_name=f"{uploaded_file.name}_mad_report.json",
                    mime="application/json"
                )
                
            # Cleanup temp files
            try:
                os.remove(tmp_path)
                if os.path.exists(heatmap_path):
                    os.remove(heatmap_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()