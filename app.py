import os
import sys
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib

# Add parent directory to sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_audio
from src.features import extract_features, get_feature_names
from src.train import run_pipeline
from src.generate_demo_data import generate_dataset

# Set Page Config
st.set_page_config(
    page_title="AcoustiShield AI - Deepfake Audio Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Color CSS Injection (yellowish-greenish in light, greyish green in dark)
st.markdown("""
<style>
    /* Custom Scoped CSS Variables to avoid overriding Streamlit's internal variables */
    @media (prefers-color-scheme: light) {
        :root {
            --app-bg: #f9fbf4;
            --app-card-bg: #ffffff;
            --app-text: #1e290a;
            --app-accent: #8abf24;
            --app-accent-hover: #729e1e;
            --app-accent-light: #eef4db;
            --app-border: #d2dfb9;
            --app-title-color: #2e3e14;
            --app-shadow: rgba(138, 191, 36, 0.08);
            --app-danger-bg: #fdf2f2;
            --app-danger-text: #9b1c1c;
            --app-success-bg: #f3faf4;
            --app-success-text: #03543f;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --app-bg: #171d15;
            --app-card-bg: #222920;
            --app-text: #e6e9e4;
            --app-accent: #a3db39;
            --app-accent-hover: #b8f04c;
            --app-accent-light: #2c3829;
            --app-border: #3b4737;
            --app-title-color: #c5dbb0;
            --app-shadow: rgba(0, 0, 0, 0.3);
            --app-danger-bg: #2d1919;
            --app-danger-text: #f9b4b4;
            --app-success-bg: #142a1e;
            --app-success-text: #8ce9ad;
        }
    }

    /* Style st.container(border=True) as custom cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--app-card-bg);
        border: 1px solid var(--app-border) !important;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 12px var(--app-shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px var(--app-shadow);
        border-color: var(--app-accent) !important;
    }
    
    /* Result Header Styling */
    .result-box {
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 25px;
        text-align: center;
        border: 2px solid transparent;
    }
    
    .genuine-box {
        background-color: var(--app-success-bg);
        color: var(--app-success-text);
        border-color: var(--app-accent);
    }
    
    .fake-box {
        background-color: var(--app-danger-bg);
        color: var(--app-danger-text);
        border-color: #ef4444;
    }
    
    .main-title {
        color: var(--app-title-color) !important;
        font-weight: 800;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        font-weight: 600;
        font-size: 16px;
        /* Let Streamlit control tab text color for default readability */
    }

    .stTabs [aria-selected="true"] {
        color: var(--app-accent) !important;
        border-bottom-color: var(--app-accent) !important;
    }

    /* Style sidebar border only */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--app-border);
    }

    /* Make slider and widgets match accent */
    .stSlider > div > div > div > div {
        background-color: var(--app-accent) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if model exists
MODEL_PATH = "saved_models/detector_model.joblib"
DEMO_DATA_DIR = "demo_data"

@st.cache_resource
def load_cached_model(path):
    if os.path.exists(path):
        from src.model import load_model
        try:
            return load_model(path)
        except Exception:
            return None
    return None

def init_app():
    """Checks model and dataset. Retrains if needed."""
    model_exists = os.path.exists(MODEL_PATH)
    data_exists = os.path.exists(DEMO_DATA_DIR)
    return model_exists, data_exists

model_exists, data_exists = init_app()
detector_model = load_cached_model(MODEL_PATH)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("<h2 class='main-title'>🛡️ AcoustiShield AI</h2>", unsafe_allow_html=True)
    st.markdown("Version 1.0.0 (Production Ready)")
    st.markdown("---")
    
    st.markdown("### System Status")
    if model_exists:
        st.success("🤖 Model Status: **LOADED**")
    else:
        st.warning("⚠️ Model Status: **NOT TRAINED**")
        
    if data_exists:
        st.info("📂 Demo Dataset: **AVAILABLE**")
    else:
        st.warning("📂 Demo Dataset: **NOT FOUND**")
        
    st.markdown("---")
    
    st.markdown("### Model Operations")
    if st.button("🔄 Generate Demo Dataset", use_container_width=True):
        with st.spinner("Synthesizing audio samples (DSP)..."):
            generate_dataset(DEMO_DATA_DIR)
            st.success("Dataset synthesized successfully!")
            st.rerun()
            
    if st.button("🚀 Train Detector Model", use_container_width=True):
        if not os.path.exists(DEMO_DATA_DIR):
            with st.spinner("Synthesizing dataset first..."):
                generate_dataset(DEMO_DATA_DIR)
        with st.spinner("Extracting features & training Random Forest..."):
            metrics = run_pipeline(DEMO_DATA_DIR, MODEL_PATH)
            st.cache_resource.clear()
            st.success("Model trained and saved!")
            st.rerun()
            
    st.markdown("---")
    st.markdown(
        "**Methodology Note:** The model utilizes **138 high-dimensional acoustic features** "
        "(MFCCs, Chroma, Mel Spectrogram, Spectral Contrast, Tonnetz, ZCR, Centroid) "
        "to classify audio as Genuine Human Speech or AI-Synthesized Deepfake."
    )

# ================= MAIN BODY =================
st.markdown("<h1 class='main-title'>Deepfake Audio Detection Portal</h1>", unsafe_allow_html=True)
st.markdown("Secure, instant verification of speech files for synthetic voice detection and biometrics security.")

# Self-healing check
if not model_exists:
    st.warning("🚨 **Getting Started:** The detector model is not trained yet. Please click the button below to generate a synthetic dataset and train the model automatically.")
    if st.button("✨ Auto-setup and Train Model", type="primary"):
        with st.spinner("Generating synthetic data and training classifier..."):
            generate_dataset(DEMO_DATA_DIR)
            run_pipeline(DEMO_DATA_DIR, MODEL_PATH)
            st.cache_resource.clear()
            st.success("Setup complete! Model is now running.")
            st.rerun()
    st.stop()

# Define tabs
tab1, tab2, tab3 = st.tabs(["🔍 Audio Analyzer", "📊 Features & Explainability", "📈 Model Performance"])

# ================= TAB 1: AUDIO ANALYZER =================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("📤 Upload Audio File")
            st.write("Upload a speech sample to check if it's Genuine (Human) or a Deepfake (AI-Generated).")
            uploaded_file = st.file_uploader("Supported Formats: WAV, MP3, M4A", type=["wav", "mp3", "m4a"])
            
            # Fast samples selector
            st.write("💡 Or try one of our validation samples:")
            sample_col1, sample_col2 = st.columns(2)
            
            selected_sample = None
            val_dir = os.path.join(DEMO_DATA_DIR, "val")
            if os.path.exists(val_dir):
                with sample_col1:
                    if st.button("🎵 Load Genuine Sample", use_container_width=True):
                        gen_dir = os.path.join(val_dir, "genuine")
                        if os.path.exists(gen_dir):
                            gen_files = [f for f in os.listdir(gen_dir) if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]
                            if gen_files:
                                selected_sample = os.path.join(gen_dir, sorted(gen_files)[0])
                            else:
                                st.error("No genuine validation files found.")
                with sample_col2:
                    if st.button("🤖 Load Deepfake Sample", use_container_width=True):
                        fake_dir = os.path.join(val_dir, "deepfake")
                        if os.path.exists(fake_dir):
                            fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac"))]
                            if fake_files:
                                selected_sample = os.path.join(fake_dir, sorted(fake_files)[0])
                            else:
                                st.error("No deepfake validation files found.")
        
        # Resolve source audio path
        audio_path = None
        if uploaded_file is not None:
            # Save uploaded buffer to a temporary file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                audio_path = tmp.name
        elif selected_sample is not None:
            audio_path = selected_sample

        if audio_path:
            with st.container(border=True):
                st.subheader("🎧 Audio Player")
                st.audio(audio_path)
                
                with st.spinner("Extracting features and conducting spectral forensic analysis..."):
                    # Run prediction
                    result = predict_audio(audio_path, detector_model if detector_model is not None else MODEL_PATH)
                    
                if result:
                    # Render results block
                    conf_pct = result["confidence"] * 100
                    is_fake = result["class_id"] == 1
                    
                    if is_fake:
                        st.markdown(f"""
                        <div class="result-box fake-box">
                            <h2>🚨 DEEPFAKE DETECTED</h2>
                            <h3>Prediction: <b>AI-Generated Speech</b></h3>
                            <h1>Confidence: {conf_pct:.2f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box genuine-box">
                            <h2>🛡️ VERIFIED GENUINE</h2>
                            <h3>Prediction: <b>Genuine Human Speech</b></h3>
                            <h1>Confidence: {conf_pct:.2f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Raw Probabilities progress bars
                    st.write("#### Forensic Score breakdown")
                    g_prob = result["probabilities"]["genuine"]
                    f_prob = result["probabilities"]["deepfake"]
                    
                    st.write(f"**Genuine Human Speech:** {g_prob * 100:.2f}%")
                    st.progress(g_prob)
                    
                    st.write(f"**AI-Generated / Synthesized Speech:** {f_prob * 100:.2f}%")
                    st.progress(f_prob)

    with col2:
        with st.container(border=True):
            st.subheader("📊 Signal Visualization")
            
            if audio_path:
                try:
                    # Load audio for display
                    y, sr = librosa.load(audio_path, sr=16000)
                    
                    # Plot Waveform
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                    fig.patch.set_facecolor('none')
                    
                    # Apply styles based on dark/light context
                    ax1.set_facecolor('none')
                    ax2.set_facecolor('none')
                    
                    # Light/Dark labels/ticks color detection
                    # Streamlit doesn't expose theme directly in Python, but we use a neutral styling
                    label_color = '#7f8c8d'
                    for ax in [ax1, ax2]:
                        ax.spines['bottom'].set_color(label_color)
                        ax.spines['top'].set_color('none')
                        ax.spines['left'].set_color(label_color)
                        ax.spines['right'].set_color('none')
                        ax.tick_params(axis='x', colors=label_color)
                        ax.tick_params(axis='y', colors=label_color)
                        ax.yaxis.label.set_color(label_color)
                        ax.xaxis.label.set_color(label_color)
                    
                    # Plot Waveform
                    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#8abf24')
                    ax1.set_title("Acoustic Waveform (Time Domain)", color='#8abf24', fontsize=12, fontweight='bold')
                    ax1.set_xlabel("Time (seconds)")
                    ax1.set_ylabel("Amplitude")
                    
                    # Plot Spectrogram
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2, cmap='viridis')
                    ax2.set_title("Log-Frequency Power Spectrogram", color='#8abf24', fontsize=12, fontweight='bold')
                    ax2.set_xlabel("Time (seconds)")
                    ax2.set_ylabel("Frequency (Hz)")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error generating signal visualizations: {e}")
            else:
                st.info("ℹ️ Upload an audio file or load a sample to display signal visualizations.")

# ================= TAB 2: FEATURES & EXPLAINABILITY =================
with tab2:
    st.subheader("💡 Acoustic Fingerprinting & Explainability")
    st.write("Understand which features differentiate genuine speech from deepfakes.")
    
    col_feat1, col_feat2 = st.columns([1, 1])
    
    with col_feat1:
        with st.container(border=True):
            st.markdown("#### Feature Importance Analysis")
            st.write("Below is the ranked importance of acoustic feature families calculated by our Random Forest model:")
            
            # Load model and calculate feature importance
            try:
                model = joblib.load(MODEL_PATH)
                feature_names = get_feature_names()
                importances = model.feature_importances_
                
                # Group importances by family
                families = {
                    "MFCCs (Envelope & Timbre)": [importances[i] for i, name in enumerate(feature_names) if "mfcc" in name],
                    "Chroma (Harmonics & Pitch)": [importances[i] for i, name in enumerate(feature_names) if "chroma" in name],
                    "Mel Spectrogram (Log Power)": [importances[i] for i, name in enumerate(feature_names) if "mel" in name],
                    "Spectral Contrast (Peaks/Valleys)": [importances[i] for i, name in enumerate(feature_names) if "contrast" in name],
                    "Tonnetz (Tonal Centroids)": [importances[i] for i, name in enumerate(feature_names) if "tonnetz" in name],
                    "Temporal Metrics (ZCR, RMS, Centroid, Rolloff)": [importances[i] for i, name in enumerate(feature_names) if any(x in name for x in ["zcr", "rms", "centroid", "rolloff"])]
                }
                
                family_means = {k: np.sum(v) for k, v in families.items()}
                df_imp = pd.DataFrame(list(family_means.items()), columns=["Feature Group", "Importance Contribution"])
                df_imp = df_imp.sort_values(by="Importance Contribution", ascending=True)
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(8, 4.5))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                # Draw horizontal bar chart
                sns.barplot(x="Importance Contribution", y="Feature Group", data=df_imp, ax=ax, palette="viridis")
                
                # Stylize plot
                label_color = '#7f8c8d'
                ax.spines['bottom'].set_color(label_color)
                ax.spines['top'].set_color('none')
                ax.spines['left'].set_color(label_color)
                ax.spines['right'].set_color('none')
                ax.tick_params(axis='x', colors=label_color)
                ax.tick_params(axis='y', colors=label_color)
                ax.xaxis.label.set_color(label_color)
                ax.yaxis.label.set_color('none')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error loading model details: {e}")
        
    with col_feat2:
        with st.container(border=True):
            st.markdown("#### Audio Feature Descriptions")
            st.markdown("""
            Our pipeline extracts **138 spectral and temporal features** from each audio clip:
            
            - **MFCCs (Mel-frequency Cepstral Coefficients)**: Captures the shape of the vocal tract and spectral envelope. Essential for voice biometrics and identifying synthetic vocoder artifacts.
            - **Mel Spectrogram Bins**: Represents energy levels across perceptually-spaced frequency bands. Deepfakes frequently show grid/checkerboard patterns or high-frequency hiss in these bands.
            - **Spectral Contrast**: Computes the difference between peaks and valleys. Deepfakes often lack the natural valley depths due to artificial smoothing.
            - **Chroma STFT**: Measures energy levels across the 12 chromatic pitches, capturing harmonic structures which are often distorted in synthesis.
            - **Tonnetz (Tonal Centroid)**: Represents harmonic relations on a helical lattice to detect artificial pitch tracking.
            - **Zero Crossing Rate & Spectral Centroid**: Measures temporal rate of sign-changes and average spectrum center-of-mass, helping capture vocoder noise levels.
            """)

# ================= TAB 3: MODEL PERFORMANCE =================
with tab3:
    st.subheader("📈 Verification & Evaluation Reports")
    st.write("AcoustiShield AI's performance validation metrics under the Problem Statement (PS) threshold guidelines.")
    
    # Check validation scores
    # We can evaluate the validation dataset on the fly to show live, true results
    try:
        from src.train import load_dataset_features
        
        if os.path.exists(os.path.join(DEMO_DATA_DIR, "val")):
            with st.spinner("Calculating validation metrics on validation dataset..."):
                X_val, y_val = load_dataset_features(os.path.join(DEMO_DATA_DIR, "val"))
                model = joblib.load(MODEL_PATH)
                from src.model import evaluate_classifier
                metrics, _, _ = evaluate_classifier(model, X_val, y_val)
                
            # Render Metrics grid
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                with st.container(border=True):
                    st.write("🎯 **Overall Accuracy**")
                    acc_val = metrics["accuracy"] * 100
                    st.markdown(f"<h1 style='text-align: center;'>{acc_val:.1f}%</h1>", unsafe_allow_html=True)
                    if acc_val >= 80:
                        st.success("Passes PS (>= 80%)")
                    else:
                        st.error("Fails PS (>= 80%)")
                
            with m_col2:
                with st.container(border=True):
                    st.write("📉 **Equal Error Rate (EER)**")
                    eer_val = metrics["eer"] * 100
                    st.markdown(f"<h1 style='text-align: center;'>{eer_val:.1f}%</h1>", unsafe_allow_html=True)
                    if eer_val <= 12:
                        st.success("Passes PS (<= 12%)")
                    else:
                        st.error("Fails PS (<= 12%)")
                
            with m_col3:
                with st.container(border=True):
                    st.write("⚖️ **F1-Score**")
                    f1_val = metrics["f1_score"] * 100
                    st.markdown(f"<h1 style='text-align: center;'>{f1_val:.1f}%</h1>", unsafe_allow_html=True)
                    if f1_val >= 80:
                        st.success("Passes PS (>= 80%)")
                    else:
                        st.error("Fails PS (>= 80%)")
                
            with m_col4:
                with st.container(border=True):
                    st.write("👥 **Per-Class Thresholds**")
                    gen_acc = metrics["per_class_accuracy"]["genuine"] * 100
                    fake_acc = metrics["per_class_accuracy"]["deepfake"] * 100
                    st.markdown(f"<h5 style='text-align: center;'>Gen: {gen_acc:.1f}%<br>Fake: {fake_acc:.1f}%</h5>", unsafe_allow_html=True)
                    if gen_acc >= 75 and fake_acc >= 75:
                        st.success("Passes PS (>= 75% ea.)")
                    else:
                        st.error("Fails PS (>= 75%)")
                
            # Confusion Matrix Plot
            col_cm1, col_cm2 = st.columns([1, 1])
            with col_cm1:
                with st.container(border=True):
                    st.markdown("#### Confusion Matrix Heatmap")
                    
                    cm = np.array(metrics["confusion_matrix"]["matrix"])
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                                xticklabels=["Genuine (Human)", "Deepfake (AI)"],
                                yticklabels=["Genuine (Human)", "Deepfake (AI)"], ax=ax)
                    
                    # Stylize heatmap
                    ax.tick_params(colors='#7f8c8d')
                    ax.set_ylabel("True Labels", color='#7f8c8d')
                    ax.set_xlabel("Predicted Labels", color='#7f8c8d')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
            with col_cm2:
                with st.container(border=True):
                    st.markdown("#### Verification Summary")
                    st.markdown(f"""
                    - **Total Evaluation Samples**: {len(X_val)} files
                    - **True Gen (TN)**: {metrics['confusion_matrix']['tn']} files (correctly classified human speech)
                    - **True Fake (TP)**: {metrics['confusion_matrix']['tp']} files (correctly identified synthetic speech)
                    - **False Fake (FP)**: {metrics['confusion_matrix']['fp']} files (human speech flagged as fake)
                    - **False Gen (FN)**: {metrics['confusion_matrix']['fn']} files (deepfake missed by model)
                    """)
                    
                    # Check compliance against PS thresholds
                    passes_acc = metrics["accuracy"] >= 0.80
                    passes_eer = metrics["eer"] <= 0.12
                    passes_f1 = metrics["f1_score"] >= 0.80
                    passes_genuine = metrics["per_class_accuracy"]["genuine"] >= 0.75
                    passes_deepfake = metrics["per_class_accuracy"]["deepfake"] >= 0.75
                    
                    all_passed = (passes_acc and passes_eer and passes_f1 and 
                                  passes_genuine and passes_deepfake)
                    
                    if all_passed:
                        verdict_html = f"""
                        <div style="background-color: var(--app-success-bg); color: var(--app-success-text); padding: 15px; border-radius: 8px; border: 1px solid var(--app-accent); margin-top: 10px; font-family: sans-serif;">
                            <strong>🛡️ System Verdict: PASSED</strong><br>
                            AcoustiShield AI fully complies with the required PS benchmarks. The Equal Error Rate of <strong>{metrics['eer']*100:.2f}%</strong> and Accuracy of <strong>{metrics['accuracy']*100:.2f}%</strong> indicate a highly balanced and reliable decision threshold.
                        </div>
                        """
                    else:
                        verdict_html = f"""
                        <div style="background-color: var(--app-danger-bg); color: var(--app-danger-text); padding: 15px; border-radius: 8px; border: 1px solid #ef4444; margin-top: 10px; font-family: sans-serif;">
                            <strong>🚨 System Verdict: FAILED</strong><br>
                            AcoustiShield AI does not meet the required benchmarks. EER: <strong>{metrics['eer']*100:.2f}%</strong> (Target: &le; 12%), Accuracy: <strong>{metrics['accuracy']*100:.2f}%</strong> (Target: &ge; 80%). Please check dataset coverage or retrain the model.
                        </div>
                        """
                    st.markdown(verdict_html, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Validation dataset is not generated. Please generate the demo dataset in the sidebar first.")
            
    except Exception as e:
        st.error(f"Error loading validation results: {e}")

# Clean up temporary file at the very end of the page execution (after all columns render)
if 'uploaded_file' in locals() and uploaded_file is not None and 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
    try:
        os.unlink(audio_path)
    except Exception:
        pass
