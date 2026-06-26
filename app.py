import os
import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import io
from audio_fingerprinter import (
    compute_spectrogram,
    get_peaks,
    generate_hashes,
    FingerprintDatabase,
    match_query,
    DEFAULT_FS
)

# Page configuration
st.set_page_config(
    page_title="Zapptain America - Sonic Signature Identifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark glassmorphism design
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #1c0a35 50%, #05040a 100%);
        color: #f3f0f7;
    }
    
    /* Header / Titles */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(45deg, #ff007f, #8b26ff, #00f0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Card / Glassmorphism Panel */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        border: 1px rgba(255, 255, 255, 0.08) solid;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 24px;
    }
    
    .result-card {
        background: rgba(139, 38, 255, 0.15);
        border: 2px solid rgba(139, 38, 255, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 0 25px rgba(139, 38, 255, 0.25);
    }
    
    .highlight-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00f0ff;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #ff007f 0%, #8b26ff 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 38, 255, 0.5);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("⚡ Zapptain America")
st.subheader("State-of-the-Art Audio Fingerprinting & Song Recognition")

# Load database
DB_PATH = "fingerprint_db.pkl"
db = FingerprintDatabase()

@st.cache_resource
def load_cached_db(path):
    database = FingerprintDatabase()
    if database.load(path):
        return database
    return None

cached_db = load_cached_db(DB_PATH)

if not cached_db:
    st.error("❌ Song fingerprint database not found. Please index the database first!")
    st.stop()

# Sidebar options
st.sidebar.markdown("<div class='glass-card'><h3>Settings</h3></div>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Choose App Mode", ["🔍 Single-Clip Identifier", "📁 Batch Processor"])

st.sidebar.markdown(f"""
<div class='glass-card'>
    <h4>Database Info</h4>
    <p><b>Songs indexed:</b> {len(cached_db.song_list)}</p>
    <p><b>Sample Rate:</b> {DEFAULT_FS} Hz</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SINGLE CLIP IDENTIFIER MODE -----------------
if app_mode == "🔍 Single-Clip Identifier":
    st.markdown("<div class='glass-card'><h3>Single-Clip Identification Mode</h3>"
                "<p>Upload an audio file (MP3, WAV, M4A) of a short query clip to identify which song it belongs to.</p></div>", 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        
        with st.spinner("Processing audio and matching signatures..."):
            # Load audio using librosa in-memory
            try:
                # Read bytes
                audio_bytes = uploaded_file.read()
                # Use io.BytesIO to feed into librosa
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=DEFAULT_FS)
                
                # Compute Spectrogram
                stft_query = librosa.stft(y, n_fft=1024, hop_length=256)
                spec_db = librosa.amplitude_to_db(np.abs(stft_query), ref=np.max)
                
                # Peaks and Hashes
                peaks = get_peaks(spec_db)
                hashes = generate_hashes(peaks)
                
                # Match
                pred_song, max_matches, offset_histograms, alignment_scores = match_query(hashes, cached_db)
                
                if pred_song:
                    st.markdown(f"""
                    <div class='result-card'>
                        <h4>🎉 Song Identified!</h4>
                        <div class='highlight-text'>{pred_song}</div>
                        <p style='margin-top: 10px; color: #bca0ff;'>Confidence Score: <b>{max_matches}</b> aligned hashes</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Columns for visualizations
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("#### 📊 Spectrogram")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        # Use specshow but since we don't have time arrays easily, just plot as image
                        im = ax.imshow(spec_db, origin='lower', aspect='auto', cmap='magma')
                        ax.set_title("Query Spectrogram")
                        ax.set_xlabel("Time (frames)")
                        ax.set_ylabel("Frequency (bins)")
                        fig.patch.set_facecolor('#0f0c1b')
                        ax.set_facecolor('#0f0c1b')
                        ax.spines['bottom'].set_color('#ffffff')
                        ax.spines['left'].set_color('#ffffff')
                        ax.xaxis.label.set_color('#ffffff')
                        ax.yaxis.label.set_color('#ffffff')
                        ax.title.set_color('#ffffff')
                        ax.tick_params(colors='#ffffff')
                        st.pyplot(fig)
                        
                    with col2:
                        st.write("#### ✨ Constellation of Peaks")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        t_idx = [p[0] for p in peaks]
                        f_idx = [p[1] for p in peaks]
                        ax.scatter(t_idx, f_idx, color='#00f0ff', s=12, edgecolors='black', linewidths=0.3)
                        ax.set_title("Peak Signatures")
                        ax.set_xlabel("Time (frames)")
                        ax.set_ylabel("Frequency (bins)")
                        ax.grid(True, linestyle='--', alpha=0.3)
                        fig.patch.set_facecolor('#0f0c1b')
                        ax.set_facecolor('#0f0c1b')
                        ax.spines['bottom'].set_color('#ffffff')
                        ax.spines['left'].set_color('#ffffff')
                        ax.xaxis.label.set_color('#ffffff')
                        ax.yaxis.label.set_color('#ffffff')
                        ax.title.set_color('#ffffff')
                        ax.tick_params(colors='#ffffff')
                        st.pyplot(fig)
                        
                    with col3:
                        st.write("#### 📈 Offset Histogram")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        offsets = offset_histograms.get(pred_song, [])
                        ax.hist(offsets, bins=50, color='#ff007f', alpha=0.8, edgecolor='black')
                        ax.set_title(f"Histogram for '{pred_song}'")
                        ax.set_xlabel("Time Offset (frames)")
                        ax.set_ylabel("Vote Count")
                        fig.patch.set_facecolor('#0f0c1b')
                        ax.set_facecolor('#0f0c1b')
                        ax.spines['bottom'].set_color('#ffffff')
                        ax.spines['left'].set_color('#ffffff')
                        ax.xaxis.label.set_color('#ffffff')
                        ax.yaxis.label.set_color('#ffffff')
                        ax.title.set_color('#ffffff')
                        ax.tick_params(colors='#ffffff')
                        st.pyplot(fig)
                else:
                    st.error("❌ Song could not be identified! No matching fingerprints in database.")
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")

# ----------------- BATCH PROCESSOR MODE -----------------
elif app_mode == "📁 Batch Processor":
    st.markdown("<div class='glass-card'><h3>Batch Processing Mode</h3>"
                "<p>Upload a zip file containing multiple query audio clips or select multiple files. The system will identify all of them and output a downloadable <code>results.csv</code> file.</p></div>", 
                unsafe_allow_html=True)
    
    upload_type = st.radio("Choose upload format", ["⚡ Multiple Files", "📦 Zip File"])
    
    files_to_process = []
    
    if upload_type == "⚡ Multiple Files":
        uploaded_files = st.file_uploader("Select multiple audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                files_to_process.append((f.name, f.read()))
                
    else:
        uploaded_zip = st.file_uploader("Upload a ZIP file containing queries", type=["zip"])
        if uploaded_zip is not None:
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
                for filename in z.namelist():
                    # Filter out directories and non-audio files
                    if not filename.endswith('/') and filename.lower().endswith(('.mp3', '.wav', '.m4a')):
                        # Get basename of file
                        base_name = os.path.basename(filename)
                        files_to_process.append((base_name, z.read(filename)))
                        
    if len(files_to_process) > 0:
        st.write(f"Loaded **{len(files_to_process)}** queries to process.")
        
        if st.button("🚀 Process Batch"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (name, data) in enumerate(files_to_process):
                status_text.text(f"Processing ({idx+1}/{len(files_to_process)}): {name}")
                try:
                    # Load audio
                    y, sr = librosa.load(io.BytesIO(data), sr=DEFAULT_FS)
                    stft_q = librosa.stft(y, n_fft=1024, hop_length=256)
                    spec_db = librosa.amplitude_to_db(np.abs(stft_q), ref=np.max)
                    peaks = get_peaks(spec_db)
                    hashes = generate_hashes(peaks)
                    pred_song, max_matches, _, _ = match_query(hashes, cached_db)
                    
                    if not pred_song:
                        pred_song = "unknown"
                        
                    results.append({"filename": name, "prediction": pred_song})
                except Exception as e:
                    st.error(f"Error processing {name}: {e}")
                    results.append({"filename": name, "prediction": "error"})
                
                progress_bar.progress((idx + 1) / len(files_to_process))
                
            status_text.text("Batch processing complete!")
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Show table
            st.dataframe(df)
            
            # CSV Download
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )
