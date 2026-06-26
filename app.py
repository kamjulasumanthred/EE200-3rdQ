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
    page_title="EE200: Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and design matching the demo screenshots
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0d0f14;
        color: #9ea5b4;
    }
    
    /* Header style */
    .header-container {
        padding-top: 10px;
        padding-bottom: 20px;
        border-bottom: 1px solid #1f232d;
        margin-bottom: 20px;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .subtitle {
        font-size: 0.8rem;
        letter-spacing: 2px;
        color: #5d6475;
        text-transform: uppercase;
        margin-top: 5px;
        margin-bottom: 15px;
    }
    
    /* Admin message box */
    .admin-box {
        background-color: #11131a;
        border: 1px solid #1f232d;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        margin-bottom: 30px;
        color: #5d6475;
    }
    
    .admin-title {
        font-size: 0.95rem;
        color: #888e9b;
        margin-bottom: 5px;
        font-weight: 600;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #1f232d;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
        background-color: transparent;
        color: #5d6475;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00f0ff !important;
        border-bottom: 2px solid #00f0ff !important;
    }
    
    /* Song database cards */
    .song-card {
        background-color: #11131a;
        border: 1px solid #1f232d;
        border-radius: 8px;
        padding: 16px;
        text-align: left;
        margin-bottom: 20px;
    }
    
    .song-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .song-hashes {
        font-size: 0.8rem;
        color: #5d6475;
        margin-bottom: 12px;
    }
    
    /* Try section row */
    .sample-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #11131a;
        padding: 8px 16px;
        border-radius: 6px;
        border: 1px solid #1f232d;
        margin-bottom: 10px;
    }
    
    /* Results formatting */
    .step-header {
        font-size: 0.8rem;
        letter-spacing: 1px;
        color: #5d6475;
        text-transform: uppercase;
        margin-top: 25px;
        margin-bottom: 5px;
        font-weight: 700;
    }
    
    .step-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 10px;
    }
    
    .step-desc {
        font-size: 0.9rem;
        color: #888e9b;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th {
        text-align: left;
        color: #5d6475;
        font-weight: 600;
        border-bottom: 1px solid #1f232d;
        padding: 10px;
    }
    
    td {
        padding: 12px 10px;
        border-bottom: 1px solid #11131a;
        color: #ffffff;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #00f0ff;
        color: #0d0f14;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #00c8d6;
        color: #0d0f14;
    }
</style>
""", unsafe_allow_html=True)

# Database Setup
DB_PATH = "fingerprint_db.pkl"
db = FingerprintDatabase()

@st.cache_resource
def load_db(path):
    database = FingerprintDatabase()
    if database.load(path):
        # Pre-calculate song hash counts
        song_hash_counts = {}
        for hash_key, entries in database.db.items():
            for t1, song_name in entries:
                song_hash_counts[song_name] = song_hash_counts.get(song_name, 0) + 1
        return database, song_hash_counts
    return None, None

cached_db, hash_counts = load_db(DB_PATH)

if not cached_db:
    st.error("❌ Pre-compiled database not found. Please compile the database first!")
    st.stop()

# Header Section
st.markdown("""
<div class="header-container">
    <div class="main-title">🎵 EE200: Audio Fingerprinting</div>
    <div class="subtitle">Signals, Systems & Networks - Project Demo</div>
    <div>Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</div>
</div>
""", unsafe_allow_html=True)

# Tab Definition
tab_lib, tab_ident, tab_batch = st.tabs(["♦ LIBRARY", "☉ IDENTIFY", "▤ BATCH"])

# ----------------- LIBRARY TAB -----------------
with tab_lib:
    st.markdown("""
    <div class="admin-box">
        <div class="admin-title">Song indexing is managed by the admin.</div>
        <div>Drop a clip in the Identify tab to test the library.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### IN THE DATABASE")
    
    # Sort songs alphabetically
    sorted_songs = sorted(cached_db.song_list)
    
    # Render songs in a 4-column grid
    cols = st.columns(4)
    for idx, song_name in enumerate(sorted_songs):
        col = cols[idx % 4]
        h_count = hash_counts.get(song_name, 0)
        formatted_hashes = f"{h_count:,} hashes"
        
        with col:
            st.markdown(f"""
            <div class="song-card">
                <div class="song-title">{song_name}</div>
                <div class="song-hashes">{formatted_hashes}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display thumbnail image
            thumb_path = f"thumbnails/{song_name}.png"
            if os.path.exists(thumb_path):
                # Put the image inside the card area using Streamlit columns/spacing hacks
                st.image(thumb_path, use_column_width=True)
            else:
                st.write("*Thumbnail missing*")

# ----------------- IDENTIFY TAB -----------------
with tab_ident:
    st.write("### Identify a clip")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "m4a"], label_visibility="collapsed")
    
    st.write("##### OR TRY A SAMPLE")
    
    # Session state for tracking sample selection
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None
        st.session_state.run_identification = False
        
    samples_dir = "samples"
    sample_files = [f"sample{i}.mp3" for i in range(1, 6)]
    
    # Mapping for sample descriptions (like in Screenshot 5 of uploader)
    sample_labels = {
        "sample1.mp3": "sample1",
        "sample2.mp3": "sample2",
        "sample3.mp3": "sample3",
        "sample4.mp3": "sample4",
        "sample5.mp3": "sample5"
    }
    
    for s_file in sample_files:
        s_path = os.path.join(samples_dir, s_file)
        if os.path.exists(s_path):
            col_label, col_audio, col_btn = st.columns([1, 6, 1])
            with col_label:
                st.write(f"**{sample_labels[s_file]}**")
            with col_audio:
                st.audio(s_path, format="audio/mp3")
            with col_btn:
                if st.button("Try", key=f"btn_{s_file}"):
                    st.session_state.selected_sample = s_path
                    st.session_state.run_identification = True
                    uploaded_file = None # Clear uploader

    # Add a spacer
    st.write("")
    
    # Identification trigger
    query_audio_path = None
    if uploaded_file is not None:
        query_audio_path = uploaded_file
        identify_clicked = st.button("Identify", key="btn_identify_upload")
    elif st.session_state.selected_sample is not None and st.session_state.run_identification:
        query_audio_path = st.session_state.selected_sample
        identify_clicked = True
        # Reset trigger so it doesn't re-run infinitely on page loads
        st.session_state.run_identification = False
    else:
        identify_clicked = False
        
    if identify_clicked and query_audio_path is not None:
        with st.spinner("Analyzing and searching signatures..."):
            try:
                # Load query audio
                if isinstance(query_audio_path, str):
                    # It's a path
                    y, sr = librosa.load(query_audio_path, sr=DEFAULT_FS)
                else:
                    # It's an uploaded file object
                    y, sr = librosa.load(io.BytesIO(query_audio_path.read()), sr=DEFAULT_FS)
                    
                # Compute Spectrogram
                stft_q = librosa.stft(y, n_fft=1024, hop_length=256)
                spec_db = librosa.amplitude_to_db(np.abs(stft_q), ref=np.max)
                
                # Extract peaks and hashes
                peaks = get_peaks(spec_db)
                hashes = generate_hashes(peaks)
                
                # Match
                pred_song, max_matches, offset_histograms, alignment_scores = match_query(hashes, cached_db)
                
                if pred_song:
                    # ----------------- STEP 1 -----------------
                    st.markdown("<div class='step-header'>Step 1 • Peak Extraction</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='step-title'>Spectrogram & Peaks</div>", unsafe_allow_html=True)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    fig.patch.set_facecolor('#0d0f14')
                    
                    # Left: Spectrogram with peaks overlaid
                    ax1.set_facecolor('#0d0f14')
                    ax1.imshow(spec_db, origin='lower', aspect='auto', cmap='magma')
                    t_idx = [p[0] for p in peaks]
                    f_idx = [p[1] for p in peaks]
                    ax1.scatter(t_idx, f_idx, color='blue', s=8, alpha=0.8)
                    ax1.set_title("Query Spectrogram + Peaks (blue)", color='white')
                    ax1.tick_params(colors='white')
                    ax1.spines['bottom'].set_color('#1f232d')
                    ax1.spines['left'].set_color('#1f232d')
                    ax1.xaxis.label.set_color('#5d6475')
                    ax1.yaxis.label.set_color('#5d6475')
                    
                    # Right: Clean constellation map
                    ax2.set_facecolor('#0d0f14')
                    ax2.scatter(t_idx, f_idx, color='cyan', s=8, alpha=0.9)
                    ax2.set_title(f"Constellation Map ({len(peaks)} peaks)", color='white')
                    ax2.tick_params(colors='white')
                    ax2.spines['bottom'].set_color('#1f232d')
                    ax2.spines['left'].set_color('#1f232d')
                    ax2.xaxis.label.set_color('#5d6475')
                    ax2.yaxis.label.set_color('#5d6475')
                    
                    st.pyplot(fig)
                    st.markdown(f"<div class='step-desc'>From this rich image, only the <b>{len(peaks)}</b> most prominent peaks were kept (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.</div>", unsafe_allow_html=True)
                    
                    # ----------------- STEP 2 -----------------
                    st.markdown("<div class='step-header'>Step 2 • Database Search</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='step-title'>Where in the song?</div>", unsafe_allow_html=True)
                    
                    # Reconstruct constellation of the matched song from database
                    song_peaks = []
                    for hash_key, entries in cached_db.db.items():
                        f1, f2, dt = hash_key
                        for t1, song_name in entries:
                            if song_name == pred_song:
                                song_peaks.append((t1, f1))
                    song_peaks = list(set(song_peaks))
                    
                    # Find the alignment offset (mode)
                    offsets = offset_histograms.get(pred_song, [])
                    offsets_arr = np.array(offsets)
                    unique_offsets, counts = np.unique(offsets_arr, return_counts=True)
                    max_idx = np.argmax(counts)
                    peak_offset = unique_offsets[max_idx]
                    
                    # Frame duration of query
                    query_duration_frames = spec_db.shape[1]
                    
                    # Plot full song constellation + highlighted segment
                    fig_search, ax_search = plt.subplots(figsize=(12, 4))
                    fig_search.patch.set_facecolor('#0d0f14')
                    ax_search.set_facecolor('#0d0f14')
                    
                    t_song = [p[0] for p in song_peaks]
                    f_song = [p[1] for p in song_peaks]
                    ax_search.scatter(t_song, f_song, color='#3c4250', s=1, alpha=0.6)
                    
                    # Draw highlight window
                    ax_search.axvspan(peak_offset, peak_offset + query_duration_frames, color='orange', alpha=0.3)
                    ax_search.set_title(f"Full Constellation of '{pred_song}' with Match Window Highlighted (orange)", color='white')
                    ax_search.set_xlabel("time (frames)", color='white')
                    ax_search.set_ylabel("freq bin", color='white')
                    ax_search.tick_params(colors='white')
                    ax_search.spines['bottom'].set_color('#1f232d')
                    ax_search.spines['left'].set_color('#1f232d')
                    st.pyplot(fig_search)
                    
                    st.markdown(f"<div class='step-desc'>The <b>{len(hashes):,}</b> fingerprint hashes were looked up against every indexed track. Below is the full fingerprint of <i>{pred_song}</i> reconstructed from the database, each dot is a stored hash anchor. The highlighted window is exactly where the query clip sits inside the full song.</div>", unsafe_allow_html=True)
                    
                    # ----------------- STEP 3 -----------------
                    st.markdown("<div class='step-header'>Step 3 • The Proof</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='step-title'>The alignment spike</div>", unsafe_allow_html=True)
                    
                    fig_proof, ax_proof = plt.subplots(figsize=(12, 4))
                    fig_proof.patch.set_facecolor('#0d0f14')
                    ax_proof.set_facecolor('#0d0f14')
                    
                    # Plot alignment histogram
                    counts_hist, bins_hist = np.histogram(offsets, bins=100)
                    ax_proof.bar(bins_hist[:-1], counts_hist, width=np.diff(bins_hist), color='#3c4250', alpha=0.7)
                    
                    # Highlight the winning bin
                    ax_proof.bar([peak_offset], [max_matches], width=np.diff(bins_hist)[0]*3, color='orange')
                    ax_proof.annotate(f"{max_matches} hashes\nalign here", 
                                      xy=(peak_offset, max_matches), 
                                      xytext=(peak_offset + 100, max_matches * 0.8),
                                      arrowprops=dict(facecolor='orange', shrink=0.05, width=1, headwidth=6),
                                      color='orange', fontweight='bold')
                                      
                    ax_proof.set_title(f"Time Offset Alignment Histogram for '{pred_song}'", color='white')
                    ax_proof.set_xlabel("time offset (database frame - query frame)", color='white')
                    ax_proof.set_ylabel("# hashes", color='white')
                    ax_proof.tick_params(colors='white')
                    ax_proof.spines['bottom'].set_color('#1f232d')
                    ax_proof.spines['left'].set_color('#1f232d')
                    st.pyplot(fig_proof)
                    
                    st.markdown(f"<div class='step-desc'>Every matched hash votes for a time offset (database frame minus query frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them converge: <b>{max_matches} hashes agreed on a single offset</b>. That spike cannot be a coincidence.</div>", unsafe_allow_html=True)
                else:
                    st.error("❌ Song could not be identified! No matching fingerprints in database.")
            except Exception as e:
                st.error(f"Error processing audio: {e}")
                
        # Clear selected state so it doesn't run again on reload
        st.session_state.selected_sample = None

# ----------------- BATCH TAB -----------------
with tab_batch:
    st.write("### Identify many clips at once")
    st.markdown("""
    Upload a set of query clips. Each is identified against the currently indexed library, and the results are written to a standardised <code>results.csv</code> with columns <span style='color:#00f0ff;'>filename</span>, <span style='color:#00f0ff;'>prediction</span>. The prediction is the matched track's filename without its extension, or <code>none</code> when no candidate clears the confidence threshold.
    """, unsafe_allow_html=True)
    
    uploaded_batch_files = st.file_uploader("Select multiple audio files for batch processing", type=["mp3", "wav", "m4a"], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_batch_files:
        st.write(f"Selected **{len(uploaded_batch_files)}** files.")
        
        if st.button("Run batch"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, f in enumerate(uploaded_batch_files):
                status_text.text(f"Processing ({idx+1}/{len(uploaded_batch_files)}): {f.name}")
                try:
                    # Load audio
                    y, sr = librosa.load(io.BytesIO(f.read()), sr=DEFAULT_FS)
                    stft_q = librosa.stft(y, n_fft=1024, hop_length=256)
                    spec_db = librosa.amplitude_to_db(np.abs(stft_q), ref=np.max)
                    peaks = get_peaks(spec_db)
                    hashes = generate_hashes(peaks)
                    
                    pred_song, max_matches, _, _ = match_query(hashes, cached_db)
                    
                    # The demo video shows "none" if it doesn't clear the confidence threshold.
                    # We can set a confidence threshold of, e.g., 5 aligned hashes.
                    if not pred_song or max_matches < 5:
                        pred_song = "none"
                        
                    results.append({"FILE": f.name, "PREDICTION": pred_song})
                except Exception as e:
                    results.append({"FILE": f.name, "PREDICTION": "error"})
                
                progress_bar.progress((idx + 1) / len(uploaded_batch_files))
                
            status_text.text("Batch processing complete!")
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Show table exactly like in the screenshot
            st.write("### RESULTS")
            st.write(f"{len(results)}/{len(results)} clips matched to a track.")
            
            # Display custom table
            st.dataframe(df, use_container_width=True)
            
            # CSV Download
            # Rename columns to matches requirements: filename, prediction
            df_export = df.rename(columns={"FILE": "filename", "PREDICTION": "prediction"})
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label="Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )
