import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from audio_fingerprinter import compute_spectrogram, get_peaks, generate_hashes, FingerprintDatabase, match_query, DEFAULT_FS

def get_single_peak_matches(query_peaks, database):
    """
    Simulate matching using only single peaks.
    Match a query peak to any database peak with the same frequency bin,
    and accumulate offsets: offset = t_song - t_query.
    """
    # First, we need to map frequency bin -> list of (t_song, song_name) in database.
    # Since our database is hash-based, let's build a temporary frequency bin index.
    freq_db = {}
    for hash_key, entries in database.db.items():
        f1, f2, dt = hash_key
        for t_song, song_name in entries:
            # Add f1
            if f1 not in freq_db:
                freq_db[f1] = []
            freq_db[f1].append((t_song, song_name))
            
    matches = {}
    for t_query, f_query in query_peaks:
        if f_query in freq_db:
            for t_song, song_name in freq_db[f_query]:
                offset = t_song - t_query
                if song_name not in matches:
                    matches[song_name] = []
                matches[song_name].append(offset)
                
    if not matches:
        return None, 0, {}, {}
        
    alignment_scores = {}
    offset_histograms = {}
    
    for song_name, offsets in matches.items():
        offsets_arr = np.array(offsets)
        unique_offsets, counts = np.unique(offsets_arr, return_counts=True)
        max_idx = np.argmax(counts)
        score = counts[max_idx]
        alignment_scores[song_name] = score
        offset_histograms[song_name] = offsets
        
    predicted_song = max(alignment_scores, key=alignment_scores.get)
    return predicted_song, alignment_scores[predicted_song], offset_histograms, alignment_scores

def add_noise(y, snr_db):
    """
    Add white Gaussian noise to signal y to achieve target SNR in dB.
    """
    sig_power = np.mean(y ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def main():
    db_path = "fingerprint_db.pkl"
    db = FingerprintDatabase()
    if not db.load(db_path):
        print("Error: Could not load database. Run index_database.py first!")
        return

    # Choose a target song for query extraction
    target_song = "Across The Universe"
    song_dir = "EE200_course_project_data_2026/Q3_data"
    song_path = os.path.join(song_dir, f"{target_song}.mp3")
    
    if not os.path.exists(song_path):
        print(f"Error: Target song file {song_path} not found!")
        return
        
    print(f"Loading '{target_song}' to extract query clip...")
    y_full, sr = librosa.load(song_path, sr=DEFAULT_FS)
    
    # Slice a 10-second clip from the middle (e.g., from 40s to 50s)
    start_sec = 40
    duration_sec = 10
    start_sample = int(start_sec * DEFAULT_FS)
    end_sample = int((start_sec + duration_sec) * DEFAULT_FS)
    y_query = y_full[start_sample:end_sample]
    
    print("Generating clean query hashes...")
    # Compute spectrogram of clean query
    stft_query = librosa.stft(y_query, n_fft=1024, hop_length=256)
    spec_db_query = librosa.amplitude_to_db(np.abs(stft_query), ref=np.max)
    peaks_query = get_peaks(spec_db_query)
    hashes_query = generate_hashes(peaks_query)
    
    # -------------------------------------------------------------
    # Experiment 1: Single Peaks vs Paired Hashes Offset Histogram
    # -------------------------------------------------------------
    print("Running Experiment 1: Single Peaks vs Paired Hashes...")
    
    # Paired Hashes Matching
    pred_pair, score_pair, hist_pair, scores_pair = match_query(hashes_query, db)
    # Single Peaks Matching
    pred_single, score_single, hist_single, scores_single = get_single_peak_matches(peaks_query, db)
    
    print(f"Paired Hashes prediction: '{pred_pair}' with score {score_pair}")
    print(f"Single Peaks prediction: '{pred_single}' with score {score_single}")
    
    # Plot offset histograms for correct and incorrect songs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Correct song - Paired Hashes
    offsets_correct_pair = hist_pair.get(target_song, [])
    axes[0, 0].hist(offsets_correct_pair, bins=100, color='blue', alpha=0.7)
    axes[0, 0].set_title(f"Paired Hashes: {target_song} (Correct)")
    axes[0, 0].set_xlabel("Offset (frames)")
    axes[0, 0].set_ylabel("Vote Count")
    
    # Incorrect song (choose one with some matches) - Paired Hashes
    inc_songs_pair = [s for s in hist_pair.keys() if s != target_song]
    incorrect_song_pair = inc_songs_pair[0] if inc_songs_pair else "Bohemian Rhapsody"
    offsets_inc_pair = hist_pair.get(incorrect_song_pair, [])
    axes[0, 1].hist(offsets_inc_pair, bins=100, color='red', alpha=0.7)
    axes[0, 1].set_title(f"Paired Hashes: {incorrect_song_pair} (Incorrect)")
    axes[0, 1].set_xlabel("Offset (frames)")
    axes[0, 1].set_ylabel("Vote Count")
    
    # Correct song - Single Peaks
    offsets_correct_single = hist_single.get(target_song, [])
    axes[1, 0].hist(offsets_correct_single, bins=100, color='blue', alpha=0.7)
    axes[1, 0].set_title(f"Single Peaks: {target_song} (Correct)")
    axes[1, 0].set_xlabel("Offset (frames)")
    axes[1, 0].set_ylabel("Vote Count")
    
    # Incorrect song - Single Peaks
    inc_songs_single = [s for s in hist_single.keys() if s != target_song]
    incorrect_song_single = inc_songs_single[0] if inc_songs_single else "Bohemian Rhapsody"
    offsets_inc_single = hist_single.get(incorrect_song_single, [])
    axes[1, 1].hist(offsets_inc_single, bins=100, color='red', alpha=0.7)
    axes[1, 1].set_title(f"Single Peaks: {incorrect_song_single} (Incorrect)")
    axes[1, 1].set_xlabel("Offset (frames)")
    axes[1, 1].set_ylabel("Vote Count")
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/exp1_single_vs_pair.png")
    plt.close()
    
    # -------------------------------------------------------------
    # Experiment 2: Robustness against Additive Noise
    # -------------------------------------------------------------
    print("Running Experiment 2: Noise Robustness...")
    snr_levels = [20, 10, 5, 0, -5, -10]
    scores_under_noise = []
    predictions_under_noise = []
    
    for snr in snr_levels:
        y_noisy = add_noise(y_query, snr)
        stft_noisy = librosa.stft(y_noisy, n_fft=1024, hop_length=256)
        spec_db_noisy = librosa.amplitude_to_db(np.abs(stft_noisy), ref=np.max)
        peaks_noisy = get_peaks(spec_db_noisy)
        hashes_noisy = generate_hashes(peaks_noisy)
        
        pred, score, _, _ = match_query(hashes_noisy, db)
        scores_under_noise.append(score if pred == target_song else 0)
        predictions_under_noise.append(pred)
        print(f"SNR: {snr} dB -> Predicted: '{pred}' (Score: {score})")
        
    plt.figure(figsize=(8, 5))
    plt.plot(snr_levels, scores_under_noise, marker='o', color='purple', linewidth=2)
    plt.title("Fingerprint Matching Score vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Correct Song Match Score (Vote Count)")
    plt.grid(True)
    plt.savefig("plots/exp2_noise_robustness.png")
    plt.close()
    
    # -------------------------------------------------------------
    # Experiment 3: Robustness against Pitch Shift and Time Stretch
    # -------------------------------------------------------------
    print("Running Experiment 3: Pitch Shift & Time Stretch...")
    
    # Pitch Shift
    pitch_shifts = [0.5, 1.0, 2.0]  # semitones
    print("Pitch Shifting...")
    for shift in pitch_shifts:
        y_shifted = librosa.effects.pitch_shift(y_query, sr=DEFAULT_FS, n_steps=shift)
        stft_shift = librosa.stft(y_shifted, n_fft=1024, hop_length=256)
        spec_db_shift = librosa.amplitude_to_db(np.abs(stft_shift), ref=np.max)
        peaks_shift = get_peaks(spec_db_shift)
        hashes_shift = generate_hashes(peaks_shift)
        
        pred, score, _, _ = match_query(hashes_shift, db)
        print(f"Pitch Shift +{shift} semitones -> Predicted: '{pred}' (Score: {score})")
        
    # Time Stretch
    stretches = [0.9, 1.1]
    print("Time Stretching...")
    for rate in stretches:
        y_stretched = librosa.effects.time_stretch(y_query, rate=rate)
        stft_stretch = librosa.stft(y_stretched, n_fft=1024, hop_length=256)
        spec_db_stretch = librosa.amplitude_to_db(np.abs(stft_stretch), ref=np.max)
        peaks_stretch = get_peaks(spec_db_stretch)
        hashes_stretch = generate_hashes(peaks_stretch)
        
        pred, score, _, _ = match_query(hashes_stretch, db)
        print(f"Time Stretch x{rate} -> Predicted: '{pred}' (Score: {score})")

    # Save a sample spectrogram and constellation for report
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db_query, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(label='Amplitude (dB)')
    plt.title(f"Spectrogram of '{target_song}' Query Clip")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (bins)")
    plt.savefig("plots/sample_spectrogram.png")
    plt.close()
    
    plt.figure(figsize=(10, 4))
    t_idx = [p[0] for p in peaks_query]
    f_idx = [p[1] for p in peaks_query]
    plt.scatter(t_idx, f_idx, color='cyan', s=10, marker='o')
    plt.title(f"Constellation Map of '{target_song}' Query Clip")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (bins)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("plots/sample_constellation.png")
    plt.close()

if __name__ == "__main__":
    main()
