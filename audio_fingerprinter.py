import os
import pickle
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt

# Configuration parameters
DEFAULT_FS = 11025
N_FFT = 1024
HOP_LENGTH = 256
PEAK_NEIGHBORHOOD_SIZE = 15  # Size of maximum filter neighborhood
MIN_PEAK_THRESHOLD_DB = 10   # Keep peaks above this amplitude (dB) in spectrogram
FAN_OUT = 3                  # Number of target peaks to pair with each anchor peak
MIN_TIME_DELTA = 2           # Minimum time difference between anchor and target (frames)
MAX_TIME_DELTA = 35          # Maximum time difference between anchor and target (frames)
FREQ_DELTA_LIMIT = 150       # Maximum frequency difference (bins)

def compute_spectrogram(audio_path, fs=DEFAULT_FS):
    """
    Load audio and compute its spectrogram in dB.
    """
    y, sr = librosa.load(audio_path, sr=fs)
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram = np.abs(stft)
    # Convert to dB scale
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram_db, y, sr

def get_peaks(spectrogram_db, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, min_threshold_db=MIN_PEAK_THRESHOLD_DB):
    """
    Find local peaks (maxima) in the spectrogram.
    Returns list of (time_idx, freq_idx) coordinates.
    """
    # 2D maximum filter
    local_max = maximum_filter(spectrogram_db, size=neighborhood_size) == spectrogram_db
    
    # Filter out weak peaks in background (since db is negative/0, we look relative to min or absolute threshold)
    # Since spectrogram_db has values between -80 and 0, we can define a threshold relative to the minimum
    # or look at absolute values. The threshold min_threshold_db can be, e.g., -60 or relative to min.
    min_val = np.min(spectrogram_db)
    threshold = min_val + min_threshold_db
    
    background = (spectrogram_db > threshold)
    eroded_background = binary_erosion(background, structure=np.ones((3, 3)))
    
    detected_peaks = local_max & eroded_background
    
    # Get coordinates of peaks
    freq_indices, time_indices = np.where(detected_peaks)
    
    # Return as list of (time, freq) tuples
    peaks = list(zip(time_indices, freq_indices))
    return peaks

def generate_hashes(peaks, song_id=None):
    """
    Pair peaks within a target zone to create hashes.
    Each hash is a tuple: ((f1, f2, dt), t1)
    """
    hashes = []
    # Sort peaks by time index
    peaks = sorted(peaks, key=lambda x: x[0])
    num_peaks = len(peaks)
    
    for i in range(num_peaks):
        t1, f1 = peaks[i]
        
        # Look at subsequent peaks
        matches_found = 0
        for j in range(i + 1, num_peaks):
            t2, f2 = peaks[j]
            dt = t2 - t1
            
            # Check if within target zone
            if dt < MIN_TIME_DELTA:
                continue
            if dt > MAX_TIME_DELTA:
                break # Since peaks are sorted by time, we can stop early
                
            df = abs(f2 - f1)
            if df <= FREQ_DELTA_LIMIT:
                # Generate hash key: (f1, f2, dt)
                hash_key = (int(f1), int(f2), int(dt))
                hashes.append((hash_key, int(t1)))
                matches_found += 1
                if matches_found >= FAN_OUT:
                    break
                    
    return hashes

class FingerprintDatabase:
    def __init__(self):
        # Dictionary mapping hash_key (f1, f2, dt) -> list of (t1, song_name)
        self.db = {}
        self.song_list = []

    def add_song(self, song_name, hashes):
        if song_name not in self.song_list:
            self.song_list.append(song_name)
        for hash_key, t1 in hashes:
            if hash_key not in self.db:
                self.db[hash_key] = []
            self.db[hash_key].append((t1, song_name))

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.db, self.song_list), f)
        print(f"Database saved with {len(self.db)} hashes and {len(self.song_list)} songs.")

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.db, self.song_list = pickle.load(f)
            print(f"Database loaded with {len(self.db)} hashes and {len(self.song_list)} songs.")
            return True
        return False

def match_query(query_hashes, database):
    """
    Matches query hashes against the database.
    Returns:
        predicted_song: name of the matched song
        max_matches: number of aligned matches
        offset_histograms: dict mapping song_name -> list of offsets for plotting
        alignment_scores: dict mapping song_name -> matching score (peak count in histogram)
    """
    # Matches will store: song_name -> list of offsets (t_song - t_query)
    matches = {}
    
    for hash_key, t_query in query_hashes:
        if hash_key in database.db:
            for t_song, song_name in database.db[hash_key]:
                offset = t_song - t_query
                if song_name not in matches:
                    matches[song_name] = []
                matches[song_name].append(offset)
                
    if not matches:
        return None, 0, {}, {}
        
    # For each song, find the largest consensus (mode of offsets)
    alignment_scores = {}
    offset_histograms = {}
    
    for song_name, offsets in matches.items():
        # Compute histogram / count occurrences of each offset
        offsets_arr = np.array(offsets)
        # Find the mode offset
        unique_offsets, counts = np.unique(offsets_arr, return_counts=True)
        max_idx = np.argmax(counts)
        peak_offset = unique_offsets[max_idx]
        score = counts[max_idx]
        
        alignment_scores[song_name] = score
        offset_histograms[song_name] = offsets
        
    # Predict the song with the highest score
    predicted_song = max(alignment_scores, key=alignment_scores.get)
    max_matches = alignment_scores[predicted_song]
    
    return predicted_song, max_matches, offset_histograms, alignment_scores
