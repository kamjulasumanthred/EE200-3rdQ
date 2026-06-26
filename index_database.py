import os
import time
from audio_fingerprinter import compute_spectrogram, get_peaks, generate_hashes, FingerprintDatabase

def main():
    song_dir = "EE200_course_project_data_2026/Q3_data"
    db_path = "fingerprint_db.pkl"
    
    db = FingerprintDatabase()
    
    start_time = time.time()
    
    # List files
    files = [f for f in os.listdir(song_dir) if f.endswith('.mp3')]
    print(f"Found {len(files)} songs to index.")
    
    for i, filename in enumerate(files):
        song_name = os.path.splitext(filename)[0]
        filepath = os.path.join(song_dir, filename)
        
        t0 = time.time()
        try:
            # Compute spectrogram
            spectrogram_db, y, sr = compute_spectrogram(filepath)
            
            # Find peaks
            peaks = get_peaks(spectrogram_db)
            
            # Generate hashes
            hashes = generate_hashes(peaks)
            
            # Add to database
            db.add_song(song_name, hashes)
            
            duration = len(y) / sr
            print(f"[{i+1}/{len(files)}] Indexed '{song_name}' ({duration:.1f}s) - {len(peaks)} peaks, {len(hashes)} hashes in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"Error indexing {filename}: {e}")
            
    print(f"Saving database to {db_path}...")
    db.save(db_path)
    print(f"Indexing completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
