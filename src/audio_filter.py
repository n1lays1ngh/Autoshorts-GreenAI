import os
import json
import librosa
import numpy as np
from scipy.signal import butter, lfilter, find_peaks


def apply_bandpass_filter(data, sample_rate, lowcut=300.0, highcut=3000.0, order=5):
    """
    Green AI Step: Isolate human vocal frequencies (laughter range).
    Filters out low-frequency mic thumps and high-frequency electronic hiss.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data


def calculate_spectral_flux(audio_path):
    """
    Implements Equation 4: Flux_t = Sum( (M_t[k] - M_t-1[k])^2 )
    to detect acoustic energy bursts (onsets of laughter).
    """
    print(f"[*] Analyzing Acoustic Spectrum: {os.path.basename(audio_path)}")

    # 1. Load at 16kHz (Standard for speech/vocal AI to save RAM)
    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # 2. Apply Bandpass Filter
    y_filtered = apply_bandpass_filter(y, sr)

    # 3. Short-Time Fourier Transform (STFT)
    # hop_length=512 at 16kHz gives us a spectral update every ~32ms
    hop_length = 512
    S = np.abs(librosa.stft(y_filtered, hop_length=hop_length))

    # 4. Spectral Flux Math (Equation 4)
    # Capture the difference between consecutive frames
    spectral_diff = np.maximum(0, S[:, 1:] - S[:, :-1])

    # Sum of squares across frequency bins (k)
    flux = np.sum(spectral_diff ** 2, axis=0)

    # Normalize for easier thresholding [0.0 - 1.0]
    if np.max(flux) > 0:
        flux_normalized = flux / np.max(flux)
    else:
        flux_normalized = flux

    # 5. Peak Detection (The Punchlines)
    # height=0.3: Peak must be at least 30% of the max burst found in the video.
    # distance: Minimum 15 seconds between detected jokes to avoid overlaps here.
    min_dist_frames = int((sr / hop_length) * 15)
    peaks, _ = find_peaks(flux_normalized, height=0.3, distance=min_dist_frames)

    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # 6. Center the Window (The "Sticky Note" Logic)
    # As discussed: 45s setup + 15s reaction = 60s total window
    candidate_windows = []
    for t in peak_times:
        start_time = max(0.0, t - 45.0)
        end_time = min(total_duration, t + 15.0)

        candidate_windows.append({
            "peak_time": round(float(t), 2),
            "start": round(float(start_time), 2),
            "end": round(float(end_time), 2),
            "duration": round(float(end_time - start_time), 2)
        })

    return candidate_windows


if __name__ == "__main__":
    # Project Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_audio = os.path.join(base_dir, "data", "raw_video", "system_test_media.wav")
    manifest_path = os.path.join(base_dir, "data", "candidate_windows.json")

    if os.path.exists(test_audio):
        windows = calculate_spectral_flux(test_audio)

        # Save the "Sticky Note" Manifest
        with open(manifest_path, "w") as f:
            json.dump(windows, f, indent=4)

        print(f"\n[+] GREEN AI CASCADE (AUDIO) COMPLETE")
        print(f"[*] Saved {len(windows)} candidate windows to: {manifest_path}")

        # Verification Output
        for i, w in enumerate(windows, 1):
            print(f"    Clip {i}: {w['start']}s -> {w['end']}s (Laughter at {w['peak_time']}s)")
    else:
        print(f"[-] Error: File not found at {test_audio}. Please run downloader.py first.")