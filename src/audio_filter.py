import os
import json
import librosa
import numpy as np
from scipy.signal import butter, lfilter, find_peaks


def apply_bandpass_filter(data, sample_rate, lowcut=300.0, highcut=3000.0, order=5):
    nyquist = 0.5 * sample_rate
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return lfilter(b, a, data)


def apply_nms(windows):
    """
    Temporal Non-Maximum Suppression (NMS)
    Merges overlapping windows so the GPU doesn't process the same frames twice.
    """
    if not windows:
        return []

    # Sort windows by start time just to be safe
    windows.sort(key=lambda x: x['start'])

    merged_windows = [windows[0]]

    for current in windows[1:]:
        previous = merged_windows[-1]

        # If the current window overlaps with the previous one
        if current['start'] <= previous['end']:
            # Merge them: Keep the earliest start and the latest end
            previous['end'] = max(previous['end'], current['end'])
            previous['duration'] = round(previous['end'] - previous['start'], 2)
            # We keep the peak_time of the first big laugh in the sequence
        else:
            # No overlap, add as a completely new window
            merged_windows.append(current)

    return merged_windows


def calculate_spectral_flux(audio_path):
    print(f"[*] Analyzing Acoustic Spectrum: {os.path.basename(audio_path)}")

    y, sr = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=y, sr=sr)

    y_filtered = apply_bandpass_filter(y, sr)
    hop_length = 512
    S = np.abs(librosa.stft(y_filtered, hop_length=hop_length))

    spectral_diff = np.maximum(0, S[:, 1:] - S[:, :-1])
    flux = np.sum(spectral_diff ** 2, axis=0)

    if np.max(flux) > 0:
        flux_normalized = flux / np.max(flux)
    else:
        flux_normalized = flux

    # INCREASED THRESHOLD: Only look for the biggest laughs (changed 0.3 to 0.4)
    min_dist_frames = int((sr / hop_length) * 15)
    peaks, _ = find_peaks(flux_normalized, height=0.6, distance=min_dist_frames)

    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    raw_windows = []
    for t in peak_times:
        start_time = max(0.0, t - 45.0)
        end_time = min(total_duration, t + 15.0)

        raw_windows.append({
            "peak_time": round(float(t), 2),
            "start": round(float(start_time), 2),
            "end": round(float(end_time), 2),
            "duration": round(float(end_time - start_time), 2)
        })

    # Apply NMS to merge overlaps
    final_windows = apply_nms(raw_windows)
    return final_windows


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_audio = os.path.join(base_dir, "data", "raw_video",
                              "system_test_media.wav")  # Make sure this path points to your new wav!
    manifest_path = os.path.join(base_dir, "data", "candidate_windows.json")

    if os.path.exists(test_audio):
        windows = calculate_spectral_flux(test_audio)

        with open(manifest_path, "w") as f:
            json.dump(windows, f, indent=4)

        print(f"\n[+] GREEN AI CASCADE (AUDIO) COMPLETE")
        print(f"[*] Saved {len(windows)} merged candidate blocks to: {manifest_path}")

        for i, w in enumerate(windows, 1):
            print(f"    Block {i}: {w['start']}s -> {w['end']}s (Duration: {w['duration']}s)")
    else:
        print(f"[-] Error: File not found at {test_audio}.")