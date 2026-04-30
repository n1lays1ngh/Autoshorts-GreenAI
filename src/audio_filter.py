"""
audio_filter.py — Green AI Cascade v1
======================================
Spectral flux peak detection with configurable window geometry.

Key addition for the window experiment:
  window_lead — seconds before the detected peak (default 45s)
  window_tail — seconds after  the detected peak (default 15s)

Both are now first-class parameters swept by batch_runner.py.
"""

import os
import json
import librosa
import numpy as np
from scipy.signal import butter, lfilter, find_peaks


def apply_bandpass_filter(data, sample_rate, lowcut=300.0, highcut=3000.0, order=5):
    nyquist = 0.5 * sample_rate
    b, a    = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return lfilter(b, a, data)


def apply_nms(windows):
    """Temporal NMS — merges overlapping windows."""
    if not windows:
        return []
    windows.sort(key=lambda x: x['start'])
    merged = [windows[0].copy()]
    for current in windows[1:]:
        prev = merged[-1]
        if current['start'] <= prev['end']:
            prev['end']      = max(prev['end'], current['end'])
            prev['duration'] = round(prev['end'] - prev['start'], 2)
        else:
            merged.append(current.copy())
    return merged


def calculate_spectral_flux(audio_path, threshold=0.6,
                             window_lead=45.0, window_tail=15.0):
    """
    Spectral flux peak detector.

    Parameters
    ----------
    audio_path   : path to .wav file
    threshold    : normalised flux sensitivity (alpha)
    window_lead  : seconds BEFORE detected peak to include in window
    window_tail  : seconds AFTER  detected peak to include in window

    Returns
    -------
    List of window dicts: {peak_time, start, end, duration, window_lead, window_tail}
    """
    print(f"[*] SF  {os.path.basename(audio_path)} "
          f"a={threshold} lead={window_lead}s tail={window_tail}s")

    y, sr          = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=y, sr=sr)

    y_filtered  = apply_bandpass_filter(y, sr)
    hop_length  = 512
    S           = np.abs(librosa.stft(y_filtered, hop_length=hop_length))

    spectral_diff   = np.maximum(0, S[:, 1:] - S[:, :-1])
    flux            = np.sum(spectral_diff ** 2, axis=0)
    flux_normalized = flux / np.max(flux) if np.max(flux) > 0 else flux

    min_dist_frames = int((sr / hop_length) * 15)
    peaks, _        = find_peaks(flux_normalized,
                                 height=threshold,
                                 distance=min_dist_frames)
    peak_times      = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    raw_windows = []
    for t in peak_times:
        start_time = max(0.0, t - window_lead)
        end_time   = min(total_duration, t + window_tail)
        raw_windows.append({
            "peak_time":   round(float(t), 2),
            "start":       round(float(start_time), 2),
            "end":         round(float(end_time), 2),
            "duration":    round(float(end_time - start_time), 2),
            "window_lead": window_lead,
            "window_tail": window_tail,
        })

    return apply_nms(raw_windows)


def calculate_rms_peaks(audio_path, threshold=0.6,
                         window_lead=45.0, window_tail=15.0):
    """RMS amplitude baseline — simplest possible audio detector."""
    y, sr          = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=y, sr=sr)
    hop_length     = 512
    rms            = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm       = rms / np.max(rms) if np.max(rms) > 0 else rms

    min_dist_frames = int((sr / hop_length) * 15)
    peaks, _        = find_peaks(rms_norm, height=threshold,
                                 distance=min_dist_frames)
    peak_times      = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    raw_windows = []
    for t in peak_times:
        start_time = max(0.0, t - window_lead)
        end_time   = min(total_duration, t + window_tail)
        raw_windows.append({
            "peak_time":   round(float(t), 2),
            "start":       round(float(start_time), 2),
            "end":         round(float(end_time), 2),
            "duration":    round(float(end_time - start_time), 2),
            "window_lead": window_lead,
            "window_tail": window_tail,
        })
    return apply_nms(raw_windows)


def calculate_mfcc_peaks(audio_path, threshold=0.6,
                          window_lead=45.0, window_tail=15.0):
    """MFCC timbral variance detector — captures sustained crowd texture."""
    y, sr          = librosa.load(audio_path, sr=16000)
    total_duration = librosa.get_duration(y=y, sr=sr)
    hop_length     = 512
    mfccs          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,
                                           hop_length=hop_length)
    mfcc_var       = np.var(mfccs, axis=0)
    mfcc_norm      = mfcc_var / np.max(mfcc_var) if np.max(mfcc_var) > 0 else mfcc_var

    min_dist_frames = int((sr / hop_length) * 15)
    peaks, _        = find_peaks(mfcc_norm, height=threshold,
                                 distance=min_dist_frames)
    peak_times      = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    raw_windows = []
    for t in peak_times:
        start_time = max(0.0, t - window_lead)
        end_time   = min(total_duration, t + window_tail)
        raw_windows.append({
            "peak_time":   round(float(t), 2),
            "start":       round(float(start_time), 2),
            "end":         round(float(end_time), 2),
            "duration":    round(float(end_time - start_time), 2),
            "window_lead": window_lead,
            "window_tail": window_tail,
        })
    return apply_nms(raw_windows)


if __name__ == "__main__":
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_id   = "wQA68Oqr1qE"
    audio_path = os.path.join(base_dir, "data", "research_dataset",
                              "wav", f"{video_id}.wav")

    candidates_dir = os.path.join(base_dir, "data", "candidate_windows")
    os.makedirs(candidates_dir, exist_ok=True)
    manifest_path  = os.path.join(candidates_dir, f"{video_id}_candidates.json")

    if os.path.exists(audio_path):
        windows = calculate_spectral_flux(audio_path, threshold=0.5,
                                          window_lead=45.0, window_tail=15.0)
        with open(manifest_path, "w") as f:
            json.dump(windows, f, indent=4)
        print(f"\n[+] {len(windows)} windows saved to {manifest_path}")
        for i, w in enumerate(windows, 1):
            print(f"    Block {i}: {w['start']}s -> {w['end']}s ({w['duration']}s)")
    else:
        print(f"[-] Audio not found: {audio_path}")