"""
feature_extractor.py — Green AI Cascade v2
==========================================
Extracts a 22-dimensional feature vector for every heatmap window in a video.
Windows are aligned to the heatmap's native ~14.29s grid so features and
labels share identical temporal boundaries.

Output per video:
    data/research_dataset/features/{video_id}_features.json
    [
        {
            "window_idx":   0,
            "start_time":   0.0,
            "end_time":     14.29,
            "features":     [f1, f2, ..., f22]   # 22-dim vector
        },
        ...
    ]

Feature vector layout (22 dims):
    Audio (18):
        0   spectral_flux_mean
        1   spectral_flux_max
        2   spectral_flux_std
        3   rms_mean
        4   rms_max
        5   rms_std
        6-8  mfcc_1_mean, mfcc_2_mean, mfcc_3_mean   (top 3 MFCCs — mean)
        9-11 mfcc_1_std,  mfcc_2_std,  mfcc_3_std    (top 3 MFCCs — std)
        12  zcr_mean          (zero crossing rate)
        13  spectral_centroid_mean
        14  spectral_centroid_std
        15  spectral_rolloff_mean
        16  spectral_bandwidth_mean
        17  onset_density     (onsets per second — crowd clap/laugh bursts)

    Visual (2):
        18  sharpness_mean    (Laplacian variance — existing signal)
        19  brightness_mean   (mean pixel intensity)

    Temporal (2):
        20  position_norm     (window centre / video duration, 0→1)
        21  local_rank        (rank of rms_mean vs 5 neighbours, 0→1)
"""

import os
import json
import numpy as np
import librosa
import cv2
from scipy.signal import butter, sosfilt

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV_DIR      = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "wav")
MP4_DIR      = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "mp4")
HEATMAP_DIR  = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "heatmaps")
FEATURE_DIR  = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "features")

os.makedirs(FEATURE_DIR, exist_ok=True)

# ── Audio constants ───────────────────────────────────────────────────────────
SR           = 22050   # librosa default sample rate
HOP_LENGTH   = 512
N_FFT        = 2048
N_MFCC       = 13
BP_LOW       = 300     # bandpass low Hz  (matches v1 filter)
BP_HIGH      = 3000    # bandpass high Hz (matches v1 filter)
FRAMES_PER_S = 1       # visual sampling rate (1 fps — cheap, sufficient)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _bandpass(y: np.ndarray, sr: int) -> np.ndarray:
    """4th-order Butterworth bandpass 300–3000 Hz (matches Phase I of v1)."""
    sos = butter(4, [BP_LOW, BP_HIGH], btype="band", fs=sr, output="sos")
    return sosfilt(sos, y)


def _safe_stat(arr: np.ndarray):
    """Return (mean, max, std) safely even if arr is empty."""
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(arr)), float(np.max(arr)), float(np.std(arr))


def extract_audio_features(y_bp: np.ndarray, y_raw: np.ndarray,
                            sr: int, start: float, end: float) -> list:
    """
    Extract 18 audio features for a single time window [start, end].
    y_bp  — bandpass-filtered signal (for flux, matches v1)
    y_raw — unfiltered signal (for MFCCs, ZCR, centroid etc.)
    """
    s = int(start * sr)
    e = int(end   * sr)

    chunk_bp  = y_bp[s:e]  if s < len(y_bp)  else np.zeros(1)
    chunk_raw = y_raw[s:e] if s < len(y_raw) else np.zeros(1)

    if chunk_bp.size < N_FFT:
        chunk_bp  = np.pad(chunk_bp,  (0, N_FFT - chunk_bp.size))
    if chunk_raw.size < N_FFT:
        chunk_raw = np.pad(chunk_raw, (0, N_FFT - chunk_raw.size))

    # ── Spectral flux (bandpass, matches v1 equation 2) ──────────────────────
    stft_bp  = np.abs(librosa.stft(chunk_bp,  n_fft=N_FFT, hop_length=HOP_LENGTH))
    flux     = np.sum(np.maximum(0, np.diff(stft_bp, axis=1)), axis=0)
    sf_mean, sf_max, sf_std = _safe_stat(flux)

    # ── RMS energy ───────────────────────────────────────────────────────────
    rms      = librosa.feature.rms(y=chunk_raw, hop_length=HOP_LENGTH)[0]
    rms_mean, rms_max, rms_std = _safe_stat(rms)

    # ── MFCCs (top 3 coefficients, mean + std) ────────────────────────────────
    mfcc     = librosa.feature.mfcc(y=chunk_raw, sr=sr,
                                    n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_means = [float(np.mean(mfcc[i])) for i in range(3)]
    mfcc_stds  = [float(np.std(mfcc[i]))  for i in range(3)]

    # ── Zero crossing rate ────────────────────────────────────────────────────
    zcr      = librosa.feature.zero_crossing_rate(chunk_raw, hop_length=HOP_LENGTH)[0]
    zcr_mean = float(np.mean(zcr))

    # ── Spectral centroid ─────────────────────────────────────────────────────
    stft_raw   = np.abs(librosa.stft(chunk_raw, n_fft=N_FFT, hop_length=HOP_LENGTH))
    centroid   = librosa.feature.spectral_centroid(S=stft_raw, sr=sr)[0]
    cent_mean  = float(np.mean(centroid))
    cent_std   = float(np.std(centroid))

    # ── Spectral rolloff ──────────────────────────────────────────────────────
    rolloff    = librosa.feature.spectral_rolloff(S=stft_raw, sr=sr)[0]
    roll_mean  = float(np.mean(rolloff))

    # ── Spectral bandwidth ────────────────────────────────────────────────────
    bandwidth  = librosa.feature.spectral_bandwidth(S=stft_raw, sr=sr)[0]
    bw_mean    = float(np.mean(bandwidth))

    # ── Onset density (laughs/claps per second) ───────────────────────────────
    onset_frames  = librosa.onset.onset_detect(y=chunk_raw, sr=sr,
                                               hop_length=HOP_LENGTH)
    window_dur    = max(end - start, 1.0)
    onset_density = float(len(onset_frames) / window_dur)

    return [
        sf_mean, sf_max, sf_std,
        rms_mean, rms_max, rms_std,
        *mfcc_means, *mfcc_stds,
        zcr_mean,
        cent_mean, cent_std,
        roll_mean,
        bw_mean,
        onset_density,
    ]  # 18 values


# ── Visual helpers ────────────────────────────────────────────────────────────

def extract_visual_features(mp4_path: str, start: float, end: float) -> list:
    """
    Extract 2 visual features for a time window by sampling at FRAMES_PER_S.
    Returns [sharpness_mean, brightness_mean].
    Falls back to [0.0, 0.0] if video can't be read.
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return [0.0, 0.0]

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sharpness_vals = []
    brightness_vals = []

    # Sample one frame per second across the window
    sample_times = np.arange(start, end, 1.0 / FRAMES_PER_S)
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness_vals.append(float(np.mean(gray)))

    cap.release()

    sharpness = float(np.mean(sharpness_vals)) if sharpness_vals else 0.0
    brightness = float(np.mean(brightness_vals)) if brightness_vals else 0.0
    return [sharpness, brightness]


# ── Temporal helpers ──────────────────────────────────────────────────────────

def compute_temporal_features(windows: list, idx: int,
                               video_duration: float,
                               rms_means: list) -> list:
    """
    Compute 2 temporal features for window at index idx.
    position_norm: normalised centre of window (0→1)
    local_rank:    rank of this window's rms_mean vs ±5 neighbours (0→1)
    """
    w = windows[idx]
    centre       = (w["start_time"] + w["end_time"]) / 2.0
    position_norm = centre / max(video_duration, 1.0)

    # Local rank: compare rms_mean against up to 5 neighbours each side
    neighbour_start = max(0, idx - 5)
    neighbour_end   = min(len(windows), idx + 6)
    local_rms       = rms_means[neighbour_start:neighbour_end]
    if len(local_rms) > 1:
        rank       = sorted(local_rms).index(rms_means[idx])
        local_rank = rank / (len(local_rms) - 1)
    else:
        local_rank = 0.5

    return [float(position_norm), float(local_rank)]


# ── Main extraction function ──────────────────────────────────────────────────

def extract_features_for_video(video_id: str) -> str | None:
    """
    Full feature extraction pipeline for one video.
    Returns path to saved feature JSON, or None on failure.
    """
    wav_path     = os.path.join(WAV_DIR,     f"{video_id}.wav")
    mp4_path     = os.path.join(MP4_DIR,     f"{video_id}.mp4")
    heatmap_path = os.path.join(HEATMAP_DIR, f"{video_id}_heatmap.json")
    out_path     = os.path.join(FEATURE_DIR, f"{video_id}_features.json")

    # Skip if already done
    if os.path.exists(out_path):
        print(f"  [~] {video_id}: already extracted, skipping.")
        return out_path

    # Validate inputs
    for p, label in [(wav_path, "WAV"), (mp4_path, "MP4"),
                     (heatmap_path, "heatmap")]:
        if not os.path.exists(p):
            print(f"  [✗] {video_id}: missing {label} at {p}")
            return None

    print(f"  [*] {video_id}: loading audio...")
    y_raw, sr = librosa.load(wav_path, sr=SR, mono=True)
    y_bp      = _bandpass(y_raw, sr)
    video_duration = len(y_raw) / sr

    with open(heatmap_path) as f:
        heatmap = json.load(f)

    print(f"  [*] {video_id}: extracting audio features "
          f"({len(heatmap)} windows)...")

    # Pass 1: audio features for all windows (need rms_means for temporal)
    audio_feats_all = []
    for seg in heatmap:
        af = extract_audio_features(y_bp, y_raw, sr,
                                    seg["start_time"], seg["end_time"])
        audio_feats_all.append(af)

    rms_means = [af[3] for af in audio_feats_all]  # index 3 = rms_mean

    print(f"  [*] {video_id}: extracting visual features...")

    results = []
    for idx, seg in enumerate(heatmap):
        af = audio_feats_all[idx]
        vf = extract_visual_features(mp4_path, seg["start_time"], seg["end_time"])
        tf = compute_temporal_features(heatmap, idx, video_duration, rms_means)

        feature_vector = af + vf + tf  # 18 + 2 + 2 = 22 dims

        results.append({
            "window_idx": idx,
            "start_time": seg["start_time"],
            "end_time":   seg["end_time"],
            "features":   feature_vector,
        })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  [+] {video_id}: saved {len(results)} windows → {out_path}")
    return out_path


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_all():
    """Extract features for all videos that have a heatmap."""
    heatmap_files = sorted(os.listdir(HEATMAP_DIR))
    video_ids = [
        f.replace("_heatmap.json", "")
        for f in heatmap_files
        if f.endswith("_heatmap.json")
    ]

    print(f"[*] Green AI v2 — Feature Extraction")
    print(f"[*] Found {len(video_ids)} videos to process\n")

    success, failed = 0, []
    for vid in video_ids:
        result = extract_features_for_video(vid)
        if result:
            success += 1
        else:
            failed.append(vid)

    print(f"\n[✓] Done: {success}/{len(video_ids)} videos extracted")
    if failed:
        print(f"[✗] Failed: {failed}")


if __name__ == "__main__":
    run_all()