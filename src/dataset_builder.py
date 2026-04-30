"""
dataset_builder.py — Green AI Cascade v2
=========================================
Pairs each feature vector with its heatmap saliency label and saves
per-video numpy arrays ready for Colab training.

Applies two preprocessing steps from the original white paper:
  1. Intro Spike Filter  — drops windows starting before 60s (Anomaly 1)
  2. Label normalisation — per-video z-score so the model learns relative
                           saliency rather than absolute replay counts
                           (handles the Viral Outlier Anomaly, Section 4.1.3)

Output per video (saved to data/research_dataset/labels/):
    {video_id}_X.npy   — shape (N, 22)  float32  feature matrix
    {video_id}_y.npy   — shape (N,)     float32  saliency labels
    {video_id}_meta.json               window metadata (start/end times)

Also saves a combined dataset manifest:
    data/research_dataset/labels/dataset_manifest.json
    {
        "video_ids": [...],
        "feature_dim": 22,
        "total_windows": ...,
        "feature_names": [...]
    }
"""

import os
import json
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR   = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "features")
HEATMAP_DIR   = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "heatmaps")
LABEL_DIR     = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "labels")

os.makedirs(LABEL_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
INTRO_CUTOFF  = 60.0   # seconds — drop windows starting before this (Anomaly 1)
FEATURE_DIM   = 22

FEATURE_NAMES = [
    # Audio (18)
    "sf_mean", "sf_max", "sf_std",
    "rms_mean", "rms_max", "rms_std",
    "mfcc1_mean", "mfcc2_mean", "mfcc3_mean",
    "mfcc1_std",  "mfcc2_std",  "mfcc3_std",
    "zcr_mean",
    "centroid_mean", "centroid_std",
    "rolloff_mean",
    "bandwidth_mean",
    "onset_density",
    # Visual (2)
    "sharpness_mean",
    "brightness_mean",
    # Temporal (2)
    "position_norm",
    "local_rank",
]


def _zscore_labels(y: np.ndarray) -> np.ndarray:
    """
    Per-video z-score normalisation of heatmap labels.
    Centres the distribution around 0 with unit variance.
    Then clips to [-3, 3] to suppress extreme outliers.
    Finally rescales to [0, 1] for stable MSE training.
    """
    std = y.std()
    if std < 1e-8:
        # All values identical — return uniform 0.5
        return np.full_like(y, 0.5, dtype=np.float32)
    z = (y - y.mean()) / std
    z = np.clip(z, -3.0, 3.0)
    # Rescale [-3,3] → [0,1]
    z = (z + 3.0) / 6.0
    return z.astype(np.float32)


def build_dataset_for_video(video_id: str) -> dict | None:
    """
    Build (X, y) arrays for one video.
    Returns a stats dict, or None on failure.
    """
    feature_path = os.path.join(FEATURE_DIR, f"{video_id}_features.json")
    heatmap_path = os.path.join(HEATMAP_DIR, f"{video_id}_heatmap.json")
    x_out        = os.path.join(LABEL_DIR,   f"{video_id}_X.npy")
    y_out        = os.path.join(LABEL_DIR,   f"{video_id}_y.npy")
    meta_out     = os.path.join(LABEL_DIR,   f"{video_id}_meta.json")

    if os.path.exists(x_out) and os.path.exists(y_out):
        print(f"  [~] {video_id}: already built, skipping.")
        X = np.load(x_out)
        return {"video_id": video_id, "windows": len(X), "skipped": True}

    for p, label in [(feature_path, "features"), (heatmap_path, "heatmap")]:
        if not os.path.exists(p):
            print(f"  [✗] {video_id}: missing {label} at {p}")
            return None

    with open(feature_path) as f:
        feature_data = json.load(f)

    with open(heatmap_path) as f:
        heatmap_data = json.load(f)

    # Build lookup: start_time → heatmap value
    heatmap_lookup = {
        round(seg["start_time"], 4): seg["value"]
        for seg in heatmap_data
    }

    X_rows, y_rows, meta_rows = [], [], []

    for win in feature_data:
        start = win["start_time"]

        # ── Anomaly 1: skip intro windows ─────────────────────────────────────
        if start < INTRO_CUTOFF:
            continue

        # ── Match heatmap label ───────────────────────────────────────────────
        label_val = heatmap_lookup.get(round(start, 4), None)
        if label_val is None:
            # Try nearest match within 1s tolerance
            closest = min(heatmap_lookup.keys(),
                         key=lambda k: abs(k - start))
            if abs(closest - start) < 1.0:
                label_val = heatmap_lookup[closest]
            else:
                print(f"  [!] {video_id}: no heatmap match for window "
                      f"start={start:.2f}s, skipping window.")
                continue

        features = win["features"]
        if len(features) != FEATURE_DIM:
            print(f"  [!] {video_id}: wrong feature dim "
                  f"({len(features)} vs {FEATURE_DIM}), skipping window.")
            continue

        X_rows.append(features)
        y_rows.append(label_val)
        meta_rows.append({
            "window_idx": win["window_idx"],
            "start_time": start,
            "end_time":   win["end_time"],
            "raw_label":  label_val,
        })

    if len(X_rows) < 5:
        print(f"  [✗] {video_id}: only {len(X_rows)} valid windows — skipping.")
        return None

    X = np.array(X_rows, dtype=np.float32)
    y_raw = np.array(y_rows, dtype=np.float32)

    # ── Anomaly 2: per-video z-score normalisation ────────────────────────────
    y_norm = _zscore_labels(y_raw)

    # ── Sanity check: replace any NaN/Inf in features with 0 ─────────────────
    nan_count = np.isnan(X).sum() + np.isinf(X).sum()
    if nan_count > 0:
        print(f"  [!] {video_id}: {nan_count} NaN/Inf values in features, "
              f"replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(x_out, X)
    np.save(y_out, y_norm)

    # Save metadata (raw labels preserved for evaluation)
    with open(meta_out, "w") as f:
        json.dump(meta_rows, f, indent=2)

    print(f"  [+] {video_id}: {len(X_rows)} windows → "
          f"X{X.shape}, y{y_norm.shape} | "
          f"label range [{y_norm.min():.3f}, {y_norm.max():.3f}]")

    return {
        "video_id":   video_id,
        "windows":    len(X_rows),
        "label_min":  float(y_norm.min()),
        "label_max":  float(y_norm.max()),
        "label_mean": float(y_norm.mean()),
        "skipped":    False,
    }


def build_all() -> None:
    """Build dataset for all videos that have extracted features."""
    feature_files = sorted(os.listdir(FEATURE_DIR))
    video_ids = [
        f.replace("_features.json", "")
        for f in feature_files
        if f.endswith("_features.json")
    ]

    print(f"[*] Green AI v2 — Dataset Builder")
    print(f"[*] Found {len(video_ids)} feature files\n")

    stats, failed = [], []
    for vid in video_ids:
        result = build_dataset_for_video(vid)
        if result:
            stats.append(result)
        else:
            failed.append(vid)

    # ── Save manifest ─────────────────────────────────────────────────────────
    successful_ids = [s["video_id"] for s in stats]
    total_windows  = sum(s["windows"] for s in stats)

    manifest = {
        "video_ids":     successful_ids,
        "feature_dim":   FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "total_windows": total_windows,
        "total_videos":  len(successful_ids),
        "intro_cutoff_s": INTRO_CUTOFF,
        "label_normalisation": "per-video zscore clipped [-3,3] rescaled [0,1]",
    }

    manifest_path = os.path.join(LABEL_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  GREEN AI v2 — DATASET BUILD COMPLETE")
    print(f"{'='*55}")
    print(f"  Videos processed:  {len(successful_ids)}/{len(video_ids)}")
    print(f"  Total windows:     {total_windows}")
    print(f"  Feature dims:      {FEATURE_DIM}")
    print(f"  Intro filtered:    windows before {INTRO_CUTOFF}s dropped")
    print(f"  Labels:            per-video z-score → [0, 1]")
    print(f"  Manifest saved:    {manifest_path}")
    if failed:
        print(f"  Failed:            {failed}")
    print(f"{'='*55}")
    print(f"\n  Ready for Colab training.")
    print(f"  Upload data/research_dataset/labels/ to Google Drive.\n")


if __name__ == "__main__":
    build_all()