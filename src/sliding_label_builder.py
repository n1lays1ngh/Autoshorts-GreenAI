"""
sliding_label_builder.py — Green AI Cascade v2
================================================
Reuses existing per-video _X.npy feature vectors (aligned to heatmap's
~14.29s windows) and builds a flat supervised dataset ready for sklearn.

Since the heatmap segments are already ~14.29s each and features are
extracted per-segment, we simply:
  1. Load existing X (features) and raw heatmap values per video
  2. Drop intro windows (start_time < 60s)
  3. Compute per-video z-score normalised labels from raw heatmap values
  4. Stack everything into one flat matrix with group IDs

Output:
    data/research_dataset/sliding_labels/dataset.npz
        X      — (N, 22) float32  feature matrix
        y      — (N,)    float32  retention labels [0,1]
        groups — (N,)    int32    video index (for GroupKFold)

    data/research_dataset/sliding_labels/dataset_info.json
        video_ids, feature_names, per-video stats
"""

import os
import json
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR    = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "features")
HEATMAP_DIR    = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "heatmaps")
SLIDING_DIR    = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "sliding_labels")

os.makedirs(SLIDING_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
INTRO_CUTOFF = 60.0   # drop windows starting before this (Anomaly 1)
FEATURE_DIM  = 22

FEATURE_NAMES = [
    "sf_mean", "sf_max", "sf_std",
    "rms_mean", "rms_max", "rms_std",
    "mfcc1_mean", "mfcc2_mean", "mfcc3_mean",
    "mfcc1_std",  "mfcc2_std",  "mfcc3_std",
    "zcr_mean",
    "centroid_mean", "centroid_std",
    "rolloff_mean", "bandwidth_mean",
    "onset_density",
    "sharpness_mean", "brightness_mean",
    "position_norm", "local_rank",
]


def _zscore_to_01(y_raw: np.ndarray) -> np.ndarray:
    """
    Per-video z-score normalisation → [0, 1].
    Handles the Viral Outlier Anomaly (Section 4.1.3 of white paper).
    """
    std = y_raw.std()
    if std < 1e-8:
        return np.full_like(y_raw, 0.5, dtype=np.float32)
    z = (y_raw - y_raw.mean()) / std
    z = np.clip(z, -3.0, 3.0)
    return ((z + 3.0) / 6.0).astype(np.float32)


def build_dataset():
    # Find all videos that have both features and heatmaps
    feature_files = sorted(f for f in os.listdir(FEATURE_DIR)
                           if f.endswith("_features.json"))
    video_ids = [f.replace("_features.json", "") for f in feature_files]

    print(f"[*] Green AI v2 — Sliding Label Builder")
    print(f"[*] Found {len(video_ids)} videos\n")

    X_all, y_all, groups_all = [], [], []
    per_video_stats = []
    failed = []

    for group_idx, vid in enumerate(video_ids):
        feature_path = os.path.join(FEATURE_DIR, f"{vid}_features.json")
        heatmap_path = os.path.join(HEATMAP_DIR, f"{vid}_heatmap.json")

        if not os.path.exists(heatmap_path):
            print(f"  [✗] {vid}: missing heatmap, skipping.")
            failed.append(vid)
            continue

        # Load feature JSON (has start_time per window)
        with open(feature_path) as f:
            feature_data = json.load(f)

        # Load heatmap
        with open(heatmap_path) as f:
            heatmap_data = json.load(f)

        # Build heatmap lookup: start_time → raw value
        heatmap_lookup = {
            round(seg["start_time"], 3): seg["value"]
            for seg in heatmap_data
        }

        # Filter windows: drop intro
        valid_windows = [
            w for w in feature_data
            if w["start_time"] >= INTRO_CUTOFF
        ]

        if len(valid_windows) < 5:
            print(f"  [✗] {vid}: only {len(valid_windows)} valid windows, skipping.")
            failed.append(vid)
            continue

        # Match features to heatmap labels
        X_rows, y_rows = [], []
        for win in valid_windows:
            features = win["features"]
            if len(features) != FEATURE_DIM:
                continue

            # Match heatmap label by start_time
            start_key = round(win["start_time"], 3)
            label = heatmap_lookup.get(start_key, None)

            if label is None:
                # Nearest match within 2s tolerance
                closest = min(heatmap_lookup.keys(),
                              key=lambda k: abs(k - win["start_time"]))
                if abs(closest - win["start_time"]) < 2.0:
                    label = heatmap_lookup[closest]
                else:
                    continue  # skip unmatched window

            X_rows.append(features)
            y_rows.append(label)

        if len(X_rows) < 5:
            print(f"  [✗] {vid}: too few matched windows, skipping.")
            failed.append(vid)
            continue

        X_vid = np.array(X_rows, dtype=np.float32)
        y_raw = np.array(y_rows, dtype=np.float32)

        # Per-video z-score normalisation
        y_norm = _zscore_to_01(y_raw)

        # Sanity: replace NaN/Inf
        nan_count = np.isnan(X_vid).sum() + np.isinf(X_vid).sum()
        if nan_count > 0:
            print(f"  [!] {vid}: {nan_count} NaN/Inf in features, replacing with 0.")
            X_vid = np.nan_to_num(X_vid, nan=0.0, posinf=0.0, neginf=0.0)

        X_all.append(X_vid)
        y_all.append(y_norm)
        groups_all.append(np.full(len(X_vid), group_idx, dtype=np.int32))

        per_video_stats.append({
            "video_id":    vid,
            "group_idx":   group_idx,
            "n_windows":   len(X_vid),
            "label_min":   float(y_norm.min()),
            "label_max":   float(y_norm.max()),
            "label_mean":  float(y_norm.mean()),
            "raw_min":     float(y_raw.min()),
            "raw_max":     float(y_raw.max()),
        })

        print(f"  [+] {vid}: {len(X_vid)} windows | "
              f"label [{y_norm.min():.3f}, {y_norm.max():.3f}]")

    # ── Stack into flat dataset ───────────────────────────────────────────────
    X      = np.concatenate(X_all,      axis=0)
    y      = np.concatenate(y_all,      axis=0)
    groups = np.concatenate(groups_all, axis=0)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_npz  = os.path.join(SLIDING_DIR, "dataset.npz")
    out_info = os.path.join(SLIDING_DIR, "dataset_info.json")

    np.savez(out_npz, X=X, y=y, groups=groups)

    successful_ids = [s["video_id"] for s in per_video_stats]
    info = {
        "video_ids":      successful_ids,
        "n_videos":       len(successful_ids),
        "n_windows":      int(X.shape[0]),
        "feature_dim":    FEATURE_DIM,
        "feature_names":  FEATURE_NAMES,
        "intro_cutoff_s": INTRO_CUTOFF,
        "label_norm":     "per-video zscore clipped [-3,3] rescaled [0,1]",
        "per_video":      per_video_stats,
    }
    with open(out_info, "w") as f:
        json.dump(info, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  GREEN AI v2 — SLIDING DATASET COMPLETE")
    print(f"{'='*55}")
    print(f"  Videos:        {len(successful_ids)}/{len(video_ids)}")
    print(f"  Total windows: {X.shape[0]}")
    print(f"  Feature dim:   {X.shape[1]}")
    print(f"  X shape:       {X.shape}")
    print(f"  y shape:       {y.shape}")
    print(f"  groups shape:  {groups.shape}")
    print(f"  Unique groups: {len(np.unique(groups))}")
    print(f"  Label range:   [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Saved npz:     {out_npz}")
    if failed:
        print(f"  Failed:        {failed}")
    print(f"{'='*55}")
    print(f"\n  Upload sliding_labels/dataset.npz to Google Drive.")
    print(f"  Upload sliding_labels/dataset_info.json to Google Drive.\n")


if __name__ == "__main__":
    build_dataset()