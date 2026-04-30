"""
tune_evaluator.py — Green AI Cascade v1
=========================================
Evaluates candidate windows against YouTube heatmap ground truth.
Accepts windows as a list (no manifest file I/O) for in-memory batch running.
Also computes random and uniform baseline scores for statistical comparison.
"""

import json
import os
import numpy as np

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


# ── Baseline window generators ────────────────────────────────────────────────

def generate_random_windows(duration_secs, n_windows, window_lead=45.0,
                             window_tail=15.0, seed=42):
    """Random baseline: n_windows placed randomly across the video body."""
    rng     = np.random.default_rng(seed)
    windows = []
    for _ in range(n_windows):
        peak  = rng.uniform(60.0, max(61.0, duration_secs - window_tail))
        start = max(0.0, peak - window_lead)
        end   = min(duration_secs, peak + window_tail)
        windows.append({
            "peak_time": round(float(peak), 2),
            "start":     round(float(start), 2),
            "end":       round(float(end), 2),
            "duration":  round(float(end - start), 2),
        })
    return windows


def generate_uniform_windows(duration_secs, n_windows, window_lead=45.0,
                              window_tail=15.0):
    """Uniform baseline: evenly spaced windows across the video body."""
    if n_windows == 0:
        return []
    body_start = 60.0
    body_end   = duration_secs - window_tail
    if body_end <= body_start:
        return []

    step    = (body_end - body_start) / n_windows
    windows = []
    for i in range(n_windows):
        peak  = body_start + i * step + step / 2
        start = max(0.0, peak - window_lead)
        end   = min(duration_secs, peak + window_tail)
        windows.append({
            "peak_time": round(float(peak), 2),
            "start":     round(float(start), 2),
            "end":       round(float(end), 2),
            "duration":  round(float(end - start), 2),
        })
    return windows


# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate_pipeline(video_id, ai_windows, top_percentile=0.10):
    """
    Evaluates ai_windows against YouTube heatmap for video_id.

    Parameters
    ----------
    video_id      : YouTube video ID string
    ai_windows    : list of window dicts from audio+visual pipeline
    top_percentile: fraction of heatmap segments treated as "viral"

    Returns
    -------
    Dict of metrics for AI pipeline + random and uniform baselines.
    None if heatmap or index is missing.
    """
    heatmap_path = os.path.join(
        _PROJECT_ROOT, "data", "research_dataset",
        "heatmaps", f"{video_id}_heatmap.json"
    )
    if not os.path.exists(heatmap_path):
        print(f"[-] Heatmap not found: {heatmap_path}")
        return None

    with open(heatmap_path) as f:
        heatmap_data = json.load(f)

    index_path = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "index.json")
    with open(index_path) as f:
        index = json.load(f)

    raw_duration = index.get(video_id, {}).get("duration_secs", 0)
    if raw_duration == 0:
        print(f"[-] duration_secs missing for {video_id}")
        return None

    # ── Filter intro (Anomaly 1) ──────────────────────────────────────────────
    valid_body = [s for s in heatmap_data if s['start_time'] > 60.0]
    if not valid_body:
        print(f"[-] No heatmap data after 60s for {video_id}")
        return None

    # ── Dynamic percentile threshold (Anomaly 2) ──────────────────────────────
    values            = sorted([s['value'] for s in valid_body], reverse=True)
    threshold_index   = max(1, int(len(values) * top_percentile)) - 1
    dynamic_threshold = values[threshold_index]

    true_viral_moments = [
        (s['start_time'] + s['end_time']) / 2
        for s in valid_body
        if s['value'] >= dynamic_threshold
    ]

    # ── Shared scoring function ───────────────────────────────────────────────
    def score_windows(windows, true_peaks, duration):
        captured  = 0
        missed    = []
        for tp in true_peaks:
            if any(w['start'] <= tp <= w['end'] for w in windows):
                captured += 1
            else:
                missed.append(round(tp, 2))

        total     = len(true_peaks)
        recall    = (captured / max(total, 1)) * 100
        useful    = sum(
            1 for w in windows
            if any(w['start'] <= tp <= w['end'] for tp in true_peaks)
        )
        noise     = ((len(windows) - useful) / max(len(windows), 1)) * 100
        gpu_secs  = sum(w['duration'] for w in windows)
        pruned    = max(0.0, 100.0 - (gpu_secs / duration * 100))

        return {
            "captured":   captured,
            "missed":     len(missed),
            "missed_at":  missed,
            "recall_pct": round(recall,  1),
            "noise_pct":  round(noise,   1),
            "pruned_pct": round(pruned,  1),
            "n_windows":  len(windows),
        }

    # ── Infer window geometry from first AI window (for fair baselines) ───────
    window_lead = ai_windows[0].get("window_lead", 45.0) if ai_windows else 45.0
    window_tail = ai_windows[0].get("window_tail", 15.0) if ai_windows else 15.0

    ai_scores   = score_windows(ai_windows, true_viral_moments, raw_duration)

    # Baselines use same n_windows and same window geometry as AI
    n_windows       = max(len(ai_windows), 1)
    rand_windows    = generate_random_windows(raw_duration, n_windows,
                                              window_lead, window_tail)
    uniform_windows = generate_uniform_windows(raw_duration, n_windows,
                                               window_lead, window_tail)
    rand_scores     = score_windows(rand_windows,    true_viral_moments, raw_duration)
    uniform_scores  = score_windows(uniform_windows, true_viral_moments, raw_duration)

    print(f"  [EVAL] {video_id} | "
          f"AI R={ai_scores['recall_pct']}% "
          f"N={ai_scores['noise_pct']}% "
          f"E={ai_scores['pruned_pct']}% | "
          f"rand R={rand_scores['recall_pct']}% | "
          f"uniform R={uniform_scores['recall_pct']}%")

    return {
        "video_id":            video_id,
        "true_peaks":          len(true_viral_moments),
        "duration_secs":       raw_duration,
        "window_lead":         window_lead,
        "window_tail":         window_tail,
        # AI
        "ai_windows":          ai_scores["n_windows"],
        "captured":            ai_scores["captured"],
        "missed":              ai_scores["missed"],
        "missed_at":           ai_scores["missed_at"],
        "recall_pct":          ai_scores["recall_pct"],
        "noise_pct":           ai_scores["noise_pct"],
        "pruned_pct":          ai_scores["pruned_pct"],
        # Baselines
        "rand_recall_pct":     rand_scores["recall_pct"],
        "rand_noise_pct":      rand_scores["noise_pct"],
        "rand_pruned_pct":     rand_scores["pruned_pct"],
        "uniform_recall_pct":  uniform_scores["recall_pct"],
        "uniform_noise_pct":   uniform_scores["noise_pct"],
        "uniform_pruned_pct":  uniform_scores["pruned_pct"],
    }


if __name__ == "__main__":
    video_id = "wQA68Oqr1qE"
    manifest = os.path.join(_PROJECT_ROOT, "data", "candidate_windows",
                            f"{video_id}_candidates.json")
    with open(manifest) as f:
        windows = json.load(f)
    result = evaluate_pipeline(video_id, windows, top_percentile=0.10)
    if result:
        print(f"\nRecall: {result['recall_pct']}%  "
              f"Noise: {result['noise_pct']}%  "
              f"Pruned: {result['pruned_pct']}%")