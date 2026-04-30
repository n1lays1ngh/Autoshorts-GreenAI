"""
batch_runner.py — Green AI Cascade v1
=======================================
Runs the full baseline comparison experiment across all 32 videos.

WHAT THIS EXPERIMENT DOES:
---------------------------
This is the random baseline validation experiment for the Green AI Cascade.
The core question it answers is:

    "Does the cascade's acoustic guidance actually add value, or would
     randomly placing the same number of windows achieve similar recall?"

For each video, we run three systems at IDENTICAL pruning levels:

  1. Green AI Cascade  — audio-guided windows (spectral flux peaks)
                         passed through a visual quality filter
  2. Random Baseline   — 1000 Monte Carlo trials of randomly placed
                         windows, same count as the cascade produced
  3. Uniform Baseline  — windows spaced evenly across the video body,
                         same count as the cascade produced

All three are evaluated against YouTube "Most Replayed" heatmap peaks
as ground truth (a behavioral signal reflecting real audience engagement).

The recall GAP between the cascade and the random baseline is the
primary empirical justification for the cascade's contribution.

FIXES APPLIED vs ORIGINAL:
---------------------------
  Fix 1: Pruning is now computed correctly for all methods by merging
          overlapping windows before measuring coverage. The original code
          inherited AI pruning for random/uniform, which was misleading
          because AI windows are NMS-merged but random windows are not.

  Fix 2: Window boundary clamping added in generate_random_windows_local()
          to prevent windows extending before t=0 or after video end.
          Without clamping, boundary windows are effectively shorter,
          giving random an unfair disadvantage.

  Fix 3: Skip logic now distinguishes between "missing data" skips
          (WAV/MP4 absent — infrastructure issue) and "cascade failure"
          skips (visual filter drops all windows — meaningful result).
          Cascade failures are recorded as R=0% rather than silently
          excluded, preventing recall inflation.

  Fix 4: Config comment added to make clear this runs the Track I
          baseline config (face ON, 60s window), NOT the optimised system.
          A second config block is provided (commented out) for the
          optimised system run.

Usage:
    python src/batch_runner.py

    To run optimised config, swap the config block at the top.
"""

import os
import json
import numpy as np
from tqdm import tqdm

from audio_filter   import calculate_spectral_flux
from video_filter   import filter_visual_quality
from tune_evaluator import evaluate_pipeline, generate_random_windows, generate_uniform_windows

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
# NOTE: This config matches Track I baseline (face ON, 60s window, α=0.6).
# These are the numbers reported in Track I Table 5.
# To reproduce Table 11 (optimised system), use the block below instead.

# ALPHA       = 0.60       # spectral flux threshold
# WINDOW_LEAD = 45.0       # seconds before peak
# WINDOW_TAIL = 15.0       # seconds after peak
# TOP_PCT     = 0.10       # top 10% heatmap = viral
# N_RANDOM    = 1000       # Monte Carlo trials for random baseline
# VIS_THRESH  = 250.0      # visual filter sharpness threshold
# USE_FACE    = True       # face detection toggle (Track III: set to False)

# ── Optimised config (Track III recommended system) ───────────────────────────
# Uncomment this block to run the optimised system comparison.
ALPHA       = 0.50
WINDOW_LEAD = 30.0
WINDOW_TAIL = 10.0
TOP_PCT     = 0.15
N_RANDOM    = 1000
VIS_THRESH  = 150.0
USE_FACE    = False


def load_index():
    path = os.path.join(_PROJECT_ROOT, "data", "research_dataset", "index.json")
    with open(path) as f:
        return json.load(f)


# ── FIX 1: Correct pruning computation ───────────────────────────────────────
def compute_pruning(duration_secs, windows):
    """
    Compute pruning efficiency correctly by merging overlapping windows
    before measuring coverage.

    This is necessary because:
    - AI windows are NMS-merged upstream, so their actual coverage is
      already de-duplicated.
    - Random/uniform windows are NOT pre-merged, so naive n_windows *
      window_size overcounts coverage when windows overlap.

    Applying the same merge logic to all three methods ensures pruning
    is a fair, apples-to-apples comparison.

    Returns: pruning efficiency as a percentage (0–100).
    """
    if not windows or duration_secs <= 0:
        return 100.0

    sorted_wins = sorted(windows, key=lambda w: w['start'])
    merged = []
    for w in sorted_wins:
        if merged and w['start'] <= merged[-1]['end']:
            # Overlapping: extend the current merged block
            merged[-1]['end'] = max(merged[-1]['end'], w['end'])
        else:
            merged.append({'start': w['start'], 'end': w['end']})

    covered = sum(w['end'] - w['start'] for w in merged)
    return max(0.0, (1.0 - covered / duration_secs) * 100.0)


# ── FIX 2: Boundary-clamped random window generation ─────────────────────────
def generate_random_windows_local(duration_secs, n_windows,
                                   window_lead, window_tail,
                                   seed=0):
    """
    Generate n_windows random windows within [0, duration_secs].

    Windows are centred on a randomly sampled peak timestamp and clamped
    to video boundaries. Without clamping, windows near t=0 or t=end
    would be effectively shorter than interior windows, giving the random
    baseline an unfair size disadvantage at boundaries.

    We sample peak centres uniformly from [window_lead, duration - window_tail]
    so that a full-size window always fits within the video body.
    If the video is too short to fit even one window, we place the window
    at the midpoint.
    """
    rng = np.random.default_rng(seed)

    # Safe zone for peak centres: ensures the full window fits in [0, duration]
    min_centre = window_lead
    max_centre = duration_secs - window_tail

    if min_centre >= max_centre:
        # Video shorter than one window — place at midpoint, return one window
        mid = duration_secs / 2.0
        return [{'start': max(0.0, mid - window_lead),
                 'end':   min(duration_secs, mid + window_tail)}]

    centres = rng.uniform(min_centre, max_centre, size=n_windows)
    return [
        {
            'start': max(0.0, c - window_lead),
            'end':   min(duration_secs, c + window_tail)
        }
        for c in centres
    ]


def run_random_baseline_trials(duration_secs, n_windows, true_peaks,
                                window_lead, window_tail, n_trials=1000):
    """
    Monte Carlo random baseline.

    Randomly places n_windows windows across the video body n_trials times
    using boundary-clamped window generation (Fix 2).

    Returns:
        mean recall (float), std recall (float), mean pruning (float)
        across all trials.

    Note: pruning is computed per-trial with the correct merge logic (Fix 1)
    and averaged, because window overlap varies by seed.
    """
    recalls  = []
    prunings = []

    for seed in range(n_trials):
        windows = generate_random_windows_local(
            duration_secs, n_windows, window_lead, window_tail, seed=seed
        )
        captured = sum(
            1 for tp in true_peaks
            if any(w['start'] <= tp <= w['end'] for w in windows)
        )
        recalls.append(captured / max(len(true_peaks), 1) * 100.0)
        prunings.append(compute_pruning(duration_secs, windows))

    return (float(np.mean(recalls)),
            float(np.std(recalls)),
            float(np.mean(prunings)))


def load_heatmap_peaks(video_id, top_percentile=0.10):
    """Load heatmap and return viral peak timestamps."""
    heatmap_path = os.path.join(
        _PROJECT_ROOT, "data", "research_dataset",
        "heatmaps", f"{video_id}_heatmap.json"
    )
    if not os.path.exists(heatmap_path):
        return None

    with open(heatmap_path) as f:
        heatmap_data = json.load(f)

    # Exclude first 60s (navigational bias / intro spike anomaly)
    valid_body = [s for s in heatmap_data if s['start_time'] > 60.0]
    if not valid_body:
        return None

    # Dynamic percentile threshold (handles viral outlier anomaly)
    values          = sorted([s['value'] for s in valid_body], reverse=True)
    threshold_index = max(1, int(len(values) * top_percentile)) - 1
    dyn_threshold   = values[threshold_index]

    return [
        (s['start_time'] + s['end_time']) / 2.0
        for s in valid_body
        if s['value'] >= dyn_threshold
    ]


def run_experiment():
    index       = load_index()
    results_dir = os.path.join(_PROJECT_ROOT, "data", "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    video_ids = [vid for vid, info in index.items()
                 if info.get("status") == "complete"]

    print(f"[*] Running baseline experiment on {len(video_ids)} videos")
    print(f"    α={ALPHA}  lead={WINDOW_LEAD}s  tail={WINDOW_TAIL}s  "
          f"top_pct={TOP_PCT}  random_trials={N_RANDOM}  "
          f"face={'ON' if USE_FACE else 'OFF'}\n")

    all_results      = []
    # ── FIX 3: Separate skip types ────────────────────────────────────────────
    skipped_missing  = []   # WAV/MP4/heatmap absent — infrastructure, not result
    skipped_no_audio = []   # Cascade found no audio peaks — content issue
    cascade_failures = []   # Visual filter dropped ALL windows — recorded as R=0%

    for video_id in tqdm(video_ids, desc="Videos"):
        info      = index[video_id]
        wav_path  = info.get("wav")
        mp4_path  = info.get("mp4")
        duration  = info.get("duration_secs", 0)

        # ── Infrastructure checks (true skips) ────────────────────────────────
        if not wav_path or not os.path.exists(wav_path):
            print(f"  [!] WAV missing for {video_id} — skip (infrastructure)")
            skipped_missing.append(video_id)
            continue
        if not mp4_path or not os.path.exists(mp4_path):
            print(f"  [!] MP4 missing for {video_id} — skip (infrastructure)")
            skipped_missing.append(video_id)
            continue

        true_peaks = load_heatmap_peaks(video_id, TOP_PCT)
        if not true_peaks:
            print(f"  [!] Heatmap missing/empty for {video_id} — skip (infrastructure)")
            skipped_missing.append(video_id)
            continue

        # ── Step 1: Audio filter ──────────────────────────────────────────────
        audio_windows = calculate_spectral_flux(
            wav_path, threshold=ALPHA,
            window_lead=WINDOW_LEAD, window_tail=WINDOW_TAIL
        )

        if not audio_windows:
            print(f"  [!] No audio peaks for {video_id} — content issue, skip")
            skipped_no_audio.append(video_id)
            continue

        # ── Step 2: Visual filter ─────────────────────────────────────────────
        ai_windows = filter_visual_quality(
            mp4_path, audio_windows,
            threshold=VIS_THRESH,
            use_face_detection=USE_FACE
        )

        # ── FIX 3: Cascade failure → R=0%, not silent skip ───────────────────
        if not ai_windows:
            print(f"  [!] Visual filter dropped all windows for {video_id} "
                  f"— recording as R=0% (cascade failure)")
            cascade_failures.append(video_id)

            # Record as zero-recall result so the aggregate is honest
            n_windows = len(audio_windows)   # use audio count for baseline match
            rand_mean, rand_std, rand_pruning = run_random_baseline_trials(
                duration, n_windows, true_peaks,
                WINDOW_LEAD, WINDOW_TAIL, N_RANDOM
            )

            # Uniform baseline
            uniform_windows = generate_uniform_windows(
                duration, n_windows, WINDOW_LEAD, WINDOW_TAIL
            )
            uniform_captured = sum(
                1 for tp in true_peaks
                if any(w['start'] <= tp <= w['end'] for w in uniform_windows)
            )
            uniform_recall  = uniform_captured / max(len(true_peaks), 1) * 100.0
            uniform_pruning = compute_pruning(duration, uniform_windows)

            result = {
                "video_id":            video_id,
                "duration_secs":       duration,
                "true_peaks":          len(true_peaks),
                "n_ai_windows":        0,
                "cascade_failure":     True,

                "ai_recall":           0.0,
                "ai_noise":            0.0,
                "ai_pruning":          100.0,

                "rand_recall_mean":    round(rand_mean, 2),
                "rand_recall_std":     round(rand_std,  2),
                "rand_pruning":        round(rand_pruning, 2),

                "uniform_recall":      round(uniform_recall,  2),
                "uniform_pruning":     round(uniform_pruning, 2),

                "recall_gap_vs_random":  round(0.0 - rand_mean, 2),
                "recall_gap_vs_uniform": round(0.0 - uniform_recall, 2),
            }
            all_results.append(result)
            continue

        # ── Step 3: Evaluate AI system ────────────────────────────────────────
        eval_result = evaluate_pipeline(video_id, ai_windows, TOP_PCT)
        if not eval_result:
            print(f"  [!] evaluate_pipeline returned None for {video_id} — skip")
            skipped_missing.append(video_id)
            continue

        n_windows = len(ai_windows)

        # ── FIX 1: Correct pruning for AI ────────────────────────────────────
        # evaluate_pipeline may use a different pruning formula; recompute
        # with our merged-window method to ensure consistency across methods.
        ai_pruning_corrected = compute_pruning(duration, ai_windows)

        # ── Step 4: Monte Carlo random baseline ───────────────────────────────
        rand_mean, rand_std, rand_pruning = run_random_baseline_trials(
            duration, n_windows, true_peaks,
            WINDOW_LEAD, WINDOW_TAIL, N_RANDOM
        )

        # ── Step 5: Uniform baseline ──────────────────────────────────────────
        uniform_windows = generate_uniform_windows(
            duration, n_windows, WINDOW_LEAD, WINDOW_TAIL
        )
        uniform_captured = sum(
            1 for tp in true_peaks
            if any(w['start'] <= tp <= w['end'] for w in uniform_windows)
        )
        uniform_recall  = uniform_captured / max(len(true_peaks), 1) * 100.0
        # FIX 1: compute uniform pruning with merge logic, not inherited from AI
        uniform_pruning = compute_pruning(duration, uniform_windows)

        # ── Step 6: Collect result ────────────────────────────────────────────
        result = {
            "video_id":            video_id,
            "duration_secs":       duration,
            "true_peaks":          len(true_peaks),
            "n_ai_windows":        n_windows,
            "cascade_failure":     False,

            # AI system
            "ai_recall":           eval_result['recall_pct'],
            "ai_noise":            eval_result['noise_pct'],
            "ai_pruning":          ai_pruning_corrected,   # FIX 1: recomputed

            # Random baseline — pruning now independently computed (FIX 1)
            "rand_recall_mean":    round(rand_mean, 2),
            "rand_recall_std":     round(rand_std,  2),
            "rand_pruning":        round(rand_pruning, 2),

            # Uniform baseline — pruning independently computed (FIX 1)
            "uniform_recall":      round(uniform_recall,  2),
            "uniform_pruning":     round(uniform_pruning, 2),

            # Gap: the core contribution metric
            "recall_gap_vs_random":  round(eval_result['recall_pct'] - rand_mean, 2),
            "recall_gap_vs_uniform": round(eval_result['recall_pct'] - uniform_recall, 2),
        }

        all_results.append(result)

        print(f"\n  [{video_id}]")
        print(f"    AI:      R={result['ai_recall']}%  "
              f"N={result['ai_noise']}%  E={result['ai_pruning']:.1f}%")
        print(f"    Random:  R={result['rand_recall_mean']:.1f}% "
              f"±{result['rand_recall_std']:.1f}%  "
              f"E={result['rand_pruning']:.1f}%  (n={N_RANDOM} trials)")
        print(f"    Uniform: R={result['uniform_recall']:.1f}%  "
              f"E={result['uniform_pruning']:.1f}%")
        print(f"    Gap vs random:  {result['recall_gap_vs_random']:+.1f}pp")
        print(f"    Gap vs uniform: {result['recall_gap_vs_uniform']:+.1f}pp")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    if not all_results:
        print("\n[-] No results to summarise.")
        return

    # Separate cascade failures from normal results for honest reporting
    normal_results  = [r for r in all_results if not r['cascade_failure']]
    failure_results = [r for r in all_results if r['cascade_failure']]

    ai_recalls      = [r['ai_recall']        for r in all_results]
    rand_recalls    = [r['rand_recall_mean']  for r in all_results]
    uniform_recalls = [r['uniform_recall']    for r in all_results]
    ai_prunings     = [r['ai_pruning']        for r in all_results]
    rand_prunings   = [r['rand_pruning']      for r in all_results]
    uniform_prunings= [r['uniform_pruning']   for r in all_results]
    gaps_rand       = [r['recall_gap_vs_random']  for r in all_results]
    gaps_uniform    = [r['recall_gap_vs_uniform'] for r in all_results]

    summary = {
        "n_videos":              len(all_results),
        "n_cascade_failures":    len(failure_results),  # FIX 3: tracked separately
        "n_skipped_missing":     len(skipped_missing),
        "n_skipped_no_audio":    len(skipped_no_audio),
        "skipped_missing":       skipped_missing,
        "skipped_no_audio":      skipped_no_audio,
        "cascade_failure_ids":   [r['video_id'] for r in failure_results],
        "config": {
            "alpha":        ALPHA,
            "window_lead":  WINDOW_LEAD,
            "window_tail":  WINDOW_TAIL,
            "top_pct":      TOP_PCT,
            "n_random":     N_RANDOM,
            "vis_thresh":   VIS_THRESH,
            "use_face":     USE_FACE,
            # FIX 4: explicit config label for paper reference
            "config_label": "Track_I_baseline" if USE_FACE else "Track_III_optimised",
        },
        "ai": {
            "mean_recall":   round(np.mean(ai_recalls),   1),
            "std_recall":    round(np.std(ai_recalls),    1),
            "mean_pruning":  round(np.mean(ai_prunings),  1),
        },
        "random_baseline": {
            "mean_recall":   round(np.mean(rand_recalls),    1),
            "std_recall":    round(np.std(rand_recalls),     1),
            # FIX 1: pruning independently computed, not copied from AI
            "mean_pruning":  round(np.mean(rand_prunings),   1),
            # NOTE: noise is not reported for random/uniform baselines because
            # noise requires ground-truth window labelling, which only the
            # cascade evaluation provides. This is intentional, not an omission.
        },
        "uniform_baseline": {
            "mean_recall":   round(np.mean(uniform_recalls),   1),
            "std_recall":    round(np.std(uniform_recalls),    1),
            "mean_pruning":  round(np.mean(uniform_prunings),  1),
        },
        "gaps": {
            "mean_gap_vs_random":    round(np.mean(gaps_rand),   1),
            "mean_gap_vs_uniform":   round(np.mean(gaps_uniform), 1),
            "pct_videos_ai_beats_random":
                round(sum(1 for g in gaps_rand if g > 0) / len(gaps_rand) * 100, 1),
            "pct_videos_ai_beats_uniform":
                round(sum(1 for g in gaps_uniform if g > 0) / len(gaps_uniform) * 100, 1),
        },
        "per_video": all_results,
    }

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON — FINAL TABLE")
    print(f"  Config: {summary['config']['config_label']}")
    print("=" * 70)
    print(f"{'Method':<38} {'Recall':>8} {'Pruning':>9} {'Noise':>7}")
    print("-" * 70)
    print(f"{'Naive (no filtering)':<38} "
          f"{'100.0%':>8}  "
          f"{'0.0%':>8}    --")
    print(f"{'Random window selection (1000 trials)':<38} "
          f"{np.mean(rand_recalls):>7.1f}%  "
          f"{np.mean(rand_prunings):>8.1f}%    --")
    print(f"{'Uniform window selection':<38} "
          f"{np.mean(uniform_recalls):>7.1f}%  "
          f"{np.mean(uniform_prunings):>8.1f}%    --")
    print(f"{'Green AI Cascade (α={})'.format(ALPHA):<38} "
          f"{np.mean(ai_recalls):>7.1f}%  "
          f"{np.mean(ai_prunings):>8.1f}%  "
          f"{np.mean([r['ai_noise'] for r in all_results]):>5.1f}%")
    print("=" * 70)
    print(f"\nCascade beats random on "
          f"{summary['gaps']['pct_videos_ai_beats_random']}% of videos")
    print(f"Cascade beats uniform on "
          f"{summary['gaps']['pct_videos_ai_beats_uniform']}% of videos")
    print(f"Mean recall gap vs random:  "
          f"{summary['gaps']['mean_gap_vs_random']:+.1f}pp")
    print(f"Mean recall gap vs uniform: "
          f"{summary['gaps']['mean_gap_vs_uniform']:+.1f}pp")

    if cascade_failures:
        print(f"\nCascade failures (visual filter dropped all windows): "
              f"{len(cascade_failures)} video(s)")
        print(f"  These are recorded as R=0% in the aggregate (not excluded).")
        for vid in cascade_failures:
            print(f"  - {vid}")

    if skipped_missing:
        print(f"\nInfrastructure skips (WAV/MP4/heatmap missing): "
              f"{len(skipped_missing)} video(s)")
        print(f"  These are excluded from all metrics.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(results_dir, "baseline_experiment.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Results saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_experiment()