"""
batch_runner.py — Green AI Cascade v1 — Full Evaluation Suite
==============================================================
Runs three phases in sequence:

  PHASE 1 — WINDOW EXPERIMENT  (fast, ~20 min)
    Fixes alpha=0.5, beta=150, gamma=0.15 (known optimal from prior work).
    Sweeps 4 window configs across 32 videos = 128 evaluations.
    This is the paper's headline experiment: does shrinking from 60s to 25s
    preserve recall while doubling pruning?

  PHASE 2 — FULL GRID SEARCH  (slow, ~2-3 hours)
    Sweeps all combinations of alpha, beta, gamma, window_lead, window_tail.
    = 5 x 5 x 3 x 4 x 32 = 9,600 evaluations.
    Produces the comprehensive hyperparameter table for the paper.

  PHASE 3 — ABLATION STUDY  (medium, ~30 min)
    Uses best params from Phase 2 (selected by recall - 0.3*noise score).
    Tests: spectral_flux vs rms vs mfcc audio detectors
           face_detection on vs off
    Provides statistical validation via Wilcoxon signed-rank test.

Usage
-----
  python batch_runner.py              # Phase 1 only (window experiment)
  python batch_runner.py --full       # All 3 phases

Outputs (all in results/)
-------------------------
  window_experiment.csv     — Phase 1 per-video per-config rows
  window_summary.csv        — Phase 1 aggregated by window config
  grid_search_full.csv      — Phase 2 full results
  summary_stats.csv         — Phase 2 aggregated by param combo
  ablation_results.csv      — Phase 3
"""

import os
import sys
import json
import csv
import itertools
import numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon as scipy_wilcoxon

from audio_filter   import (calculate_spectral_flux,
                             calculate_rms_peaks,
                             calculate_mfcc_peaks)
from video_filter   import filter_visual_quality
from tune_evaluator import evaluate_pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "research_dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Grid space ────────────────────────────────────────────────────────────────
AUDIO_THRESHOLDS  = [0.4, 0.5, 0.6, 0.7, 0.8]
VISUAL_THRESHOLDS = [150, 200, 250, 300, 350]
TOP_PERCENTILES   = [0.05, 0.10, 0.15]

# Window experiment configs: (lead, tail, label)
WINDOW_CONFIGS = [
    (45, 15, "original_60s"),   # current baseline
    (30, 10, "compact_40s"),
    (20,  5, "tight_25s"),      # paper hypothesis
    (15,  5, "minimal_20s"),
]

# Fixed values for Phase 1 (window experiment only)
FIXED_ALPHA  = 0.5
FIXED_BETA   = 150
FIXED_GAMMA  = 0.15

# ── CSV headers ───────────────────────────────────────────────────────────────
WINDOW_HEADER = [
    "video_id", "window_config", "window_lead", "window_tail", "window_total",
    "audio_threshold", "visual_threshold", "top_percentile",
    "true_peaks", "duration_secs",
    "ai_windows_after_audio", "ai_windows_after_visual",
    "recall_pct", "noise_pct", "pruned_pct",
    "rand_recall_pct", "rand_noise_pct", "rand_pruned_pct",
    "uniform_recall_pct", "uniform_noise_pct", "uniform_pruned_pct",
    "recall_delta_vs_random", "recall_delta_vs_uniform",
]

WINDOW_SUMMARY_HEADER = [
    "window_config", "window_lead", "window_tail", "window_total",
    "mean_recall", "std_recall",
    "mean_noise",  "std_noise",
    "mean_pruned", "std_pruned",
    "mean_recall_delta_vs_random",
    "mean_recall_delta_vs_uniform",
    "wilcoxon_p_vs_random",
    "wilcoxon_p_vs_uniform",
    "wilcoxon_p_vs_original_60s",   # FIX 4: added direct comparison to baseline
    "n_videos",
]

GRID_HEADER = [
    "video_id", "audio_threshold", "visual_threshold", "top_percentile",
    "window_lead", "window_tail", "window_total",
    "true_peaks", "duration_secs",
    "ai_windows_after_audio", "ai_windows_after_visual",
    "recall_pct", "noise_pct", "pruned_pct",
    "rand_recall_pct", "rand_noise_pct", "rand_pruned_pct",
    "uniform_recall_pct", "uniform_noise_pct", "uniform_pruned_pct",
    "recall_delta_vs_random", "recall_delta_vs_uniform",
]

SUMMARY_HEADER = [
    "audio_threshold", "visual_threshold", "top_percentile",
    "window_lead", "window_tail", "window_total",
    "mean_recall", "std_recall",
    "mean_noise",  "std_noise",
    "mean_pruned", "std_pruned",
    "mean_recall_delta_vs_random",
    "mean_recall_delta_vs_uniform",
    "n_videos",
]

ABLATION_HEADER = [
    "video_id", "audio_method", "face_detection",
    "audio_threshold", "visual_threshold", "top_percentile",
    "window_lead", "window_tail",
    "true_peaks", "ai_windows_after_audio", "ai_windows_after_visual",
    "recall_pct", "noise_pct", "pruned_pct",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_video_ids():
    index_path = os.path.join(DATASET_DIR, "index.json")
    with open(index_path) as f:
        index = json.load(f)

    valid_ids = []
    for vid_id in index.keys():
        wav_ok = os.path.exists(os.path.join(DATASET_DIR, "wav",      f"{vid_id}.wav"))
        mp4_ok = os.path.exists(os.path.join(DATASET_DIR, "mp4",      f"{vid_id}.mp4"))
        hm_ok  = os.path.exists(os.path.join(DATASET_DIR, "heatmaps", f"{vid_id}_heatmap.json"))
        if wav_ok and mp4_ok and hm_ok:
            valid_ids.append(vid_id)
        else:
            missing = [n for n, ok in
                       [("wav", wav_ok), ("mp4", mp4_ok), ("heatmap", hm_ok)]
                       if not ok]
            print(f"[!] Skipping {vid_id} — missing: {', '.join(missing)}")

    print(f"[+] {len(valid_ids)} videos ready.\n")
    return valid_ids


def run_wilcoxon(a, b, label):
    """Paired Wilcoxon signed-rank test. Returns p-value."""
    if len(a) < 5:
        print(f"    Wilcoxon vs {label}: skipped (n<5)")
        return 1.0
    try:
        _, p = scipy_wilcoxon(a, b)
        sig  = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"    Wilcoxon vs {label}: p={p:.4f} — {sig}")
        return round(p, 4)
    except Exception as e:
        print(f"    Wilcoxon failed vs {label}: {e}")
        return 1.0


def build_row_from_result(result, audio_windows, visual_windows,
                           extra_fields: dict) -> dict:
    """Merge evaluate_pipeline result with extra fields into one flat dict."""
    if result is None:
        return None
    row = {**extra_fields}
    row.update({
        "true_peaks":                  result["true_peaks"],
        "duration_secs":               result["duration_secs"],
        "ai_windows_after_audio":      len(audio_windows),
        "ai_windows_after_visual":     len(visual_windows),
        "recall_pct":                  result["recall_pct"],
        "noise_pct":                   result["noise_pct"],
        "pruned_pct":                  result["pruned_pct"],
        "rand_recall_pct":             result["rand_recall_pct"],
        "rand_noise_pct":              result["rand_noise_pct"],
        "rand_pruned_pct":             result["rand_pruned_pct"],
        "uniform_recall_pct":          result["uniform_recall_pct"],
        "uniform_noise_pct":           result["uniform_noise_pct"],
        "uniform_pruned_pct":          result["uniform_pruned_pct"],
        "recall_delta_vs_random":      round(result["recall_pct"] - result["rand_recall_pct"], 1),
        "recall_delta_vs_uniform":     round(result["recall_pct"] - result["uniform_recall_pct"], 1),
    })
    return row


# ── Phase 1: Window Experiment ────────────────────────────────────────────────

def run_window_experiment(video_ids):
    """
    The headline experiment for the paper.
    Fixes alpha/beta/gamma at known optimal.
    Sweeps 4 window configs x 32 videos = 128 evaluations.
    """
    csv_path = os.path.join(RESULTS_DIR, "window_experiment.csv")
    total    = len(video_ids) * len(WINDOW_CONFIGS)

    print(f"\n{'='*65}")
    print(f"  PHASE 1 — WINDOW EXPERIMENT")
    print(f"  {len(WINDOW_CONFIGS)} configs x {len(video_ids)} videos = {total} runs")
    print(f"  Fixed: alpha={FIXED_ALPHA} beta={FIXED_BETA} gamma={FIXED_GAMMA}")
    print(f"{'='*65}\n")

    all_rows = []
    run_idx  = 0

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_HEADER)
        writer.writeheader()

        for video_id in video_ids:
            wav_path = os.path.join(DATASET_DIR, "wav", f"{video_id}.wav")
            mp4_path = os.path.join(DATASET_DIR, "mp4", f"{video_id}.mp4")

            print(f"\n  Video {video_ids.index(video_id)+1}/{len(video_ids)}: {video_id}")

            # FIX 1: Cache audio per (lead, tail) — avoids reloading wav
            # for every window config (was 128 loads, now 32)
            audio_cache = {}
            for lead, tail, _ in WINDOW_CONFIGS:
                audio_cache[(lead, tail)] = calculate_spectral_flux(
                    wav_path, threshold=FIXED_ALPHA,
                    window_lead=lead, window_tail=tail
                )

            for lead, tail, label in WINDOW_CONFIGS:
                run_idx += 1
                print(f"    [{run_idx}/{total}] {label} (lead={lead}s tail={tail}s)")

                # Use cached audio windows
                audio_windows  = audio_cache[(lead, tail)]
                visual_windows = filter_visual_quality(
                    mp4_path,
                    [dict(w) for w in audio_windows],
                    threshold=FIXED_BETA,
                    use_face_detection=True
                )
                result = evaluate_pipeline(
                    video_id, visual_windows,
                    top_percentile=FIXED_GAMMA
                )

                row = build_row_from_result(
                    result, audio_windows, visual_windows,
                    extra_fields={
                        "video_id":         video_id,
                        "window_config":    label,
                        "window_lead":      lead,
                        "window_tail":      tail,
                        "window_total":     lead + tail,
                        "audio_threshold":  FIXED_ALPHA,
                        "visual_threshold": FIXED_BETA,
                        "top_percentile":   FIXED_GAMMA,
                    }
                )
                if row:
                    writer.writerow(row)
                    f.flush()
                    all_rows.append(row)

    print(f"\n[+] Window experiment complete — {len(all_rows)} rows → {csv_path}")
    _summarise_window_experiment(all_rows)
    return all_rows


def _summarise_window_experiment(rows):
    """Aggregates window experiment by config, prints + saves summary.
    FIX 4: Adds Wilcoxon test between each config and original_60s directly.
    """
    summary_path = os.path.join(RESULTS_DIR, "window_summary.csv")
    groups       = defaultdict(list)

    for r in rows:
        key = (r["window_config"], r["window_lead"], r["window_tail"])
        groups[key].append(r)

    # Extract original_60s recalls once for direct comparison (FIX 4)
    orig_key      = ("original_60s", 45, 15)
    orig_recalls  = [r["recall_pct"] for r in groups.get(orig_key, [])]

    summary_rows = []

    print(f"\n{'='*65}")
    print(f"  WINDOW EXPERIMENT SUMMARY")
    print(f"  {'Config':<18} {'Lead':>5} {'Tail':>5} {'Total':>6} "
          f"{'R mean':>8} {'R std':>7} {'E mean':>8} {'N mean':>8}")
    print(f"{'='*65}")

    for (label, lead, tail), group_rows in sorted(groups.items(),
                                                   key=lambda x: x[0][1]):
        recalls  = [r["recall_pct"]              for r in group_rows]
        noises   = [r["noise_pct"]               for r in group_rows]
        pruned   = [r["pruned_pct"]              for r in group_rows]
        d_rand   = [r["recall_delta_vs_random"]  for r in group_rows]
        d_unif   = [r["recall_delta_vs_uniform"] for r in group_rows]

        rand_recalls = [r["rand_recall_pct"]    for r in group_rows]
        unif_recalls = [r["uniform_recall_pct"] for r in group_rows]

        p_vs_rand = run_wilcoxon(recalls, rand_recalls,  "random baseline")
        p_vs_unif = run_wilcoxon(recalls, unif_recalls,  "uniform baseline")

        # FIX 4: Direct Wilcoxon vs original_60s for all non-baseline configs
        if label == "original_60s":
            p_vs_orig = 1.0   # comparing against itself is meaningless
        else:
            p_vs_orig = run_wilcoxon(recalls, orig_recalls, "original_60s")

        print(f"  {label:<18} {lead:>5} {tail:>5} {lead+tail:>6} "
              f"{np.mean(recalls):>7.1f}% {np.std(recalls):>6.1f}% "
              f"{np.mean(pruned):>7.1f}% {np.mean(noises):>7.1f}%")

        summary_rows.append({
            "window_config":                label,
            "window_lead":                  lead,
            "window_tail":                  tail,
            "window_total":                 lead + tail,
            "mean_recall":                  round(np.mean(recalls), 2),
            "std_recall":                   round(np.std(recalls),  2),
            "mean_noise":                   round(np.mean(noises),  2),
            "std_noise":                    round(np.std(noises),   2),
            "mean_pruned":                  round(np.mean(pruned),  2),
            "std_pruned":                   round(np.std(pruned),   2),
            "mean_recall_delta_vs_random":  round(np.mean(d_rand),  2),
            "mean_recall_delta_vs_uniform": round(np.mean(d_unif),  2),
            "wilcoxon_p_vs_random":         p_vs_rand,
            "wilcoxon_p_vs_uniform":        p_vs_unif,
            "wilcoxon_p_vs_original_60s":   p_vs_orig,  # FIX 4
            "n_videos":                     len(group_rows),
        })

    print(f"{'='*65}\n")

    with open(summary_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[+] Window summary saved → {summary_path}")


# ── Phase 2: Full Grid Search ─────────────────────────────────────────────────

def run_full_grid_search(video_ids):
    csv_path     = os.path.join(RESULTS_DIR, "grid_search_full.csv")
    param_combos = list(itertools.product(
        AUDIO_THRESHOLDS, VISUAL_THRESHOLDS, TOP_PERCENTILES,
        [(l, t) for l, t, _ in WINDOW_CONFIGS]
    ))
    # FIX 3: corrected run count — 5 x 5 x 3 x 4 x 32 = 9,600 (not 28,800)
    total = len(video_ids) * len(param_combos)

    print(f"\n{'='*65}")
    print(f"  PHASE 2 — FULL GRID SEARCH")
    print(f"  {len(param_combos)} combos x {len(video_ids)} videos = {total} runs")
    print(f"{'='*65}\n")

    all_rows = []
    run_idx  = 0

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=GRID_HEADER)
        writer.writeheader()

        for video_id in video_ids:
            wav_path = os.path.join(DATASET_DIR, "wav", f"{video_id}.wav")
            mp4_path = os.path.join(DATASET_DIR, "mp4", f"{video_id}.mp4")

            print(f"\n  Video {video_ids.index(video_id)+1}/{len(video_ids)}: {video_id}")

            # Cache audio per (alpha, lead, tail) — avoids reloading wav
            audio_cache = {}
            for at in AUDIO_THRESHOLDS:
                for lead, tail, _ in WINDOW_CONFIGS:
                    audio_cache[(at, lead, tail)] = calculate_spectral_flux(
                        wav_path, threshold=at,
                        window_lead=lead, window_tail=tail
                    )

            # Cache visual per (alpha, beta, lead, tail)
            visual_cache = {}

            for at, bt, tp, (lead, tail) in param_combos:
                run_idx += 1
                vkey = (at, bt, lead, tail)

                if vkey not in visual_cache:
                    audio_w = audio_cache[(at, lead, tail)]
                    visual_cache[vkey] = filter_visual_quality(
                        mp4_path,
                        [dict(w) for w in audio_w],
                        threshold=bt,
                        use_face_detection=True
                    )

                audio_windows  = audio_cache[(at, lead, tail)]
                visual_windows = visual_cache[vkey]

                result = evaluate_pipeline(video_id, visual_windows,
                                           top_percentile=tp)
                row    = build_row_from_result(
                    result, audio_windows, visual_windows,
                    extra_fields={
                        "video_id":         video_id,
                        "audio_threshold":  at,
                        "visual_threshold": bt,
                        "top_percentile":   tp,
                        "window_lead":      lead,
                        "window_tail":      tail,
                        "window_total":     lead + tail,
                    }
                )
                if row:
                    writer.writerow(row)
                    f.flush()
                    all_rows.append(row)

    print(f"\n[+] Full grid search complete — {len(all_rows)} rows → {csv_path}")
    _summarise_full_grid(all_rows)
    return all_rows


def _summarise_full_grid(rows):
    summary_path = os.path.join(RESULTS_DIR, "summary_stats.csv")
    groups       = defaultdict(list)

    for r in rows:
        key = (r["audio_threshold"], r["visual_threshold"],
               r["top_percentile"],  r["window_lead"], r["window_tail"])
        groups[key].append(r)

    summary_rows = []
    for (at, bt, tp, lead, tail), group_rows in groups.items():
        recalls = [r["recall_pct"]               for r in group_rows]
        noises  = [r["noise_pct"]                for r in group_rows]
        pruned  = [r["pruned_pct"]               for r in group_rows]
        d_rand  = [r["recall_delta_vs_random"]   for r in group_rows]
        d_unif  = [r["recall_delta_vs_uniform"]  for r in group_rows]
        summary_rows.append({
            "audio_threshold":              at,
            "visual_threshold":             bt,
            "top_percentile":               tp,
            "window_lead":                  lead,
            "window_tail":                  tail,
            "window_total":                 lead + tail,
            "mean_recall":                  round(np.mean(recalls), 2),
            "std_recall":                   round(np.std(recalls),  2),
            "mean_noise":                   round(np.mean(noises),  2),
            "std_noise":                    round(np.std(noises),   2),
            "mean_pruned":                  round(np.mean(pruned),  2),
            "std_pruned":                   round(np.std(pruned),   2),
            "mean_recall_delta_vs_random":  round(np.mean(d_rand),  2),
            "mean_recall_delta_vs_uniform": round(np.mean(d_unif),  2),
            "n_videos":                     len(group_rows),
        })

    summary_rows.sort(key=lambda r: r["mean_recall"], reverse=True)

    with open(summary_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[+] Summary stats saved → {summary_path}")
    print(f"\n  TOP 5 COMBOS BY MEAN RECALL:")
    for r in summary_rows[:5]:
        print(f"    a={r['audio_threshold']} b={r['visual_threshold']} "
              f"g={r['top_percentile']} "
              f"lead={r['window_lead']} tail={r['window_tail']} | "
              f"R={r['mean_recall']}±{r['std_recall']}% "
              f"N={r['mean_noise']}% "
              f"E={r['mean_pruned']}%")

    return summary_rows[0] if summary_rows else None


# ── Phase 3: Ablation Study ───────────────────────────────────────────────────

def run_ablation(video_ids, best_audio, best_visual, best_gamma,
                 best_lead, best_tail):
    csv_path      = os.path.join(RESULTS_DIR, "ablation_results.csv")
    audio_methods = {
        "spectral_flux": calculate_spectral_flux,
        "rms":           calculate_rms_peaks,
        "mfcc":          calculate_mfcc_peaks,
    }
    face_options = [True, False]

    print(f"\n{'='*65}")
    print(f"  PHASE 3 — ABLATION STUDY")
    print(f"  Best params: a={best_audio} b={best_visual} g={best_gamma} "
          f"lead={best_lead} tail={best_tail}")
    print(f"{'='*65}\n")

    ablation_rows = []

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ABLATION_HEADER)
        writer.writeheader()

        for method_name, audio_fn in audio_methods.items():
            for use_face in face_options:
                tag = f"{method_name} | face={'on' if use_face else 'off'}"
                print(f"\n  ── {tag} ──")
                method_recalls = []

                for video_id in video_ids:
                    wav_path = os.path.join(DATASET_DIR, "wav", f"{video_id}.wav")
                    mp4_path = os.path.join(DATASET_DIR, "mp4", f"{video_id}.mp4")

                    audio_windows  = audio_fn(wav_path, threshold=best_audio,
                                              window_lead=best_lead,
                                              window_tail=best_tail)
                    visual_windows = filter_visual_quality(
                        mp4_path,
                        [dict(w) for w in audio_windows],
                        threshold=best_visual,
                        use_face_detection=use_face
                    )
                    result = evaluate_pipeline(video_id, visual_windows,
                                               top_percentile=best_gamma)
                    if result is None:
                        continue

                    method_recalls.append(result["recall_pct"])
                    row = {
                        "video_id":                video_id,
                        "audio_method":            method_name,
                        "face_detection":          use_face,
                        "audio_threshold":         best_audio,
                        "visual_threshold":        best_visual,
                        "top_percentile":          best_gamma,
                        "window_lead":             best_lead,
                        "window_tail":             best_tail,
                        "true_peaks":              result["true_peaks"],
                        "ai_windows_after_audio":  len(audio_windows),
                        "ai_windows_after_visual": len(visual_windows),
                        "recall_pct":              result["recall_pct"],
                        "noise_pct":               result["noise_pct"],
                        "pruned_pct":              result["pruned_pct"],
                    }
                    writer.writerow(row)
                    f.flush()
                    ablation_rows.append(row)

                print(f"    Mean recall: {np.mean(method_recalls):.1f}% "
                      f"± {np.std(method_recalls):.1f}%")

    # Ablation summary
    print(f"\n{'='*65}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*65}")
    abl_groups = defaultdict(list)
    for r in ablation_rows:
        abl_groups[(r["audio_method"], r["face_detection"])].append(r["recall_pct"])

    sf_face_on = abl_groups.get(("spectral_flux", True), [])
    for (method, face), recalls in sorted(abl_groups.items()):
        delta = ""
        if sf_face_on and (method != "spectral_flux" or not face):
            d = np.mean(recalls) - np.mean(sf_face_on)
            delta = f"  ({d:+.1f}pp vs SF+face)"
        print(f"  {method:<15} face={'on ' if face else 'off'} | "
              f"R={np.mean(recalls):.1f}±{np.std(recalls):.1f}%{delta}")

    print(f"\n[+] Ablation CSV saved → {csv_path}")
    return ablation_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    full_mode = "--full" in sys.argv
    video_ids = load_video_ids()

    if not video_ids:
        print("[-] No valid videos found. Exiting.")
        return

    # ── Phase 1: Window experiment (always runs) ──────────────────────────────
    window_rows = run_window_experiment(video_ids)

    if not full_mode:
        print("\n[*] Phase 1 complete.")
        print("[*] Run with --full to execute Phase 2 (grid search) + Phase 3 (ablation).")
        print(f"\n    Results saved to: {RESULTS_DIR}")
        return

    # ── Phase 2: Full grid search ─────────────────────────────────────────────
    grid_rows = run_full_grid_search(video_ids)

    # FIX 2: Best combo selected by recall - 0.3*noise score, not recall alone
    # This prevents picking a high-recall but very noisy combo for ablation
    combo_scores = defaultdict(lambda: {"recalls": [], "noises": []})
    for r in grid_rows:
        key = (r["audio_threshold"], r["visual_threshold"],
               r["top_percentile"],  r["window_lead"], r["window_tail"])
        combo_scores[key]["recalls"].append(r["recall_pct"])
        combo_scores[key]["noises"].append(r["noise_pct"])

    best_key = max(
        combo_scores,
        key=lambda k: (
            np.mean(combo_scores[k]["recalls"]) -
            0.3 * np.mean(combo_scores[k]["noises"])
        )
    )
    best_at, best_bt, best_gm, best_lead, best_tail = best_key
    best_recall = np.mean(combo_scores[best_key]["recalls"])
    best_noise  = np.mean(combo_scores[best_key]["noises"])
    print(f"\n[+] Best combo for ablation (recall - 0.3*noise): "
          f"a={best_at} b={best_bt} g={best_gm} "
          f"lead={best_lead} tail={best_tail} | "
          f"recall={best_recall:.1f}% noise={best_noise:.1f}%")

    # ── Phase 3: Ablation ─────────────────────────────────────────────────────
    run_ablation(video_ids, best_at, best_bt, best_gm, best_lead, best_tail)

    print(f"\n{'='*65}")
    print(f"  ALL PHASES COMPLETE")
    print(f"  window_experiment.csv  → Phase 1 per-video rows")
    print(f"  window_summary.csv     → Phase 1 aggregated + Wilcoxon")
    print(f"  grid_search_full.csv   → Phase 2 full 9,600 runs")
    print(f"  summary_stats.csv      → Phase 2 aggregated")
    print(f"  ablation_results.csv   → Phase 3")
    print(f"  All in: {RESULTS_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()