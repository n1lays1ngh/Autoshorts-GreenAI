import yt_dlp
import json
import os


def fetch_youtube_data(youtube_url):
    """
    Extracts the heatmap AND the total video duration.
    """
    print(f"[*] Fetching crowd-sourced Heatmap data for: {youtube_url} ...")
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        heatmap = info.get('heatmap')
        duration = info.get('duration')  # Get raw video length in seconds

        if not heatmap:
            print("[-] No heatmap data available for this video.")
            return None, duration
        return heatmap, duration


def evaluate_pipeline(youtube_url, manifest_path, top_percentile=0.10):
    if not os.path.exists(manifest_path):
        print(f"[-] Error: Manifest not found at {manifest_path}.")
        return None

    with open(manifest_path, 'r') as f:
        ai_windows = json.load(f)

    # 1. Grab both heatmap and raw duration
    heatmap_data, raw_duration = fetch_youtube_data(youtube_url)
    if not heatmap_data:
        return None

    # 2. Calculate Pruning Percentage (Efficiency)
    total_gpu_seconds = sum(w['duration'] for w in ai_windows)
    pruned_pct = 0.0
    if raw_duration and raw_duration > 0:
        pruned_pct = max(0.0, 100.0 - ((total_gpu_seconds / raw_duration) * 100))

    # 3. Filter out the fake intro data completely (first 60 seconds)
    valid_body_segments = [s for s in heatmap_data if s['start_time'] > 60.0]

    if not valid_body_segments:
        print("[-] No heatmap data exists past the 60-second mark.")
        return None

    # 4. Extract all values and sort them to find the percentile threshold
    values = [s['value'] for s in valid_body_segments]
    values.sort(reverse=True)

    threshold_index = max(1, int(len(values) * top_percentile)) - 1
    dynamic_threshold = values[threshold_index]

    # 5. Isolate the "Viral Peaks"
    true_viral_moments = []
    for segment in valid_body_segments:
        if segment['value'] >= dynamic_threshold:
            peak_time = (segment['start_time'] + segment['end_time']) / 2
            true_viral_moments.append(peak_time)

    # 6. Grade the AI
    captured_peaks = 0
    missed_peaks = []
    for true_peak in true_viral_moments:
        is_captured = any(w['start'] <= true_peak <= w['end'] for w in ai_windows)
        if is_captured:
            captured_peaks += 1
        else:
            missed_peaks.append(round(true_peak, 2))

    total_true_peaks = len(true_viral_moments)
    capture_rate = (captured_peaks / max(total_true_peaks, 1)) * 100

    useful_ai_windows = 0
    for w in ai_windows:
        has_peak = any(w['start'] <= tp <= w['end'] for tp in true_viral_moments)
        if has_peak:
            useful_ai_windows += 1

    noise_rate = ((len(ai_windows) - useful_ai_windows) / max(len(ai_windows), 1)) * 100

    # 7. Print the Tuning Report
    print(f"\n" + "=" * 50)
    print(f"      SMART HYPERPARAMETER TUNING REPORT")
    print(f"=" * 50)
    print(f"Ground Truth (Top {int(top_percentile * 100)}% YouTube Peaks): {total_true_peaks}")
    print(f"AI Candidate Windows:                  {len(ai_windows)}")
    print(f"-" * 50)
    print(f"✅ True Positives Captured:            {captured_peaks}")
    print(f"❌ Viral Moments Missed:               {len(missed_peaks)}")
    print(f"🎯 Capture Rate (Recall):              {capture_rate:.1f}%")
    print(f"🗑️  AI 'Noise' Rate:                     {noise_rate:.1f}%")
    print(f"✂️  Efficiency (Pruned):                 {pruned_pct:.1f}%")
    print(f"=" * 50)

    # 8. Return the full dictionary (INCLUDING pruned_pct) back to batch_runner.py
    return {
        "true_peaks": total_true_peaks,
        "ai_windows": len(ai_windows),
        "captured": captured_peaks,
        "missed": len(missed_peaks),
        "recall_pct": round(capture_rate, 1),
        "noise_pct": round(noise_rate, 1),
        "pruned_pct": round(pruned_pct, 1)
    }


if __name__ == "__main__":
    # Test URL (Gaurav Kapoor Standup)
    test_url = "https://www.youtube.com/watch?v=MoBkkw66NWY"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = os.path.join(base_dir, "data", "candidate_windows.json")

    evaluate_pipeline(test_url, manifest, top_percentile=0.10)