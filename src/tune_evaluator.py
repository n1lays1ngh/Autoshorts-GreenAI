import yt_dlp
import json
import os


def fetch_youtube_heatmap(youtube_url):
    """
    Extracts the 'Most Replayed' heatmap data from YouTube without downloading the video.
    """
    print(f"[*] Fetching crowd-sourced Heatmap data for: {youtube_url} ...")
    ydl_opts = {
        'quiet': True,
        'skip_download': True,  # We only want the metadata
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        heatmap = info.get('heatmap')

        if not heatmap:
            print("[-] No heatmap data available for this video (might not have enough views).")
            return None
        return heatmap


def evaluate_pipeline(youtube_url, manifest_path, top_percentile=0.10):
    """
    Compares our AI-generated windows against YouTube's actual user-replay peaks.
    Uses 'Percentile Thresholding' to prevent one mega-viral outlier from ruining the curve.
    """
    if not os.path.exists(manifest_path):
        print(f"[-] Error: Manifest not found at {manifest_path}. Run audio filter first.")
        return

    with open(manifest_path, 'r') as f:
        ai_windows = json.load(f)

    heatmap_data = fetch_youtube_heatmap(youtube_url)
    if not heatmap_data:
        return

    # 1. Filter out the fake intro data completely (first 60 seconds)
    valid_body_segments = [s for s in heatmap_data if s['start_time'] > 60.0]

    if not valid_body_segments:
        print("[-] No heatmap data exists past the 60-second mark.")
        return

    # 2. Extract all values and sort them to find the percentile threshold
    values = [s['value'] for s in valid_body_segments]
    values.sort(reverse=True)

    # 3. Find the threshold for the Top 10% (or whatever top_percentile is set to)
    # This completely ignores extreme outliers ruining the curve!
    threshold_index = max(1, int(len(values) * top_percentile)) - 1
    dynamic_threshold = values[threshold_index]

    # 4. Isolate the "Viral Peaks" using the new statistical threshold
    true_viral_moments = []
    for segment in valid_body_segments:
        if segment['value'] >= dynamic_threshold:
            peak_time = (segment['start_time'] + segment['end_time']) / 2
            true_viral_moments.append(peak_time)

    # 5. Grade the AI
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
    print(f"=" * 50)

    if missed_peaks:
        print(f"[*] Missed Peak Timestamps for debugging: {missed_peaks[:5]}...")


if __name__ == "__main__":
    # Ensure this is the URL for your 1-hour video
    test_url = "https://www.youtube.com/watch?v=HxjDgR8itZM"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = os.path.join(base_dir, "data", "candidate_windows.json")

    # THE FIX: Call the function with top_percentile=0.10 instead of relative_threshold
    evaluate_pipeline(test_url, manifest, top_percentile=0.10)