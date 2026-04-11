import os
import json
import csv
import subprocess
from tune_evaluator import evaluate_pipeline
from audio_filter import calculate_spectral_flux

def download_audio_only(url, output_path):
    """Uses yt-dlp to grab just the WAV file for quick testing"""
    print(f"[*] Downloading audio for {url}...")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_batch_experiment(video_urls, thresholds_to_test):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_audio_path = os.path.join(base_dir, "data", "temp_batch_audio.wav")
    manifest_path = os.path.join(base_dir, "data", "temp_candidate_windows.json")
    report_path = os.path.join(base_dir, "batch_report.csv")

    # Set up the CSV Writer with the new Pruned (%) column
    with open(report_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Video URL", "Threshold", "Ground Truth Peaks", "AI Windows", "Captured", "Missed", "Recall (%)",
             "Noise (%)", "Pruned (%)"]
        )

        for url in video_urls:
            print(f"\n========================================")
            print(f"🎬 Processing: {url}")

            # 1. Download the audio (overwrites the temp file each loop)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            download_audio_only(url, temp_audio_path)

            for thresh in thresholds_to_test:
                print(f"   ➤ Testing Threshold: {thresh}")

                # 2. Run your Audio Filter
                windows = calculate_spectral_flux(temp_audio_path, threshold=thresh)

                # 3. Save the JSON for the evaluator
                with open(manifest_path, "w") as f:
                    json.dump(windows, f, indent=4)

                # 4. Grade it!
                results = evaluate_pipeline(url, manifest_path, top_percentile=0.10)

                # 5. Log it to the CSV (Now including pruned_pct)
                if results:
                    writer.writerow([
                        url,
                        thresh,
                        results["true_peaks"],
                        results["ai_windows"],
                        results["captured"],
                        results["missed"],
                        results["recall_pct"],
                        results["noise_pct"],
                        results["pruned_pct"]
                    ])

    print(f"\n[+] BATCH COMPLETE! Report saved to {report_path}")

if __name__ == "__main__":
    # Your 11 diverse stand-up URLs
    test_videos = [
        "https://www.youtube.com/watch?v=MoBkkw66NWY",  # 30 min
        "https://www.youtube.com/watch?v=ewKEe8pL8_w",  # 10 min
        "https://www.youtube.com/watch?v=wdGLOmdpq_k",  # 10 min
        "https://www.youtube.com/watch?v=wQA68Oqr1qE",  # 10 min
        "https://www.youtube.com/watch?v=EiL5bMvDkNA",  # 10 min
        "https://www.youtube.com/watch?v=pjSxOnCkHIA",  # 10 min
        "https://www.youtube.com/watch?v=IEfBBYmxtIo",  # 30 min
        "https://www.youtube.com/watch?v=4Z-KLkjICoY",  # 45 min
        "https://www.youtube.com/watch?v=5hHM3LdKcRY",  # 30 min (Added missing comma here!)
        "https://www.youtube.com/watch?v=yZMFBsYyJL0",  # 30 min
        "https://www.youtube.com/watch?v=igUiMu6sZm0"   # 15 min
    ]

    # We will test the 3 most likely candidates
    thresholds = [0.5, 0.6, 0.7]

    run_batch_experiment(test_videos, thresholds)