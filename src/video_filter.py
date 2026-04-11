import cv2
import json
import os
import numpy as np


def calculate_laplacian_variance(frame):
    """
    Computes Var(∇²I) to detect blurry or black frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def filter_visual_quality(video_path, manifest_path, threshold=250.0):
    """
    Final Gatekeeper: Prunes windows where the punchline moment is visually
    non-salient (dark/blurry).
    """
    if not os.path.exists(manifest_path):
        print(f"[-] Error: Manifest not found at {manifest_path}")
        return

    with open(manifest_path, 'r') as f:
        windows = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video file {video_path}")
        return

    # Metadata for metrics
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / fps

    valid_windows = []
    print(f"[*] Starting Visual Cascade on {len(windows)} candidates...")

    for i, w in enumerate(windows):
        peak_time = w['peak_time']

        # Jump to the peak frame (the punchline)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(peak_time * fps))
        ret, frame = cap.read()

        if not ret:
            continue

        score = calculate_laplacian_variance(frame)

        if score > threshold:
            w['visual_score'] = round(score, 2)
            valid_windows.append(w)
            print(f"    [+] Keep {i}: Score {score:.1f} (Clear)")
        else:
            print(f"    [-] Drop {i}: Score {score:.1f} (Blurry/Dark)")

    cap.release()

    # --- THE PRUNING REPORT ---
    original_duration = video_duration_sec
    # Calculate unique seconds to be processed by GPU (accounts for overlaps)
    # We create a set of all second-integers covered by valid windows
    gpu_seconds_set = set()
    for w in valid_windows:
        for sec in range(int(w['start']), int(w['end'])):
            gpu_seconds_set.add(sec)

    gpu_duration = len(gpu_seconds_set)
    pruning_ratio = (1 - (gpu_duration / original_duration)) * 100

    print(f"\n" + "=" * 45)
    print(f"      GREEN AI CASCADE: PRUNING REPORT")
    print(f"=" * 45)
    print(f"Raw Video Duration:   {original_duration / 60:.2f} mins")
    print(f"GPU Workload:         {gpu_duration / 60:.2f} mins")
    print(f"Efficiency Gain:      {pruning_ratio:.1f}% PRUNED")
    print(f"Status:               {len(valid_windows)} Windows Ready for A100")
    print(f"=" * 45)

    # Save the cleaned manifest
    with open(manifest_path, 'w') as f:
        json.dump(valid_windows, f, indent=4)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Path to the 720p video from downloader.py
    video_file = os.path.join(base_dir, "data", "raw_video", "system_test_media.mp4")
    manifest = os.path.join(base_dir, "data", "candidate_windows.json")

    filter_visual_quality(video_file, manifest)