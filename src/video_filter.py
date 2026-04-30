"""
video_filter.py — Green AI Cascade v1
======================================
Visual quality gate using Laplacian sharpness + optional face detection.
Accepts window list directly (no manifest file I/O) so batch_runner
can pass windows in-memory without hitting disk.
"""

import cv2
import json
import os
import numpy as np

# Load face detector once at module level
_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade      = cv2.CascadeClassifier(_FACE_CASCADE_PATH)


def calculate_laplacian_variance(frame):
    """
    Grid-based sharpness: splits frame into 3x3 blocks,
    returns max Laplacian variance across all blocks.
    """
    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w        = gray.shape
    block_h     = h // 3
    block_w     = w // 3
    max_score   = 0.0

    for row in range(3):
        for col in range(3):
            block = gray[row * block_h:(row + 1) * block_h,
                         col * block_w:(col + 1) * block_w]
            score = cv2.Laplacian(block, cv2.CV_64F).var()
            if score > max_score:
                max_score = score
    return max_score


def calculate_face_score(frame):
    """
    Returns a face confidence multiplier:
      1.0 — face detected
      0.5 — no face but frame is bright (performer may be off-centre)
      0.0 — dark / no face (likely a cut or b-roll)
    """
    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    faces           = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        return 1.0
    elif mean_brightness > 40:
        return 0.5
    else:
        return 0.0


def score_frame(frame, use_face_detection=True):
    """Combined visual score: sharpness x face_multiplier."""
    sharpness = calculate_laplacian_variance(frame)
    if use_face_detection:
        face_mult = calculate_face_score(frame)
        return sharpness * face_mult, round(sharpness, 2), round(face_mult, 2)
    return sharpness, round(sharpness, 2), 1.0


def filter_visual_quality(video_path, windows, threshold=250.0,
                           use_face_detection=True):
    """
    Parameters
    ----------
    video_path         : path to .mp4 file
    windows            : list of window dicts from audio_filter
    threshold          : minimum combined visual score to keep a window
    use_face_detection : toggle for ablation study

    Returns
    -------
    Filtered list of windows with visual_score, sharpness, face_score added.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Could not open video: {video_path}")
        return []

    fps            = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    valid_windows = []
    face_tag      = 'on' if use_face_detection else 'off'
    print(f"[*] Visual {len(windows)} windows | "
          f"threshold={threshold} | face={face_tag}")

    for i, w in enumerate(windows):
        peak_time = w['peak_time']
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(peak_time * fps))
        ret, frame = cap.read()

        if not ret:
            print(f"    [!] Cannot read frame at {peak_time}s — skipping")
            continue

        combined, sharpness, face_mult = score_frame(frame, use_face_detection)

        if combined > threshold:
            w = dict(w)   # don't mutate caller's list
            w['visual_score'] = round(combined,   2)
            w['sharpness']    = sharpness
            w['face_score']   = face_mult
            valid_windows.append(w)
            print(f"    [+] Keep {i}: combined={combined:.1f} "
                  f"sharp={sharpness} face={face_mult}")
        else:
            print(f"    [-] Drop {i}: combined={combined:.1f} "
                  f"sharp={sharpness} face={face_mult}")

    cap.release()
    return valid_windows


if __name__ == "__main__":
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_id   = "wQA68Oqr1qE"
    video_path = os.path.join(base_dir, "data", "research_dataset",
                              "mp4", f"{video_id}.mp4")
    manifest   = os.path.join(base_dir, "data", "candidate_windows",
                              f"{video_id}_candidates.json")

    with open(manifest) as f:
        windows = json.load(f)

    valid = filter_visual_quality(video_path, windows)
    print(f"\n[+] {len(valid)} windows passed visual filter.")