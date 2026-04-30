import os
import json
import subprocess

# ── Cookie configuration ──────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
COOKIES_FILE  = os.path.normpath(os.path.join(_PROJECT_ROOT, "cookies.txt"))


def verify_cookies():
    """Test cookies before burning through all 32 URLs."""
    print("[*] Verifying cookies...")
    cmd = [
        "yt-dlp",
        "--cookies", COOKIES_FILE,
        "--remote-components", "ejs:github",
        "--dump-json",
        "--no-warnings",
        "https://www.youtube.com/watch?v=wQA68Oqr1qE",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[✗] Cookies are stale. Re-export cookies.txt from YouTube and retry.")
        return False
    print("[✓] Cookies verified.\n")
    return True


def fetch_info(url):
    cmd = [
        "yt-dlp",
        "--cookies", COOKIES_FILE,
        "--remote-components", "ejs:github",
        "--skip-download",
        "--dump-json",
        "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [✗] yt-dlp failed: {result.stderr.strip()}")
        return None
    return json.loads(result.stdout)


def load_index(index_path):
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    return {}


def save_index(index_path, index):
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=4)


def build_research_dataset(youtube_urls: list, base_dir: str = "data/research_dataset"):
    mp4_dir     = os.path.join(base_dir, "mp4")
    wav_dir     = os.path.join(base_dir, "wav")
    heatmap_dir = os.path.join(base_dir, "heatmaps")
    index_path  = os.path.join(base_dir, "index.json")

    for directory in [mp4_dir, wav_dir, heatmap_dir]:
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(COOKIES_FILE):
        print(f"[✗] cookies.txt not found at: {COOKIES_FILE}")
        return

    if not verify_cookies():      # ← exits immediately if cookies are stale
        return

    print(f"[*] Using cookies: {COOKIES_FILE}")
    print(f"[*] Starting Dataset Generation: {len(youtube_urls)} videos queued.\n")

    index = load_index(index_path)

    for i, url in enumerate(youtube_urls):
        print(f"[{i + 1}/{len(youtube_urls)}] Processing: {url}")

        info = fetch_info(url)
        if not info:
            print("  [-] Could not fetch metadata. Skipping.")
            index[url] = {"status": "failed", "reason": "metadata_fetch_failed"}
            save_index(index_path, index)
            continue

        video_id = info.get('id')
        title    = info.get('title', 'unknown')
        duration = info.get('duration', 0)

        if not video_id:
            print("  [-] Could not extract video ID. Skipping.")
            index[url] = {"status": "failed", "reason": "no_video_id"}
            save_index(index_path, index)
            continue

        mp4_path     = os.path.join(mp4_dir,     f"{video_id}.mp4")
        wav_path     = os.path.join(wav_dir,     f"{video_id}.wav")
        heatmap_path = os.path.join(heatmap_dir, f"{video_id}_heatmap.json")

        if (
            video_id in index
            and index[video_id].get("status") == "complete"
            and os.path.exists(mp4_path)
            and os.path.exists(wav_path)
            and os.path.exists(heatmap_path)
        ):
            print(f"  [✓] {video_id} already complete. Skipping.")
            continue

        heatmap_data = info.get('heatmap')
        if not heatmap_data:
            print(f"  [!] No heatmap for {video_id} — skipping (no ground truth).")
            index[video_id] = {
                "url": url, "title": title, "duration_secs": duration,
                "status": "skipped", "reason": "no_heatmap"
            }
            save_index(index_path, index)
            continue

        with open(heatmap_path, 'w') as f:
            json.dump(heatmap_data, f, indent=4)
        print(f"  [+] Heatmap saved: {video_id}_heatmap.json")

        output_template = mp4_path.replace(".mp4", ".%(ext)s")
        cmd_download = [
            "yt-dlp",
            "--cookies", COOKIES_FILE,
            "--remote-components", "ejs:github",
            "--format", (
                "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[height<=720]+bestaudio/"
                "bestvideo+bestaudio/best"
            ),
            "--merge-output-format", "mp4",
            "--output", output_template,
            "--no-warnings",
            url,
        ]

        print("  [*] Downloading MP4...")
        result = subprocess.run(cmd_download)
        if result.returncode != 0:
            print(f"  [-] Download failed. Skipping {video_id}.")
            index[video_id] = {
                "url": url, "title": title, "duration_secs": duration,
                "status": "failed", "reason": "download_failed"
            }
            save_index(index_path, index)
            continue

        if not os.path.exists(mp4_path):
            candidates = [
                f for f in os.listdir(mp4_dir)
                if f.startswith(video_id) and f.endswith((".mkv", ".webm"))
            ]
            if candidates:
                found = os.path.join(mp4_dir, candidates[0])
                print(f"  [*] Re-muxing {candidates[0]} → mp4...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", found, "-c", "copy", mp4_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                os.remove(found)
            else:
                print(f"  [-] No video file found for {video_id}.")
                index[video_id] = {
                    "url": url, "title": title, "duration_secs": duration,
                    "status": "failed", "reason": "no_output_file"
                }
                save_index(index_path, index)
                continue

        print(f"  [+] MP4 saved: {video_id}.mp4")

        print("  [*] Extracting WAV...")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", mp4_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    wav_path,
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            print(f"  [+] WAV saved: {video_id}.wav\n")
        except subprocess.CalledProcessError as e:
            print(f"  [-] WAV extraction failed: {e.stderr.decode()}")
            index[video_id] = {
                "url": url, "title": title, "duration_secs": duration,
                "status": "failed", "reason": "wav_extraction_failed"
            }
            save_index(index_path, index)
            continue

        index[video_id] = {
            "url": url, "title": title, "duration_secs": duration,
            "status": "complete",
            "mp4": mp4_path, "wav": wav_path, "heatmap": heatmap_path,
        }
        save_index(index_path, index)
        print(f"  [✓] {video_id} indexed as complete.\n")

    complete = sum(1 for v in index.values() if v.get("status") == "complete")
    skipped  = sum(1 for v in index.values() if v.get("status") == "skipped")
    failed   = sum(1 for v in index.values() if v.get("status") == "failed")

    print("=" * 50)
    print(f"  DATASET BUILD COMPLETE")
    print("=" * 50)
    print(f"  ✅ Complete : {complete}")
    print(f"  ⏭️  Skipped  : {skipped}  (no heatmap)")
    print(f"  ❌ Failed   : {failed}")
    print(f"  📄 Index    : {index_path}")
    print("=" * 50)


if __name__ == "__main__":
    target_dataset = [
        "https://www.youtube.com/watch?v=wQA68Oqr1qE",
        "https://www.youtube.com/watch?v=8PtsKRBgLrA",
        "https://www.youtube.com/watch?v=Tqsz6fjvhZM",
        "https://www.youtube.com/watch?v=z12bz7adLKI",
        "https://www.youtube.com/watch?v=EiL5bMvDkNA",
        "https://www.youtube.com/watch?v=VgxHwN6qZxo",
        "https://www.youtube.com/watch?v=ikr-zKaxSF0",
        "https://www.youtube.com/watch?v=pzPCJj1U8bE",
        "https://www.youtube.com/watch?v=GxLa6rdww8g",
        "https://www.youtube.com/watch?v=pjSxOnCkHIA",
        "https://www.youtube.com/watch?v=g1T8-WSYcdI",
        "https://www.youtube.com/watch?v=0guSWBSO8lo",
        "https://www.youtube.com/watch?v=_fWyWcZB7VA",
        "https://www.youtube.com/watch?v=0bGQ_I5NXXo",
        "https://www.youtube.com/watch?v=Ug-nsXBgCJk",
        "https://www.youtube.com/watch?v=4JB24wITVEU",
        "https://www.youtube.com/watch?v=5Q5QWdR42Is",
        "https://www.youtube.com/watch?v=pqnhlrlrfmw",
        "https://www.youtube.com/watch?v=IEfBBYmxtIo",
        "https://www.youtube.com/watch?v=IcAV5qiko8M",
        "https://www.youtube.com/watch?v=DZ6q5X7fAmU",
        "https://www.youtube.com/watch?v=t8HrZTLRCeU",
        "https://www.youtube.com/watch?v=MIoNEQeC7_s",
        "https://www.youtube.com/watch?v=qxXYC15dmpc",
        "https://www.youtube.com/watch?v=lbMs5zG4TbM",
        "https://www.youtube.com/watch?v=MoBkkw66NWY",
        "https://www.youtube.com/watch?v=ewKEe8pL8_w",
        "https://www.youtube.com/watch?v=wdGLOmdpq_k",
        "https://www.youtube.com/watch?v=4Z-KLkjICoY",
        "https://www.youtube.com/watch?v=5hHM3LdKcRY",
        "https://www.youtube.com/watch?v=yZMFBsYyJL0",
        "https://www.youtube.com/watch?v=igUiMu6sZm0",
    ]

    base_output = os.path.join(_PROJECT_ROOT, "data", "research_dataset")
    build_research_dataset(target_dataset, base_output)