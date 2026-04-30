import os
import subprocess

# ── Configuration ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_media(youtube_url: str, output_directory: str, file_prefix: str = "target_video") -> str:
    os.makedirs(output_directory, exist_ok=True)

    final_mp4        = os.path.join(output_directory, f"{file_prefix}.mp4")
    final_wav        = os.path.join(output_directory, f"{file_prefix}.wav")
    output_template  = os.path.join(output_directory, f"{file_prefix}.%(ext)s")

    format_string = (
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
        "bestvideo[height<=720]+bestaudio/"
        "bestvideo+bestaudio/"
        "best"
    )

    cmd_download = [
        "yt-dlp",
        "--cookies-from-browser", "chrome",   # ← replaces stale cookies.txt
        "--remote-components", "ejs:github",
        "--format", format_string,
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-warnings",
        youtube_url,
    ]

    print(f"[*] Initializing Green AI Ingestion for: {youtube_url}")
    print("[*] Step 1/2 — Downloading video...")

    try:
        subprocess.run(cmd_download, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[-] yt-dlp download failed (exit code {e.returncode})")
        return None

    # ── Confirm MP4 exists ────────────────────────────────────────────────────
    if not os.path.exists(final_mp4):
        candidates = [
            f for f in os.listdir(output_directory)
            if f.startswith(file_prefix) and f.endswith((".mkv", ".webm"))
        ]
        if candidates:
            found = os.path.join(output_directory, candidates[0])
            print(f"[*] Re-muxing {os.path.basename(found)} → mp4 (no re-encode)...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", found, "-c", "copy", final_mp4],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            os.remove(found)
        else:
            print(f"[-] No video file found in {output_directory}")
            return None

    print(f"[+] MP4 ready: {final_mp4}")

    # ── Step 2: Extract WAV ───────────────────────────────────────────────────
    print("[*] Step 2/2 — Extracting WAV for Spectral Flux analysis...")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", final_mp4,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                final_wav,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        print(f"[+] WAV ready: {final_wav}")
    except subprocess.CalledProcessError as e:
        print(f"[-] ffmpeg WAV extraction failed: {e.stderr.decode()}")
        return None

    return final_mp4


if __name__ == "__main__":
    test_url   = "https://www.youtube.com/watch?v=wQA68Oqr1qE"
    target_dir = os.path.join(_PROJECT_ROOT, "data", "raw_video")

    result = fetch_media(test_url, target_dir, "system_test_media")
    print(f"\n[✓] Pipeline complete: {result}" if result else "\n[✗] Pipeline failed.")