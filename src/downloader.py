import os
import yt_dlp


def fetch_media(youtube_url: str, output_directory: str, file_prefix: str = "target_video") -> str:
    """
    Downloads 720p video and best audio, merging them into an MP4.
    Also ensures the audio is available for the Spectral Flux analysis.
    """
    # 1. Ensure target directory exists
    os.makedirs(output_directory, exist_ok=True)

    # 2. Define output paths
    # We save to data/raw_video/ now as it's our master source
    output_template = os.path.join(output_directory, f'{file_prefix}.%(ext)s')

    ydl_options = {
        # 1. Reject AV1/VP9 completely. FORCE H.264 (avc) and AAC.
        'format': 'bestvideo[vcodec^=avc][height<=720]+bestaudio[ext=m4a]/best[vcodec^=avc]/best',
        'merge_output_format': 'mp4',
        'outtmpl': output_template,

        # 2. Extract the WAV for librosa
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],

        'keepvideo': True,
        'quiet': False,
        'no_warnings': True
    }


    print(f"[*] Initializing Green AI Ingestion for: {youtube_url}")

    try:
        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            ydl.download([youtube_url])

        final_mp4 = os.path.join(output_directory, f"{file_prefix}.mp4")
        final_wav = os.path.join(output_directory, f"{file_prefix}.wav")

        print(f"[+] Success! Master MP4: {final_mp4}")
        print(f"[+] Success! Analysis WAV: {final_wav}")
        return final_mp4

    except Exception as e:
        print(f"[-] Error during media ingestion: {e}")
        return None


if __name__ == "__main__":
    # Using your provided URL
    test_url = "https://www.youtube.com/watch?v=PPjYWaqCffQ"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # We will store the master files in data/raw_video
    # The audio filter can point here to get the .wav
    target_dir = os.path.join(base_dir, "data", "raw_video")

    fetch_media(test_url, target_dir, "system_test_media")