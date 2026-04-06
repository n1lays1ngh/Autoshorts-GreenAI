import os
import yt_dlp


def fetch_audio_stream(youtube_url: str, output_directory: str, file_prefix: str = "raw_source") -> str:
    """
    Downloads only the best available audio stream from a YouTube video
    and converts it to a WAV format for DSP analysis.

    Args:
        youtube_url (str): The link to the source video.
        output_directory (str): Where to save the file (e.g., data/raw_audio/).
        file_prefix (str): The name of the file before the .wav extension.

    Returns:
        str: The absolute path to the downloaded .wav file, or None if it fails.
    """
    # 1. Ensure our target directory exists
    os.makedirs(output_directory, exist_ok=True)

    # 2. Define the exact output path structure
    output_template = os.path.join(output_directory, f'{file_prefix}.%(ext)s')
    final_wav_path = os.path.join(output_directory, f'{file_prefix}.wav')

    # 3. Configure yt-dlp to prioritize audio and force WAV conversion
    ydl_options = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',  # 192kbps is plenty for our 300-3000Hz analysis
        }],
        'quiet': False,  # Keep this False so we can see the download progress in terminal
        'no_warnings': True
    }

    print(f"[*] Initializing smart download for: {youtube_url}")

    try:
        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            ydl.download([youtube_url])

        print(f"[+] Success! Audio extracted and saved to: {final_wav_path}")
        return final_wav_path

    except Exception as e:
        print(f"[-] Error downloading audio: {e}")
        return None


# Simple block to let us test this specific file in isolation
if __name__ == "__main__":
    # A short, 30-second copyright-free sound effect video for rapid testing
    test_url = "https://www.youtube.com/watch?v=2PYpGGn4l7s"

    # Navigate up one level from 'src' to find the 'data/raw_audio' folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "raw_audio")

    fetch_audio_stream(test_url, target_dir, "system_test_audio")