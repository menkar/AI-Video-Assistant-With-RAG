import os

import static_ffmpeg
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

static_ffmpeg.add_paths(weak=True)
_ffmpeg_exe, _ffprobe_exe = get_or_fetch_platform_executables_else_raise()
_ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
os.environ["PATH"] = os.pathsep.join([_ffmpeg_dir, os.environ.get("PATH", "")])

import yt_dlp
from pydub import AudioSegment

AudioSegment.converter = _ffmpeg_exe
AudioSegment.ffprobe = _ffprobe_exe

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _configure_ffmpeg() -> str:
    """Return ffmpeg directory for yt-dlp."""
    return _ffmpeg_dir


def _base_ydl_opts(output_path: str) -> dict:
    """Common yt-dlp options shared by all strategies."""
    return {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "ffmpeg_location": _configure_ffmpeg(),
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        "noplaylist": True,
        "quiet": True,
    }


def _build_ydl_opts(output_path: str, player_client: list | None = None) -> dict:
    """Build yt-dlp options, optionally pinning a player client."""
    opts = _base_ydl_opts(output_path)
    if player_client:
        opts["extractor_args"] = {"youtube": {"player_client": player_client}}
    return opts


# Strategies tried in order.
# Strategy None = no override (yt-dlp picks the default client automatically).
# Named clients are the officially supported values across yt-dlp >=2023.06.
_YDL_STRATEGIES: list[list | None] = [
    None,               # yt-dlp default — always valid regardless of version
    ["ios"],
    ["android"],
    ["web"],
    ["web_embedded"],
    ["tv_embedded"],
]

_ACTIONABLE_KEYWORDS = (
    "sign in", "bot", "confirm you", "cookies", "auth", "403",
    "no player clients",        # "No player clients have been requested"
    "player client",
    "private video",
)


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio, trying multiple strategies before giving up."""
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    last_error: Exception | None = None

    for clients in _YDL_STRATEGIES:
        try:
            ydl_opts = _build_ydl_opts(output_path, player_client=clients)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                requested = info.get("requested_downloads") or []
                if requested and requested[0].get("filepath"):
                    return requested[0]["filepath"]
                return os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
        except Exception as exc:
            last_error = exc
            print(f"yt-dlp strategy {clients!r} failed: {exc}")

    # All strategies exhausted — surface a clear, actionable message.
    err_lower = str(last_error).lower()
    if any(kw in err_lower for kw in _ACTIONABLE_KEYWORDS):
        raise RuntimeError(
            "YouTube could not be downloaded from this server.\n\n"
            "This usually happens because:\n"
            "  • The server IP is flagged by YouTube as a bot, OR\n"
            "  • The video requires sign-in / age verification.\n\n"
            "Quick fix: download the video/audio on your own computer and upload it "
            "using the 'Upload local file' option instead of pasting a YouTube URL."
        )
    raise last_error


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) # 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format = "wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
   

