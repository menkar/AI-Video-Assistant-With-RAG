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


def _build_ydl_opts(output_path: str) -> dict:
    """Build yt-dlp options resilient to YouTube 403 / SABR blocking."""
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
        "js_runtimes": {"node": {}},
        "remote_components": {"ejs:github"},
        "extractor_args": {"youtube": {"player_client": ["default", "-android_vr"]}},
        "noplaylist": True,
        "quiet": True,
    }


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = _build_ydl_opts(output_path)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        requested = info.get("requested_downloads") or []
        if requested and requested[0].get("filepath"):
            filename = requested[0]["filepath"]
        else:
            filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    return filename


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
   

