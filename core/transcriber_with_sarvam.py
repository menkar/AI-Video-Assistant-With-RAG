import os
import re

import requests
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL          = os.getenv("WHISPER_MODEL", "base")
SARVAM_API_KEY         = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL           = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

# Cloud-mode flag — set RENDER=true in Render env vars.
# When True, torch/whisper are not installed; only Sarvam AI is available.
_CLOUD_MODE = os.getenv("RENDER", "").lower() in ("1", "true", "yes")

_model = None


# ── Whisper helpers (local only) ──────────────────────────────────────────────

def _chunk_duration_minutes(chunk_path: str) -> float:
    # torch/whisper imported lazily — only when Whisper path is actually used
    import whisper as _whisper
    from whisper.audio import SAMPLE_RATE
    samples = _whisper.load_audio(chunk_path)
    return len(samples) / SAMPLE_RATE / 60


def load_model():
    global _model
    if _model is None:
        # Lazy import — torch and whisper are not installed in cloud mode
        import torch
        import whisper as _whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{WHISPER_MODEL}' on {device}...")
        if device == "cpu":
            print(
                "Note: CPU transcription is slow. "
                "Use WHISPER_MODEL=tiny or base in .env for faster runs."
            )
        _model = _whisper.load_model(WHISPER_MODEL, device=device)
        print("Whisper model loaded successfully.")
    return _model


def transcribe_chunk_whisper(chunk_path: str, translate: bool = False) -> str:
    if _CLOUD_MODE:
        raise RuntimeError(
            "English / Whisper transcription is not available on the cloud "
            "deployment (torch is not installed). Please select "
            "'Hinglish — Sarvam AI' as the language."
        )
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


# ── Sarvam AI helpers (cloud API — works in both local and cloud mode) ────────

def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data  = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )
    if not response.ok:
        print(f"\n Sarvam returned {response.status_code}")
        print(f"Response body : {response.text} \n")
        response.raise_for_status()
    return response.json().get("transcript", "")


def _clean_sarvam_transcript(text: str) -> str:
    """Remove Sarvam language tags and normalize whitespace."""
    text = re.sub(r"<\|[^|]+\|>", " ", text)
    text = re.sub(r"\s+",          " ", text)
    return text.strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. Split each chunk into 25s
    pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio    = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text    = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece      = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            print(f" -> Sarvam piece {i + 1}/{total_pieces}...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return _clean_sarvam_transcript(full_text)


# ── Router ────────────────────────────────────────────────────────────────────

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper (local) or Sarvam AI (cloud API).
      english  → Whisper  (requires torch; not available in cloud mode)
      hinglish → Sarvam AI (cloud API; works everywhere)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription Complete.")
    return full_transcript.strip()
