import os
import time

import torch
import whisper
from whisper.audio import SAMPLE_RATE

# WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

_model = None


def _chunk_duration_minutes(chunk_path: str) -> float:
    samples = whisper.load_audio(chunk_path)
    return len(samples) / SAMPLE_RATE / 60


def load_model():
    global _model

    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{WHISPER_MODEL}' on {device}...")
        if device == "cpu":
            print(
                "Note: CPU transcription is slow. "
                "Use WHISPER_MODEL=tiny or base in .env for faster runs."
            )
        _model = whisper.load_model(WHISPER_MODEL, device=device)
        print("Whisper model loaded successfully.")

    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"

    # verbose=False enables Whisper's internal tqdm progress bar.
    result = model.transcribe(
        chunk_path,
        task=task,
        verbose=False,
        fp16=torch.cuda.is_available(),
    )

    return result["text"]


def transcribe_all(chunks: list, translate: bool = False):
    full_transcript = ""
    total = len(chunks)

    print(f"Transcribing {total} chunk(s)...")

    for i, chunk in enumerate(chunks, start=1):
        duration_min = _chunk_duration_minutes(chunk)
        print(
            f"\nChunk {i}/{total} (~{duration_min:.1f} min): "
            f"{os.path.basename(chunk)}"
        )
        started = time.perf_counter()
        text = transcribe_chunk(chunk, translate=translate)
        elapsed_min = (time.perf_counter() - started) / 60
        print(f"Chunk {i}/{total} finished in {elapsed_min:.1f} min.")

        full_transcript += text + " "

    print("\nTranscription completed.")

    return full_transcript.strip()
