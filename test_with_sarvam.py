from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber_with_sarvam import transcribe_all

source = "https://youtu.be/6hgLvj8GVss"
language = 'hinglish' # 'english' -> Whisper, "hinglish" -> Sarvam

chunks = process_input(source)


transcript = transcribe_all(chunks, language=language)
print("\n" + "=" * 60)
print("📝 TRANSCRIPT")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)


