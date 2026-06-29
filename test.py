from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://youtu.be/6hgLvj8GVss"

chunks = process_input(source)

print(transcribe_all(chunks))
