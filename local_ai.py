import json
import sys
from faster_whisper import WhisperModel

if len(sys.argv) != 3 or sys.argv[1] != "transcribe":
    raise SystemExit("usage: local_ai.py transcribe AUDIO_FILE")

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, _ = model.transcribe(sys.argv[2], word_timestamps=True, vad_filter=True)
words = []
text = []
for segment in segments:
    text.append(segment.text.strip())
    for word in segment.words or []:
        words.append({"word": word.word.strip(), "start": word.start, "end": word.end})
print(json.dumps({"text": " ".join(text), "words": words}))
