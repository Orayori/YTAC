import json
import re
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
words = []
for event in data.get("events", []):
    base = event.get("tStartMs", 0) / 1000
    duration = event.get("dDurationMs", 0) / 1000
    segs = [s for s in event.get("segs", []) if s.get("utf8", "").strip()]
    for index, seg in enumerate(segs):
        start = base + seg.get("tOffsetMs", 0) / 1000
        next_start = base + (segs[index + 1].get("tOffsetMs", 0) / 1000 if index + 1 < len(segs) else duration)
        tokens = re.findall(r"\S+", seg["utf8"])
        for token_index, token in enumerate(tokens):
            token_start = start + (next_start - start) * token_index / len(tokens)
            token_end = start + (next_start - start) * (token_index + 1) / len(tokens)
            words.append({"word": token, "start": token_start, "end": max(token_start + 0.05, token_end)})

deduped = []
for word in words:
    if not deduped or word["word"] != deduped[-1]["word"] or word["start"] - deduped[-1]["start"] > 0.2:
        deduped.append(word)
print(json.dumps({"text": " ".join(w["word"] for w in deduped), "words": deduped}))
