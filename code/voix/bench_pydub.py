"""Mesure le chemin de fusion actuel d'Alexandria (pydub) sur N segments."""
import os
import resource
import sys
import time

from pydub import AudioSegment

root = os.environ.get("ALEX_ROOT", "/alex")
sys.path.insert(0, "/data")

import json

chunks = json.load(open(f"{root}/runtime-data/chunks.json"))
chunks = chunks[: int(sys.argv[1]) if len(sys.argv) > 1 else 400]

t0 = time.perf_counter()
combined = None
prev = None
for ch in chunks:
    seg = AudioSegment.from_file(os.path.join(root, "data/output", os.path.basename(ch["audio_path"])))
    if combined is None:
        combined = seg
    else:
        gap = AudioSegment.silent(duration=250 if ch["speaker"] == prev else 500)
        combined = combined + gap + seg
    prev = ch["speaker"]
t1 = time.perf_counter()
out = "/out/pydub_test.mp3"
combined.export(out, format="mp3")
t2 = time.perf_counter()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print(f"{len(chunks)} segments | fusion {t1-t0:.1f} s | export {t2-t1:.1f} s "
      f"| total {t2-t0:.1f} s | RAM pic {peak:.2f} Go")
