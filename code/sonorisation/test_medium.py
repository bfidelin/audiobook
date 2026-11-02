"""Verifie si Stable Audio 3 Medium se charge sans Flash Attention."""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import time

import soundfile as sf
from stable_audio_3 import StableAudioModel

t0 = time.perf_counter()
model = StableAudioModel.from_pretrained("medium")
print(f"charge en {time.perf_counter() - t0:.0f} s", flush=True)

t1 = time.perf_counter()
audio = model.generate(
    prompt="a busy French bar at night, people talking, glasses clinking", duration=10.0)
arr = audio[0] if isinstance(audio, (list, tuple)) else audio
arr = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else arr
while arr.ndim > 2:
    arr = arr[0]
if arr.ndim == 2:
    arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
sf.write("/out/medium_bar.wav", arr, 44100)
elapsed = time.perf_counter() - t1
print(f"10 s generees en {elapsed:.1f} s -> {10 / elapsed:.2f}x", flush=True)
