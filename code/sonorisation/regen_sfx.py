"""Regenere un effet avec plusieurs graines et plusieurs formulations, puis
selectionne sur criteres objectifs (pas a l'oreille, que je n'ai pas).

Telephone : on veut une cadence lente (0,4-3 s) et pas de raie unique dominante.
Bar : on veut de la modulation de parole et un spectre large, pas un sifflement.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

OUT = Path("/out")
RATE = 44_100

VARIANTS = {
    "telephone": {
        "duration": 10.0,
        "prompts": [
            "a classic landline telephone ringing in a quiet room, repeated double ring cadence, close perspective",
            "an old fixed telephone ring, two short bursts then silence, then again, interior of an apartment",
        ],
    },
    "bar": {
        "duration": 6.0,
        "prompts": [
            "bustling French cafe interior, indistinct conversation and murmur, cutlery on plates, warm crowded room",
            "people talking in a busy bar, layered voices, background murmur, glasses set down on a wooden counter",
        ],
    },
}


def features(x: np.ndarray, sr: int) -> dict:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    energy = spec ** 2
    centroid = float((freqs * energy).sum() / (energy.sum() + 1e-12))
    flatness = float(np.exp(np.mean(np.log(spec + 1e-10))) / (np.mean(spec) + 1e-10))
    raie = float(energy.max() / (energy.sum() + 1e-12))  # domination d'une raie

    hop = int(0.02 * sr)
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2))
                    for i in range(0, len(x) - hop, hop)])
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = 20, min(len(ac) - 1, 150)  # 0,4 s a 3 s
    lag = int(np.argmax(ac[lo:hi]) + lo) if hi > lo else 0
    period = float(ac[lag]) if hi > lo else 0.0

    es = np.abs(np.fft.rfft(env - env.mean()))
    ef = np.fft.rfftfreq(len(env), hop / sr)
    band = (ef >= 3) & (ef <= 8)
    modulation = float(es[band].sum() / (es.sum() + 1e-12))
    return {"centroid": centroid, "flatness": flatness, "raie": raie,
            "period": period, "lag_ms": lag * 20, "modulation": modulation}


def score(kind: str, f: dict) -> float:
    s = 0.0
    s -= f["raie"] * 40                      # eviter le bourdonnement pur
    if kind == "telephone":
        s += f["period"] * 30                # cadence nette
        s += f["modulation"] * 10
        s -= abs(f["centroid"] - 1200) / 300
    else:
        s += f["modulation"] * 60            # voix
        s += min(f["flatness"], 0.6) * 8
        s -= max(0.0, f["centroid"] - 2500) / 200
    return s


def main() -> None:
    kind = sys.argv[1]
    cfg = VARIANTS[kind]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"chargement small-sfx pour '{kind}'...", flush=True)
    model = StableAudioModel.from_pretrained("small-sfx")

    results = []
    for pi, prompt in enumerate(cfg["prompts"]):
        for seed in (0, 7):
            torch.manual_seed(seed)
            audio = model.generate(prompt=prompt, duration=cfg["duration"])
            data = audio[0] if isinstance(audio, (list, tuple)) else audio
            arr = data.detach().float().cpu().numpy() if hasattr(data, "detach") else data
            arr = np.asarray(arr)
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim == 2:
                arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
            name = f"{kind}_p{pi}_s{seed}"
            sf.write(OUT / f"{name}.wav", arr, RATE)
            f = features(arr, RATE)
            sc = score(kind, f)
            results.append((sc, name, f))
            print(f"  {name}: note {sc:6.2f} | raie {f['raie']:.3f} | "
                  f"periodicite {f['period']:.2f} ({f['lag_ms']} ms) | "
                  f"modulation {f['modulation']:.2f} | centroide {f['centroid']:.0f} Hz",
                  flush=True)

    results.sort(reverse=True)
    print(f"\nMEILLEUR : {results[0][1]} (note {results[0][0]:.2f})", flush=True)


if __name__ == "__main__":
    main()
