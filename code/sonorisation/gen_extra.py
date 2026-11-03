"""Genere les deux fonds qui manquaient : musique de disque et sirene de police.

Le pipeline principal interdisait la musique dans son prompt negatif, d'ou son
absence totale du livre. Ici chaque ambiance a son propre negatif.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel
from transformers import ClapModel, ClapProcessor

OUT = Path("/out")
DEST = Path("/data/out23_loop")
LOOP_SECONDS = 60.0
SECONDS = 20.0
SEEDS = (0, 1, 2, 3, 4)
MIN_SCORE = 0.18

CIBLES = [
    {
        "name": "17_musique_disque",
        "prompt": ("A vinyl rock record playing on a living-room stereo, warm "
                   "mid-range, slightly muffled through the walls, steady groove"),
        "label": "a song playing from a stereo",
        "negative": "speech, voice, shouting, clipping, distortion, silence",
    },
    {
        "name": "18_sirene_police",
        "prompt": ("Police sirens wailing in a Paris street at night, approaching "
                   "then passing, distant traffic underneath"),
        "label": "a police siren",
        "negative": "music, melody, speech, voice, clipping, distortion",
    },
]


def loop_to(src: Path, dst: Path, seconds: float) -> None:
    """Boucle avec fondu en puissance constante (racines carrees)."""
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 24000:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 24000 / sr)),
                      np.arange(len(x)), x).astype(np.float32)
    n = int(seconds * 24000)
    fade = int(5 * 24000)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)
    pos = 0
    while pos < n:
        seg = x.copy()
        if pos + len(seg) > n:
            seg = seg[: n - pos]
        if len(seg) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[:fade] *= np.sqrt(ramp)
            seg[-fade:] *= np.sqrt(ramp[::-1])
        out[pos:pos + len(seg)] += seg
        pos += len(seg) - fade
    out = out[:n]
    peak = float(np.max(np.abs(out))) or 1.0
    sf.write(str(dst), (out / peak * 0.9).astype(np.float32), 24000, subtype="PCM_16")
    _ = rng


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    model = StableAudioModel.from_pretrained("medium")
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    for cible in CIBLES:
        tf = clap.get_text_features(**proc(text=[cible["label"]], return_tensors="pt",
                                           padding=True))
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        best = None
        for seed in SEEDS:
            audio = model.generate(prompt=cible["prompt"], seconds_total=SECONDS,
                                   seed=seed, cfg_scale=7.0,
                                   negative_prompt=cible["negative"])
            y = np.asarray(audio, dtype=np.float32)
            if y.ndim > 1:
                y = y.mean(axis=0)
            tmp = OUT / f"_{cible['name']}_s{seed}.wav"
            sf.write(str(tmp), y, 24000, subtype="PCM_16")
            z = np.interp(np.linspace(0, len(y) - 1, int(20 * 48000)),
                          np.arange(len(y)), y).astype("float32")
            with torch.no_grad():
                af = clap.get_audio_features(**proc(audio=z, sampling_rate=48000,
                                                    return_tensors="pt"))
                af = getattr(af, "pooler_output", af)
                af = af / af.norm(dim=-1, keepdim=True)
            score = float((af @ tf.T)[0][0])
            print(f"  {cible['name']} seed {seed} : {score:.3f}", flush=True)
            if best is None or score > best[0]:
                best = (score, tmp)
        score = best[0] if best else 0.0
        if best and score >= MIN_SCORE:
            loop_to(best[1], DEST / f"{cible['name']}.bed.wav", LOOP_SECONDS)
            print(f"{cible['name']} retenu ({score:.2f})", flush=True)
        else:
            print(f"{cible['name']} ECARTE (meilleur {score:.2f})", flush=True)
            sys.exit(1)
    subprocess.run(["true"], check=False)


if __name__ == "__main__":
    main()
