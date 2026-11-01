"""Genere des effets sonores avec Stable Audio 3 Small-SFX.

Modele entraine pour les effets (pas la musique) : 433 M de parametres,
jusqu'a 120 s de sortie, fonctionne sans Flash Attention.
Sortie mono 24 kHz, -26 LUFS, fondus courts : des evenements ponctuels.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

OUT = Path("/out")
MODEL = os.environ.get("SA3_MODEL", "stabilityai/stable-audio-3-small-sfx")
SA3_RATE = 44_100  # frequence de sortie du modele (stereo, moyennee en mono)
JOBS = [
    ("porte_claque", "a heavy wooden door slamming shut in an empty apartment, close perspective", 4.0),
    ("pas_escalier", "footsteps on old wooden stairs, slow and irregular, interior reverb", 6.0),
    ("telephone_sonne", "an old landline telephone ringing in a dark room", 5.0),
    ("cles_serrure", "keys turning in a door lock, close up, metal on metal", 4.0),
    ("verres_bar", "glasses clinking in a busy bar, close up", 4.0),
    ("chaise_racle", "a wooden chair scraping on a cafe floor", 3.0),
    ("briquet", "a cigarette lighter flicking open and igniting, close up", 3.0),
    ("sirene_loin", "a police siren passing far away on a city street at night", 8.0),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"chargement {MODEL}...", flush=True)
    model = StableAudioModel.from_pretrained(MODEL)
    print(f"pret | GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
          flush=True)

    manifest = []
    for name, prompt, dur in JOBS:
        raw = OUT / f"{name}.wav"
        if raw.exists() and raw.stat().st_size > 20_000:
            print(f"[skip] {name}", flush=True)
        else:
            torch.manual_seed(0)
            audio = model.generate(prompt=prompt, duration=dur)
            data = audio[0] if isinstance(audio, (list, tuple)) else audio
            arr = data.detach().float().cpu().numpy() if hasattr(data, "detach") else data
            arr = np.asarray(arr)
            # Forme (lot, canaux, echantillons) : retirer le lot, moyenner les canaux.
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim == 2:
                arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
            sf.write(raw, arr, SA3_RATE)
            print(f"[gen ] {name} ({dur:.0f} s, {arr.shape[-1] / SA3_RATE:.1f} s reels)",
                  flush=True)

        mono = OUT / f"{name}.sfx.wav"
        fade_out = max(0.0, dur - 0.2)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
             "-ac", "1", "-ar", "24000",
             "-af", f"loudnorm=I=-26:TP=-4:LRA=11,afade=t=in:st=0:d=0.03,"
                    f"afade=t=out:st={fade_out:.2f}:d=0.2",
             str(mono)],
            check=True,
        )
        manifest.append({"name": name, "prompt": prompt, "duration": dur,
                         "file": mono.name})

    try:
        with open(OUT / "sfx_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
    except OSError as exc:
        print(f"manifeste non ecrit : {exc}", file=sys.stderr)
    print(f"\n{len(manifest)} effets dans {OUT}", flush=True)


if __name__ == "__main__":
    main()
