"""Genere des effets sonores avec AudioLDM 2 (diffusers).

AudioLDM 2 est entraine sur des effets sonores, la ou MusicGen fait de la
musique : portes, pas, telephone, verres, briquet. Sortie mono 24 kHz
normalisee a -26 LUFS (effets ponctuels, poses ponctuellement).
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from diffusers import AudioLDM2Pipeline

OUT = Path("/out")
MODEL = os.environ.get("SFX_MODEL", "cvssp/audioldm2")
JOBS = [
    ("porte_claque", "a heavy wooden door slamming shut in an empty apartment, close perspective"),
    ("pas_escalier", "footsteps on old wooden stairs, slow and irregular, interior reverb"),
    ("telephone_sonne", "an old landline telephone ringing in a dark room"),
    ("cles_serrure", "keys turning in a door lock, close up, metal on metal"),
    ("verres_bar", "glasses clinking in a busy bar, close up"),
    ("chaise_racle", "a wooden chair scraping on a cafe floor"),
    ("briquet", "a cigarette lighter flicking open and igniting, close up"),
    ("sirene_loin", "a police siren passing far away on a city street at night"),
]
SECONDS = 5.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"chargement {MODEL}...", flush=True)
    pipe = AudioLDM2Pipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
    print(f"pret | GPU {torch.cuda.get_device_name(0)}", flush=True)

    manifest = []
    for name, prompt in JOBS:
        raw = OUT / f"{name}.wav"
        if raw.exists() and raw.stat().st_size > 20_000:
            print(f"[skip] {name}", flush=True)
        else:
            torch.manual_seed(0)
            audio = pipe(
                prompt,
                num_inference_steps=200,
                audio_length_in_s=SECONDS,
                num_waveforms_per_prompt=1,
            ).audios
            sf.write(raw, audio[0], 16000)
            print(f"[gen ] {name} ({SECONDS:.0f} s)", flush=True)

        mono = OUT / f"{name}.sfx.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
             "-ac", "1", "-ar", "24000",
             "-af", "loudnorm=I=-26:TP=-4:LRA=11,afade=t=in:st=0:d=0.05,"
                    "afade=t=out:st=4.8:d=0.2",
             str(mono)],
            check=True,
        )
        manifest.append({"name": name, "prompt": prompt, "file": mono.name})

    try:
        with open(OUT / "sfx_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
    except OSError as exc:
        print(f"manifeste non ecrit : {exc}", file=sys.stderr)
    print(f"\n{len(manifest)} effets dans {OUT}", flush=True)


if __name__ == "__main__":
    main()
