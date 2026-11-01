"""Genere les fonds sonores du livre audio avec MusicGen (transformers).

Stable Audio Open est « gated » sur Hugging Face (token requis) ; MusicGen-medium
n'a pas de restriction. Chaque ambiance : 30 s, mono 24 kHz, normalisee a
-33 LUFS avec fondus, pour se poser sous la narration sans calcul de gain.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

OUT = Path("/out")
MODEL = os.environ.get("MUSICGEN", "facebook/musicgen-medium")
JOBS = [
    ("01_appartement_vide", "empty Parisian apartment, bare wooden floor, distant traffic through closed windows, fridge hum, quiet room tone"),
    ("02_rue_paris_nuit", "Paris street at night, light rain on pavement, an occasional car passing, distant voices, urban night ambience"),
    ("03_cafe_foule", "crowded French cafe interior, muffled conversation, cups and saucers, espresso machine in the distance"),
    ("04_squat_fete", "underground warehouse squat party, muffled bass and drums through walls, crowd murmur, concrete reverb"),
    ("05_salon_bourgeois", "quiet luxury Parisian living room, ticking clock, faint street noise outside, parquet floor, calm room tone"),
    ("06_magasin_disques", "small record shop interior, faint rock music from ceiling speakers, occasional door chime, vinyl crates"),
    ("07_metro_quai", "Paris metro platform, train arriving and braking, distant station announcement, echoing tiled walls"),
    ("08_voiture_ville", "inside a small car in city traffic, engine idle, indicators ticking, faint radio, wet road"),
    ("09_hopital_couloir", "hospital corridor at night, fluorescent lights hum, distant doors and footsteps, quiet clinical room tone"),
    ("10_foret_aube", "forest edge at dawn, birdsong, light wind in leaves, distant cuckoo, peaceful natural ambience"),
    ("11_ferme_cuisine", "farmhouse kitchen, wood stove crackling, wind outside the window, old clock, warm domestic room tone"),
    ("12_bar_nuit", "busy bar at night, glasses clinking, laughter and chatter, jukebox in the background, warm crowded room"),
    ("13_pluie_fenetre", "heavy rain against a window, occasional distant thunder, gutters overflowing, grey afternoon ambience"),
    ("14_open_space_aube", "empty open plan office at six in the morning, ventilation hum, faint computer fans, silent fluorescent lights"),
    ("15_television_nuit", "dark living room with a television on low, muffled news voices, static hiss, late night room tone"),
    ("16_manifestation", "street demonstration in the distance, chanted slogans, whistles, crowd movement, urban square"),
]
SECONDS = 30.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"chargement {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16
    ).to("cuda")
    sr = model.config.audio_encoder.sampling_rate
    max_tokens = int(SECONDS * 50)  # MusicGen : 50 jetons par seconde d'audio
    print(f"pret | GPU {torch.cuda.get_device_name(0)} | {sr} Hz", flush=True)

    manifest = []
    for name, prompt in JOBS:
        raw = OUT / f"{name}.wav"
        if raw.exists() and raw.stat().st_size > 50_000:
            print(f"[skip] {name}", flush=True)
        else:
            torch.manual_seed(0)
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                audio = model.generate(
                    **inputs, do_sample=True, guidance_scale=3.0,
                    max_new_tokens=max_tokens,
                )
            data = audio[0, 0].float().cpu().numpy()
            sf.write(raw, data, sr)
            print(f"[gen ] {name} ({len(data) / sr:.1f} s)", flush=True)

        mono = OUT / f"{name}.bed.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
             "-ac", "1", "-ar", "24000",
             "-af", "loudnorm=I=-33:TP=-6:LRA=11,afade=t=in:st=0:d=1.5,"
                    "areverse,afade=t=in:st=0:d=1.5,areverse",
             str(mono)],
            check=True,
        )
        manifest.append({"name": name, "prompt": prompt, "file": mono.name})

    try:
        with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
    except OSError as exc:
        print(f"manifeste non ecrit : {exc}", file=sys.stderr)
    print(f"\n{len(manifest)} fonds dans {OUT}", flush=True)


if __name__ == "__main__":
    main()
