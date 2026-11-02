"""Genere les 16 scenes avec Stable Audio 3 Medium, puis les valide par CLAP.

Une scene n'est conservee que si CLAP la reconnait au-dessus du seuil. Les
fichiers sont ecrits en mono 24 kHz -23 LUFS, pretes pour mix_events.py.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel
from transformers import ClapModel, ClapProcessor

OUT = Path("/out")
RAW = OUT / "raw"
MIN_SCORE = 0.20
SECONDS = 20.0

SCENES = {
    "01_appartement_vide": "an empty Parisian apartment, bare wooden floor, faint city traffic "
                           "through closed windows, quiet room tone",
    "02_rue_paris_nuit": "a Paris street at night, light rain on the pavement, a car passing, "
                         "distant voices, urban ambience",
    "03_cafe_foule": "a crowded French cafe interior, people talking, cups and cutlery, "
                     "espresso machine in the distance",
    "04_squat_fete": "an underground warehouse party, muffled bass and drums through walls, "
                     "crowd murmur, concrete reverb",
    "05_salon_bourgeois": "a quiet luxury Parisian living room, ticking clock, faint street noise, "
                          "calm room tone",
    "06_magasin_disques": "a small record shop interior, faint rock music from ceiling speakers, "
                          "a door chime, vinyl crates",
    "07_metro_quai": "a Paris metro platform, a train arriving and braking, echoing tiled walls",
    "08_voiture_ville": "inside a small car in city traffic, engine idle, indicator ticking, "
                        "faint radio, wet road",
    "09_hopital_couloir": "a hospital corridor at night, fluorescent lights humming, distant doors, "
                          "quiet clinical room tone",
    "10_foret_aube": "a forest edge at dawn, birdsong, light wind in the leaves, peaceful ambience",
    "11_ferme_cuisine": "a farmhouse kitchen, wood stove crackling, wind outside the window, "
                        "an old clock",
    "12_bar_nuit": "a busy French bar at night, people talking, glasses clinking, "
                   "warm crowded room ambience",
    "13_pluie_fenetre": "heavy rain falling against a window, distant thunder, gutters overflowing",
    "14_open_space_aube": "an empty open plan office at six in the morning, ventilation hum, "
                          "silent fluorescent lights",
    "15_television_nuit": "a dark living room with a television playing quietly, muffled voices, "
                          "late night room tone",
    "16_manifestation": "a street demonstration in the distance, chanted slogans, whistles, "
                        "crowd movement",
}
CLAP_LABELS = [
    "a busy bar with people talking and glasses clinking",
    "a cafe with conversation and cutlery",
    "a city street with traffic and passers-by",
    "a quiet empty apartment room",
    "a forest with birds singing",
    "heavy rain falling",
    "a metro or subway station",
    "footsteps walking on gravel or stone",
    "a crowd of people in a large hall",
    "a car engine and road noise",
    "a hospital corridor",
    "wind blowing through trees",
    "an empty quiet room tone",
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = StableAudioModel.from_pretrained("medium")
    print(f"medium charge en {time.perf_counter() - t0:.0f} s", flush=True)

    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    with torch.no_grad():
        tf = clap.get_text_features(**proc(text=CLAP_LABELS, return_tensors="pt", padding=True))
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    report = {}
    for name, prompt in SCENES.items():
        raw = RAW / f"{name}.wav"
        if not raw.exists() or raw.stat().st_size < 50_000:
            t1 = time.perf_counter()
            audio = model.generate(prompt=prompt, duration=SECONDS)
            arr = audio[0] if isinstance(audio, (list, tuple)) else audio
            arr = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else arr
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim == 2:
                arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
            sf.write(str(raw), arr, 44100)
            print(f"  gen {name:22s} {time.perf_counter() - t1:5.1f} s", flush=True)

        y, sr = sf.read(str(raw), dtype="float32", always_2d=True)
        y = y.mean(axis=1)[: int(20 * sr)]
        if sr != 48000:
            y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * 48000 / sr)),
                          np.arange(len(y)), y).astype("float32")
        with torch.no_grad():
            af = clap.get_audio_features(**proc(audio=y, sampling_rate=48000, return_tensors="pt"))
            af = getattr(af, "pooler_output", af)
            af = af / af.norm(dim=-1, keepdim=True)
        sims = (af @ tf.T)[0]
        score, idx = float(sims.max()), int(sims.argmax())
        label = CLAP_LABELS[idx]
        ok = score >= MIN_SCORE
        print(f"  {name:22s} -> {label[:42]:42s} {score:.2f} {'garde' if ok else 'ECARTE'}",
              flush=True)
        if ok:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(raw), "-ac", "1", "-ar", "24000",
                            "-af", "loudnorm=I=-23:TP=-6:LRA=11",
                            str(OUT / f"{name}.bed.wav")], check=True)
        report[name] = {"prompt": prompt, "label": label, "score": round(score, 3),
                        "retenu": ok}

    (OUT / "medium_mapping.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    gardes = sum(1 for r in report.values() if r["retenu"])
    print(f"\n{gardes}/{len(SCENES)} scenes retenues", flush=True)


if __name__ == "__main__":
    main()
