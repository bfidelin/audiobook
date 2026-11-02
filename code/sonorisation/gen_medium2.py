"""Genere les 16 scenes avec Medium, prompt negatif, plusieurs graines,
et ne garde que les candidats ou l'etiquette ATTENDUE gagne.

Le critere n'est plus « un lieu quelconque au-dessus d'un seuil » mais
« la bonne etiquette en tete ». C'est ce qui distingue un son plausible
d'un son juste.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel
from transformers import ClapModel, ClapProcessor

OUT = Path("/out")
RAW = OUT / "raw"
SEEDS = (0, 1, 2, 3)
SECONDS = 20.0
NEG = "music, melody, singing, voice, speech, loud drums, distorted, clipping"

LABELS = [
    "a busy bar with people talking and glasses clinking",
    "a cafe with conversation and cutlery",
    "a city street with traffic and passers-by",
    "a quiet empty apartment room",
    "a forest with birds singing",
    "heavy rain falling",
    "a metro or subway station",
    "a crowd of people in a large hall",
    "a car engine and road noise",
    "a hospital corridor",
    "wind blowing through trees",
    "an empty quiet room tone",
]

# scene -> (prompt, etiquette attendue)
SCENES = {
    "01_appartement_vide": ("an empty Parisian apartment, bare wooden floor, faint city traffic "
                            "through closed windows, quiet room tone", 3),
    "02_rue_paris_nuit": ("a Paris street at night, light rain on the pavement, a car passing, "
                          "distant voices", 2),
    "03_cafe_foule": ("a crowded French cafe interior, people talking, cups and cutlery", 1),
    "04_squat_fete": ("an underground warehouse party, muffled bass through walls, crowd murmur", 7),
    "05_salon_bourgeois": ("a quiet luxury Parisian living room, ticking clock, calm room tone", 11),
    "06_magasin_disques": ("a small shop interior, faint background music, a door chime", 1),
    "07_metro_quai": ("a Paris metro platform, a train arriving and braking, echoing tiles", 6),
    "08_voiture_ville": ("inside a small car in city traffic, engine idle, wet road", 8),
    "09_hopital_couloir": ("a hospital corridor at night, fluorescent lights humming, distant doors", 9),
    "10_foret_aube": ("a forest edge at dawn, birdsong, light wind in the leaves", 4),
    "11_ferme_cuisine": ("a farmhouse kitchen, wood stove crackling, wind outside the window", 10),
    "12_bar_nuit": ("a busy French bar at night, people talking, glasses clinking", 0),
    "13_pluie_fenetre": ("heavy rain falling against a window, distant thunder", 5),
    "14_open_space_aube": ("an empty open plan office at dawn, ventilation hum, silent lights", 11),
    "15_television_nuit": ("a dark living room, a television playing quietly, late night room tone", 11),
    "16_manifestation": ("a street demonstration in the distance, chanted slogans, crowd", 7),
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    model = StableAudioModel.from_pretrained("medium")
    print("medium charge", flush=True)

    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    with torch.no_grad():
        tf = clap.get_text_features(**proc(text=LABELS, return_tensors="pt", padding=True))
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    report = {}
    for name, (prompt, expected) in SCENES.items():
        best = None
        for seed in SEEDS:
            raw = RAW / f"{name}_s{seed}.wav"
            if not raw.exists() or raw.stat().st_size < 50_000:
                audio = model.generate(prompt=prompt, duration=SECONDS, seed=seed,
                                       negative_conditioning=NEG)
                arr = audio[0] if isinstance(audio, (list, tuple)) else audio
                arr = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else arr
                while arr.ndim > 2:
                    arr = arr[0]
                if arr.ndim == 2:
                    arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
                sf.write(str(raw), arr, 44100)

            y, sr = sf.read(str(raw), dtype="float32", always_2d=True)
            y = y.mean(axis=1)[: int(20 * sr)]
            if sr != 48000:
                y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * 48000 / sr)),
                              np.arange(len(y)), y).astype("float32")
            with torch.no_grad():
                af = clap.get_audio_features(**proc(audio=y, sampling_rate=48000,
                                                    return_tensors="pt"))
                af = getattr(af, "pooler_output", af)
                af = af / af.norm(dim=-1, keepdim=True)
            sims = (af @ tf.T)[0]
            own = float(sims[expected])
            win = int(sims.argmax()) == expected
            if best is None or (win, own) > (best[0], best[1]):
                best = (win, own, raw, seed)

        win, own, raw, seed = best
        if win:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(raw), "-ac", "1", "-ar", "24000",
                            "-af", "loudnorm=I=-23:TP=-6:LRA=11",
                            str(OUT / f"{name}.bed.wav")], check=True)
        print(f"  {name:22s} {'OK ' if win else 'KO '} score propre {own:.2f} "
              f"(graine {seed})", flush=True)
        report[name] = {"prompt": prompt, "attendu": LABELS[expected],
                        "score_propre": round(own, 2), "gagne": win, "graine": seed}

    (OUT / "mapping_v2.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    gagnes = sum(1 for r in report.values() if r["gagne"])
    print(f"\n{gagnes}/{len(SCENES)} scenes ou la bonne etiquette gagne", flush=True)


if __name__ == "__main__":
    main()
