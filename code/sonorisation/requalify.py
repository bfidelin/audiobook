"""Requalifie le pool deja telecharge avec CLAP, puis affecte une ambiance par scene.

Garde-fou : un fichier n'est retenu que si sa duree atteint MIN_DUR et si son
meilleur score CLAP depasse MIN_SCORE. Un score negatif signifie « ne ressemble
a rien de connu » : rejet, quelle que soit la duree.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

POOL = Path("/data/pool")
BEDS = Path("/data/out23_real3")
MIN_DUR = 10.0
MIN_SCORE = 0.30

LABELS = {
    "01_appartement_vide": "a quiet empty apartment room with faint street noise",
    "02_rue_paris_nuit": "a city street with traffic and passers-by at night",
    "03_cafe_foule": "a cafe with conversation and cutlery",
    "04_squat_fete": "a crowd of people in a large hall, party",
    "05_salon_bourgeois": "a calm quiet living room interior",
    "06_magasin_disques": "a small shop interior with faint music",
    "07_metro_quai": "a metro or subway station platform",
    "08_voiture_ville": "a car engine and road noise inside a vehicle",
    "09_hopital_couloir": "a quiet hospital corridor, fluorescent hum",
    "10_foret_aube": "a forest with birds singing",
    "11_ferme_cuisine": "a farmhouse kitchen, wood stove crackling",
    "12_bar_nuit": "a busy bar with people talking and glasses clinking",
    "13_pluie_fenetre": "heavy rain falling against a window",
    "14_open_space_aube": "an empty office with ventilation hum",
    "15_television_nuit": "a television playing quietly in a dark room",
    "16_manifestation": "a street demonstration, crowd chanting",
}


def main() -> None:
    BEDS.mkdir(parents=True, exist_ok=True)
    files = sorted(POOL.glob("*.mp3")) + sorted(POOL.glob("*.wav"))
    print(f"{len(files)} fichiers dans le pool", flush=True)

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    names = list(LABELS)
    with torch.no_grad():
        t = model.get_text_features(**proc(text=[LABELS[n] for n in names],
                                           return_tensors="pt", padding=True))
        t = getattr(t, "pooler_output", t)
        t = t / t.norm(dim=-1, keepdim=True)

    best: dict[str, tuple[float, Path]] = {}
    rejects = []
    for f in files:
        try:
            y, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except (OSError, ValueError):
            rejects.append((f.name, "illisible"))
            continue
        y = y.mean(axis=1)
        dur = len(y) / sr
        if dur < MIN_DUR:
            rejects.append((f.name, f"court ({dur:.1f}s)"))
            continue
        seg = y[: int(20 * sr)]
        if sr != 48000:
            seg = np.interp(np.linspace(0, len(seg) - 1, int(len(seg) * 48000 / sr)),
                            np.arange(len(seg)), seg).astype("float32")
        with torch.no_grad():
            a = model.get_audio_features(**proc(audio=seg, sampling_rate=48000,
                                                return_tensors="pt"))
            a = getattr(a, "pooler_output", a)
            a = a / a.norm(dim=-1, keepdim=True)
        sims = (a @ t.T)[0]
        score, idx = float(sims.max()), int(sims.argmax())
        name = names[idx]
        if score < MIN_SCORE:
            rejects.append((f.name, f"score {score:.2f} < {MIN_SCORE}"))
            continue
        if name not in best or score > best[name][0]:
            best[name] = (score, f)

    report = {}
    for name, (score, f) in best.items():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(f),
                        "-ac", "1", "-ar", "24000",
                        "-af", "loudnorm=I=-23:TP=-6:LRA=11",
                        str(BEDS / f"{name}.bed.wav")], check=True)
        report[name] = {"source": f.name, "score": round(score, 3)}
        print(f"  {name:24s} <- {f.name} ({score:.2f})", flush=True)

    (BEDS / "mapping.json").write_text(json.dumps(
        {"retenus": report, "rejets": [{"fichier": a, "raison": b} for a, b in rejects]},
        indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(report)} scenes affectees / {len(names)} | "
          f"{len(rejects)} fichiers rejetes", flush=True)
    manquantes = [n for n in names if n not in report]
    if manquantes:
        print("sans fond :", ", ".join(manquantes), flush=True)


if __name__ == "__main__":
    main()
