"""Recolte ciblee : telephone des candidats, les identifie avec CLAP, affecte.

Principe : au lieu d'attribuer un fichier a une scene au hasard, on constitue un
pool, on demande a CLAP a quoi ressemble chaque fichier, et on ne l'affecte que
si la correspondance depasse un seuil. Faute de candidat credible, la scene
reste sans fond plutot que d'en recevoir un faux.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

POOL = Path("/data/pool")
BEDS = Path("/data/out23_real2")
MIN_DUR = 25.0
MIN_SCORE = 0.30

PAGES = [
    "https://sfxmint.com/sounds/ambience-forest",
    "https://sfxmint.com/sounds/ambience-rain",
    "https://sfxmint.com/sounds/ambience-city",
    "https://sfxmint.com/category/footsteps",
    "https://mixkit.co/free-sound-effects/crowd/",
    "https://mixkit.co/free-sound-effects/rain/",
    "https://mixkit.co/free-sound-effects/restaurant/",
    "https://mixkit.co/free-sound-effects/city/",
]

BED_LABELS = {
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


def harvest() -> list[Path]:
    POOL.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    for page in PAGES:
        try:
            req = urllib.request.Request(page, headers={"User-Agent": "Mozilla/5.0"})
            html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore")
        except Exception:
            html = ""
        for line in html.split('"'):
            if (line.startswith("http") and line.endswith((".mp3", ".wav"))
                    and line not in urls):
                urls.append(line)
    print(f"{len(urls)} candidats trouves", flush=True)
    files = []
    for i, u in enumerate(urls[:40], start=1):
        dst = POOL / f"c{i:02d}{Path(u).suffix}"
        if dst.exists():
            files.append(dst)
            continue
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            dst.write_bytes(urllib.request.urlopen(req, timeout=45).read())
        except Exception:
            pass
        if dst.exists() and dst.stat().st_size > 20_000:
            files.append(dst)
    return files


def main() -> None:
    BEDS.mkdir(parents=True, exist_ok=True)
    files = harvest()
    print(f"{len(files)} fichiers telecharges", flush=True)

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    names = list(BED_LABELS)
    with torch.no_grad():
        t = model.get_text_features(**proc(text=[BED_LABELS[n] for n in names],
                                           return_tensors="pt", padding=True))
        t = getattr(t, "pooler_output", t)
        t = t / t.norm(dim=-1, keepdim=True)

    best: dict[str, tuple[float, Path]] = {}
    for f in files:
        try:
            y, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except (OSError, ValueError):
            continue
        y = y.mean(axis=1)
        dur = len(y) / sr
        if dur < MIN_DUR:
            continue
        y = y[: int(20 * sr)]
        if sr != 48000:
            y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * 48000 / sr)),
                          np.arange(len(y)), y).astype("float32")
        with torch.no_grad():
            a = model.get_audio_features(**proc(audio=y, sampling_rate=48000, return_tensors="pt"))
            a = getattr(a, "pooler_output", a)
            a = a / a.norm(dim=-1, keepdim=True)
        sims = (a @ t.T)[0]
        score, idx = float(sims.max()), int(sims.argmax())
        name = names[idx]
        if score >= MIN_SCORE and (name not in best or score > best[name][0]):
            best[name] = (score, f)
            print(f"  {f.name} -> {name} ({score:.2f})", flush=True)

    report = {}
    for name, (score, f) in best.items():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(f),
                        "-ac", "1", "-ar", "24000",
                        "-af", "loudnorm=I=-23:TP=-6:LRA=11",
                        str(BEDS / f"{name}.bed.wav")], check=True)
        report[name] = {"source": f.name, "score": round(score, 3)}

    (BEDS / "mapping.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\n{len(report)} scenes affectees sur {len(names)}", flush=True)
    manquantes = [n for n in names if n not in report]
    if manquantes:
        print("sans fond :", ", ".join(manquantes), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
