"""Fusionne les deux campagnes de generation et garde le meilleur par scene.

96 candidats au total (prompts simples et prompts « guide »), meme critere :
l'etiquette attendue doit gagner ET le score depasser le seuil. Aucune
regeneration : tout est deja sur disque.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

POOLS = [Path("/data/med16c/raw"), Path("/data/med4/raw")]
BEDS = Path("/out")
MIN_SCORE = 0.20

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
    "a hospital corridor with fluorescent hum",
    "a farmhouse kitchen with a wood stove crackling",
    "a record shop interior with background music",
    "a television playing in a dark room",
    "an empty open plan office with ventilation hum",
    "a quiet living room with a ticking clock",
    "a street demonstration with chanting crowd",
    "an underground party with muffled bass through walls",
]

EXPECTED = {
    "01_appartement_vide": 3, "02_rue_paris_nuit": 2, "03_cafe_foule": 1,
    "04_squat_fete": 16, "05_salon_bourgeois": 14, "06_magasin_disques": 11,
    "07_metro_quai": 6, "08_voiture_ville": 8, "09_hopital_couloir": 9,
    "10_foret_aube": 4, "11_ferme_cuisine": 10, "12_bar_nuit": 0,
    "13_pluie_fenetre": 5, "14_open_space_aube": 13, "15_television_nuit": 12,
    "16_manifestation": 15,
}


def main() -> None:
    BEDS.mkdir(parents=True, exist_ok=True)
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    with torch.no_grad():
        tf = clap.get_text_features(**proc(text=LABELS, return_tensors="pt", padding=True))
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    report, retenus = {}, []
    for name, expected in EXPECTED.items():
        best = None
        for pool in POOLS:
            for f in sorted(pool.glob(f"{name}_s*.wav")):
                y, sr = sf.read(str(f), dtype="float32", always_2d=True)
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
                    best = (win, own, f)

        win, own, f = best
        garde = win and own >= MIN_SCORE
        if garde:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(f), "-ac", "1", "-ar", "24000",
                            "-af", "loudnorm=I=-23:TP=-6:LRA=11",
                            str(BEDS / f"{name}.bed.wav")], check=True)
            retenus.append(name)
        print(f"  {name:22s} {'GARDE ' if garde else 'ECARTE'} {own:.2f}  {f.parent.parent.name}",
              flush=True)
        report[name] = {"fichier": str(f), "score": round(own, 2),
                        "gagne": win, "retenu": garde}

    (BEDS / "selection_finale.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(retenus)}/{len(EXPECTED)} fonds retenus", flush=True)


if __name__ == "__main__":
    main()
