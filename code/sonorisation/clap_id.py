"""Identifie le contenu d'un fichier audio par comparaison texte (CLAP).

On donne une liste de descriptions candidates ; le modele renvoie laquelle
correspond le mieux. Remplace l'ecoute humaine pour nommer des fichiers
recoltes sans titre.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

MODEL = "laion/clap-htsat-unfused"
LABELS = [
    "a busy bar with people talking and glasses clinking",
    "a cafe with conversation and cutlery",
    "a city street with traffic and passers-by",
    "a quiet empty apartment room",
    "a forest with birds singing",
    "heavy rain falling",
    "a metro or subway station",
    "footsteps walking on gravel or stone",
    "footsteps on wooden stairs",
    "a crowd of people in a large hall",
    "a car engine and road noise",
    "wind blowing through trees",
]


def main() -> None:
    model = ClapModel.from_pretrained(MODEL)
    proc = ClapProcessor.from_pretrained(MODEL)
    ref = proc(text=LABELS, return_tensors="pt", padding=True)
    with torch.no_grad():
        tfeat = model.get_text_features(**ref)
        tfeat = getattr(tfeat, "pooler_output", tfeat)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    for name in sys.argv[1:]:
        path = Path(name)
        try:
            y, sr0 = sf.read(str(path), dtype="float32", always_2d=True)
            y = y.mean(axis=1)[: int(20.0 * sr0)]
            if sr0 != 48000:
                y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * 48000 / sr0)),
                              np.arange(len(y)), y).astype("float32")
        except (OSError, ValueError) as exc:
            print(f"{path.name}: illisible ({exc})")
            continue
        af = proc(audio=y, sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            afeat = model.get_audio_features(**af)
            afeat = getattr(afeat, "pooler_output", afeat)
            afeat = afeat / afeat.norm(dim=-1, keepdim=True)
        sims = (afeat @ tfeat.T)[0]
        top = torch.topk(sims, 3)
        print(f"\n{path.name}")
        for score, idx in zip(top.values.tolist(), top.indices.tolist(), strict=False):
            print(f"   {score:.3f}  {LABELS[idx]}")


if __name__ == "__main__":
    main()
