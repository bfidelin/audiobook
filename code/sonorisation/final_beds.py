"""Selection finale + bouclage : etiquettes fusionnees, fonds etendus a 60 s.

Deux corrections par rapport a la version livree :
  - les etiquettes d'interieurs silencieux sont regroupees (elles se disputaient
    le meme son et faisaient echouer l'argmax) ;
  - chaque fond retenu est boucle a 60 s, car les plages durent jusqu'a une
    minute alors que les clips n'en font que 20.
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
MIN_SCORE = 0.18
LOOP_SECONDS = 60.0

# familles : une etiquette par famille, plus de doublons concurrents
LABELS = [
    "a busy bar with people talking and glasses clinking",   # 0  bar
    "a cafe with conversation and cutlery",                   # 1  cafe
    "a city street with traffic and passers-by",              # 2  rue
    "a quiet interior room tone, no voices",                  # 3  interieur calme
    "a forest with birds singing",                            # 4  foret
    "heavy rain falling",                                     # 5  pluie
    "a metro or subway station",                              # 6  metro
    "a crowd of people in a large hall",                      # 7  foule
    "a car engine and road noise",                            # 8  voiture
    "a record shop interior with background music",           # 9  boutique
    "an underground party with muffled bass through walls",   # 10 squat
    "a street demonstration with chanting crowd",             # 11 manif
]

EXPECTED = {
    "01_appartement_vide": 3, "02_rue_paris_nuit": 2, "03_cafe_foule": 1,
    "04_squat_fete": 10, "05_salon_bourgeois": 3, "06_magasin_disques": 9,
    "07_metro_quai": 6, "08_voiture_ville": 8, "09_hopital_couloir": 3,
    "10_foret_aube": 4, "11_ferme_cuisine": 12, "12_bar_nuit": 0,
    "13_pluie_fenetre": 5, "14_open_space_aube": 3, "15_television_nuit": 3,
    "16_manifestation": 11,
}


def loop_to(src: Path, dst: Path, seconds: float) -> None:
    """Boucle avec fondu en puissance constante et decalage aleatoire."""
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    fade = int(3.0 * sr)
    hop = len(x) - fade
    if hop <= 0:
        sf.write(str(dst), x, sr)
        return
    out = x[:hop].copy()
    rng = np.random.default_rng(0)
    while len(out) < int(seconds * sr):
        shift = int(rng.uniform(0, 0.9) * sr) % len(x)
        nxt = np.roll(x, -shift)
        w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[-fade:] = out[-fade:] * np.sqrt(1 - w) + nxt[:fade] * np.sqrt(w)
        out = np.concatenate([out, nxt[fade:hop]])
    sf.write(str(dst), out[: int(seconds * sr)], sr)


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
            tmp = BEDS / f"_tmp_{name}.wav"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(f), "-ac", "1", "-ar", "24000", str(tmp)], check=True)
            loop_to(tmp, BEDS / f"{name}.bed.wav", LOOP_SECONDS)
            tmp.unlink(missing_ok=True)
            retenus.append(name)
        print(f"  {name:22s} {'GARDE ' if garde else 'ECARTE'} {own:.2f}", flush=True)
        report[name] = {"score": round(own, 2), "gagne": win, "retenu": garde}

    (BEDS / "selection_loop.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(retenus)}/{len(EXPECTED)} fonds boucles a {LOOP_SECONDS:.0f} s", flush=True)


if __name__ == "__main__":
    main()
