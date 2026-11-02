"""Etire un clip court (10-25 s) en ambiance longue par bouclage en fondu croise.

Le fondu croise supprime le clic de jonction ; un decalage aleatoire d'une copie
a l'autre empeche le motif de tomber au meme endroit et rend la repetition
beaucoup moins perceptible. Le script mesure ensuite les discontinuites pour
verifier qu'aucun clic ne subsiste.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 24000


def crossfade_loop(src: np.ndarray, target_s: float, fade_s: float = 5.0,
                   jitter_s: float = 0.9, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Concatene des copies du clip avec recouvrement et fondu, decalage aleatoire."""
    rng = np.random.default_rng(seed)
    fade = int(fade_s * RATE)
    hop = len(src) - fade
    if hop <= 0:
        raise ValueError("clip plus court que le fondu")

    out = src[:hop].copy()
    junctions = []
    while len(out) < int(target_s * RATE):
        # copie suivante, demarree a un offset aleatoire pour casser le motif
        shift = int(rng.uniform(0.0, jitter_s) * RATE) % len(src)
        nxt = np.roll(src, -shift)
        tail = out[-fade:]
        head = nxt[:fade]
        # fondu lineaire croise
        w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        # fondu en puissance constante : indispensable sur des signaux decorreles
        out[-fade:] = tail * np.sqrt(1 - w) + head * np.sqrt(w)
        out = np.concatenate([out, nxt[fade:hop]])
        junctions.append(len(out) - len(nxt[fade:hop]))
    return out[: int(target_s * RATE)], junctions


def clicks(x: np.ndarray, junctions: list[int], win: int = 240) -> tuple[float, float]:
    """Compare la discontinuite aux jonctions au 99,9e percentile global."""
    d = np.abs(np.diff(x))
    global_p999 = float(np.percentile(d, 99.9))
    worst = 0.0
    for j in junctions:
        lo, hi = max(0, j - win), min(len(d), j + win)
        if hi > lo:
            worst = max(worst, float(np.max(d[lo:hi])))
    return worst, global_p999


def main() -> None:
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    target = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0

    x, sr = sf.read(str(src_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != RATE:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * RATE / sr)),
                      np.arange(len(x)), x).astype(np.float32)

    long, junctions = crossfade_loop(x, target)
    peak = float(np.max(np.abs(long)))
    if peak > 0.89:
        long *= 0.89 / peak
    sf.write(str(out_path), long, RATE, subtype="PCM_16")

    worst, p999 = clicks(long, junctions)
    print(f"{src_path.name} ({len(x) / RATE:.1f} s) -> {out_path.name} "
          f"({len(long) / RATE:.1f} s, {len(junctions)} jonctions)")
    print(f"  discontinuite aux jonctions : {worst:.4f} | 99,9e percentile global : {p999:.4f}")
    print(f"  verdict : {'OK' if worst <= p999 * 1.5 else 'CLIC PROBABLE'}")


if __name__ == "__main__":
    main()
