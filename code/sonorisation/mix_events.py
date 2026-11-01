"""Mixeur a evenements : fonds continus ET sons ponctuels, ducking en numpy.

Remplace la chaine ffmpeg (sidechaincompress + amix + loudnorm) par un calcul
direct dans un buffer : plus de fichiers intermediaires, placement a
l'echantillon, et un cout proportionnel a la duree une seule fois.

Evenements (JSON) :
  {"voice": "part.wav", "out": "mix.wav",
   "beds":   [{"t": 24.5, "d": 103.4, "file": "01_appartement_vide.bed.wav"}],
   "events": [{"t": 412.5, "file": "porte_claque.sfx.wav", "gain_db": -6}]}

Tous les niveaux sont en dBFS relatif au fichier source ; les fonds sont censes
etre normalises a -33 LUFS et les effets a -26 LUFS.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 24000
ATTACK = 0.15   # s
RELEASE = 0.60  # s
DUCK_DB = -4.0  # attenuation du fond quand la voix parle


def db(x: float) -> float:
    return 10 ** (x / 20)


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != RATE:
        idx = np.linspace(0, len(x) - 1, int(len(x) * RATE / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def place(buf: np.ndarray, clip: np.ndarray, start_s: float, gain_db: float,
          fade: float = 0.15) -> None:
    start = int(start_s * RATE)
    if start >= len(buf):
        return
    n = min(len(clip), len(buf) - start)
    if n <= 0:
        return
    seg = clip[:n] * db(gain_db)
    k = min(int(fade * RATE), n // 2)
    if k > 0:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        seg[:k] *= ramp
        seg[-k:] *= ramp[::-1]
    buf[start:start + n] += seg


def duck_curve(voice: np.ndarray, block: int = 2400) -> np.ndarray:
    """Courbe de gain pour le fond : -8 dB quand la voix est presente."""
    n_blocks = max(1, len(voice) // block)
    env = np.array([np.sqrt(np.mean(voice[i * block:(i + 1) * block] ** 2))
                    for i in range(n_blocks)], dtype=np.float32)
    peak = env.max() + 1e-9
    active = np.clip(env / (0.05 * peak), 0.0, 1.0) if peak > 0.01 else np.zeros_like(env)
    target = 1.0 + (db(DUCK_DB) - 1.0) * active

    # lissage asymetrique : attaque rapide, relachement lent
    a = np.exp(-block / (ATTACK * RATE))
    r = np.exp(-block / (RELEASE * RATE))
    out = np.empty_like(target)
    g = 1.0
    for i, t in enumerate(target):
        coeff = a if t < g else r
        g = coeff * g + (1 - coeff) * t
        out[i] = g
    curve = np.repeat(out, block)
    if len(curve) < len(voice):
        curve = np.concatenate([curve, np.full(len(voice) - len(curve), out[-1], dtype=curve.dtype)])
    return curve[: len(voice)]


def main() -> None:
    try:
        cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.exit(f"config illisible : {exc}")
    voice = load(Path(cfg["voice"]))
    n = len(voice)
    print(f"voix : {n / RATE / 60:.1f} min", flush=True)

    bed = np.zeros(n, dtype=np.float32)
    for b in cfg.get("beds", []):
        place(bed, load(Path(cfg["beds_dir"], b["file"])), b["t"], b.get("gain_db", 0.0),
              fade=b.get("fade", 2.0))

    if bed.any():
        bed *= duck_curve(voice)
        print(f"ducking applique sur {np.count_nonzero(bed)} echantillons", flush=True)

    for e in cfg.get("events", []):
        place(bed, load(Path(cfg["events_dir"], e["file"])), e["t"],
              e.get("gain_db", 0.0), fade=e.get("fade", 0.02))

    mix = voice + bed
    peak = float(np.max(np.abs(mix))) if n else 0.0
    if peak > 0.89:                      # marge avant ecriture
        mix *= 0.89 / peak
        print(f"crete ramenee de {20 * np.log10(peak):.1f} dBFS a -1.0", flush=True)

    sf.write(cfg["out"], mix, RATE, subtype="PCM_16")
    print(f"ecrit : {cfg['out']} ({len(mix) / RATE / 60:.1f} min, "
          f"{Path(cfg['out']).stat().st_size / 1e6:.0f} Mo)", flush=True)


if __name__ == "__main__":
    main()
