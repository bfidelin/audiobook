"""Rend les fonds audibles : coupe le rumble <90 Hz et renormalise sur la bande parole.

Constat : plusieurs fonds Stable Audio ont 95-99 % de leur energie sous 60 Hz.
Inaudibles sur enceintes/casque, alors que leur RMS est correct.
On filtre, puis on egalise le niveau dans la bande ou l'oreille travaille (200 Hz - 8 kHz).
"""
import subprocess
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]  (image sa3-sfx)
import soundfile as sf  # pyright: ignore[reportMissingImports]

SRC = Path("/data/out23_loop")
DST = Path("/data/out23_fix")
RATE = 24000
CIBLE_RMS = 0.045          # RMS vise DANS la bande 200 Hz - 8 kHz
HP = 90                    # frequence de coupure du passe-haut

DST.mkdir(exist_ok=True)


def charge(p):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", str(RATE),
         "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768


def rms_bande(x, lo=200, hi=8000):
    X = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / RATE)
    m = (fr >= lo) & (fr < hi)
    # Parseval : puissance d'une bande = 2 * somme(X) / N^2 (spectre unilateral)
    n = len(x)
    return float(np.sqrt(2.0 * X[m].sum() / (n * n)))


for p in sorted(SRC.glob("*.bed.wav")):
    tmp = DST / (p.stem + ".hp.wav")
    # passe-haut 2 pôles en cascade = 24 dB/octave, plus un leger boost de presence
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(p),
         "-af", f"highpass=f={HP}:poles=2,highpass=f={HP}:poles=2,"
                f"equalizer=f=300:t=q:w=1.0:g=3",
         "-ac", "1", "-ar", str(RATE), str(tmp)], check=True)
    x = charge(tmp)
    if len(x) == 0:
        print(f"{p.name:<32} VIDE")
        continue
    r = rms_bande(x)
    gain = CIBLE_RMS / max(r, 1e-6)
    y = np.clip(x * gain, -0.89, 0.89)
    out = DST / p.name
    sf.write(out, y, RATE, subtype="PCM_16")
    # controle
    X = np.abs(np.fft.rfft(y)) ** 2
    fr = np.fft.rfftfreq(len(y), 1 / RATE)
    tot = X.sum() + 1e-20
    s60 = 100 * X[fr < 60].sum() / tot
    s_par = 100 * X[(fr >= 200) & (fr < 2000)].sum() / tot
    print(f"{p.name:<32} gain {20*np.log10(gain):+5.1f} dB | <60Hz {s60:4.0f}% | 200-2k {s_par:3.0f}%")
    tmp.unlink(missing_ok=True)
