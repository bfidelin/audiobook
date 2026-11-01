"""Analyse objective d'un effet sonore : periodicity, centroide, modulation.

Un telephone doit montrer une cadence (autocorrelation de l'enveloppe forte) et
des pics spectraux etroits. Un bar doit montrer une modulation de parole dans la
bande 3-8 Hz et un spectre large (flatness elevee).
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def analyse(path: Path) -> None:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr

    # spectre global
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    energy = spec ** 2
    total = energy.sum() + 1e-12
    centroid = float((freqs * energy).sum() / total)
    flatness = float(np.exp(np.mean(np.log(spec + 1e-10))) / (np.mean(spec) + 1e-10))
    top = np.argsort(spec)[-5:][::-1]
    peaks = [(round(float(freqs[i])), round(float(spec[i] / (spec.max() + 1e-12)), 2))
             for i in top]

    # enveloppe 20 ms puis autocorrelation -> cadence
    hop = int(0.02 * sr)
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2))
                    for i in range(0, len(x) - hop, hop)])
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = 3, min(len(ac) - 1, 60)  # 60 ms a 1.2 s
    peak_lag = int(np.argmax(ac[lo:hi]) + lo) if hi > lo else 0
    periodicity = float(ac[peak_lag]) if hi > lo else 0.0

    # modulation de parole : energie du spectre d'enveloppe dans 3-8 Hz
    es = np.abs(np.fft.rfft(env - env.mean()))
    ef = np.fft.rfftfreq(len(env), hop / sr)
    band = (ef >= 3) & (ef <= 8)
    speech_mod = float(es[band].sum() / (es.sum() + 1e-12))

    rms = float(np.sqrt(np.mean(x ** 2)))
    print(f"\n{path.name}  ({dur:.1f} s, {sr} Hz)")
    print(f"  RMS {20 * np.log10(rms + 1e-12):6.1f} dBFS | centroide {centroid:6.0f} Hz | "
          f"flatness {flatness:.3f}")
    print(f"  periodicite enveloppe {periodicity:.2f} (cadence ~{peak_lag * 20} ms)")
    print(f"  modulation 3-8 Hz {speech_mod:.2f} (parole : > 0,15)")
    print(f"  pics spectraux {peaks}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        analyse(Path(arg))
