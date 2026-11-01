"""sfxcheck — controle generique de qualite des effets sonores.

Au lieu d'un jugement a l'oreille (impossible en automatique) ou de regles
tapees pour un son precis, on mesure des descripteurs normalises et on les
confronte a un *archetype* :

  event    un impact isole (porte, claquement, coup)
  cadence  une repetition (sonnerie, alarme, pas)
  ambience un lieu continu (rue, bar, couloir, pluie)
  texture  un bruit large sans structure (souffle, friture)

Chaque archetype definit des bornes ; le fichier passe ou echoue, avec la
raison. C'est ce qui permet de valider n'importe quelle generation sans
ecouter, et de comparer plusieurs variantes.

Usage :
  ./sfxcheck.py check fichier1.wav fichier2.wav
  ./sfxcheck.py check fichier.wav --type cadence --json
  ./sfxcheck.py compare var1.wav var2.wav --type cadence
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# bornes par archetype : (min, max) ou None si non contraignant
ARCHETYPES: dict[str, dict[str, tuple[float | None, float | None]]] = {
    "event": {
        "duration": (0.4, 8.0),
        "onsets": (0.8, 60.0),          # peu d'attaques : un impact, pas un flux
        "periodicity": (None, 0.45),    # pas de repetition marquee
        "silence_ratio": (0.05, 0.95),
        "flatness": (0.02, 0.85),
    },
    "cadence": {
        "duration": (1.5, 30.0),
        "periodicity": (0.25, 1.0),     # une repetition nette
        "lag_ms": (250.0, 4000.0),      # dans une plage plausible
        "tonal_dominance": (None, 0.35),  # pas un bourdonnement pur
        "silence_ratio": (0.08, 0.95),
    },
    "ambience": {
        "duration": (2.0, 600.0),
        "tonal_dominance": (None, 0.20),  # pas de raie : ce serait un sifflement
        "flatness": (0.10, 0.90),
        "modulation": (0.05, 0.60),
        "centroid_hz": (200.0, 5000.0),
    },
    "texture": {
        "duration": (1.5, 600.0),
        "tonal_dominance": (None, 0.15),
        "modulation": (None, 0.25),
    },
}

# verifications communes a tous les archetypes
COMMON = {
    "peak_dbfs": (None, -0.5),
    "dc_offset": (None, 0.02),
    "clipping_ratio": (None, 0.001),
}


def features(path: Path) -> dict[str, float]:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr

    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    energy = spec ** 2
    total = energy.sum() + 1e-12
    centroid = float((freqs * energy).sum() / total)
    flatness = float(np.exp(np.mean(np.log(spec + 1e-10))) / (np.mean(spec) + 1e-10))
    tonal = float(energy.max() / total)

    hop = int(0.02 * sr)
    env = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2))
                    for i in range(0, max(hop, len(x) - hop), hop)])
    silence_ratio = float(np.mean(env < 10 ** (-50 / 20)))

    centered = env - env.mean()
    ac = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = 5, min(len(ac) - 1, int(4.0 / 0.02))
    lag = int(np.argmax(ac[lo:hi]) + lo) if hi > lo else 0
    periodicity = float(ac[lag]) if hi > lo else 0.0

    es = np.abs(np.fft.rfft(centered))
    ef = np.fft.rfftfreq(len(centered), hop / sr)
    band = (ef >= 3) & (ef <= 8)
    modulation = float(es[band].sum() / (es.sum() + 1e-12))

    # attaques : montees d'energie franches
    diff = np.diff(env)
    onsets = float(np.sum(diff > 0.25 * (diff.max() + 1e-9)) / max(dur, 1e-6))

    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return {
        "duration": dur,
        "peak_dbfs": 20 * np.log10(peak + 1e-12),
        "dc_offset": abs(float(np.mean(x))),
        "clipping_ratio": float(np.mean(np.abs(x) > 0.999)),
        "centroid_hz": centroid,
        "flatness": flatness,
        "tonal_dominance": tonal,
        "silence_ratio": silence_ratio,
        "periodicity": periodicity,
        "lag_ms": lag * 20.0,
        "modulation": modulation,
        "onsets": onsets,
    }


def infer_type(f: dict[str, float]) -> str:
    # une repetition forte suffit, meme si la periode sort de la fenetre :
    # une sonnerie de 10 s peut avoir une autocorrelation a 4 s et plus.
    if f["periodicity"] >= 0.5 and f["silence_ratio"] >= 0.10:
        return "cadence"
    if f["periodicity"] >= 0.3 and 250 <= f["lag_ms"] <= 4000:
        return "cadence"
    if f["onsets"] >= 1.5 and f["silence_ratio"] > 0.25:
        return "event"
    if f["modulation"] >= 0.05 and f["tonal_dominance"] < 0.20:
        return "ambience"
    return "texture"


def check(f: dict[str, float], kind: str) -> list[tuple[str, bool, str]]:
    rules = {**COMMON, **ARCHETYPES[kind]}
    out = []
    names = {
        "duration": ("duree", "s"),
        "peak_dbfs": ("crete", "dBFS"),
        "dc_offset": ("offset DC", ""),
        "clipping_ratio": ("ecretage", ""),
        "centroid_hz": ("centroide", "Hz"),
        "flatness": ("flatness", ""),
        "tonal_dominance": ("raie dominante", ""),
        "silence_ratio": ("silences", ""),
        "periodicity": ("periodicite", ""),
        "lag_ms": ("cadence", "ms"),
        "modulation": ("modulation 3-8 Hz", ""),
        "onsets": ("attaques/s", ""),
    }
    for key, bounds in rules.items():
        if bounds is None:
            continue
        low, high = bounds
        value = f.get(key)
        if value is None:
            continue
        ok = (low is None or value >= low) and (high is None or value <= high)
        label, unit = names.get(key, (key, ""))
        limit = f"[{'' if low is None else f'{low:g}'} … {'' if high is None else f'{high:g}'}]"
        out.append((f"{label} {value:.3g}{unit}", ok, limit))
    return out


def report(path: Path, forced: str | None, as_json: bool) -> bool:
    f = features(path)
    kind = forced or infer_type(f)
    checks = check(f, kind)
    passed = all(ok for _, ok, _ in checks)
    if as_json:
        print(json.dumps({"file": str(path), "type": kind, "passed": passed,
                          "features": {k: round(v, 4) for k, v in f.items()},
                          "checks": [{"mesure": m, "ok": o, "attendu": a}
                                     for m, o, a in checks]},
                         ensure_ascii=False))
        return passed
    print(f"\n{path.name}  →  archetype « {kind} »  {'PASS' if passed else 'ECHEC'}")
    for mesure, ok, attendu in checks:
        print(f"   {'ok  ' if ok else 'KO  '} {mesure:28s} attendu {attendu}")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "compare"):
        p = sub.add_parser(name)
        p.add_argument("files", nargs="+")
        p.add_argument("--type", choices=list(ARCHETYPES), help="forcer un archetype")
        p.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = []
    for name in args.files:
        path = Path(name)
        if not path.exists():
            print(f"absent : {path}", file=sys.stderr)
            continue
        try:
            results.append((path, report(path, args.type, args.json)))
        except (OSError, ValueError) as exc:
            print(f"illisible {path} : {exc}", file=sys.stderr)

    if args.cmd == "compare" and not args.json:
        print("\nclassement :")
        for path, ok in sorted(results, key=lambda r: not r[1]):
            print(f"  {'PASS' if ok else 'ECHEC'}  {path.name}")
    sys.exit(0 if results and all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
