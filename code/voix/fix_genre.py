# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Regenere une reference VoiceDesign jusqu'a ce que le genre corresponde.

Le modele de conception de voix inverse parfois le sexe demande : Daniel est
ressorti a 250 Hz pour un homme, Marie-Ange a 103 Hz pour une femme. On tire
plusieurs variantes et on garde la premiere dont la frequence fondamentale
tombe dans la plage attendue.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path("/home/calmos/alexandria-audiobook")
CFG = ROOT / "runtime-data" / "voice_config.json"
PREVIEWS = ROOT / "data" / "designed_voices" / "previews"
API = "http://localhost:4200/api/voice_design/preview"

SAMPLE = (
    "C'est alors qu'il comprit que la nuit ne finirait pas de sitôt. Il resta "
    "immobile un long moment, à écouter le silence de l'appartement, le "
    "claquement lointain d'une porte, le passage d'une voiture dans la rue. "
    "Rien ne pressait. Plus rien ne pressait, désormais. Il avait tout son "
    "temps, et cette idée, loin de le rassurer, lui serrait la poitrine."
)

CIBLES = {
    "AMELIE": ("femme", "Une voix de FEMME française d'une trentaine d'années. "
                        "Timbre féminin clair et chaleureux, débit naturel.", 155, 400),
    "DANIEL": ("homme", "Une voix d'HOMME français de 45 ans. Timbre masculin "
                        "médium, posé et réfléchi, débit régulier.", 70, 165),
    "JULIEN": ("homme", "Une voix d'HOMME français de 30 ans. Timbre masculin "
                        "jeune et neutre, débit naturel.", 70, 165),
    "MARIE-ANGE": ("femme", "Une voix de FEMME française de 70 ans. Timbre "
                            "féminin âgé, un peu chevrotant, débit lent.", 155, 400),
}


def f0(path: Path) -> float:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
                          "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(0.05 * 16000)
    vals = []
    for i in range(0, len(x) - n, n // 2):
        w = x[i:i + n]
        if np.sqrt((w ** 2).mean()) < 0.008:
            continue
        w = w - w.mean()
        c = np.correlate(w, w, "full")[n - 1:]
        lo, hi = int(16000 / 400), int(16000 / 70)
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.35 * c[0]:
            vals.append(16000 / k)
    return float(np.median(vals)) if vals else 0.0


def design(desc: str) -> Path:
    body = json.dumps({"description": desc, "sample_text": SAMPLE,
                       "language": "french"}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:  # noqa: S310
            rep = json.loads(r.read())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"appel API impossible : {e}") from e
    return PREVIEWS / Path(rep["audio_url"]).name


def main() -> None:
    try:
        cfg = json.loads(CFG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"voice_config illisible : {e}")
    for voix, (_genre, desc, lo, hi) in CIBLES.items():
        for essai in range(1, 5):
            wav = design(desc)
            f = f0(wav)
            ok = lo <= f <= hi
            print(f"{voix:<11} essai {essai} : {f:5.0f} Hz -> "
                  f"{'OK' if ok else 'rejete'}", flush=True)
            if ok:
                wav.with_suffix(".txt").write_text(SAMPLE, encoding="utf-8")
                cfg[voix]["ref_audio"] = f"designed_voices/previews/{wav.name}"
                cfg[voix]["ref_text"] = SAMPLE
                cfg[voix]["description"] = desc
                CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                break
        else:
            print(f"{voix:<11} aucun tirage valide en 4 essais", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
