"""Controle qualite apres re-rendu : duree par segment vs longueur du texte.

La duree est deduite de la taille du fichier (MP3 CBR 32 kb/s) pour eviter
4 162 appels ffprobe, qui depassent le temps d'execution utile.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import statistics
from pathlib import Path

ROOT = Path("/home/calmos/alexandria-audiobook")
BITRATE = 32_000  # bits/s, segments exportes en MP3 CBR 32k mono

try:
    chunks = json.loads((ROOT / "runtime-data" / "chunks.json").read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"chunks.json illisible : {exc}") from exc

out = ROOT / "data" / "output"
rows = []
missing = 0
for ch in chunks:
    if ch.get("speaker") != "NARRATOR" or not ch.get("audio_path"):
        continue
    f = out / Path(ch["audio_path"]).name
    if not f.exists():
        missing += 1
        continue
    dur = f.stat().st_size * 8 / BITRATE
    rows.append((ch.get("id", -1), dur, len(ch["text"])))

print(f"segments narrateur : {len(rows)} | audio manquant : {missing}")
rates = [r[2] / r[1] for r in rows if r[1] > 0.2]
print(f"caracteres/seconde : median {statistics.median(rates):.1f} | "
      f"min {min(rates):.1f} | max {max(rates):.1f}")
print(f"duree totale narrateur : {sum(r[1] for r in rows) / 3600:.2f} h")

suspects = sorted((r for r in rows if r[1] < 0.6 or r[2] / r[1] > 30), key=lambda r: r[1])
print(f"segments suspects (trop courts pour leur texte) : {len(suspects)}")
for r in suspects[:12]:
    print(f"  id={r[0]:5d} {r[1]:5.2f}s pour {r[2]:4d} car ({r[2] / r[1]:5.1f} c/s)")
