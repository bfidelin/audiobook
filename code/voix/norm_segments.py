"""Ramene chaque segment a -18 LUFS / -1.5 dBTP, sinon un personnage parle
8 dB plus fort qu'un autre. Ecrit dans un fichier temporaire puis renomme."""
import os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path("/home/calmos/alexandria-audiobook/data/output")
TARGET = -18.0

def norm(p: Path) -> str:
    tmp = p.with_suffix(".tmp.mp3")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(p),
                        "-af", f"loudnorm=I={TARGET}:TP=-1.5:LRA=11",
                        "-c:a", "libmp3lame", "-b:a", "64k", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"ECHEC {p.name}"
    tmp.replace(p)
    return ""

files = sorted(SRC.glob("*.mp3"))
print(f"{len(files)} segments", flush=True)
ok = bad = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    for i, res in enumerate(ex.map(norm, files), 1):
        if res: bad += 1
        else: ok += 1
        if i % 500 == 0:
            print(f"  {i}/{len(files)} — {bad} echecs", flush=True)
print(f"termine : {ok} normalises, {bad} echecs", flush=True)
