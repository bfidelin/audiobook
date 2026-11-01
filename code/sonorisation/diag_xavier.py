"""Diagnostic de la voix XAVIER."""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/calmos/alexandria-audiobook")


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path.name} illisible : {exc}") from exc


vc = read(ROOT / "runtime-data" / "voice_config.json")
v = vc["XAVIER"]
ref = v["ref_audio"]
print("ref_audio :", ref)
print("  fichier present :", (ROOT / "data" / ref).exists())
print("  ref_text :", len(v.get("ref_text", "")), "caracteres")
print("  seed :", v.get("seed"), "| style :", (v.get("character_style") or "")[:50])

chunks = read(ROOT / "runtime-data" / "chunks.json")
x = [c for c in chunks if c.get("speaker") == "XAVIER"]
print("\nrepliques XAVIER :", len(x))
for ins, n in Counter(c.get("instruct", "") for c in x).most_common(4):
    print(f"  {n:3d} x {ins[:80]}")

print("\nfichiers xavier* disponibles :")
for f in sorted((ROOT / "data" / "designed_voices").glob("xavier*")):
    print(f"  {f.stat().st_size:8d} o  {f.name}")
