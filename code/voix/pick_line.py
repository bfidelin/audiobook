"""Choisit une phrase du narrateur pour le test A/B."""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
from pathlib import Path

path = Path("/home/calmos/alexandria-audiobook/runtime-data/chunks.json")
try:
    chunks = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"chunks.json illisible : {exc}") from exc

matches = [c["text"] for c in chunks
           if c["speaker"] == "NARRATOR" and 180 < len(c["text"]) < 240]
print(matches[3] if len(matches) > 3 else matches[0])
