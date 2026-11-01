"""Repare la voix XAVIER : reference valide + registre sobre."""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/calmos/alexandria-audiobook")
RUNTIME = ROOT / "runtime-data"
INSTRUCT = "Low, tired, slightly raspy. Dry, understated delivery."


def parse(text: str, label: str):
    try:
        return json.loads(text)
    except ValueError as exc:
        raise SystemExit(f"{label} illisible : {exc}") from exc


vc_path = RUNTIME / "voice_config.json"
shutil.copy2(vc_path, vc_path.with_suffix(".json.bak-xavier"))
try:
    vc = parse(vc_path.read_text(encoding="utf-8"), "voice_config.json")
except OSError as exc:
    raise SystemExit(f"lecture impossible : {exc}") from exc
avant = vc["XAVIER"]["ref_audio"]
vc["XAVIER"]["ref_audio"] = "designed_voices/xavier_1784633447.wav"
vc_path.write_text(json.dumps(vc, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"ref_audio : {avant}\n         -> {vc['XAVIER']['ref_audio']}")

chunks_path = RUNTIME / "chunks.json"
raw = subprocess.run(["sudo", "-n", "cat", str(chunks_path)],
                     capture_output=True, text=True, check=True).stdout
chunks = parse(raw, "chunks.json")
n = 0
for c in chunks:
    if c.get("speaker") == "XAVIER" and c.get("instruct") != INSTRUCT:
        c["instruct"] = INSTRUCT
        n += 1
tmp = Path("/home/calmos/ambience-gen/chunks_xavier.json")
tmp.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
subprocess.run(["sudo", "-n", "cp", str(tmp), str(chunks_path)], check=True)
subprocess.run(["sudo", "-n", "chown", "calmos:calmos", str(chunks_path)], check=True)
print(f"{n} indications de jeu remplacees par : {INSTRUCT}")
