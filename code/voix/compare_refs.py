"""Compare les references actuelles d'Alexandria aux nouvelles extraites."""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import subprocess
from pathlib import Path

import voicefix

ROOT = Path("/home/calmos/alexandria-audiobook")
try:
    vc = json.loads((ROOT / "runtime-data" / "voice_config.json").read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"voice_config.json illisible : {exc}") from exc
NEW = {
    "NARRATOR": Path("refs/narrateur_proust.wav"),
    "VERNON": Path("refs/vernon_test.wav"),
    "LA HYÈNE": Path("refs/la_hyene_test.wav"),
}


def dur(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


print(f"{'voix':10s} {'source':12s} {'durée':>7s} {'F0':>6s} {'voisé':>6s}")
print("-" * 48)
for voice, new in NEW.items():
    old_rel = vc.get(voice, {}).get("ref_audio", "")
    old = None
    for base in (ROOT / "data", ROOT / "runtime-data", ROOT):
        cand = base / old_rel
        if cand.exists():
            old = cand
            break
    for label, path in (("actuelle", old), ("nouvelle", new)):
        if path is None or not path.exists():
            print(f"{voice:10s} {label:12s} {'absente':>7s}")
            continue
        f0, voiced = voicefix.pitch_stats(path)
        genre = "H" if 0 < f0 < 165 else "F"
        print(f"{voice:10s} {label:12s} {dur(path):6.2f}s {f0:5.0f}Hz {voiced * 100:5.0f}%  {genre}")
    print()
