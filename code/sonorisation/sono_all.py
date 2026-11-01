"""Sonorise les 28 parties du livre, une par une, en series bornees.

Aucune parallelisation : un seul ffmpeg a la fois, pour ne pas saturer la
machine (les gels precedents venaient de traitements massivement paralleles).
Produit 28 WAV, puis un M4B chapitre.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import sonorisation as S

OUT = Path("/out")
PARTS = 0  # calcule dans main() depuis la duree reelle


def loadavg() -> float:
    try:
        with open("/proc/loadavg", encoding="utf-8") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError):
        return 0.0


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg: {proc.stderr[-500:]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, total = S.load_segments()
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(S.ROOT / "runtime-data" / "audiobook.m4b")], capture_output=True, text=True).stdout.strip())
    parts = int(total // S.PART_SECONDS) + 1
    print(f"{parts} parties a traiter", flush=True)
    print(f"{total / 3600:.2f} h de narration, {len(rows)} segments", flush=True)

    produced = []
    for part in range(1, parts + 1):
        start = (part - 1) * S.PART_SECONDS
        end = min(start + S.PART_SECONDS, total)
        out = OUT / f"partie_{part:02d}.wav"

        # garde-fou : on attend que la machine respire
        while loadavg() > 8.0:
            print(f"  charge {loadavg():.1f}, pause de 60 s", flush=True)
            time.sleep(60)

        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"[skip] partie {part}", flush=True)
            produced.append(out)
            continue

        t0 = time.time()
        runs = S.build_runs(rows, start, end)
        if not runs:
            print(f"  partie {part} : aucune plage, copie de la voix seule", flush=True)

        work = Path(tempfile.mkdtemp(prefix=f"sono_part{part:02d}_"))
        voice = work / "voice.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-i", str(S.ROOT / "runtime-data" / "audiobook.m4b"),
             "-ac", "1", "-ar", "24000", str(voice)])

        if runs:
            track = S.build_bed_track(runs, start, end, work)
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(voice), "-i", str(track), "-filter_complex",
                 "[1:a][0:a]sidechaincompress=threshold=0.02:ratio=8:"
                 "attack=200:release=800[bed];"
                 "[0:a][bed]amix=inputs=2:normalize=0:duration=first,"
                 "loudnorm=I=-23:TP=-3:LRA=11[mix]",
                 "-map", "[mix]", "-ac", "1", "-ar", "24000", str(out)])
        else:
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(voice), "-af", "loudnorm=I=-23:TP=-3:LRA=11",
                 "-ac", "1", "-ar", "24000", str(out)])

        cover = sum(r["end"] - r["start"] for r in runs)
        print(f"partie {part:2d}/{parts} : {len(runs)} plages, "
              f"{cover / 60:.1f} min sonorisees, {time.time() - t0:.0f} s "
              f"(load {loadavg():.1f})", flush=True)
        produced.append(out)

    # chapitrage + assemblage
    meta = OUT / "chapters.txt"
    lines = [";FFMETADATA1"]
    acc = 0.0
    for i, wav in enumerate(produced, start=1):
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav)], capture_output=True, text=True).stdout.strip())
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(acc * 1000)}",
                  f"END={int((acc + dur) * 1000)}", f"title=Partie {i}"]
        acc += dur
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

    listing = OUT / "parts.txt"
    listing.write_text("\n".join(f"file '{p}'" for p in produced) + "\n", encoding="utf-8")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", str(listing), "-i", str(meta), "-map_metadata", "1",
         "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "24000",
         "-movflags", "+faststart", str(OUT / "audiobook_sonorise.m4b")])
    print(f"\nM4B : {OUT / 'audiobook_sonorise.m4b'} ({acc / 3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
