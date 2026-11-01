"""Remixe les 30 parties avec le mixeur numpy (fonds -23 LUFS, ducking -4 dB),
puis assemble le M4B chapitre.

Une partie coute ~1 s : les 30 se refont en une minute au lieu de 8.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


OUT = Path("/out")
BEDS = Path("/data/out23_real")


def sh(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg: {proc.stderr[-400:]}")


def main() -> None:
    import mix_events as M
    import sonorisation as S
    src = S.ROOT / "runtime-data" / "audiobook.m4b"
    OUT.mkdir(parents=True, exist_ok=True)
    rows, _ = S.load_segments()
    total = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip())
    parts = int(total // S.PART_SECONDS) + 1
    print(f"{total / 3600:.2f} h, {parts} parties", flush=True)

    produced = []
    for part in range(1, parts + 1):
        start = (part - 1) * S.PART_SECONDS
        end = min(start + S.PART_SECONDS, total)
        out = OUT / f"p{part:02d}.wav"
        voice = Path(tempfile.mkdtemp(prefix=f"mx{part:02d}_")) / "v.wav"
        sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
            "-ac", "1", "-ar", "24000", str(voice)])
        v = M.load(voice)

        bed = np.zeros(len(v), dtype=np.float32)
        runs = S.build_runs(rows, start, end)
        for r in runs:
            clip = M.load(BEDS / f"{r['bed']}.bed.wav")
            M.place(bed, clip, r["start"] - start, 0.0, fade=2.0)
        if bed.any():
            bed *= M.duck_curve(v)

        mix = v + bed
        peak = float(np.max(np.abs(mix)))
        if peak > 0.89:
            mix *= 0.89 / peak
        sf.write(out, mix, M.RATE, subtype="PCM_16")
        voice.unlink(missing_ok=True)
        cov = sum(r["end"] - r["start"] for r in runs)
        print(f"  partie {part:2d}/{parts} : {len(runs)} plages, {cov / 60:.1f} min", flush=True)
        produced.append(out)

    meta = OUT / "ch.txt"
    lines, acc = [";FFMETADATA1"], 0.0
    for i, wav in enumerate(produced, start=1):
        d = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(wav)], capture_output=True, text=True).stdout.strip())
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(acc * 1000)}",
                  f"END={int((acc + d) * 1000)}", f"title=Partie {i}"]
        acc += d
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

    listing = OUT / "list.txt"
    listing.write_text("\n".join(f"file '{p}'" for p in produced) + "\n", encoding="utf-8")
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
        "-safe", "0", "-i", str(listing), "-i", str(meta), "-map_metadata", "1",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "24000",
        "-movflags", "+faststart", str(OUT / "audiobook_sonorise_v2.m4b")])
    print(f"\nM4B : {acc / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
