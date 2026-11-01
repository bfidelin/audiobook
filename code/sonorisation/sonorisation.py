"""Sonorise un chapitre : pose des fonds sonores sous la narration avec ducking.

Le mapping est piloté par le texte : chaque réplique est étiquetée par mots-clés
(rue, bar, appartement, musique...), les étiquettes consécutives sont fusionnées
en « plages », et une ambiance est posée sur chaque plage assez longue.

Deux passes ffmpeg, pour garder des graphes de filtres bornés :
  1. piste d'ambiance = concaténation de [silence, fond tronqué, silence, ...]
  2. mixage voix + ambiance, l'ambiance compressée en sidechain par la voix

Usage :
  ./sonorisation.py --chapter 4 --dry-run
  ./sonorisation.py --chapter 4 --out /home/calmos/alexandria-bench/partie4.mp3
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/calmos/alexandria-audiobook")
BEDS = Path("/home/calmos/ambience-gen/out")
BITRATE = 32_000
PART_SECONDS = 1200.0

# mot-cle -> fichier d'ambiance
RULES = [
    (r"\b(rue|trottoir|boulevard|extérieur|exterieur|dehors|quartier)\w*", "02_rue_paris_nuit"),
    (r"\b(bar|bistrot|verre|bière|biere|comptoir|juke|café|cafe)\w*", "12_bar_nuit"),
    (r"\b(fête|fete|soirée|soiree|teuf|dancefloor|dance|boucan|rave)\w*", "04_squat_fete"),
    (r"\b(magasin|boutique|rayon|caissier|vendeur|vitrine)\w*", "06_magasin_disques"),
    (r"\b(musique|chanson|concert|groupe|enceinte|casque|guitare)\w*", "04_squat_fete"),
    (r"\b(métro|metro|station|quai|gare|rame)\w*", "07_metro_quai"),
    (r"\b(voiture|bagnole|moteur|klaxon|taxi|bus|scooter|moto|autoroute)\w*", "08_voiture_ville"),
    (r"\b(hôpital|hopital|urgences|infirmière|infirmiere|patient|couloir)\w*", "09_hopital_couloir"),
    (r"\b(pluie|orage|tonnerre|averse)\w*", "13_pluie_fenetre"),
    (r"\b(télé|tele|écran|ecran|radio|journal télévisé|info)\w*", "15_television_nuit"),
    (r"\b(appartement|chambre|salon|cuisine|couloir|palier|immeuble)\w*", "01_appartement_vide"),
    (r"\b(squat|cave|entrepôt|entrepot|usine|hangar)\w*", "04_squat_fete"),
]
MIN_RUN_SECONDS = 12.0
MAX_RUN_SECONDS = 60.0   # au-dela, l'ambiance tourne en boucle et fatigue
COOLDOWN = 90.0     # secondes minimales entre deux plages
GAP = 2.5           # silence entre deux ambiances differentes
LEAD = 1.5          # la plage commence un peu avant la première réplique
FADE = 2.0


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"sonorisation: echec ffmpeg\n{proc.stderr[-700:]}")


def load_segments() -> tuple[list[tuple[str, str, float, float]], float]:
    try:
        chunks = json.loads((ROOT / "runtime-data" / "chunks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.exit(f"chunks.json illisible : {exc}")
    out = ROOT / "data" / "output"
    rows, acc = [], 0.0
    for ch in chunks:
        f = out / Path(ch.get("audio_path", "")).name
        dur = f.stat().st_size * 8 / BITRATE if f.exists() else 0.0
        rows.append((ch.get("text", ""), ch.get("speaker", ""), acc, dur))
        acc += dur
    return rows, acc


def label(text: str) -> str | None:
    for pattern, bed in RULES:
        if re.search(pattern, text, re.I):
            return bed
    return None


def establishes(seq: list[tuple[str, str, float, float]], i: int,
               window: float = 20.0) -> bool:
    """Le lieu est installe si les ~20 s qui suivent sont de la narration.

    Un personnage qui evoque un bar ne justifie pas qu'on entende un bar : il
    faut que le recit decrive l'endroit, sans dialogue.
    """
    t0 = seq[i][2]
    seen = 0
    for _text, speaker, s, _d in seq[i:]:
        if s - t0 > window:
            break
        if speaker != "NARRATOR":
            return False
        seen += 1
    return seen >= 2


def build_runs(rows: list[tuple[str, str, float, float]], start: float,
               end: float) -> list[dict]:
    """Pose une ambiance seulement quand le recit installe un lieu.

    Trois garde-fous : narration (pas de dialogue) apres le repere, COOLDOWN
    secondes depuis la fin de la plage precedente, duree plafonnee.
    """
    seq = [(t, sp, s, d) for t, sp, s, d in rows if start <= s < end]
    runs: list[dict] = []
    cur: dict | None = None
    last_end = start - COOLDOWN
    for idx, (text, _speaker, s, d) in enumerate(seq):
        lab = label(text)
        if lab is None:
            continue
        if cur is not None and lab == cur["bed"]:
            cur["end"] = s + d
            cur["n"] += 1
            continue
        if s - last_end < COOLDOWN or not establishes(seq, idx):
            continue
        if cur is not None:
            runs.append(cur)
            last_end = cur["end"]
        cur = {"bed": lab, "start": max(start, s - LEAD), "end": s + d, "n": 1}
    if cur is not None:
        runs.append(cur)

    merged: list[dict] = []
    for r in runs:
        if merged and r["end"] - r["start"] < MIN_RUN_SECONDS:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(r)
    for r in merged:
        if r["end"] - r["start"] > MAX_RUN_SECONDS:
            r["end"] = r["start"] + MAX_RUN_SECONDS
    return [r for r in merged if r["end"] - r["start"] >= MIN_RUN_SECONDS]

def build_bed_track(runs: list[dict], start: float, end: float, workdir: Path) -> Path:
    """Piste d'ambiance de la duree exacte du chapitre.

    Chaque silence est dimensionne au trou reel : un silence de la longueur du
    chapitre, reutilise pour chaque trou, produisait une piste de plusieurs heures.
    """

    def silence_of(seconds: float, tag: str) -> Path:
        path = workdir / f"sil_{tag}.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "anullsrc=r=24000:cl=mono", "-t", f"{seconds:.3f}", str(path)])
        return path

    pieces, cursor = [], start
    for i, r in enumerate(runs):
        gap_start = r["start"] - GAP if i > 0 else start
        if gap_start > cursor:
            pieces.append(silence_of(gap_start - cursor, f"gap{i:02d}"))
        r["start"] = max(r["start"], gap_start)
        dur = round(r["end"] - r["start"], 2)
        bed = workdir / f"bed_{i:03d}.wav"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-stream_loop", "-1", "-i", str(BEDS / f"{r['bed']}.bed.wav"),
             "-t", f"{dur:.3f}", "-ac", "1", "-ar", "24000",
             "-af", f"afade=t=in:st=0:d={FADE},"
                    f"afade=t=out:st={max(0.0, dur - FADE):.2f}:d={FADE}",
             str(bed)])
        pieces.append(bed)
        cursor = r["end"]
    if cursor < end:
        pieces.append(silence_of(end - cursor, "tail"))

    listing = workdir / "beds.txt"
    listing.write_text("\n".join(f"file '{p}'" for p in pieces) + "\n", encoding="utf-8")
    track = workdir / "bed_track.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", str(listing), "-ac", "1", "-ar", "24000",
         "-t", f"{end - start:.3f}", str(track)])
    return track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", type=int, required=True, help="numero de partie (1-28)")
    ap.add_argument("--out", help="fichier de sortie (mp3 ou m4a)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start = (args.chapter - 1) * PART_SECONDS
    end = start + PART_SECONDS
    rows, total = load_segments()
    if end > total:
        end = total
    runs = build_runs(rows, start, end)

    print(f"partie {args.chapter} : {start / 60:.1f} min -> {end / 60:.1f} min "
          f"({(end - start) / 60:.1f} min de narration)")
    print(f"{len(runs)} plages sonorisees :")
    for r in runs:
        print(f"  {r['start'] - start:7.1f}s -> {r['end'] - start:7.1f}s "
              f"({r['end'] - r['start']:5.1f}s, {r['n']:2d} repliques)  {r['bed']}")
    if args.dry_run:
        return

    workdir = Path(tempfile.mkdtemp(prefix="sono-"))
    voice = workdir / "voice.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", str(ROOT / "runtime-data" / "audiobook.m4b"),
         "-ac", "1", "-ar", "24000", str(voice)])

    track = build_bed_track(runs, start, end, workdir)
    dry = workdir / "mix.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(voice), "-i", str(track),
         "-filter_complex",
         "[1:a][0:a]sidechaincompress=threshold=0.02:ratio=8:attack=200:release=800[bed];"
         "[0:a][bed]amix=inputs=2:normalize=0:duration=first,"
         "loudnorm=I=-23:TP=-3:LRA=11[mix]",
         "-map", "[mix]", "-ac", "1", "-ar", "24000", str(dry)])

    out = Path(args.out) if args.out else workdir / "mix.mp3"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(dry),
         "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "96k", str(out)])
    size = out.stat().st_size / 1e6
    print(f"\nsortie : {out} ({size:.1f} Mo)")
    print(f"travail : {workdir}")


if __name__ == "__main__":
    main()
