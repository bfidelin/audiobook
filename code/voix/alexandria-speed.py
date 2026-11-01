#!/usr/bin/env python3
"""alexandria-speed — accelere l'encodage et la generation d'Alexandria Audiobook.

Sous-commandes :

  encode   Reconstruit M4B / MP3 / Opus depuis les segments deja generes en un
           seul passage ffmpeg (au lieu de la fusion pydub). Debit et
           chapitrage corriges.
  bench    Compare le moteur actuel (qwen-tts, batch) a Faster Qwen3-TTS
           (CUDA graphs, batch=1) sur des chunks reels du projet. GPU requis.
  patch    Applique les optimisations dans le code de l'app (segments sans
           perte, fusion ffmpeg, debit M4B). Dry-run par defaut.

Exemples :
  ./alexandria-speed.py encode --m4b 64k --mp3 64k
  ./alexandria-speed.py encode --limit 400 --m4b 48k --out-dir /tmp/test
  ./alexandria-speed.py bench --samples 16
  ./alexandria-speed.py patch --apply merge-ffmpeg --no-dry-run
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, NoReturn

DEFAULT_ROOT = Path(os.environ.get("ALEX_ROOT", "/home/calmos/alexandria-audiobook"))
SRC_RATE = 24000  # frequence de sortie de Qwen3-TTS 12 Hz
TOOLS_DIR = Path(__file__).resolve().parent


def die(msg: str) -> NoReturn:
    sys.exit(f"alexandria-speed: {msg}")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        if default is not None:
            return default
        die(f"{path} illisible: {exc}")


def run(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and proc.returncode != 0:
        die(f"commande echouee ({cmd[0]}):\n{proc.stderr[-600:]}")
    return proc


def run_stream(cmd: list[str], timeout: int | None = None) -> None:
    """Execute en laissant stdout/stderr aller au terminal (sortie des benches)."""
    proc = subprocess.run(cmd, timeout=timeout)
    if proc.returncode != 0:
        die(f"commande echouee ({cmd[0]}), code {proc.returncode}")


def human(nbytes: float) -> str:
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} To"


class Project:
    """Vue lecture seule d'un projet Alexandria (runtime-data monte sur /alexandria)."""

    def __init__(self, root: Path):
        self.root = root
        self.runtime = root / "runtime-data"
        self.chunks_path = self.runtime / "chunks.json"
        if not self.chunks_path.exists():
            die(f"{self.chunks_path} introuvable (verifier --root)")
        chunks = read_json(self.chunks_path, [])
        self.chunks: list[dict] = chunks if isinstance(chunks, list) else []
        config = read_json(self.runtime / "config" / "config.json", {})
        if not isinstance(config, dict):
            config = {}
        tts = config.get("tts", {}) if isinstance(config, dict) else {}
        self.pause_ms = int(tts.get("pause_between_speakers_ms", 500))
        self.same_pause_ms = int(tts.get("pause_same_speaker_ms", 250))
        self.segment_dirs = [d for d in (
            self.root / "data" / "output",
            self.runtime / "voicelines",
            self.root / "voicelines",
        ) if d.is_dir()]
        if not self.segment_dirs:
            die("aucun dossier de segments (data/output ou voicelines/)")

    def segment_path(self, chunk: dict) -> Path | None:
        rel = chunk.get("audio_path")
        if not rel:
            return None
        stem = Path(rel).stem
        for d in self.segment_dirs:
            for ext in (".mp3", ".flac", ".wav"):
                cand = d / f"{stem}{ext}"
                if cand.exists():
                    return cand
        return None

    def with_audio(self, limit: int | None = None) -> tuple[list[dict], int]:
        items, missing = [], 0
        for chunk in self.chunks:
            path = self.segment_path(chunk)
            if path is None:
                missing += 1
                continue
            items.append({**chunk, "_path": path})
            if limit and len(items) >= limit:
                return items, missing
        return items, missing


def probe_duration(path: Path) -> float:
    proc = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(path)])
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


# ── encode ───────────────────────────────────────────────────────────────────


def make_silence(workdir: Path, ms: int) -> Path:
    target = workdir / f"sil_{ms}.mp3"
    if not target.exists():
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"anullsrc=r={SRC_RATE}:cl=mono",
             "-t", f"{ms / 1000:.3f}", "-c:a", "libmp3lame", "-b:a", "32k", str(target)])
    return target


def build_concat_lines(project: Project, items: list[dict], workdir: Path) -> list[str]:
    """Lignes de concatenation, pauses inter-segments incluses."""
    lines, prev = [], None
    for chunk in items:
        if prev is not None:
            gap = (project.same_pause_ms if chunk["speaker"] == prev["speaker"]
                   else project.pause_ms)
            override = prev.get("pause_after")
            if override is not None:
                gap = int(override)
            if gap > 0:
                lines.append(f"file '{make_silence(workdir, gap)}'")
        lines.append(f"file '{chunk['_path']}'")
        prev = chunk
    return lines


def encode_parts(lines: list[str], codec: str, bitrate: str, suffix: str,
                 batch_size: int, workdir: Path) -> list[Path]:
    """Encode par tranches : chaque ffmpeg ne voit qu'un lot de fichiers.

    Un seul passage sur des milliers de segments sature l'I/O (mesure : load
    average 38 sur 4 795 fichiers, machine bloquee). Les tranches bornent le
    nombre d'entrees ouvertes simultanement.
    """
    per_batch = max(1, batch_size) * 2  # un segment + sa pause par entree
    parts: list[Path] = []
    total = (len(lines) + per_batch - 1) // per_batch
    for n, start in enumerate(range(0, len(lines), per_batch)):
        chunk = lines[start:start + per_batch]
        part = workdir / f"part_{n:04d}{suffix}"
        listing = workdir / f"part_{n:04d}.txt"
        listing.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        print(f"  tranche {n + 1}/{total} : {len(chunk)} entrees -> {part.name}")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(listing),
             "-ac", "1", "-ar", str(SRC_RATE), "-c:a", codec, "-b:a", bitrate,
             str(part)])
        parts.append(part)
    return parts


def concat_parts(parts: list[Path], suffix: str, workdir: Path) -> Path:
    """Recolle les tranches sans reencodage (-c copy)."""
    listing = workdir / "parts.txt"
    listing.write_text("\n".join(f"file '{p}'" for p in parts) + "\n", encoding="utf-8")
    joined = workdir / f"joined{suffix}"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(joined)])
    return joined


def segment_offsets(items: list[dict], workdir: Path) -> tuple[list[float], float]:
    """Durees reelles de chaque segment (ffprobe parallele) + decalages cumules."""
    listing = workdir / "paths.txt"
    listing.write_text("\n".join(str(i["_path"]) for i in items), encoding="utf-8")
    proc = run(["xargs", "-a", str(listing), "-d", "\n", "-P", "2", "-n", "1",
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0"], check=False)
    durations = []
    for line in proc.stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            durations.append(float(line))
        except ValueError:
            durations.append(0.0)
    if len(durations) != len(items):
        return [], 0.0
    offsets, acc = [], 0.0
    for dur in durations:
        offsets.append(acc)
        acc += dur
    return offsets, acc


def write_chapter_meta(project: Project, items: list[dict], offsets: list[float],
                       total: float, mode: str, workdir: Path) -> Path | None:
    if mode == "none":
        return None
    entries: list[tuple[float, str]] = []
    if mode.startswith("file:"):
        src = Path(mode.split(":", 1)[1])
        if not src.exists():
            die(f"fichier de chapitres introuvable: {src}")
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, title = line.split(None, 1)
            entries.append((float(ts), title))
    elif not offsets:
        print("  (chapitrage ignore : durees de segments indisponibles)")
        return None
    else:
        minutes = float(mode.split(":", 1)[1]) if ":" in mode else 20.0
        step = minutes * 60
        idx, n = 0, 1
        # `idx` avance strictement a chaque tour : la boucle termine toujours.
        while idx < len(items):
            entries.append((offsets[idx], f"Partie {n}"))
            n += 1
            limit = offsets[idx] + step
            idx += 1
            while idx < len(items) and offsets[idx] < limit:
                idx += 1

    lines = [";FFMETADATA1"]
    for i, (start, title) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else total
        safe = title.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;")
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(start * 1000)}", f"END={int(end * 1000)}", f"title={safe}"]
    meta = workdir / "chapters.txt"
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta


def cmd_encode(args: argparse.Namespace) -> None:
    project = Project(Path(args.root))
    items, missing = project.with_audio(limit=args.limit)
    if not items:
        die("aucun segment audio trouve")
    print(f"projet   : {project.root}")
    print(f"segments : {len(items)} avec audio"
          + (f", {missing} ignores (sans audio)" if missing else ""))
    print(f"pauses   : {project.pause_ms} ms (voix differente) / "
          f"{project.same_pause_ms} ms (meme voix)")

    targets = []
    if args.m4b:
        targets.append(("m4b", "aac", args.m4b, ".m4b"))
    if args.mp3:
        targets.append(("mp3", "libmp3lame", args.mp3, ".mp3"))
    if args.opus:
        targets.append(("opus", "libopus", args.opus, ".opus"))
    if not targets:
        die("rien a faire : --m4b 64k / --mp3 64k / --opus 32k")

    out_dir = Path(args.out_dir) if args.out_dir else project.runtime
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="alex-enc-"))
    cover = Path(args.cover) if args.cover else None
    try:
        lines = build_concat_lines(project, items, workdir)
        print(f"entrees  : {len(lines)} lignes de concatenation")
        offsets, total = ([], 0.0)
        meta = None
        if any(t[1] == "aac" and args.chapters != "none" for t in targets):
            print("durees  : mesure des segments...")
            offsets, total = segment_offsets(items, workdir)
            print(f"          {total / 3600:.2f} h d'audio, {len(offsets)} durees")
        for label, codec, bitrate, suffix in targets:
            out = Path(args.out) if (args.out and len(targets) == 1) else out_dir / f"{args.name}{suffix}"
            t0 = time.perf_counter()
            parts = encode_parts(lines, codec, bitrate, suffix, args.batch_size, workdir)
            joined = concat_parts(parts, suffix, workdir)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(joined)]
            cover_idx = 1
            if codec == "aac" and args.chapters != "none":
                meta = write_chapter_meta(project, items, offsets, total, args.chapters, workdir)
            if meta:
                cmd += ["-i", str(meta), "-map_metadata", "1"]
                cover_idx = 2
            if codec == "aac" and cover:
                cmd += ["-i", str(cover), "-map", "0:a", "-map", f"{cover_idx}:v",
                        "-c:v", "copy", "-disposition:v:0", "attached_pic"]
            cmd += ["-c:a", "copy"]
            if codec == "aac":
                cmd += ["-movflags", "+faststart"]
            cmd.append(str(out))
            run(cmd)
            elapsed = time.perf_counter() - t0
            dur = probe_duration(out)
            size = out.stat().st_size
            speed = dur / elapsed if elapsed else 0
            print(f"{label:5s} {bitrate:>4s}  {human(size):>9s}  {dur / 3600:5.2f} h  "
                  f"en {elapsed:5.1f} s ({speed:.0f}x temps reel)  -> {out}")
            for part in parts:
                part.unlink(missing_ok=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── bench ────────────────────────────────────────────────────────────────────


def cmd_bench(args: argparse.Namespace) -> None:
    project = Project(Path(args.root))
    items, _ = project.with_audio()
    sample = [i for i in items
              if i["speaker"] == args.speaker and 200 <= len(i["text"]) < 600][: args.samples]
    if len(sample) < 2:
        die(f"pas assez de chunks '{args.speaker}' exploitables")

    voice_config = read_json(project.runtime / "voice_config.json", {})
    if not isinstance(voice_config, dict):
        die("voice_config.json invalide")
    voice = voice_config.get(args.speaker, {})
    ref_rel, ref_text = voice.get("ref_audio"), voice.get("ref_text")
    if not ref_rel or not ref_text:
        die(f"voix '{args.speaker}' sans ref_audio/ref_text")
    rel = Path(ref_rel)
    candidates = [rel] if rel.is_absolute() else [
        project.root / rel, project.root / "data" / rel, project.runtime / rel,
    ]
    ref = next((c for c in candidates if c.exists()), None)
    if ref is None:
        die(f"audio de reference introuvable, essaye : {[str(c) for c in candidates]}")

    workdir = Path(tempfile.mkdtemp(prefix="alex-bench-"))
    (workdir / "out").mkdir()
    (workdir / "chunks.json").write_text(
        json.dumps([{"text": c["text"]} for c in sample], ensure_ascii=False), encoding="utf-8")
    try:
        shutil.copy(ref, workdir / "ref.wav")
    except OSError as exc:
        die(f"copie de la reference impossible: {exc}")
    for worker in ("bench_current.py", "bench_faster.py"):
        src = TOOLS_DIR / worker
        if not src.exists():
            die(f"{src} manquant (les deux workers doivent etre a cote du script)")
        try:
            shutil.copy(src, workdir / worker)
        except OSError as exc:
            die(f"copie de {src} impossible: {exc}")

    hf = os.environ.get("ALEX_HF_CACHE", "alexandria-audiobook_hf_cache")
    snap = f"/hf/hub/models--Qwen--{args.model}/snapshots"
    common = ["docker", "run", "--rm", "--gpus", "all",
              "-v", f"{hf}:/hf", "-v", f"{workdir}:/data", "-v", f"{workdir}/out:/out",
              "-e", "HF_HOME=/hf", "-e", f"QWEN_MODEL_PATH={snap}/{args.snapshot}",
              "-e", "HF_HUB_OFFLINE=1", "--entrypoint", "python"]

    gpu = run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
               "--format=csv,noheader,nounits"]).stdout.strip()
    try:
        used_mb, total_mb = (int(x) for x in gpu.split(","))
    except ValueError:
        used_mb, total_mb = 0, 0
    print(f"corpus : {len(sample)} chunks {args.speaker} "
          f"({sum(len(c['text']) for c in sample)} caracteres)")
    print(f"GPU    : {used_mb} / {total_mb} MiB utilises")
    if used_mb > total_mb * 0.4:
        print("ATTENTION : le GPU est deja largement occupe, le bench va echouer (OOM).")
        if not args.force:
            die("relancer avec --force pour tenter quand meme")

    print("\n[A] engine actuel — qwen-tts, generation par lots")
    run_stream(common + ["alexandria-audiobook-alexandria", "/data/bench_current.py",
                        "/data/chunks.json", "/data/ref.wav", ref_text,
                        *[str(b) for b in args.batches]], timeout=args.timeout)
    print("\n[B] Faster Qwen3-TTS — CUDA graphs, batch=1")
    run_stream(common + ["alexandria-faster-qwen3tts-test-qwen3tts-faster-test",
                         "/data/bench_faster.py", "/data/chunks.json", "/data/ref.wav", ref_text],
                timeout=args.timeout)
    print(f"\nresultats bruts : {workdir}/out")


# ── patch ────────────────────────────────────────────────────────────────────

SEGMENT_OLD = '''                    mp3_filename = f"{filename_base}.mp3"'''
SEGMENT_NEW = '''                    mp3_filename = f"{filename_base}.flac"'''

PATCHES = {
    "segment-lossless": {
        "file": "app/project.py",
        "why": "segments ecrits en FLAC sans perte au lieu de MP3 32 kb/s "
               "(supprime une generation lossy dans les exports)",
        "edits": [
            (SEGMENT_OLD, SEGMENT_NEW),
            ('                    segment.export(mp3_filepath, format="mp3")',
             '                    segment.export(mp3_filepath, format="flac")'),
        ],
    },
    "m4b-bitrate": {
        "file": "app/project.py",
        "why": "debit AAC du M4B configurable (128k inutile en mono 24 kHz)",
        "edits": [
            ('                "-c:a", "aac",\n                "-b:a", "128k",',
             '                "-c:a", "aac",\n'
             '                "-b:a", os.environ.get("ALEX_M4B_BITRATE", "64k"),'),
        ],
    },
    "merge-ffmpeg": {
        "file": "app/project.py",
        "why": "remplace la fusion pydub par un concat ffmpeg "
               "(~9x plus rapide, RAM constante, un seul encodage)",
        "edits": [
            ('''        final_audio = combine_audio_with_pauses(
            audio_segments, speakers, pause_ms, same_speaker_pause_ms, pause_overrides
        )
        output_filename = "cloned_audiobook.mp3"
        output_path = os.path.join(self.root_dir, output_filename)
        final_audio.export(output_path, format="mp3")
''',
             '''        # Fusion ffmpeg : un seul decodage de tous les segments, un seul encodage.
        import tempfile
        workdir = tempfile.mkdtemp(prefix="alex-merge-")
        try:
            gaps = {pause_ms, same_speaker_pause_ms} | {
                int(p) for p in pause_overrides if p is not None
            }
            silences = {}
            for ms in gaps:
                if ms <= 0:
                    continue
                sfile = os.path.join(workdir, f"sil_{ms}.mp3")
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                     "-t", f"{ms / 1000:.3f}", "-c:a", "libmp3lame", "-b:a", "32k", sfile],
                    check=True,
                )
                silences[ms] = sfile
            lines, prev = [], None
            for chunk, _segment, _start in timeline:
                if prev is not None:
                    gap = (same_speaker_pause_ms if chunk["speaker"] == prev["speaker"]
                           else pause_ms)
                    override = chunk.get("pause_after")
                    if override is not None:
                        gap = int(override)
                    if gap > 0:
                        lines.append("file '%s'" % silences[gap])
                lines.append("file '%s'" % os.path.join(self.root_dir, chunk["audio_path"]))
                prev = chunk
            concat_path = os.path.join(workdir, "concat.txt")
            with open(concat_path, "w", encoding="utf-8") as fh:
                fh.write("\\n".join(lines) + "\\n")
            output_filename = "cloned_audiobook.mp3"
            output_path = os.path.join(self.root_dir, output_filename)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "concat", "-safe", "0", "-i", concat_path,
                 "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame",
                 "-b:a", os.environ.get("ALEX_MP3_BITRATE", "64k"), output_path],
                check=True,
            )
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
'''),
        ],
    },
    "free-warmup-vram": {
        "file": "app/tts.py",
        "why": "libere le modele CustomVoice apres le warmup : il ne sert qu'a "
               "chauffer le GPU mais reste en VRAM (~3,5 Go) et plafonne la "
               "taille de lot du clonage a 10 au lieu de ~15",
        "edits": [
            (
                """        if self._warmup_needed:
            warmup_model = self._init_local_custom()
            print("Running batch warmup generation...")
            self._warmup_model(warmup_model)
            self._warmup_needed = False
""",
                """        if self._warmup_needed:
            warmup_model = self._init_local_custom()
            print("Running batch warmup generation...")
            self._warmup_model(warmup_model)
            self._warmup_needed = False
            # Le CustomVoice ne sert qu'au warmup : le liberer rend ~3,5 Go de
            # VRAM, ce qui fait passer la taille de lot du clonage de 10 a ~15.
            self._local_custom_model = None
            del warmup_model
            self._clear_gpu_cache()
""",
            ),
            (
                """        if self._warmup_needed:
            warmup_model = self._init_local_custom()
            print("Running batch warmup generation...")
            self._warmup_model(warmup_model)
            self._warmup_needed = False
""",
                """        if self._warmup_needed:
            warmup_model = self._init_local_custom()
            print("Running batch warmup generation...")
            self._warmup_model(warmup_model)
            self._warmup_needed = False
            self._local_custom_model = None
            del warmup_model
            self._clear_gpu_cache()
""",
            ),
        ],
    },
}


def cmd_patch(args: argparse.Namespace) -> None:
    root = Path(args.root)
    names = args.apply or []
    if args.list or not names:
        print("patches disponibles :")
        for name, spec in PATCHES.items():
            print(f"  {name:18s} {spec['file']:16s} {spec['why']}")
        if not names:
            print("\nappliquer : --apply <nom> [--no-dry-run]")
        return

    applied = 0
    for name in names:
        spec = PATCHES.get(name)
        if spec is None:
            die(f"patch inconnu: {name}")
        path = root / spec["file"]
        if not path.exists():
            die(f"{path} introuvable")
        source = path.read_text(encoding="utf-8")
        absent = [old for old, _ in spec["edits"] if old not in source]
        if absent:
            print(f"[{name}] NON APPLICABLE : {len(absent)} motif(s) absent(s) "
                  f"(deja applique ou code different)")
            continue
        patched = source
        for old, new in spec["edits"]:
            patched = patched.replace(old, new, 1)
        if args.dry_run:
            print(f"[{name}] DRY-RUN : {len(spec['edits'])} remplacement(s) dans {path}")
            print(f"          {spec['why']}")
            continue
        backup = path.with_suffix(path.suffix + f".bak-{int(time.time())}")
        shutil.copy2(path, backup)
        path.write_text(patched, encoding="utf-8")
        applied += 1
        print(f"[{name}] applique a {path} (sauvegarde {backup.name})")
    if applied and not args.dry_run:
        print("\nreconstruire l'image pour que le conteneur prenne le changement :")
        print("  docker compose --project-directory " + str(root)
              + " -p alexandria-audiobook -f docker-compose.yml "
              "-f docker-compose.override.yml build alexandria")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="racine du depot")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="reconstruire les exports via ffmpeg")
    enc.add_argument("--m4b", metavar="DEBIT", help="ex: 64k (AAC mono chapitre)")
    enc.add_argument("--mp3", metavar="DEBIT", help="ex: 64k")
    enc.add_argument("--opus", metavar="DEBIT", help="ex: 32k (plus petit)")
    enc.add_argument("--chapters", default="auto:20",
                     help="none | auto:<minutes> | file:<chemin>")
    enc.add_argument("--name", default="audiobook")
    enc.add_argument("--out", help="fichier unique de sortie")
    enc.add_argument("--out-dir", help="dossier de sortie (defaut: runtime-data)")
    enc.add_argument("--cover", help="couverture (M4B)")
    enc.add_argument("--limit", type=int, help="ne traiter que N segments")
    enc.add_argument("--batch-size", type=int, default=200,
                     help="segments par tranche ffmpeg (defaut 200 ; baisser si la "
                          "machine souffre)")
    enc.set_defaults(func=cmd_encode)

    ben = sub.add_parser("bench", help="comparer les moteurs TTS (GPU libre requis)")
    ben.add_argument("--samples", type=int, default=16)
    ben.add_argument("--speaker", default="NARRATOR")
    ben.add_argument("--batches", type=int, nargs="*", default=[1, 4, 8, 16])
    ben.add_argument("--model", default="Qwen3-TTS-12Hz-1.7B-Base")
    ben.add_argument("--snapshot", required=True, help="hash du snapshot HF du modele")
    ben.add_argument("--timeout", type=int, default=2400)
    ben.add_argument("--force", action="store_true",
                     help="ignorer l'avertissement GPU deja occupe")
    ben.set_defaults(func=cmd_bench)

    pat = sub.add_parser("patch", help="optimisations dans le code de l'app")
    pat.add_argument("--list", action="store_true")
    pat.add_argument("--apply", nargs="*", metavar="NOM")
    pat.add_argument("--dry-run", action="store_true", default=True)
    pat.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    pat.set_defaults(func=cmd_patch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
