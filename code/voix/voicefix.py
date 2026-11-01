#!/home/calmos/voicefix-venv/bin/python
"""voicefix — fabrique des references de clonage a partir de voix humaines reelles.

Le probleme corrige ici : les 38 voix d'Alexandria utilisent des extraits
VoiceDesign de 1 a 12 s (mediane 3,6 s), donc une voix de synthese servant de
reference a une autre synthese. D'ou le rendu metallique.

Ce script prend une source reelle (interview YouTube ou lecture LibriVox),
trouve automatiquement une plage de parole propre, la transcrit, la normalise
et ecrit un WAV mono 24 kHz pret pour l'import de voix d'Alexandria.

Sous-commandes :
  find    <recherche>                  liste les videos candidates (yt-dlp)
  grab    <url|id> [--min 18] [--max 28]
  librivox <identifiant archive.org> [--fichier 1]
  install <VOIX> <wav> "<texte>"       ecrit la voix dans voice_config.json

Exemples :
  ./voicefix.py find "Romain Duris interview longue"
  ./voicefix.py grab RYlYEnr6VEc --min 20 --max 26
  ./voicefix.py librivox voyageaucentredelaterre_1210_librivox
  ./voicefix.py install VERNON refs/vernon.wav "texte exact de l'extrait"
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(os.environ.get("ALEX_ROOT", "/home/calmos/alexandria-audiobook"))
OUT_DIR = Path(__file__).resolve().parent / "refs"
VENV = Path("/home/calmos/voicefix-venv/bin")
YTDLP = VENV / "yt-dlp"


def die(msg: str) -> NoReturn:
    sys.exit(f"voicefix: {msg}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


# ── sources ──────────────────────────────────────────────────────────────────


def cmd_find(args: argparse.Namespace) -> None:
    proc = run([str(YTDLP), "--flat-playlist", "--print",
                "%(id)s|%(duration)s|%(title)s",
                f"ytsearch{args.count}:{args.query}"])
    if proc.returncode != 0:
        die(proc.stderr[-300:])
    for line in proc.stdout.strip().splitlines():
        vid, dur, title = line.split("|", 2)
        try:
            minutes = f"{int(float(dur)) // 60} min"
        except (TypeError, ValueError):
            minutes = "?"
        print(f"  {vid}  {minutes:>7s}  {title[:78]}")


def download_youtube(video: str, workdir: Path) -> Path:
    target = workdir / "source.%(ext)s"
    proc = run([str(YTDLP), "-f", "bestaudio/best", "-o", str(target),
                "--no-playlist", "--quiet", video])
    if proc.returncode != 0:
        die(f"telechargement impossible : {proc.stderr[-400:]}")
    files = list(workdir.glob("source.*"))
    if not files:
        die("aucun fichier telecharge")
    return files[0]


def download_librivox(identifier: str, index: int, workdir: Path) -> Path:
    meta = run(["curl", "-s", "--max-time", "30",
                f"https://archive.org/metadata/{identifier}"])
    if meta.returncode != 0:
        die("archive.org injoignable")
    try:
        data: Any = json.loads(meta.stdout)
    except (ValueError, TypeError) as exc:
        die(f"reponse archive.org invalide : {exc}")
    mp3s = [f["name"] for f in data.get("files", [])
            if f["name"].endswith("_64kb.mp3")]
    if not mp3s:
        mp3s = [f["name"] for f in data.get("files", []) if f["name"].endswith(".mp3")]
    if index >= len(mp3s):
        die(f"index {index} hors limites ({len(mp3s)} fichiers)")
    name = mp3s[index]
    out = workdir / "source.mp3"
    got = run(["curl", "-sL", "--max-time", "300",
               f"https://archive.org/download/{identifier}/{name}", "-o", str(out)])
    if got.returncode != 0 or not out.exists() or out.stat().st_size < 10_000:
        die("telechargement LibriVox echoue")
    print(f"  fichier : {name}")
    return out


# ── analyse audio ────────────────────────────────────────────────────────────


def speech_windows(src: Path, workdir: Path, min_s: float, max_s: float,
                   wanted: int) -> list[tuple[float, float]]:
    """Plages de parole continue, via silencedetect d'ffmpeg."""
    mono = workdir / "mono16k.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-ac", "1", "-ar", "16000", str(mono)], check=True)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(mono), "-af",
         "silencedetect=noise=-32dB:d=0.35", "-f", "null", "-"],
        capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]

    total = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "csv=p=0", str(mono)]).stdout.strip())
    bounds = [0.0] + [b for pair in zip(starts, ends, strict=False) for b in pair] + [total]
    speech = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds) - 1, 2)
              if bounds[i + 1] - bounds[i] > 0.4]

    windows, start = [], None
    for s0, s1 in speech:
        if start is None:
            start = s0
        if s1 - start > max_s:
            if s1 - start >= min_s:
                windows.append((start, min(s1, start + max_s)))
            start = s0
        elif s1 - start >= min_s:
            windows.append((start, s1))
            start = None
    if not windows:
        die(f"aucune plage de parole de {min_s}-{max_s} s trouvee")
    # repartir sur toute la duree plutot que de tout prendre au debut
    step = max(1, len(windows) // wanted)
    return windows[::step][:wanted]


_MODEL_CACHE: dict[str, Any] = {}


def load_model(model_size: str) -> Any:
    """Charge whisper une seule fois, pas par candidat."""
    from faster_whisper import WhisperModel

    if model_size not in _MODEL_CACHE:
        print(f"  chargement whisper '{model_size}' (une fois)...")
        _MODEL_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODEL_CACHE[model_size]


def transcribe(wav: Path, model_size: str) -> str:
    """Whisper sur un tableau numpy : contourne un bug de decode_audio avec av 19."""
    import wave

    import numpy as np

    with wave.open(str(wav), "rb") as fh:
        channels, width, rate, frames = (fh.getnchannels(), fh.getsampwidth(),
                                         fh.getframerate(), fh.readframes(fh.getnframes()))
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio /= float(np.iinfo(dtype).max) if width > 1 else 1.0
    del rate
    model = load_model(model_size)
    segments, _info = model.transcribe(audio, language="fr", vad_filter=True,
                                       beam_size=1, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


def pitch_stats(wav: Path) -> tuple[float, float]:
    """F0 median (Hz) et proportion de frames voisees, par autocorrelation."""
    import wave

    import numpy as np

    with wave.open(str(wav), "rb") as fh:
        rate = fh.getframerate()
        width = fh.getsampwidth()
        data = fh.readframes(fh.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    x = np.frombuffer(data, dtype=dtype).astype(np.float64)
    if x.size == 0:
        return 0.0, 0.0
    frame, hop = int(0.04 * rate), int(0.02 * rate)
    lo, hi = int(rate / 350), int(rate / 60)
    energies, f0s = [], []
    for start in range(0, len(x) - frame, hop):
        f = x[start:start + frame]
        energies.append(float(np.sqrt(np.mean(f ** 2))))
    if not energies:
        return 0.0, 0.0
    gate = np.percentile(energies, 60)
    for i, start in enumerate(range(0, len(x) - frame, hop)):
        if energies[i] < gate:
            continue
        f = x[start:start + frame] - x[start:start + frame].mean()
        ac = np.correlate(f, f, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        window = ac[lo:hi]
        if window.size == 0:
            continue
        peak = int(np.argmax(window)) + lo
        if ac[peak] > 0.3:
            f0s.append(rate / peak)
    return (float(np.median(f0s)) if f0s else 0.0,
            len(f0s) / max(1, len(energies)))


def score_text(text: str, duration: float) -> float:
    """Note heuristique : on veut une phrase francaise propre et bien calibree."""
    if not text:
        return -100.0
    words = text.split()
    if len(words) < 8:
        return -50.0
    bad = ("♪", "[musique]", "[applaudissements]", "sous-titres", "amara.org",
           "merci d'avoir regardé", "abonnez-vous",
           "domaine public", "enregistrement libre", "librevox", "librivox",
           "lue par", "lu par", "chapitre 1", "chapitre 2")
    if any(b in fold(text) for b in bad):
        return -50.0
    wps = len(words) / duration
    score = 0.0
    score += 10 if text.rstrip()[-1] in ".!?" else -5
    score -= abs(wps - 2.6) * 4
    score += min(len(words), 60) / 10
    # une hallucination boucle : meme trigramme repete
    trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    if trigrams and len(set(trigrams)) < len(trigrams) * 0.6:
        score -= 20
    return score


def extract_best(src: Path, out_wav: Path, min_s: float, max_s: float,
                 wanted: int, model_size: str) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="voicefix-"))
    try:
        windows = speech_windows(src, workdir, min_s, max_s, wanted)
        print(f"  {len(windows)} plages candidates")
        best: tuple[float, float, float, str] | None = None
        for i, (start, end) in enumerate(windows):
            clip = workdir / f"cand_{i}.wav"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src),
                            "-ac", "1", "-ar", "16000", str(clip)], check=True)
            text = transcribe(clip, model_size)
            s = score_text(text, end - start)
            print(f"    [{i}] {start:7.1f}-{end:7.1f} s ({end - start:4.1f} s) note {s:5.1f} | {text[:70]}")
            if best is None or s > best[0]:
                best = (s, start, end, text)
        if best is None:
            die("aucune plage exploitable")
        s, start, end, text = best
        if s < 0:
            die(f"meilleure plage rejetee (note {s:.1f}) — essayer une autre source")
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src),
                        "-ac", "1", "-ar", "24000",
                        "-af", "loudnorm=I=-23:TP=-3:LRA=11,"
                               "afade=t=in:st=0:d=0.03,areverse,"
                               "afade=t=in:st=0:d=0.03,areverse",
                        str(out_wav)], check=True)
        dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "csv=p=0", str(out_wav)]).stdout.strip())
        f0, voiced = pitch_stats(out_wav)
        genre = "masculin" if 0 < f0 < 165 else ("feminin" if f0 else "?")
        print(f"\n  MEILLEUR [{start:.1f}-{end:.1f} s] note {s:.1f} -> {out_wav} ({dur:.1f} s)")
        print(f"  F0 median {f0:.0f} Hz ({genre}), {voiced * 100:.0f}% de frames voisees")
        print(f"  TEXTE : {text}")
        side = out_wav.with_suffix(".txt")
        side.write_text(text, encoding="utf-8")
        print(f"  transcription : {side}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def cmd_grab(args: argparse.Namespace) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="voicefix-src-"))
    try:
        print(f"telechargement : {args.source}")
        src = download_youtube(args.source, workdir)
        print(f"  {src.name} ({src.stat().st_size / 1e6:.1f} Mo)")
        extract_best(src, Path(args.out), args.min, args.max, args.candidates, args.model)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def cmd_librivox(args: argparse.Namespace) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="voicefix-lv-"))
    try:
        print(f"archive.org : {args.identifier}")
        src = download_librivox(args.identifier, args.fichier, workdir)
        extract_best(src, Path(args.out), args.min, args.max, args.candidates, args.model)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── installation dans le projet ──────────────────────────────────────────────


def cmd_install(args: argparse.Namespace) -> None:
    wav = Path(args.wav)
    if not wav.exists():
        die(f"{wav} introuvable")
    text = args.text
    if text == "-":
        sidecar = wav.with_suffix(".txt")
        if not sidecar.exists():
            die(f"pas de transcription a cote de {wav} (attendu : {sidecar})")
        text = sidecar.read_text(encoding="utf-8").strip()
    if not text:
        die("ref_text vide : c'est la transcription exacte de l'extrait")
    config_path = ROOT / "runtime-data" / "voice_config.json"
    try:
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        die(f"voice_config.json illisible : {exc}")
    key = args.voice.upper()
    if key not in config:
        proches = [k for k in config if fold(k).startswith(fold(key)[:4])]
        die(f"voix '{key}' inconnue. Proches : {proches}")
    dest_dir = ROOT / "data" / "designed_voices" / "refs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{fold(key).replace(' ', '_')}_real.wav"
    shutil.copy2(wav, dest)
    backup = config_path.with_suffix(f".json.bak-{int(os.path.getmtime(wav))}")
    if not backup.exists():
        shutil.copy2(config_path, backup)
    config[key]["ref_audio"] = str(dest.relative_to(ROOT / "data"))
    config[key]["ref_text"] = text
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    sidecar = dest.with_suffix(".txt")
    sidecar.write_text(text, encoding="utf-8")
    print(f"{key} : ref_audio -> {config[key]['ref_audio']}")
    print(f"{key} : ref_text  -> {text[:70]}...")
    print(f"sauvegarde de voice_config.json : {backup.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("find", help="chercher des videos sources")
    f.add_argument("query")
    f.add_argument("--count", type=int, default=8)
    f.set_defaults(func=cmd_find)

    for name, fn, helptext in (("grab", cmd_grab, "extraire depuis YouTube"),
                               ("librivox", cmd_librivox, "extraire depuis archive.org")):
        p = sub.add_parser(name, help=helptext)
        if name == "grab":
            p.add_argument("source")
        else:
            p.add_argument("identifier")
            p.add_argument("--fichier", type=int, default=1)
        p.add_argument("--out", required=True, help="wav de sortie")
        p.add_argument("--min", type=float, default=18.0)
        p.add_argument("--max", type=float, default=28.0)
        p.add_argument("--candidates", type=int, default=5)
        p.add_argument("--model", default="small", help="taille du modele whisper")
        p.set_defaults(func=fn)

    ins = sub.add_parser("install", help="ecrire la reference dans voice_config.json")
    ins.add_argument("voice")
    ins.add_argument("wav")
    ins.add_argument("text")
    ins.set_defaults(func=cmd_install)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
