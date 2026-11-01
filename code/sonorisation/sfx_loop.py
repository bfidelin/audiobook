"""Boucle generation -> controle -> conservation des seuls fichiers conformes.

Pour chaque son : on genere des variantes (graines et formulations differentes),
on les passe a sfxcheck avec le type DECLARE, et on garde la premiere conforme.
Si aucune ne passe au bout de N essais, on garde la moins mauvaise et on le dit.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel


def _load_checker():
    """Import local : evite un bloc « first-party » dans les imports du module."""
    import sfxcheck  # noqa: PLC0415

    return sfxcheck

OUT = Path("/out")
RATE = 44_100
MAX_TRIES = 6

SPECS = [
    ("porte_claque", "event", 4.0, [
        "a heavy wooden door slamming shut, sharp impact, short decay, empty room",
        "a door banging closed, hard wooden slam, close perspective, no reverb tail"]),
    ("pas_escalier", "cadence", 6.0, [
        "footsteps climbing old wooden stairs, regular pace, interior reverb"]),
    ("telephone_sonne", "cadence", 10.0, [
        "a classic landline telephone ringing in a quiet room, repeated double ring, close perspective",
        "an old fixed telephone, two short ring bursts then a pause, then again"]),
    ("verres_bar", "ambience", 6.0, [
        "bustling French cafe interior, indistinct conversation and murmur, cutlery, warm room",
        "people talking in a busy bar, layered voices, background murmur"]),
    ("cles_serrure", "event", 4.0, [
        "keys turning in a door lock, metal on metal, close up, single action"]),
    ("chaise_racle", "event", 3.0, [
        "a wooden chair dragged once across a cafe floor, short scrape, single movement"]),
    ("briquet", "event", 3.0, [
        "a metal cigarette lighter flicking open and igniting, sharp click, close up"]),
    ("sirene_loin", "ambience", 8.0, [
        "a distant city street at night, a siren passing far away, traffic hum, wide perspective"]),
]


def convert(raw: Path, out: Path, dur: float) -> None:
    import subprocess
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
         "-ac", "1", "-ar", "24000",
         "-af", f"loudnorm=I=-26:TP=-4:LRA=11,afade=t=in:st=0:d=0.03,"
                f"afade=t=out:st={max(0.0, dur - 0.2):.2f}:d=0.2", str(out)],
        check=True)


def main() -> None:
    checker = _load_checker()
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print("chargement small-sfx...", flush=True)
    model = StableAudioModel.from_pretrained("small-sfx")

    report = []
    for name, kind, dur, prompts in SPECS:
        best = None
        for attempt in range(MAX_TRIES):
            prompt = prompts[attempt % len(prompts)]
            torch.manual_seed(attempt * 13)
            audio = model.generate(prompt=prompt, duration=dur)
            data = audio[0] if isinstance(audio, (list, tuple)) else audio
            arr = data.detach().float().cpu().numpy() if hasattr(data, "detach") else data
            arr = np.asarray(arr)
            while arr.ndim > 2:
                arr = arr[0]
            if arr.ndim == 2:
                arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)

            raw = OUT / f"{name}_try{attempt}.wav"
            sf.write(raw, arr, RATE)
            feats = checker.features(raw)
            checks = checker.check(feats, kind)
            fails = [m for m, ok, _ in checks if not ok]
            print(f"  {name} essai {attempt + 1}/{MAX_TRIES} ({kind}) : "
                  f"{'PASS' if not fails else 'KO ' + ', '.join(fails[:2])}", flush=True)
            score = len(checks) - len(fails)
            if best is None or score > best[0]:
                best = (score, raw, prompt, fails)
            if not fails:
                break

        if best is None:
            raise RuntimeError(f"aucune variante produite pour {name}")
        score, raw, prompt, fails = best
        final = OUT / f"{name}.sfx.wav"
        convert(raw, final, dur)
        report.append({"name": name, "type": kind, "passed": not fails,
                       "fails": fails, "prompt": prompt, "file": final.name})
        for tmp in OUT.glob(f"{name}_try*.wav"):
            tmp.unlink(missing_ok=True)

    try:
        (OUT / "sfx_report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"rapport non ecrit : {exc}", file=sys.stderr)

    ok = sum(1 for r in report if r["passed"])
    print(f"\n{ok}/{len(report)} sons conformes")
    sys.exit(0 if ok == len(report) else 1)


if __name__ == "__main__":
    main()
