"""Decoupe le livre en scenes avec un lieu, d'apres le texte de narration.

Une ambiance par scene, tenue du debut a la fin, au lieu de plages de 60 s
saupoudrees sur des mots isoles. Un lieu s'ouvre quand la narration l'annonce et
se referme quand un autre lieu apparait, quand la scene depasse sa duree
maximale, ou a la fin du livre.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path("/home/calmos/alexandria-audiobook")
OUT = Path(__file__).parent / "scenes.json"

# lieu -> ambiance. Les expressions sont cherchées dans la NARRATION uniquement :
# un personnage qui parle d'un bar ne justifie pas qu'on l'entende.
LIEUX = [
    (r"\b(tourne-disque|musique chaude|il passe un disque|il met un disque|"
     r"il monte le son|il lance le disque)\b", "17_musique_disque"),
    (r"\b(chez lui|chez elle|chez eux|dans son appartement|dans l'appartement|"
     r"dans sa chambre|dans le salon|dans la cuisine|à la maison|dans la maison|"
     r"dans son studio|sur le palier|dans l'escalier)\b", "01_appartement_vide"),
    (r"\b(dans la rue|sur le trottoir|dans le quartier|dehors|dans la nuit|"
     r"au coin de la rue|sur le boulevard)\b", "02_rue_paris_nuit"),
    (r"\b(au bar|dans un bar|au bistrot|au comptoir|au café|chez le trader|"
     r"au restaurant|dans un resto)\b", "12_bar_nuit"),
    (r"\b(dans le métro|à la gare|sur le quai|dans le RER|dans le train)\b",
     "07_metro_quai"),
    (r"\b(en voiture|dans la voiture|au volant|en taxi|dans le bus)\b",
     "08_voiture_ville"),
    (r"\b(à l'hôpital|aux urgences|dans le couloir de l'hôpital|à l'hosto)\b",
     "09_hopital_couloir"),
    (r"\b(au magasin|dans le magasin|dans la boutique|devant la vitrine|"
     r"au rayon|dans la librairie)\b", "06_magasin_disques"),
    (r"\b(dans le squat|au squat|dans la cave|dans l'entrepôt|dans l'usine)\b",
     "04_squat_fete"),
    (r"\b(il pleut|sous la pluie|la pluie|orage|averse)\b", "13_pluie_fenetre"),
    (r"\b(à la télé|à la radio|le journal télévisé|les infos)\b", "15_television_nuit"),
]

MIN_SCENE = 45.0        # sous ce seuil, une scene n'est pas sonorisee
MAX_SCENE = 900.0       # au-dela, on coupe : la boucle fatigue
BITRATE_RATE = 0.0


def durees(out_dir: Path) -> dict[str, float]:
    cache_path = Path(__file__).parent / "seg_durations.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    for f in out_dir.glob("*.mp3"):
        if f.name in cache:
            continue
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                               "format=duration", "-of", "csv=p=0", str(f)],
                              capture_output=True, text=True)
        try:
            cache[f.name] = float(proc.stdout.strip())
        except ValueError:
            cache[f.name] = 0.0
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def localise(texte: str) -> str | None:
    for motif, ambiance in LIEUX:
        if re.search(motif, texte, re.I):
            return ambiance
    return None


def main() -> None:
    try:
        chunks = json.loads((ROOT / "runtime-data" / "chunks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"chunks.json illisible : {exc}") from exc
    repo = ROOT / "data" / "output"
    dur = durees(repo)

    scenes: list[dict] = []
    courant: dict | None = None
    acc, prev = 0.0, None
    for ch in chunks:
        nom = Path(ch.get("audio_path", "")).name
        d = dur.get(nom, 0.0)
        if prev is not None:
            acc += 0.5 if ch.get("speaker") != prev else 0.25
        # la narration annonce le lieu
        if ch.get("speaker") == "NARRATOR":
            lieu = localise(ch.get("text", ""))
            if lieu:
                if courant and acc - courant["start"] >= MIN_SCENE:
                    courant["end"] = acc
                    scenes.append(courant)
                elif courant:
                    courant["start"] = acc      # scene trop courte : on la deplace
                courant = {"bed": lieu, "start": acc, "end": None}
        if courant and acc - courant["start"] > MAX_SCENE:
            courant["end"] = acc
            scenes.append(courant)
            courant = {"bed": courant["bed"], "start": acc, "end": None}
        acc += d
        prev = ch.get("speaker")
    if courant:
        courant["end"] = acc
        scenes.append(courant)

    scenes = [s for s in scenes if (s["end"] or 0) - s["start"] >= MIN_SCENE]
    OUT.write_text(json.dumps(scenes, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(s["end"] - s["start"] for s in scenes)
    print(f"{len(scenes)} scenes, {total / 3600:.2f} h sonorisees sur {acc / 3600:.2f} h "
          f"({100 * total / acc:.0f} %)")
    par: dict[str, float] = {}
    for s in scenes:
        par[s["bed"]] = par.get(s["bed"], 0) + (s["end"] - s["start"])
    for k, v in sorted(par.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22}{v / 60:7.1f} min")


if __name__ == "__main__":
    main()
