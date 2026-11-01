"""Trouve la partie du livre la plus riche en occasions sonores.

Les 28 chapitres du M4B sont decoupes toutes les 20 minutes ; on reconstruit la
meme grille a partir de la duree reelle de chaque segment (MP3 CBR 32 kb/s) et
on note chaque partie sur des mots-cles d'ambiance.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import re
from pathlib import Path

ROOT = Path("/home/calmos/alexandria-audiobook")
BITRATE = 32_000
PART_SECONDS = 1200.0  # 20 min, comme le chapitrage auto
KEYWORDS = {
    "musique": r"\b(musique|chanson|disque|vinyle|album|concert|groupe|guitare|baffe|enceinte|dj)\w*",
    "rue": r"\b(rue|trottoir|boulevard|métro|metro|station|quai|gare|carrefour)\w*",
    "bar": r"\b(bar|café|cafe|bistrot|verre|bière|biere|comptoir|juke)\w*",
    "fete": r"\b(fête|fete|soirée|soiree|teuf|dancefloor|boucan|rave)\w*",
    "cris": r"\b(cri|crie|hurle|gueule|vocifère|engueul|dispute|siffle)\w*",
    "porte": r"\b(porte|sonnette|frappe|toc|clenche|serrure|interphone)\w*",
    "telephone": r"\b(téléphone|telephone|sonnerie|appel|portable|répondeur|repondeur)\w*",
    "voiture": r"\b(voiture|bagnole|moteur|klaxon|taxi|bus|scooter|moto)\w*",
    "pluie": r"\b(pluie|orage|tonnerre|averse|mouillé|mouille)\w*",
    "tele": r"\b(télé|tele|écran|ecran|radio|journal|info)\w*",
    "lieu": r"\b(appartement|squat|cave|escalier|ascenseur|hôtel|hotel|chambre|cuisine|salon)\w*",
    "corps": r"\b(souffle|respire|halète|halete|tousse|vomit|sanglote|pleure)\w*",
}


def main() -> None:
    try:
        chunks = json.loads((ROOT / "runtime-data" / "chunks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"chunks.json illisible : {exc}") from exc
    out = ROOT / "data" / "output"

    rows, acc = [], 0.0
    for ch in chunks:
        f = out / Path(ch.get("audio_path", "")).name
        dur = f.stat().st_size * 8 / BITRATE if f.exists() else 0.0
        rows.append((ch.get("text", ""), acc, dur))
        acc += dur
    total = acc
    print(f"{total / 3600:.2f} h, {len(rows)} segments")

    n_parts = int(total // PART_SECONDS) + 1
    parts = [[] for _ in range(n_parts)]
    for text, start, _dur in rows:
        parts[min(int(start // PART_SECONDS), n_parts - 1)].append(text)

    scored = []
    for i, texts in enumerate(parts):
        blob = " ".join(texts)
        counts = {k: len(re.findall(p, blob, re.I)) for k, p in KEYWORDS.items()}
        total_hits = sum(counts.values())
        scored.append((total_hits, i, counts, len(blob)))

    scored.sort(reverse=True)
    print("\nparties les plus riches en occasions sonores :")
    for hits, i, counts, chars in scored[:5]:
        top = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4] if v)
        mm = int(i * PART_SECONDS // 60)
        print(f"  partie {i + 1:2d} ({mm // 60}:{mm % 60:02d}) : {hits:4d} indices "
              f"({chars} car.) — {top}")


if __name__ == "__main__":
    main()
