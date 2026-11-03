"""Redessine en francais les voix dont la reference vient d'une consigne anglaise.

Toutes les voix sauf celles deja remplacees par un enregistrement humain reel
(narrateur, Vernon, La Hyene, Olga, Patrice, Gaelle, Pamela, Alexandre) sont
regenerees via /api/voice_design/preview avec une description francaise, puis
installees dans voice_config.json avec leur transcription.
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/calmos/alexandria-audiobook")
CFG = ROOT / "runtime-data" / "voice_config.json"
PREVIEWS = ROOT / "data" / "designed_voices" / "previews"
API = "http://localhost:4200/api/voice_design/preview"

# voix dont la reference est deja une vraie voix humaine : on n'y touche pas
REAL = {"NARRATOR", "VERNON", "LA HYÈNE", "OLGA", "PATRICE", "GAËLLE",
        "PAMELA", "ALEXANDRE"}

# passage long et neutre : une reference de clonage doit faire 20-30 s, sinon le
# modele n'a pas d'ancrage et produit un timbre metallique.
SAMPLE = (
    "C'est alors qu'il comprit que la nuit ne finirait pas de sitot. Il resta "
    "immobile un long moment, a ecouter le silence de l'appartement, le "
    "claquement lointain d'une porte, le passage d'une voiture dans la rue. "
    "Rien ne pressait. Plus rien ne pressait, desormais. Il avait tout son "
    "temps, et cette idee, loin de le rassurer, lui serrait la poitrine. Au "
    "matin, il faudrait descendre, affronter le jour, repondre aux questions, "
    "faire semblant."
)

# description francaise par personnage : sexe, age, timbre
DESC = {
    "ALEX":       "Un homme de 46 ans, chanteur de rock francais a la voix grave et eraillee, un peu cassee par les annees. Debit lent, ton desabuse.",
    "AMELIE":     "Une femme francaise d'une trentaine d'annees. Voix claire et chaleureuse, debit naturel de conversation.",
    "ANAIS":      "Une adolescente francaise de 17 ans. Voix jeune et vive, articulation relachee, debit rapide.",
    "AÏCHA":      "Une jeune femme francaise de 25 ans. Voix douce mais assuree, prononciation nette.",
    "BERTRAND":   "Un homme francais de 55 ans, journaliste. Voix posee, un peu professorale, debit regulier.",
    "CASSANDRE":  "Une femme francaise de 35 ans. Voix seche et ironique, debit tranchant.",
    "CECILE":     "Une femme francaise de 45 ans. Voix posee, calme, legerement rauque.",
    "CELESTE":    "Une jeune femme francaise de 28 ans. Voix legere et claire, debit souple.",
    "DANIEL":     "Un homme francais de 45 ans. Voix posee et reflechie, debit regulier, timbre medium.",
    "DEBORAH":    "Une femme francaise de 40 ans. Voix grave et assuree, debit franc.",
    "EMILIE":     "Une femme francaise de 33 ans. Voix energetique, debit vif, timbre clair.",
    "JP":         "Un homme francais de 50 ans. Voix legerement nasillarde, debit familier.",
    "JULIEN":     "Un homme francais de 30 ans. Voix neutre et jeune, debit naturel.",
    "KIKO":       "Un jeune homme francais de 22 ans. Voix jeune, debit rapide, ton decontracte.",
    "LA CLODOTE": "Une femme francaise d'environ 60 ans, sans abri. Voix eraillee et fatiguee, debit irregulier.",
    "LAURENT":    "Un homme francais de 62 ans, producteur de musique. Voix autoritaire et grave, debit lent.",
    "LE CHEF":    "Un homme francais de 55 ans. Voix grave et seche, debit bref.",
    "LOIC":       "Un homme francais de 35 ans. Voix ordinaire, debit neutre, timbre medium.",
    "MARCIA":     "Une femme francaise de 50 ans. Voix chaleureuse et maternelle, debit pose.",
    "MARIE-ANGE": "Une femme francaise de 70 ans. Voix agee, un peu chevrotante, debit lent.",
    "NOEL":       "Un homme francais de 65 ans. Voix fatiguee et grave, debit lent.",
    "PAUL":       "Un homme francais de 40 ans. Voix neutre, debit regulier.",
    "SOPHIE":     "Une femme francaise de 38 ans. Voix claire et neutre, debit regulier.",
    "SÉLIM":      "Un homme francais de 30 ans. Voix douce et posee, debit calme.",
    "WOMAN":      "Une femme francaise d'une trentaine d'annees. Voix claire et energique, debit vif.",
    "XAVIER":     "Un homme francais de 45 ans. Voix basse, fatiguee, legerement rauque. Debit sobre et retenu.",
}


def design(desc: str, texte: str) -> Path:
    body = json.dumps({"description": desc, "sample_text": texte,
                       "language": "french"}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    if not API.startswith("http://localhost:"):
        raise ValueError(f"URL non locale refusee : {API}")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:  # noqa: S310
            rep = json.loads(r.read())
    except (OSError, ValueError) as e:
        raise RuntimeError(f"appel API impossible : {e}") from e
    return PREVIEWS / Path(rep["audio_url"]).name


def main() -> None:
    try:
        cfg = json.loads(CFG.read_text(encoding="utf-8"))
        shutil.copy(CFG, str(CFG) + f".bak-redesign-{int(time.time())}")
    except (OSError, ValueError) as e:
        raise SystemExit(f"voice_config.json illisible : {e}") from e
    faits = []
    for voix, desc in DESC.items():
        entree = cfg.get(voix)
        if entree is None:
            print(f"{voix:<12} absente de voice_config", flush=True)
            continue
        if voix in REAL:
            continue
        texte = SAMPLE
        try:
            wav = design(desc, texte)
        except Exception as e:  # noqa: BLE001
            print(f"{voix:<12} ECHEC api : {e}", flush=True)
            continue
        wav.with_suffix(".txt").write_text(texte, encoding="utf-8")
        entree["ref_audio"] = f"designed_voices/previews/{wav.name}"
        entree["ref_text"] = texte
        entree["description"] = desc
        faits.append(voix)
        print(f"{voix:<12} ok -> {wav.name}", flush=True)
        CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(faits)} voix redessinees : {', '.join(faits)}", flush=True)


if __name__ == "__main__":
    main()
