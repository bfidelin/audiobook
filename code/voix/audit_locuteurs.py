#!/usr/bin/env python3
"""Audit des locuteurs produits par l'annotation LLM.

Le modele invente parfois des noms de personnages absents du livre (un
« LE COACH » est apparu sur un essai du tome 2). Ce script confronte la liste
des locuteurs du script annote aux voix reellement configurees, et signale
tout ce qui n'a pas de voix avant le rendu.

    python3 audit_locuteurs.py ../annotated_script.json voice_config.json
"""
import json
import sys

# Locuteurs sans voix dediee : ils reviennent au narrateur, c'est voulu.
NARRATEURS = {"NARRATOR", "NARRATEUR", "NARRATION"}


def main(script_path: str, voices_path: str) -> int:
    entries = json.load(open(script_path, encoding="utf-8"))
    voices = set(json.load(open(voices_path, encoding="utf-8")))

    compte: dict[str, int] = {}
    for e in entries:
        sp = (e.get("speaker") or "?").strip()
        compte[sp] = compte.get(sp, 0) + 1

    total = sum(compte.values())
    sans_voix = {k: v for k, v in compte.items()
                 if k not in voices and k.upper() not in NARRATEURS and k not in NARRATEURS}

    print(f"{total} entrees, {len(compte)} locuteurs distincts, {len(voices)} voix configurees\n")
    print(f"{'locuteur':<18} {'lignes':>7}  etat")
    for nom, n in sorted(compte.items(), key=lambda x: -x[1]):
        if nom in voices:
            etat = "ok"
        elif nom.upper() in NARRATEURS or nom in NARRATEURS:
            etat = "narrateur"
        else:
            etat = ">>> AUCUNE VOIX <<<"
        print(f"{nom:<18} {n:>7}  {etat}")

    if sans_voix:
        lignes = sum(sans_voix.values())
        print(f"\n{len(sans_voix)} locuteur(s) sans voix, {lignes} lignes :")
        for nom, n in sorted(sans_voix.items(), key=lambda x: -x[1]):
            print(f"  {nom} ({n})")
        print("\nAvant le rendu : soit creer la voix, soit reverser sur le narrateur.")
        return 1

    print("\nTous les locuteurs ont une voix.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
