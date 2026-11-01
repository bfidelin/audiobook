#!/usr/bin/env python3
"""Relance le rendu audio d'un sous-ensemble de chunks via l'API d'Alexandria.

Usage :
  ./rerender.py --speaker NARRATOR --limit 2      (test rapide)
  ./rerender.py --speaker NARRATOR --all          (tout le narrateur)
  ./rerender.py --speaker NARRATOR --dry-run

L'app doit tourner sur le port 4200. Les chunks non regeneres gardent leur
ancien audio : l'operation est reprise-able.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://localhost:4200"
ALLOWED = ("/api/",)


def call(path: str, payload: dict | None = None) -> dict:
    """Appel JSON sur l'API locale, chemin restreint a /api/."""
    if not path.startswith(ALLOWED):
        sys.exit(f"voicefix: chemin refuse ({path})")
    url = urllib.parse.urljoin(BASE, path)
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        sys.exit(f"voicefix: schema refuse ({url})")
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        sys.exit(f"voicefix: {path} -> HTTP {exc.code} {exc.read()[:300]!r}")
    except urllib.error.URLError as exc:
        sys.exit(f"voicefix: API injoignable ({exc}). Lancer l'app sur {BASE}")


def narrator_indices(chunks: list[dict], speaker: str) -> list[int]:
    return [c.get("index", i) for i, c in enumerate(chunks) if c.get("speaker") == speaker]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="NARRATOR")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=20)
    args = ap.parse_args()

    payload = call("/api/chunks")
    chunks = payload.get("chunks", []) if isinstance(payload, dict) else payload
    indices = narrator_indices(chunks, args.speaker)
    print(f"{len(indices)} chunks '{args.speaker}' sur {len(chunks)}")

    if args.limit:
        indices = indices[: args.limit]
    elif not args.all:
        sys.exit("preciser --limit N ou --all")

    statuses = {c.get("index", i): c.get("status") for i, c in enumerate(chunks)}
    done = [i for i in indices if statuses.get(i) == "done"]
    print(f"  dont deja rendus : {len(done)} (ils seront regeneres)")
    if args.dry_run:
        print("dry-run : rien envoye")
        return

    resp = call("/api/generate_batch_fast", {"indices": indices})
    print(f"  reponse : {resp}")
    t0 = time.time()
    status: dict = {}
    while True:
        time.sleep(args.poll)
        status = call("/api/status/audio")
        logs = status.get("logs") or []
        last = logs[-1] if logs else ""
        print(f"  [{time.time() - t0:6.0f} s] running={status.get('running')} | {last[:110]}")
        if not status.get("running"):
            break
    print(f"\ntermine en {(time.time() - t0) / 60:.1f} min")
    for line in (status.get("logs") or [])[-6:]:
        print(f"  {line[:140]}")


if __name__ == "__main__":
    main()
