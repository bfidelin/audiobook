"""Benchmark de l'engine actuel d'Alexandria (qwen-tts 0.1.1, Qwen3TTSModel).

Mesure le RTF du clonage de voix par lots, pour plusieurs tailles de batch.
Les dependances (torch, qwen_tts, soundfile) ne sont presentes que dans l'image
Docker, jamais sur l'hote : les imports sont donc marques non verifiables.

Usage (dans l'image alexandria):
  python bench_current.py /data/bench_chunks.json /data/narrator.wav "ref texte" [batch sizes...]
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def load_texts(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sorted((c["text"] for c in json.load(f)), key=len)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        sys.exit(f"Corpus illisible ({path}): {exc}")


def save_results(path, results):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    except OSError as exc:
        print(f"Resultats non ecrits ({path}): {exc}")


def main():
    chunks_path, ref_wav, ref_text = sys.argv[1], sys.argv[2], sys.argv[3]
    batches = [int(x) for x in sys.argv[4:]] or [1, 4, 8, 16]

    model_path = os.environ["QWEN_MODEL_PATH"]
    texts = load_texts(chunks_path)
    print(f"{len(texts)} textes, {sum(map(len, texts))} caracteres")

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    print(
        f"Chargement: {time.perf_counter() - t0:.1f} s"
        f" | GPU: {torch.cuda.get_device_name(0)}"
    )

    audio, sr = sf.read(ref_wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(
        ref_audio=(audio, sr), ref_text=ref_text
    )
    print(f"Clone prompt: {time.perf_counter() - t0:.2f} s (ref {len(audio) / sr:.1f} s)")

    t0 = time.perf_counter()
    model.generate_voice_clone(
        text=[texts[0]],
        voice_clone_prompt=prompt,
        non_streaming_mode=True,
        max_new_tokens=2048,
    )
    print(f"Warmup: {time.perf_counter() - t0:.1f} s")

    results = {}
    for bs in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_audio = 0.0
        t_start = time.perf_counter()
        for i in range(0, len(texts), bs):
            wavs, osr = model.generate_voice_clone(
                text=texts[i : i + bs],
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            for w in wavs:
                a = np.concatenate(w) if isinstance(w, list) else w
                total_audio += len(a) / osr
        wall = time.perf_counter() - t_start
        vram = torch.cuda.max_memory_allocated() / 1e9
        results[bs] = total_audio / wall
        print(
            f"batch={bs:>2} | {wall:6.1f} s pour {total_audio:6.1f} s audio"
            f" | RTF {results[bs]:5.2f}x | VRAM pic {vram:.1f} Go"
        )

    best = max(results.items(), key=lambda kv: kv[1])
    print(
        f"\nMEILLEUR: batch={best[0]} -> {best[1]:.2f}x temps reel"
        f" ({len(texts) / best[1]:.0f} min pour {len(texts)} chunks)"
    )
    save_results("/out/current.json", results)


if __name__ == "__main__":
    main()
