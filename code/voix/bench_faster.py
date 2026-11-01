"""Benchmark de Faster Qwen3-TTS (CUDA graphs, backend torch).

Meme corpus, meme voix de reference, batch=1 (le moteur faster n'expose pas de
generation multi-textes). Les dependances ne sont presentes que dans l'image
Docker.

Usage: python bench_faster.py /data/bench_chunks.json /data/narrator.wav "ref texte"
Env: BACKEND=torch|ggml, QUANT=BF16|...
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import os
import sys
import time

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS


def load_texts(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sorted((c["text"] for c in json.load(f)), key=len)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        sys.exit(f"Corpus illisible ({path}): {exc}")


def save_results(path, payload):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
    except OSError as exc:
        print(f"Resultats non ecrits ({path}): {exc}")


def main():
    chunks_path, ref_wav, ref_text = sys.argv[1], sys.argv[2], sys.argv[3]
    model_path = os.environ["QWEN_MODEL_PATH"]
    backend = os.environ.get("BACKEND", "torch")
    quant = os.environ.get("QUANT", "BF16")

    texts = load_texts(chunks_path)
    print(f"{len(texts)} textes, {sum(map(len, texts))} caracteres | backend={backend}")

    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        model_path, dtype=torch.bfloat16, backend=backend, quant=quant
    )
    print(
        f"Chargement: {time.perf_counter() - t0:.1f} s"
        f" | GPU: {torch.cuda.get_device_name(0)}"
    )

    t0 = time.perf_counter()
    model.warmup(prefill_len=len(texts[0]))
    print(f"Warmup CUDA graphs: {time.perf_counter() - t0:.1f} s")

    torch.cuda.reset_peak_memory_stats()
    total_audio = 0.0
    t_start = time.perf_counter()
    for text in texts:
        wavs, sr = model.generate_voice_clone(
            text=text, language="French", ref_audio=ref_wav, ref_text=ref_text
        )
        a = np.concatenate(wavs) if isinstance(wavs, list) else wavs
        total_audio += len(a) / sr
    wall = time.perf_counter() - t_start
    vram = torch.cuda.max_memory_allocated() / 1e9
    rtf = total_audio / wall
    print(
        f"batch= 1 | {wall:6.1f} s pour {total_audio:6.1f} s audio"
        f" | RTF {rtf:5.2f}x | VRAM pic {vram:.1f} Go"
    )
    save_results("/out/faster.json", {"backend": backend, "batch_1": rtf})


if __name__ == "__main__":
    main()
