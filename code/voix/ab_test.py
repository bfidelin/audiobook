"""A/B : meme phrase, deux references de clonage, pour comparer a l'oreille.

Usage (dans l'image alexandria) :
  python ab_test.py "<texte>" <ref1.wav> "<ref_text1>" <out1.wav> <ref2.wav> "<ref_text2>" <out2.wav>
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    text = sys.argv[1]
    pairs = [
        (sys.argv[2], sys.argv[3], sys.argv[4]),
        (sys.argv[5], sys.argv[6], sys.argv[7]),
    ]
    model = Qwen3TTSModel.from_pretrained(
        os.environ["QWEN_MODEL_PATH"], dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"modele charge | GPU {torch.cuda.get_device_name(0)}")

    for ref_path, ref_text, out_path in pairs:
        if ref_text == "-":
            ref_text = Path(ref_path).with_suffix(".txt").read_text(encoding="utf-8").strip()
        audio, sr = sf.read(ref_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        prompt = model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=ref_text)
        t0 = time.perf_counter()
        wavs, osr = model.generate_voice_clone(
            text=[text], voice_clone_prompt=prompt,
            non_streaming_mode=True, max_new_tokens=2048,
        )
        elapsed = time.perf_counter() - t0
        wav = wavs[0]
        data = np.concatenate(wav) if isinstance(wav, list) else wav
        sf.write(out_path, data, osr)
        print(f"{out_path}: {len(data) / osr:.1f} s d'audio en {elapsed:.1f} s "
              f"(ref {len(audio) / sr:.1f} s)")

    print("TEXTE GENERER :", text)


if __name__ == "__main__":
    main()
