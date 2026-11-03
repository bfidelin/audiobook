"""Test AudioX en texte seul : trois ambiances, comparaison avec MusicGen.

Charge le checkpoint AudioX (5,96 Go) via stable-audio-tools et genere a partir
d'une description, sans image ni video.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

OUT = Path("/out")
PROMPTS = [
    ("bar", "a busy French bar at night, people talking, glasses clinking, "
            "warm crowded room ambience"),
    ("rue", "a Paris street at night, light rain on the pavement, a car passing, "
            "distant voices, urban ambience"),
    ("appartement", "a quiet empty apartment, bare wooden floor, faint traffic "
                    "through closed windows, room tone"),
]
SECONDS = 20.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ckpt = hf_hub_download("HKUSTAudio/AudioX-Turbo", "model.ckpt")
    cfg_path = hf_hub_download("HKUSTAudio/AudioX-Turbo", "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    model = create_model_from_config(cfg)
    model.load_state_dict(load_ckpt_state_dict(ckpt))
    model = model.to("cuda").eval()
    sr = cfg["sample_rate"]
    print(f"modele charge en {time.perf_counter() - t0:.0f} s | {sr} Hz | "
          f"GPU {torch.cuda.get_device_name(0)}", flush=True)

    report = []
    for name, prompt in PROMPTS:
        t1 = time.perf_counter()
        audio = generate_diffusion_cond(
            model,
            steps=50,
            cfg_scale=7,
            conditioning=[{"prompt": prompt, "seconds_start": 0,
                           "seconds_total": SECONDS}],
            sample_size=int(SECONDS * sr),
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device="cuda",
        )
        elapsed = time.perf_counter() - t1
        import soundfile as sf
        arr = audio[0].float().cpu().numpy().T
        sf.write(str(OUT / f"{name}.wav"), arr, sr)
        rtf = SECONDS / elapsed
        print(f"  {name:12s} {elapsed:5.1f} s pour {SECONDS:.0f} s -> {rtf:5.2f}x", flush=True)
        report.append({"scene": name, "prompt": prompt, "secondes": SECONDS,
                       "temps": round(elapsed, 1), "rtf": round(rtf, 2)})

    (OUT / "audiox_texte.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(report)} ambiances dans {OUT}", flush=True)


if __name__ == "__main__":
    main()
