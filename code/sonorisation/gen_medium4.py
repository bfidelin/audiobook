"""Generation avec les regles du guide officiel : source, action, production, tag.

Ajoute cfg_scale 7.0 (guidage fort, defaut 1.0) et le tag TrackType: SFX dont la
documentation dit qu'il produit des effets plus coherents. Selection identique :
etiquette attendue en tete ET score au-dessus du seuil.
"""
# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel
from transformers import ClapModel, ClapProcessor

OUT = Path("/out")
RAW = OUT / "raw"
SEEDS = (0, 1, 2)
SECONDS = 20.0
MIN_SCORE = 0.20
NEG = "music, melody, singing, speech, loud drums, clipping, distortion"

LABELS = [
    "a busy bar with people talking and glasses clinking",
    "a cafe with conversation and cutlery",
    "a city street with traffic and passers-by",
    "a quiet empty apartment room",
    "a forest with birds singing",
    "heavy rain falling",
    "a metro or subway station",
    "a crowd of people in a large hall",
    "a car engine and road noise",
    "a hospital corridor with fluorescent hum",
    "a farmhouse kitchen with a wood stove crackling",
    "a record shop interior with background music",
    "a television playing in a dark room",
    "an empty open plan office with ventilation hum",
    "a quiet living room with a ticking clock",
    "a street demonstration with chanting crowd",
    "an underground party with muffled bass through walls",
]

# scene -> (prompt, etiquette attendue)
SCENES = {
    "01_appartement_vide": (
        "Room tone of an unfurnished Haussmannian apartment, bare parquet floor, "
        "microphone at one metre, distant traffic through closed double-glazed windows, "
        "slight low-frequency build-up, no voices. TrackType: SFX", 3),
    "02_rue_paris_nuit": (
        "Field recording of a Paris side street at night, close stereo pair on the pavement, "
        "light rain on stone, a car passing left to right, footsteps fading, "
        "wide urban reverb. TrackType: SFX", 2),
    "03_cafe_foule": (
        "Field recording inside a busy French cafe, close pair above a marble table, "
        "dense overlapping conversation, cups and cutlery, espresso machine hiss in the back, "
        "lively midrange. TrackType: SFX", 1),
    "04_squat_fete": (
        "Recording from inside an underground concrete warehouse party, microphone near a wall, "
        "muffled bass and drums bleeding through, crowd murmur in the foreground, "
        "long concrete reverb, distorted lows. TrackType: SFX", 16),
    "05_salon_bourgeois": (
        "Room tone of a quiet bourgeois Parisian living room, microphone at two metres, "
        "a clock ticking, faint street noise behind heavy curtains, dry acoustics, "
        "warm low mids. TrackType: SFX", 14),
    "06_magasin_disques": (
        "Interior of a cramped record shop, close perspective, faint rock music from a ceiling "
        "speaker, cardboard crates being flipped through, a bell above the door, "
        "dry narrow acoustics. TrackType: SFX", 11),
    "07_metro_quai": (
        "Recording on a Paris metro platform, microphone under the tiled vault, a train "
        "braking and squealing into the station, air displacement, echoing footsteps. "
        "TrackType: SFX", 6),
    "08_voiture_ville": (
        "Interior of a small car in city traffic, microphone clipped to the dashboard, "
        "engine at idle, indicator relay ticking, faint radio, wet road under the tyres, "
        "close dry cabin. TrackType: SFX", 8),
    "09_hopital_couloir": (
        "Night recording in a hospital corridor, microphone at head height, fluorescent tubes "
        "humming, a distant door closing, slow footsteps on linoleum, long sterile reverb. "
        "TrackType: SFX", 9),
    "10_foret_aube": (
        "Dawn field recording at a forest edge, spaced pair between trees, layered birdsong, "
        "light wind in the leaves, a cuckoo far away, wide natural ambience. TrackType: SFX", 4),
    "11_ferme_cuisine": (
        "Interior of an old farmhouse kitchen, microphone near the stove, wood crackling and "
        "settling, wind pressing against the window, a wall clock ticking, warm wooden room. "
        "TrackType: SFX", 10),
    "12_bar_nuit": (
        "Field recording inside a small stone-walled bar, close stereo pair, dense overlapping "
        "conversation at two metres, glasses and bottles on a wooden counter, warm low reverb, "
        "no music. TrackType: SFX", 0),
    "13_pluie_fenetre": (
        "Heavy rain recorded from inside against a window, microphone half a metre from the "
        "glass, water running in the gutters, distant thunder rolling, dense broadband hiss. "
        "TrackType: SFX", 5),
    "14_open_space_aube": (
        "Room tone of an empty open plan office before dawn, microphone in the middle of the "
        "floor, constant ventilation hum, a single fluorescent ballast buzzing, very dry. "
        "TrackType: SFX", 13),
    "15_television_nuit": (
        "Recording of a dark living room with a television playing quietly, microphone across "
        "the room, muffled dialogue through the speaker, refrigerator hum behind, "
        "close domestic room tone. TrackType: SFX", 12),
    "16_manifestation": (
        "Field recording of a street demonstration in the distance, microphone at street level, "
        "a crowd chanting in unison, whistles, sporadic shouting, urban canyon reverb. "
        "TrackType: SFX", 15),
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    model = StableAudioModel.from_pretrained("medium")
    print("medium charge", flush=True)

    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    with torch.no_grad():
        tf = clap.get_text_features(**proc(text=LABELS, return_tensors="pt", padding=True))
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    report, retenus = {}, []
    for name, (prompt, expected) in SCENES.items():
        best = None
        for seed in SEEDS:
            raw = RAW / f"{name}_s{seed}.wav"
            if not raw.exists() or raw.stat().st_size < 50_000:
                kwargs = {"prompt": prompt, "duration": SECONDS, "seed": seed,
                          "negative_conditioning": NEG}
                try:
                    audio = model.generate(**kwargs, sampler_kwargs={"cfg_scale": 7.0})
                except (TypeError, ValueError, KeyError):
                    audio = model.generate(**kwargs)
                arr = audio[0] if isinstance(audio, (list, tuple)) else audio
                arr = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else arr
                while arr.ndim > 2:
                    arr = arr[0]
                if arr.ndim == 2:
                    arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
                sf.write(str(raw), arr, 44100)

            y, sr = sf.read(str(raw), dtype="float32", always_2d=True)
            y = y.mean(axis=1)[: int(20 * sr)]
            if sr != 48000:
                y = np.interp(np.linspace(0, len(y) - 1, int(len(y) * 48000 / sr)),
                              np.arange(len(y)), y).astype("float32")
            with torch.no_grad():
                af = clap.get_audio_features(**proc(audio=y, sampling_rate=48000,
                                                    return_tensors="pt"))
                af = getattr(af, "pooler_output", af)
                af = af / af.norm(dim=-1, keepdim=True)
            sims = (af @ tf.T)[0]
            own = float(sims[expected])
            win = int(sims.argmax()) == expected
            if best is None or (win, own) > (best[0], best[1]):
                best = (win, own, raw)

        win, own, raw = best
        garde = win and own >= MIN_SCORE
        if garde:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(raw), "-ac", "1", "-ar", "24000",
                            "-af", "loudnorm=I=-23:TP=-6:LRA=11",
                            str(OUT / f"{name}.bed.wav")], check=True)
            retenus.append(name)
        print(f"  {name:22s} {'GARDE ' if garde else 'ECARTE'} {own:.2f}", flush=True)
        report[name] = {"score": round(own, 2), "gagne": win, "retenu": garde}

    (OUT / "mapping_v4.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(retenus)}/{len(SCENES)} fonds retenus", flush=True)


if __name__ == "__main__":
    main()
