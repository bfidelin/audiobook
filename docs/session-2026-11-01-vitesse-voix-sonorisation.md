# Session du 1er novembre 2026 — vitesse, voix, sonorisation

Complément au document `tts-voices-vernon-subutex.md`. Tout ce qui suit a été
mesuré sur la machine locale (RTX 5060 Ti 16 Go, Ryzen AI HX370, 92 Go de RAM).

## 1. État d'Alexandria en amont

| | |
|---|---|
| Dernier commit amont | `be6343d`, 2 août 2026 |
| Dernière release | v1.5, 3 mars 2026 |
| Étoiles / forks | 1 083 / 138 |
| Issues ouvertes | 17 |

Le dépôt est en veille. L'installation locale porte **4 patches** au-delà de la
pointe : `merge-ffmpeg`, `m4b-bitrate`, `segment-lossless`, `free-warmup-vram`.

## 2. Débit de génération : 4,9× → 8,2× temps réel

Trois freins, tous mesurés dans les journaux du conteneur.

| Frein | Correction | Effet |
|---|---|---|
| Le modèle CustomVoice restait en VRAM après le warmup (~3,5 Go) | libéré après usage | 8,0 → 12,2 Go libres, `max_batch` 10 → 16 |
| `sub_batch_min_size = 4`, `sub_batch_ratio = 5.0` | 16 et 40 | sous-lots de 7-11 chunks → 16 (956 sous-lots pour 4 162 chunks) |
| `parallel_workers = 20` | 64 | fin des tranches résiduelles de 4 chunks |

Résultat : **8,03 à 8,18× temps réel** contre 4,32-5,71× avant. Rendu complet
du narrateur : 100 min → ~60 min.

Bench de référence, 16 chunks réels, voix Proust, même carte :

| batch | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| RTF | 1,67× | 5,34× | 8,51× | **11,37×** |
| VRAM | 5,0 Go | 7,2 Go | 10,2 Go | 12,6 Go |

## 3. Voix : la cause racine des timbres metalliques

Les 38 voix utilisaient des **previews VoiceDesign** comme référence de
clonage : 1,04 à 12,48 s, **médiane 3,60 s**, 31 sur 38 sous 6 s. Une voix de
synthèse servant de référence à une seconde synthèse.

Correctif appliqué au narrateur (4 162 lignes sur 4 795, soit 87 % de
l'écoute) : référence de **22,3 s** de narration humaine réelle, 98 Hz,
−23 LUFS. Les 4 162 lignes ont été régénérées, 0 échec.

Conséquence : le livre passe de 10 h 14 à **9 h 42** — la voix réelle lit 5 %
plus vite.

### Xavier Fardin

Deux défauts cumulés : `ref_audio` pointait vers un fichier **inexistant**
(`xavier_1784632966777165830_preview.wav`), et les 44 répliques portaient 44
indications de jeu différentes, toutes outrancières (« Enthusiastic, teasing,
fast-paced banter », « loud tone »). Remplacées par un registre unique
(« Low, tired, slightly raspy. Dry, understated delivery. ») et référence
réparée sur `xavier_1784633447.wav` (15,1 s).

## 4. Encodage et exports

| | Avant | Après |
|---|---|---|
| M4B | 480 Mo, AAC 108 kb/s, 3 chapitres cassés | 292 Mo, AAC 64 kb/s, 30 chapitres |
| MP3 | 140 Mo, 32 kb/s | 266 Mo, 64 kb/s |
| Fusion pydub (41 min) | 43 s | **4,6 s** avec ffmpeg |

Les segments sont écrits en MP3 32 kb/s : la perte est déjà faite à ce niveau.
Le patch `segment-lossless` fait écrire du FLAC pour les **futurs** rendus.

## 5. Sonorisation : chaîne et résultats

Trois briques écrites et validées.

**`sonorisation.py`** — carte de scènes par mots-clés du texte. Une ambiance
persiste jusqu'au repère suivant ; garde-fous : les 20 s qui suivent doivent
être de la narration (pas de dialogue), 90 s de repos entre deux plages, 60 s
maximum par plage. Couverture obtenue : **~25 % du livre**.

**`mix_events.py`** — mixeur numpy : fonds et événements placés à
l'échantillon, ducking calculé par enveloppe RMS (attaque 150 ms, relâchement
600 ms), pas de ffmpeg, pas de fichier intermédiaire. **0,95 s par partie de
20 min**, contre 16 s avec la chaîne ffmpeg.

**`sfxcheck.py`** — contrôle générique : 12 descripteurs normalisés (durée,
crête, offset DC, écrêtage, centroïde, flatness, raie dominante, silences,
périodicité, cadence, modulation 3-8 Hz, attaques/s) confrontés à 4 archétypes
(`event`, `cadence`, `ambience`, `texture`). Sortie PASS/ECHEC par critère.

## 6. Fonds sonores : ce qui marche et ce qui ne marche pas

**Génération** — MusicGen-medium (`facebook/musicgen-medium`, CC-BY-NC 4.0) :
16 ambiances de 30 s. Correct pour des textures, **jamais du cinéma** : c'est un
modèle de musique.

**Stable Audio 3 Small-SFX** (`stabilityai/stable-audio-3-small-sfx`, 433 M,
2,27 Go, licence communautaire) : entraîné pour les effets. 8 bruits générés,
**3 sur 8 seulement passent le contrôle** — porte trop résonante, téléphone sans
cadence (100 ms au lieu de 2,4 s), chaise périodique, briquet sans attaque,
sirène trop tonale.

**Identification** — `clap_id.py` (CLAP `laion/clap-htsat-unfused`) identifie le
contenu d'un WAV par comparaison à des descriptions. A démontré qu'un fichier
supposé « bar de nuit » était de la **pluie** : le mapping par durée était faux.

**Récolte réelle** — SFXMint et Mixkit se téléchargent sans compte ni clé. Mais
**aucun clip ne dépasse 25 s** : sur 40 fichiers récoltés, zéro n'a passé le
filtre de durée. La solution serait le bouclage avec fondu croisé sur des clips
de 10-15 s.

## 7. Écosystème : ce qui n'existe pas

| Piste | Constat |
|---|---|
| Forks de Qwen3-TTS | `dffdeeq/Qwen3-TTS-streaming` (276 ★) annonce 6× : **aucun gain** mesuré en mode lot, et il perd le batch 16 (OOM). Les réimplémentations C++/Rust plafonnent au niveau du batch actuel. |
| OmniVoice (k2-fsa) | annoncé 87× temps réel sur H100 ; mesuré **4,08× en batch=1** sur cette carte, et le batching le **ralentit** (3,25× à batch 8). |
| AudioGen (Meta) | épingle torch 2.1.0 → **antérieur à Blackwell, ne peut pas tourner** |
| TangoFlux | épingle torch 2.4.0 → même problème |
| AudioLDM 2 | conflit `diffusers` / `transformers` insoluble (3 tentatives) |
| Modèle d'ambiance sur Hugging Face | **n'existe pas** (recherche « ambience » : 1 résultat sans rapport) |
| Foley de pointe (HunyuanVideo-Foley, LoRA LTX) | **conditionné par une vidéo** — inapplicable à un livre audio |
| Ollama | héberge `phonon`, `lfm3-audio`, `lfm3-omni` — mais **aucun clonage de voix**, et pas de batch |
| Outils « audiobook creator » du marché | TTS + mastering ACX ; la sonorisation est **hors périmètre** |
| Open source (GitHub) | maximum **6 étoiles** sur les projets de sonorisation de livres audio |

## 8. Pièges rencontrés, à ne pas répéter

- **Parallélisme massif** : un `ffmpeg` sur 4 795 fichiers et un `xargs -P 16`
  ont gelé la machine deux fois (load 38). Un seul ffmpeg séquentiel sur des
  fichiers de 20 min passe sans problème à load 1,3.
- **`areverse` sur un flux infini** (`-stream_loop -1`) : ffmpeg bufferise sans
  fin. Utiliser `afade=t=out` à la place.
- **`-filter_complex` et `-af` sur la même sortie** : refusé.
- **Durée estimée par la taille des fichiers** (`taille × 8 / 32000`) : ignore
  les pauses inter-répliques, soit 820 s d'écart sur ce livre. Toujours
  interroger le fichier de référence avec `ffprobe`.
- **Contrôle appliqué au fichier brut** plutôt qu'au fichier normalisé : un
  modèle sort à 0 dBFS, le contrôle de crête échoue systématiquement.
- **Sortie stéréo aplatie** : une forme `(lot, canaux, échantillons)` passée à
  `reshape(-1)` met les canaux bout à bout et double la durée.
- **`huggingface_hub` 2.x** est incompatible avec `diffusers` et avec les
  `transformers` qui savent charger AudioLDM 2. Toute pile audio doit être
  isolée dans sa propre image.

## 9. Inventaire des scripts

`~/ambience-gen/` : `sonorisation.py`, `mix_events.py`, `mix_all.py`,
`sfxcheck.py`, `clap_id.py`, `harvest.py`, `gen_beds.py`, `gen_sfx3.py`,
`sono_all.py`, `voicefix.py` (côté voix), `alexandria-speed.py` (`encode`,
`bench`, `patch`), `rerender.py`, `qc_narrator.py`.

Sauvegardes : `~/alexandria-backups/` (4 points), `~/systemd-units-retires/`.
