# Session du 2 novembre — rendre le fond sonore réellement audible

Suite directe de `session-2026-11-01-vitesse-voix-sonorisation.md`. Le tome 1 était
livré en 9 h 42 avec 156 scènes sonorisées, mais **le fond ne s'entendait pas**.
Trois livraisons ont été nécessaires pour corriger cela : v7, v8, v9.

## 1. Deux erreurs de mesure successives

### Erreur 1 — mesurer l'énergie globale

Premier réflexe : comparer le RMS du fond et celui de la voix sur une fenêtre de
25 s. Le fond ressortait à 0,02-0,05 contre 0,08 pour la voix, donc « présent à
−24 dB », puis « −12 dB », puis « −3 dB ». **L'utilisateur n'entendait toujours
rien.** Le fond était en réalité 50 dB sous la parole : la voix, continue et
forte, écrase toute moyenne glissante.

**Méthode correcte** — mesurer le *plancher de bruit* : découper en trames de
150-200 ms, ne garder que les trames situées dans les silences **entre deux
répliques**, et prendre le 10ᵉ-15ᵉ percentile. C'est la seule grandeur qui
corresponde à ce que l'oreille perçoit quand personne ne parle.

### Erreur 2 — calibrer sur le RMS large bande

Une fois le plancher mesuré correctement, l'écart restait faible alors que le
fond demeurait inaudible. Analyse spectrale par bandes : **la majorité de
l'énergie des fonds Stable Audio se situe sous 60 Hz**, c'est-à-dire dans un
rumble que ni une enceinte de bureau ni un casque ne restitue.

| Fond | Part de l'énergie sous 60 Hz | Part dans 200 Hz – 2 kHz |
|---|---|---|
| `15_television_nuit` | 99 % | — |
| `05_salon_bourgeois` | 97 % | — |
| `14_open_space_aube` | 76 % | — |
| `01_appartement_vide` | 43 % | — |
| `17_musique_disque` | 57 % | **0 %** |

`01_appartement_vide` représente à lui seul 284 min du livre : le fond le plus
utilisé était donc à moitié inaudible.

## 2. Correctif — `fix_beds.py`

Pour chaque fond de `out23_loop/` :

1. passe-haut 90 Hz à 2 pôles **en cascade** (24 dB/oct), pas un seul biquad ;
2. `equalizer=f=300:t=q:w=1.0:g=3` pour remonter le bas-médium ;
3. renormalisation sur la bande **200 Hz – 8 kHz** à `CIBLE_RMS = 0.045` ;
4. sortie dans `/data/out23_fix`.

Gains appliqués : **+5 à +36 dB** selon le fond. Part du rumble retombée à 0-9 %.
`mix_all2.py` pointe désormais sur `out23_fix` avec `BED_DB = 0.0`.

### Piège de calcul

`rms_bande` sans facteur de Parseval donne des gains aberrants : il faut

```python
np.sqrt(2.0 * X[m].sum() / (n * n))
```

Sans le `sqrt(2 * ...)`, les gains mesurés sortaient à −60 dB et le correctif
était rejeté à tort.

## 3. Résultat mesuré en v9

Écart de plancher de bruit entre les silences avec fond et les silences sans fond :

| Scène | Écart |
|---|---|
| Pluie | +27,1 dB |
| Musique | +22,8 dB |
| Bar | +12,4 dB |
| Rue | +8,5 dB |
| Magasin | −5,3 dB (aucun fond) |

## 4. Limites assumées à la livraison

- **`17_musique_disque`** — après filtrage il ne reste qu'un bourdon grave. C'est
  audible, mais ce n'est pas de la musique. Le modèle Stable Audio `medium` a
  disparu du disque : toute nouvelle génération de fond est impossible sans
  retéléchargement (~10 Go).
- **`06_magasin_disques`** — seule scène sans aucun fond. CLAP rejette le fichier
  existant (scores 0,11 puis 0,03). Garde-fou : on ne mixe pas une scène sans
  fond plutôt que d'en mettre un faux.
- **`18_sirene_police`** — jamais généré, `gen_extra.py` échoue faute de modèle.
- **Scène 3086-3126** — dialogue attribué à NARRATOR au lieu de Sélim. Le
  détecteur automatique produit 281 candidats, trop de faux positifs pour une
  réécriture non supervisée.

## 5. Livraisons

| Version | Contenu | Taille |
|---|---|---|
| v7 | détection de scènes par lieux | 279 741 810 o |
| v8 | fond remonté (`BED_DB` −3) | 279 203 611 o |
| v9 | fond filtré (`out23_fix`) | 279 149 172 o |

v9 : 9,48 h, 29 parties, AAC 64 kb/s. Vérification par MD5 à chaque copie.

## 6. Temps de la chaîne, mesurés

Chiffres relevés pour dimensionner le tome 2 (même longueur : 107 169 mots
contre 106 965).

| Étape | Mesure |
|---|---|
| Rendu TTS | 8,03-8,18× temps réel (bench batch 16 : 11,37×) |
| Normalisation LUFS | 0,52 s/segment, 3 threads → ~14 min pour 4 835 |
| Décodage pour le mixage | 624 s d'audio en 11,5 s (54×) |
| Fusion M4B | 4,6 s avec ffmpeg (contre 41 min avec pydub) |
| Copie WebDAV | 27,9 Mo/s |

L'écart entre les 11,37× du bench et les 8,1× de la production reste inexpliqué :
fragmentation des sous-lots, encodage MP3 par segment, warmup répété.

## 7. Points de fragilité à ne pas perdre

- Les patchs moteur (`free-warmup-vram`, `sub_batch_min_size`, `parallel_workers`,
  `merge-ffmpeg`, `m4b-bitrate`, `segment-lossless`) sont dans
  `code/moteur/patchs-vitesse-et-encodage.diff`. Ils ne sont **pas** commités dans
  le dépôt amont : ils vivent dans l'image Docker `alexandria-audiobook-alexandria`,
  construite le 31 octobre.
- Le mapping des 35 voix est dans `data/voice_config.json`. C'est le travail
  éditorial le plus coûteux du projet ; il n'existait jusqu'ici que dans
  `runtime-data/`.
- `chunks.json` redevient `root:root` après chaque rendu : `chown` avant lecture.
