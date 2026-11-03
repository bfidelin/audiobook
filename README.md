# Vernon Subutex — chaîne de production audio

Documentation et code de la chaîne utilisée pour produire l'audiobook *Vernon
Subutex Tome 1* : génération TTS multi-voix, clonage de voix humaines réelles,
sonorisation par fonds générés, mixage et encodage M4B.

## Contenu

- `docs/` — études et comptes rendus de session, avec les mesures.
- `code/voix/` — mesure du débit TTS, conception et correction des voix, QC.
- `code/sonorisation/` — détection de scènes, sélection CLAP des fonds,
  filtrage, mixage avec ducking.
- `code/moteur/` — patchs appliqués au moteur Alexandria (non commités en amont).
- `data/` — configuration des 35 voix, carte des 156 scènes, durées mesurées.

## Repères

- 4 795 segments, 9,14 h de narration, 35 voix.
- 156 scènes sonorisées, 94 % du livre couvert.
- Rendu TTS : 8,03-8,18× temps réel.
- Livraison finale : 9,48 h, 29 parties, AAC 64 kb/s.

## État

Tome 1 livré. Le tome 2 fait la même longueur (107 169 mots) et réutilise
l'intégralité de l'outillage.
