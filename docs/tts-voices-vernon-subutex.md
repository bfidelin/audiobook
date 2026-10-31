# Audiobook local — recherches voix, casting et moteurs TTS

Mise à jour : 31 octobre 2026. Notes de recherche pour Alexandria / Qwen3-TTS. Les propositions de casting sont subjectives et les liens YouTube de recherche **ne sont pas** des URL de vidéos vérifiées.

## Vidéo principale : BigIron

- **Turn Your Ebooks Into Audiobooks Locally (and the Truth About Voice Cloning)** : https://youtu.be/LBaHFL_NFq8
- Chapitres : https://youtu.be/LBaHFL_NFq8?t=0 (introduction) ; https://youtu.be/LBaHFL_NFq8?t=31 (pipeline) ; https://youtu.be/LBaHFL_NFq8?t=116 (choisir la voix) ; https://youtu.be/LBaHFL_NFq8?t=348 (clonage) ; https://youtu.be/LBaHFL_NFq8?t=492 (bibliothèque).
- Guides de l'auteur :
  - https://www.bigiron.cc/guides/self-hosted-read-aloud-tts-for-your-ebook-library
  - https://www.bigiron.cc/guides/voice-cloning-self-hosted-coqui-tts-vs-piper-vs-xtts
  - https://www.bigiron.cc/guides/piper-vs-kokoro-vs-xtts-tts-for-home-assistant-voice
  - https://www.bigiron.cc/guides/audiobook-chapterization-and-metadata-the-m4b-pipeline
  - https://www.bigiron.cc/guides/audiobookshelf-vs-booksonic-vs-libation

### Faits annoncés dans la vidéo (non benchmarkés sur notre machine)
- Pipeline : EPUB → Calibre / texte par chapitre → TTS local → M4B chapitré → Audiobookshelf.
- Kokoro-82M : environ **25× temps réel sur GTX 1070** et **5× sur CPU 12 cœurs** selon l'auteur. Une APU testée y était moins performante via GPU que via CPU.
- Piper : léger et rapide, voix potentiellement moins naturelles ; Kokoro : voix naturelles et rapide ; XTTS-v2 : clonage zero-shot avec environ **6 s** de référence, ~17 langues selon la vidéo.
- XTTS-v2 : vérifier la Coqui Public Model License (usage non commercial) et le consentement. Ne pas confondre usage local et absence de droits.
- Une API compatible OpenAI `/v1/audio/speech` peut exposer un moteur rapide et un moteur qualité.
- La prononciation des noms propres et les corrections phonétiques représentent une part du travail.

### Conséquences pour Alexandria
- Mesure utilisateur historique : ~10 h d'audio produites en un peu plus de 2 h (~5× temps réel) avec Alexandria. Comparer sur **le même chapitre français**, la même durée, la qualité, les erreurs, le temps de calcul, VRAM et le temps de préparation.
- Conserver Qwen3-TTS comme moteur principal pour narration expressive multivoix ; évaluer Kokoro pour lecture rapide et XTTS-v2 pour la fidélité de clonage, sans supposer que l'un est meilleur.
- Alexandria dispose déjà d'une fonctionnalité d'import de voix : **ne pas redévelopper un import YouTube/voice-import**.
- Préférer de vraies voix de référence naturelles au changement de timbre par instruction de contexte ; utiliser l'instruction surtout pour une émotion modérée.
- Référence conseillée : 10–30 s de parole isolée, peu de musique/bruit, sans chevauchement, et transcription correcte quand nécessaire. C'est une recommandation de test, non une exigence universelle.
- Mesurer les résultats avant d'importer 20–25 voix.

## Casting vocal proposé — Vernon Subutex

**Ce tableau est un casting créatif, pas une attribution officielle du roman.** Les URL de recherche ci-dessous permettent de trouver une interview ; aucune plage temporelle n'a été écoutée et validée.

| Personnage | Choix A | Choix B | Timbre recherché |
|---|---|---|---|
| Vernon | Romain Duris | Mathieu Amalric | détaché, fatigué, légèrement rauque |
| Alex Bleach | Athaya Mokonzi | Omar Sy | grave, chaleureux |
| Xavier Fardin | Philippe Rebbot | François Morel | rocailleux, râleur |
| La Hyène | Céline Sallette | Marina Foïs | sec, rapide, affirmé |
| Sylvie | Florence Thomassin | Karin Viard | mature, nerveux |
| Lydia Bazooka | Calypso Valois | Léa Drucker | clair, intellectuel |
| Laurent Dopalet | Laurent Lucas | Denis Podalydès | bourgeois, calme, inquiétant |
| Émilie | Émilie Gavois-Kahn | Corinne Masiero | direct, populaire |
| Pamela Kant | Juana Acosta | Béatrice Dalle | chaud, grave |
| Olga | Yolande Moreau | Corinne Masiero | usé, tendre |
| Sophie | Josiane Balasko | Miou-Miou | âgé, mordant |
| Kiko | Benoît Magimel | Vincent Cassel | nerveux, décadent |
| Patrice | Vincent Lindon | Reda Kateb | tendu, sec |
| Sélim | Roschdy Zem | Sami Bouajila | grave, posé |
| Aïcha | Zita Hanrot | Déborah Lukumuena | jeune, déterminé |
| Vodka Satana / Faïza | Béatrice Dalle | autre voix féminine grave | cassé, provocant |
| Marcia | Inès Rau | voix francophone brésilienne | féminin, légère coloration brésilienne |
| Gaëlle | Valérie Donzelli | Florence Loiret Caille | naturel |
| Marie-Ange | Léa Drucker | Agnès Jaoui | posé |
| Julien | Finnegan Oldfield | Anthony Bajon | jeune, tendu |
| Noël | Vincent Macaigne | Damien Bonnard | instable |
| Loïc | Anthony Bajon | Paul Kircher | jeune, clair |

### URL YouTube directes mentionnées précédemment (à revalider avant import)

- Béatrice Dalle, interview RTBF 1992 : https://www.youtube.com/watch?v=fJ4Z9eWja0g
- Béatrice Dalle, France 2 : https://www.youtube.com/watch?v=aAD3lwtFykM
- Béatrice Dalle, Clique : https://youtu.be/Mct3hlUfHOI
- Josiane Balasko, INA : https://www.youtube.com/watch?v=WnPjimEYsmM
- Josiane Balasko, Canal+ : https://www.youtube.com/watch?v=MCppMRw9eIE
- Benoît Magimel, interview France TV : https://www.youtube.com/watch?v=WNM1U1M5k_A
- Benoît Magimel, Cannes : https://www.youtube.com/watch?v=6GC4SQc2Rlc

**Attention :** ces identifiants ont été fournis au cours de la conversation mais n'ont pas tous été revérifiés individuellement. Aucun timestamp « meilleur segment » n'a été vérifié.

### Liens vers interviews / recherches

- Romain Duris : https://www.youtube.com/results?search_query=Romain+Duris+interview+longue
- Philippe Rebbot : https://www.youtube.com/results?search_query=Philippe+Rebbot+interview+longue
- Céline Sallette : https://www.youtube.com/results?search_query=Celine+Sallette+interview+longue
- Florence Thomassin : https://www.youtube.com/results?search_query=Florence+Thomassin+interview
- Yolande Moreau : https://www.youtube.com/results?search_query=Yolande+Moreau+interview+longue
- Vincent Lindon : https://www.youtube.com/results?search_query=Vincent+Lindon+interview+longue
- Roschdy Zem : https://www.youtube.com/results?search_query=Roschdy+Zem+interview+longue
- Philippe Rebbot, entretien vidéo Pétaouchnok (page tierce) : https://leschroniquesdecliffhanger.com/2022/11/12/petaouchnok-entretien-video-avec-edouard-deluc-et-philippe-rebbot-philippe-me-donne-envie-de-faire-du-cinema/
- Céline Sallette, leçon de cinéma (page tierce) : https://www.sooner.fr/articles/celine-sallette

## Archives françaises / domaine public

- Gallica, voix des femmes : https://gallica.bnf.fr/selections/fr/html/enregistrements-sonores/voix-des-femmes
- Gallica, voix du théâtre et du music-hall : https://gallica.bnf.fr/selections/fr/html/enregistrements-sonores/voix-du-theatre-et-du-music-hall
- Candidates historiques à étudier : Sarah Bernhardt, Julia Bartet, Cécile Sorel, Polaire, Mistinguett, Berthe Reynold, Camille Pottecher.
- Ne pas assimiler « texte littéraire du domaine public » à « enregistrement sonore libre de droits ». Vérifier pour chaque phonogramme publication, interprétation, producteur et conditions de réutilisation. Le clonage d'une voix d'acteur vivant peut soulever des droits distincts, même pour un usage privé.
- Références juridiques générales : https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000043499688 et https://www.cnil.fr/fr/hypertrucage-deepfake

## À faire
1. Vérifier les 7 URL YouTube directes et sélectionner 10 interviews supplémentaires avec URL `watch?v=` précise.
2. Faire écouter / tester dans Alexandria, sélectionner les segments via son import existant.
3. Évaluer une phrase neutre + une phrase expressive + un passage de 3–5 min pour chaque clone.
4. Documenter naturel, fidélité, erreurs de prononciation, stabilité, vitesse, VRAM.
5. Conserver un registre de provenance et de droits pour chaque voix ; ne pas publier de clones d'acteurs sans autorisation.
