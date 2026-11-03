# Casting du tome 2 — ce qui change par rapport au tome 1

Analyse du 3 novembre sur l'EPUB du tome 2 (`579,7 Ko`, **107 169 mots** contre
106 965 au tome 1 : même longueur à 0,2 % près).

Méthode : extraction du texte, comptage des noms propres en milieu de phrase,
puis rattachement des répliques en tiret au locuteur par les verbes
d'attribution, et enfin examen des blocs narratifs où chaque nom apparaît pour
compter les répliques réellement présentes.

## Deux personnages à cloner

### CHARLES — majeur

Absent du tome 1 (`0` occurrence), **68 mentions** dans le tome 2. Il apparaît
dans deux grands blocs narratifs :

| Bloc (paragraphes) | Taille | Répliques |
|---|---|---|
| 3741-3906 | 166 par. | 39 |
| 4950-5085 | 136 par. | 38 |

C'est un personnage qui parle autant que Patrice ou Gaëlle. Il vit avec la Véro,
il est décrit comme « le vieux Charles » — donc une voix d'homme âgé. **Sans voix
dédiée, ses 77 répliques tomberaient sur le narrateur.**

### ANTOINE — secondaire

Absent du tome 1, **40 mentions**. C'est le fils de Dopalet. Trois blocs :

| Bloc | Répliques |
|---|---|
| 1450-1591 | 2 |
| 1617-1640 | 7 |
| 2630-2683 | 10 |

Il a de vraies scènes de dialogue, dont l'altercation sur la passerelle avec
Aïcha et Céleste. Homme jeune.

### VÉRO — mineur

**22 mentions**, une seule réplique proche. Compagne de Charles. Une voix
optionnelle ; à défaut, une seule réplique peut rester au narrateur sans
dommage.

## Faux positifs écartés

Vérification faite, ces noms ne sont pas de nouveaux personnages :

| Nom | Pourquoi ce n'est pas un nouveau rôle |
|---|---|
| **Bleach** (41) | Alex Bleach — la voix `ALEX` existe déjà |
| **Kant** (39) | Pamela Kant — la voix `PAMELA` existe déjà |
| **Rosa** (29) | Le bar *Rosa Bonheur*, un lieu |
| **Bazooka** (12) | Lydia Bazooka — la voix `LYDIA` existe déjà |
| **Satana** (48) | Faïza, qui a changé de nom ; mentionnée, ne parle pas |
| **Sylvie** (34) | Amie d'Olga, citée, zéro réplique |
| **Pénélope** (9) | Citée, zéro réplique |
| **Clara** (12) | Un bébé, zéro réplique |
| **Kevin** (9) | Cité, zéro réplique |
| **Francesca** (5) | Citée, zéro réplique |
| **Max** (4) | Max Romeo, musicien |
| **Gabriel** (4) | Un ami cité |
| **Buttes-Chaumont** | Un lieu |
| **Marie-Ange** (30) | La voix `MARIE-ANGE` existe déjà |

## Poids des personnages entre les deux tomes

| Personnage | T1 | T2 | Évolution |
|---|---|---|---|
| Loïc | 21 | **86** | ×4 |
| Anaïs | 8 | **67** | ×8 |
| Céleste | 4 | **64** | ×16 |
| Olga | 5 | **56** | ×11 |
| Sélim | 17 | 42 | ×2,5 |
| Pamela | 50 | 89 | ×1,8 |
| Xavier | 103 | 126 | stable |

| Personnage | T1 | T2 | Évolution |
|---|---|---|---|
| Marcia | 44 | 3 | quasi disparu |
| Cécile | 29 | 2 | quasi disparu |
| Kiko | 34 | 10 | en retrait |
| Alexandre | 18 | 2 | quasi disparu |
| Bertrand | 12 | 0 | absent |
| Sophie | 13 | 0 | absent |
| Déborah | 7 | 0 | absent |
| Cassandre | 9 | 0 | absent |
| JP | 3 | 0 | absent |
| Paul | 10 | 1 | absent |

Les voix des absents peuvent être archivées, pas supprimées : elles resserviront
au tome 3 (`735 Ko`, le plus long des trois).

## État des références de clonage

Les 35 références sont présentes et valides, toutes au-dessus de 16 s — le
problème des previews de 3,6 s relevé au tome 1 est corrigé.

Mais **26 voix sur 35 sont encore clonées à partir d'une preview VoiceDesign**,
c'est-à-dire d'une voix de synthèse servant de référence à une seconde synthèse.
C'est exactement la cause du timbre métallique identifiée au tome 1, corrigée
seulement pour 9 voix (NARRATOR, VERNON, LA HYÈNE, OLGA, PATRICE, GAËLLE,
PAMELA, ALEXANDRE, LYDIA).

Pour Charles et Antoine, il faut donc des enregistrements humains réels de
10-30 s, pas des previews.

## Prérequis bloquant

Le point d'entrée LLM configuré (`http://192.168.1.190:8888/v1`) **ne répond
pas**. Sans lui, pas d'annotation du script : c'est l'étape qui produit les
4 795 entrées `{speaker, text, instruct}`. Le rendu TTS ne peut pas démarrer
avant que ce service soit de nouveau disponible.

## Voix retenues

| Rôle | Voix | Source | F0 |
|---|---|---|---|
| CHARLES | Jean-Pierre Darroussin | YouTube `oKWuSROw0wU`, 357-381 s (entretien autour de *Des hommes*) | 98 Hz |
| ANTOINE | Pierre Niney | YouTube `fLWvbnxVOnI`, 558-578 s (« L'interview sans filtre ») | 136 Hz |
| VÉRO | voix `CECILE` existante | — | — |

Charles et Antoine viennent de parole humaine réelle, pas de previews
synthétiques. Les deux voix masculines sont nettement distinctes (98 contre
136 Hz), ce qui les rend reconnaissables en dialogue.

Véro n'a **qu'une réplique** dans tout le tome 2 : plutôt que de courser une
source propre pour un seul mot, la rôle est rattaché à `CECILE`, voix féminine
déjà conçue et absente du tome 2.

Une tentative sur Josiane Balasko a été écartée : l'extrait choisi
automatiquement (`fbnLwDtIljs`, 808-828 s) mesurait 133 Hz, dans la plage
masculine — c'était l'intervieweur, pas elle. Leçon : toujours vérifier le F0
avant d'accepter un extrait, la note de qualité du script ne détecte pas un
changement de locuteur.

## Contrôle avant rendu

`code/voix/audit_locuteurs.py` confronte les locuteurs du script annoté aux voix
configurées et signale ceux qui n'en ont pas. À lancer systématiquement : le
modèle invente des personnages.
