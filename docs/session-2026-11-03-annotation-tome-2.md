# Annotation du script du tome 2 — contournement de l'endpoint LLM

## Le blocage

L'étape qui transforme le texte en script (`{speaker, text, instruct}` par
segment) appelle un LLM. Le point d'entrée configuré pour le tome 1,
`http://192.168.1.190:8888/v1`, **ne répond plus**.

Sans lui, pas de `chunks.json`, donc pas de rendu TTS.

## Le contournement

`app/generate_script.py` construit un client `OpenAI(base_url=..., api_key=...)`.
N'importe quel service compatible OpenAI convient, donc Ollama Cloud :

```json
"llm": {
  "base_url": "https://ollama.com/v1",
  "api_key": "<clé OLLAMA_API_KEY>",
  "model_name": "deepseek-v4.1-flash"
}
```

## Le piège qui aurait tout fait échouer

Les modèles récents **raisonnent avant de répondre**, y compris ceux déclarés
sans raisonnement. Ce raisonnement est décompté de `max_tokens`.

Avec la valeur du tome 1 (`max_tokens: 4096`) et un chunk de 3 000 caractères :

| Modèle | Jetons de sortie | Contenu produit |
|---|---|---|
| `deepseek-v4.1-flash` | 4 096 | **vide** (raisonnement de 20 000 caractères) |
| `mistral-large-4` | 4 096 | **vide** (raisonnement de 15 265 caractères) |
| `gpt-oss:120b` | 4 096 | JSON tronqué |
| `deepseek-v4.1-flash` | **16 384** | JSON valide, 15 entrées |

Chaque chunk serait ressorti vide ou tronqué, avec `finish_reason=length`.
**Il faut `max_tokens: 16384`.**

Résultat sur un vrai passage du tome 2, avec la valeur corrigée : 15 entrées
valides, locuteurs `NARRATOR`, `LAURENT`, `CHARLES` correctement identifiés.

Coût observé : environ 2 000 jetons d'entrée et 4 600-7 000 de sortie par chunk
de 3 000 caractères, soit à peu près **2 $ pour les 200 chunks du tome 2**.

## Isolation du chantier

`_write_script_output()` écrit `../annotated_script.json` et **supprime
`../chunks.json`**. Lancer l'annotation depuis le dossier du tome 1 détruirait
donc les 4 795 segments.

Le tome 2 tourne dans un espace séparé :

```
~/alexandria-tome2/
  app/              copie des scripts + config.json dédiée
  t2.txt            texte extrait de l'EPUB
  default_prompts.txt, persona_prompts.txt, review_prompts.txt
  logs/             llm_responses.log
  annotated_script.json   (sortie)
```

Lancement, sans GPU et sans toucher au tome 1 :

```bash
docker run --rm -v ~/alexandria-tome2:/work -w /work/app \
  alexandria-audiobook-alexandria:latest python generate_script.py ../t2.txt
```

L'image Alexandria est utilisée uniquement parce qu'elle contient le paquet
`openai`, absent de l'hôte.

## Débit

Environ **6 chunks par minute**, soit une trentaine de minutes pour les 201
chunks du tome 2. La sortie standard est bufferisée : la progression se lit dans
`logs/llm_responses.log`, où chaque réponse est préfixée par
`CHUNK n/N | attempt k | finish_reason=...`.

## À surveiller

Certains passages produisent des locuteurs inventés par le modèle — un
`LE COACH` est apparu sur un essai. Il faut confronter la liste finale aux 35
voix existantes et aux 3 rôles du tome 2 (Charles, Antoine, Véro) avant de lancer
le rendu, et reverser les inconnus sur le narrateur ou sur une voix existante.
