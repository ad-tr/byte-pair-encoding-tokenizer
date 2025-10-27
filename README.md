# Byte Pair Encoding Tokenizer

Implémentation from-scratch d'un tokenizer BPE avec support des tokens spéciaux, inspiré de GPT-2.

## Installation
```bash
pip install regex streamlit
```

## Interface de visualisation

Une interface Streamlit interactive pour explorer le BPE en temps réel :

```bash
streamlit run src/visualizer.py
```

**Fonctionnalités** :
- Training interactif avec datasets intégrés
- Visualisation colorée des tokens
- Mode document et conversation
- Métriques (compression ratio, temps d'encoding)
- Historique complet des merges



## Usage rapide
```python
from bpe import BytePairEncoder

# Training
tokenizer = BytePairEncoder()
tokenizer.train(text="Votre corpus d'entraînement", vocab_size=512)
tokenizer.save("merges.pkl")

# Encoding
ids = tokenizer.encode("Hello world", mode="document")
text = tokenizer.decode(ids)
```

## Features

- Algorithme BPE classique
- Support tokens spéciaux (`<begin_of_text>`, `<im_start>`, etc.)
- Modes document/conversation
- Regex splitting pattern GPT-2
- Sauvegarde/chargement des merges

## Modes d'encodage

**Document** : Ajoute automatiquement `<begin_of_text>` et `<end_of_text>`

**Conversation** : Structure les dialogues avec `<im_start>`, `<im_sep>`, `<im_end>`
```python
messages = [
    {"role": "System", "message": "Tu es un assistant"},
    {"role": "user", "message": "Bonjour"},
    {"role": "assistant", "message": "Bonjour"}
]
ids = tokenizer.encode(messages, mode="conversation")
```

## Corpus d'entraînement

Ajoutez vos fichiers texte (`.txt`) dans le dossier `/data`. Vous pourrez ensuite les sélectionner dans le visualiseur pour entraîner le tokenizer.

## Limites

- Nécessite un corpus suffisamment large (erreur si vocab_size > paires disponibles)