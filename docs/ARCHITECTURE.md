# Architecture technique

## Vue d'ensemble

Implémentation Python pure d'un tokenizer BPE basé sur l'approche GPT-2, avec ajouts :
- Gestion des tokens spéciaux via Unicode Private Use Area
- Structure de données optimisée pour le training itératif

## Pipeline de tokenization
```
Texte brut 
  → Injection tokens spéciaux (via placeholders Unicode)
  → Regex splitting (pattern GPT-2)
  → Conversion UTF-8 bytes
  → Application des merges appris
  → Liste d'IDs
```

## Décisions de design

### 1. Structure de données pour l'entraînement

**Choix** : `List[List[int]]` au lieu de `List[int]` flat

**Raison** : Préserve les frontières de mots pendant le training. Évite les merges inter-mots qui polluent le vocabulaire.

**Trade-off** : Overhead mémoire léger, mais gain en qualité du vocab.

### 2. Gestion des tokens spéciaux

**Problème** : Comment insérer des tokens non-UTF-8 dans le pipeline ?

**Solution** : Unicode Private Use Area (U+E000) comme placeholders temporaires

- Injection avant le regex split
- Mapping vers IDs réservés (100001-100005)
- Pattern regex adapté pour capturer ces placeholders

**Alternative rejetée** : Réserver des bytes (ex: 0xFF, 0xFE) → conflits avec textes multilingues

### 3. Regex pattern GPT-2
```python
r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

Capture les unités linguistiques naturelles (mots, nombres, ponctuation) avec gestion des espaces optionnels. Améliore la cohérence du vocabulaire vs. split caractère par caractère.

## Complexité algorithmique

### Training
- `_get_stats()` : O(n) où n = nb total de tokens
- Boucle principale : O(vocab_size × n)
- **Pire cas** : O(vocab_size × n²) si merges fragmentent beaucoup

### Encoding/Decoding
- Encode : O(m × len(merges)) où m = longueur du texte
- Decode : O(k) où k = nb d'IDs

## Composants du projet
```
├── src/
│   ├── tokenizer.py       # Core BPE
│   ├── basic-usage.py     # Utilisation basique du BPE
│   └── visualizer.py      # Interface Streamlit
├── data/
│   ├── save/              # Sauvegardes post training
│   ├── conversation/      # Exemple de conversation
│   ├── *.txt              # Corpus d'entraînement d'exemple
│   └── save/              # Modèles sauvegardés (.pkl)
└── docs/                  # Documentation
```

### Séparation des responsabilités

- `BytePairEncoder` : logique pure, agnostique à l'UI
- `visualizer.py` : layer de présentation, aucune logique métier