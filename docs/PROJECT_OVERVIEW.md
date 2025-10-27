# BPE Tokenizer - Project Overview

## Contexte

Implémentation from-scratch d'un tokenizer Byte Pair Encoding pour comprendre en profondeur les mécanismes de tokenization des LLMs modernes.

## Objectifs techniques

1. Reproduire l'algorithme BPE de GPT-2
2. Gérer les tokens spéciaux
3. Optimiser pour la lisibilité et la pédagogie du code

## Défis rencontrés

**Challenge #1** : Intégration des special tokens dans le pipeline UTF-8
- Problème : Les tokens comme `<begin_of_text>` ne sont pas des bytes valides
- Solution : Unicode Private Use Area comme couche d'abstraction

**Challenge #2** : Gestion des erreurs de vocab_size
- Problème : Si corpus trop petit, impossible d'atteindre le vocab demandé
- Solution : Détection early avec message explicite

## Résultats

- ~130 lignes de Python pur (hors commentaires)
- Compatible avec le format de tokenization GPT-2