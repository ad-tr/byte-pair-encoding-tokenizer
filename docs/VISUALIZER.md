# Interface de visualisation

## Architecture technique

### Stack
- **Frontend** : Streamlit
- **Coloration des tokens** : HSL dynamique basée sur token_id
- **State management** : `st.session_state` pour persistance du modèle

### Design choices

**1. Coloration des tokens**
```python
hue = (token_id * 137.5) % 360  # Golden angle
```
Utilise le golden angle (137.5°) pour maximiser la distinction visuelle entre tokens adjacents. Meilleure lisibilité que random ou hash-based.

**2. Mode conversation**
Structure JSON dynamique avec `st.session_state.lines`. Permet de modifier des dialogue avant encoding.

**3. Affichage des caractères spéciaux**
- Espace → `␣` (U+2423)
- Newline → `↵` (U+21B5)
- Tab → `→` (U+2192)

Rend visible la tokenization des whitespaces, critique pour debug.

## Métriques calculées

| Métrique | Calcul | Utilité |
|----------|--------|---------|
| Compression ratio | bytes UTF-8 / nb tokens | Efficacité du vocab |
| Temps encoding | `time.time()` en ms | Perf sur textes longs |
| Historique merges | Top 50 ou total | Audit du training |

## Limites actuelles

- Pas de streaming pour gros textes (>10k tokens lag le navigateur)
- Mode conversation limité

## Extensions possibles

- Export des tokens en JSON/CSV
- Comparaison side-by-side de 2 vocabs
- Heatmap des fréquences de tokens