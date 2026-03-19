# Clustering Bâtiments — Bordeaux Métropole

Identification de 15 typologies de bâtiments sur les 28 communes de Bordeaux Métropole par clustering non supervisé, à partir de la BDNB et de sources complémentaires.

## Objectif

Classifier les bâtiments de la métropole en ~15 types interprétables (pavillonnaire récent, collectif dense ancien, tertiaire, industriel…) en combinant plusieurs sources de données urbaines et en comparant différents algorithmes de clustering.

## Démarrage rapide

```bash
# 1. Cloner et installer
git clone https://github.com/VOTRE_USER/bordeaux-building-clustering.git
cd bordeaux-building-clustering
pip install -e ".[dev]"

# 2. Placer les données BDNB dans data/raw/bdnb_33/

# 3. Lancer le pipeline complet
make all
```

## Pipeline

| Étape | Commande | Module |
|-------|----------|--------|
| Chargement BDNB | `make data` | `src/data_loader.py` |
| Feature engineering | `make features` | `src/features.py` |
| Sélection features | `make select` | `src/feature_selection.py` |
| Clustering | `make cluster` | `src/clustering.py` |
| Évaluation | `make evaluate` | `src/evaluation.py` |
| Cartographie | `make map` | `src/mapping.py` |

## Configuration

Tous les paramètres sont centralisés dans `config.yaml` : codes INSEE, algorithmes, seuils, chemins. Aucun paramètre en dur dans le code.

## Sources de données

Voir `data/sources.md` pour le détail des sources, liens de téléchargement et dates.

- **BDNB** (source primaire) : surface, hauteur, âge, DPE, usage
- **DVF** (à venir) : prix au m², transactions
- **Fichiers fonciers** (à venir) : propriétaires, parcelles
- **BD TOPO** (à venir) : environnement bâti
- **INSEE carroyées** (à venir) : revenus, densité

## Résultats

- Cartes interactives : `outputs/maps/`
- Figures statiques : `outputs/figures/`
- Log des expériences : `outputs/experiments.json`

## Méthodologie

Voir `docs/methodology.md`.
