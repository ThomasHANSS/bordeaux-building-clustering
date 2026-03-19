# Clustering Bâtiments — Bordeaux Métropole

## Rôle
Data scientist spécialisé en analyse urbaine et géospatiale.
Projet : clustering de ~15 typologies de bâtiments sur Bordeaux Métropole (28 communes, département 33).

## Données
- **Source primaire** : BDNB (Base de Données Nationale des Bâtiments), dept 33, CSV/Parquet/GeoJSON
- **Périmètre** : 28 communes de Bordeaux Métropole (codes INSEE dans config.yaml)
- **Sources futures** : DVF, Fichiers fonciers, BD TOPO, INSEE carroyées, PLU
- **Volume** : plusieurs Go → toujours lire par chunks (pandas `chunksize`)
- **Référence** : `data/sources.md` pour les liens et dates de téléchargement

## Stack
Python 3.11+ | pandas | geopandas | scikit-learn | hdbscan |
umap-learn | folium | matplotlib | seaborn | pysal (esda, libpysal) |
pyyaml | scipy | kneed | python-dotenv | pyarrow | statsmodels | pytest

## Conventions code
- Tout paramètre va dans `config.yaml`, jamais en dur dans le code
- Lire la config via `src/config.py → load_config()`
- Initialiser le projet via `src/config.py → setup_project()` (config + logging)
- Code réutilisable dans `src/`, exploration dans `notebooks/`
- `random_state=42` partout pour reproductibilité
- Typage des fonctions (type hints)
- Docstrings numpy-style
- Chaque module `src/*.py` doit avoir un bloc `if __name__ == "__main__"` pour être exécutable via `python -m src.module_name`

## Column mapping — IMPORTANT
- Ne JAMAIS utiliser de noms de colonnes BDNB en dur dans le code
- Toujours passer par le mapping défini dans `config.yaml → features.column_mapping`
- Le notebook 01 sert à explorer le schéma réel de la BDNB et compléter le mapping
- Quand le mapping est incomplet, lever une erreur explicite, ne pas deviner

## Géométries — IMPORTANT
- Toujours valider les géométries (`make_valid`, filtrer `is_empty`, `notna`)
- Exploser les MultiPolygon (`explode`) et créer un `uid` unique post-explode
- Filtrer les micro-géométries < 1 m² (artefacts)
- Calculs de distance/surface en EPSG:2154, affichage carte en EPSG:4326
- Si un CRS est absent → hypothèse WGS84 avec warning
- Utiliser `src/geo_utils.py` pour toute manipulation géométrique

## Format pivot — IMPORTANT
- Le format intermédiaire entre les étapes du pipeline est le **GeoParquet**
- Les géométries doivent TOUJOURS être conservées à travers le pipeline
- Chemins définis dans `config.yaml → pipeline`
- `features_ready.geoparquet` : features + géométries (sortie de features.py)
- `clustered.geoparquet` : features + géométries + labels clusters (sortie de clustering.py)

## Pipeline complet

### Étape 0 — Audit de complétude (notebook 01)
- AVANT TOUT : calculer le % de remplissage par colonne sur la zone
- Ne retenir que les features renseignées à >70%
- Compléter le column_mapping dans config.yaml
- Documenter dans `data/sources.md`

### Étape 1 — Chargement (data_loader.py)
- Charger BDNB par chunks, filtrer sur codes INSEE métropole
- Appeler `geo_utils.prepare_geodf()` systématiquement en fin de chargement
- Sauvegarder en GeoParquet dans `data/processed/`
- Vérifier qu'aucune commune ne produit 0 bâtiment (warning si c'est le cas)

### Étape 2 — Feature engineering (features.py)
- Appliquer le column_mapping
- Encoder les catégorielles : cardinalité ≤ seuil config → one-hot, sinon ordinal
- Normaliser avec StandardScaler, sauvegarder le scaler dans `outputs/`
- Sauvegarder `features_ready.geoparquet` (features normalisées + géométries)

### Étape 3 — Sélection de features (feature_selection.py)
- Matrice de corrélation → supprimer si > seuil config
- VIF → supprimer si > seuil config
- Optionnel : PCA/UMAP si activé dans config (`features.reduction`)
- Logger les features supprimées et conservées

### Étape 4 — Clustering (clustering.py)
- Déterminer k optimal : Elbow (inertie), Gap Statistic, Silhouette par k sur la plage `k_range`
- Si le k optimal détecté est très différent de 15, documenter l'écart, tester les deux, présenter les résultats comparatifs
- Lancer les 4 algos définis dans config
- Sauvegarder `clustered.geoparquet`

### Étape 5 — Évaluation (evaluation.py)
- Calculer les métriques : silhouette, Davies-Bouldin, Calinski-Harabasz
- Stabilité : bootstrap (n=config), mesurer ARI entre runs
- Cohérence spatiale : Moran's I sur les labels (spatial_analysis.py)
- Tableau comparatif + visualisations radar/parallel coordinates
- Logger chaque run dans `outputs/experiments.json`

### Étape 6 — Cartographie (mapping.py)
- Folium pour cartes interactives (HTML dans outputs/maps/)
- Palette 15 couleurs distinctes, colorblind-safe
- Popups avec infos bâtiment + cluster
- Couche par cluster activable/désactivable
- GeoPandas `.plot()` pour figures statiques dans outputs/figures/

## Bruit HDBSCAN (label = -1)
- Toujours reporter le % de bruit dans experiments.json
- Si bruit > 20% : ajuster `min_cluster_size` / `min_samples`
- Pour la carto finale : réaffecter le bruit au cluster le plus proche (nearest centroid), mais les marquer visuellement (transparence réduite)

## Validation visuelle obligatoire
Après chaque clustering, vérifier sur carte au minimum :
- Centre historique de Bordeaux (bâti ancien dense)
- Mériadeck (grands ensembles)
- Caudéran / Le Bouscat (pavillonnaire)
- Bassins à flot / Euratlantique (construction récente)
- Zone d'activité de Pessac-Bersol (tertiaire/industriel)
Si un clustering mélange ces quartiers dans les mêmes types, il est probablement mauvais.

## Jointures entre sources — TOUJOURS SPATIALES PAR DÉFAUT
- Utiliser `src/geo_utils.spatial_join()` pour toute jointure inter-sources
- AVANT tout sjoin : reprojeter en EPSG:2154, valider les géométries
- Gérer systématiquement le cas 1:N (`smart_aggregate`)
- Numériques → area_weighted. Catégorielles → majority.
- Jointure par identifiant UNIQUEMENT au sein de la BDNB (tables internes)
- Après chaque jointure d'enrichissement : reporter le taux de couverture
- Si une feature enrichie est renseignée à < 50%, la traiter comme optionnelle et tester le clustering avec et sans
- Documenter chaque jointure : source gauche, source droite, prédicat, taux de matching, stratégie d'agrégation

## Jointures spatiales — Gestion des 4 risques
1. **CRS** : `prepare_geodf()` reprojette tout en 2154. Si CRS absent → hypothèse WGS84 avec warning.
2. **1:N** : Toujours passer par `smart_aggregate()`. Numériques → area_weighted. Catégorielles → majority.
3. **Géométries** : `prepare_geodf()` fait make_valid + explode + filtre <1m². Toujours vérifier le log du nombre d'entités perdues.
4. **Performance** : Chunking auto au-delà du seuil config. Simplification des géométries avant sjoin (tolérance dans config). Ne JAMAIS faire d'overlay complet sans nécessité — sjoin d'abord, overlay seulement si min_overlap est requis.

## Enrichissement itératif
Quand une nouvelle source est ajoutée :
1. Créer une fonction de chargement dédiée dans `data_loader.py`
2. Documenter dans `data/sources.md`
3. Joindre via `geo_utils.spatial_join()` aux données existantes
4. Mettre à jour `config.yaml` (column_mapping + features)
5. Relancer le pipeline complet : `make all`
6. Comparer les métriques avant/après dans experiments.json
7. Documenter le delta dans `docs/methodology.md`

## Gestion mémoire
- Lire les CSV par chunks (`pd.read_csv(..., chunksize=config)`)
- Convertir en Parquet / GeoParquet dès que possible
- Utiliser `dtype` explicites pour réduire l'empreinte
- `del` + `gc.collect()` après les grosses opérations
- Si nécessaire, travailler d'abord sur 1 commune test (Bordeaux, 33063)

## Logging
- Utiliser le module `logging` standard de Python
- Configuration dans config.yaml (level, file)
- Initialiser dans chaque module : `logger = logging.getLogger(__name__)`
- Setup centralisé dans `src/config.py → setup_project()`

## Format experiments.json
```json
{
  "runs": [{
    "id": "run_001",
    "date": "2025-01-15T14:30:00",
    "algorithm": "hdbscan",
    "params": {"min_cluster_size": 100},
    "n_clusters": 14,
    "n_noise": 230,
    "noise_pct": 0.12,
    "features_used": ["surface_bat", "hauteur_bat"],
    "n_features": 8,
    "n_samples": 185000,
    "metrics": {
      "silhouette": 0.42,
      "davies_bouldin": 0.87,
      "calinski_harabasz": 1250,
      "moran_i": 0.65
    },
    "data_sources": ["bdnb"],
    "notes": "Premier run BDNB seule"
  }]
}
```

## Git — OBLIGATOIRE
- Format des messages : "type: description courte"
  - types : data, features, clustering, eval, map, fix, config, docs
- Ne JAMAIS accumuler plusieurs changements non liés dans un seul commit
- Commiter le config.yaml ET le code quand les deux changent ensemble
- AVANT de commiter : vérifier que le code s'exécute sans erreur
- Ne PAS commiter automatiquement après chaque action
- Proposer le commit à l'utilisateur avec le message prévu, et attendre sa validation avant d'exécuter git commit

## Encodage catégoriel
- Cardinalité ≤ seuil config (`encoding.onehot_max_cardinality`) → one-hot encoding
- Cardinalité > seuil → ordinal encoding ou target encoding
- Seuil configurable dans config.yaml

## État actuel du projet (mars 2026)

### Données
- La BDNB est en format **GeoPackage** : `data/raw/bdnb_33/gpkg/bdnb.gpkg`
- Table principale : `batiment_groupe_compile` (277 colonnes)
- Charger avec filtre SQL pour éviter les problèmes mémoire :
```python
  gdf = gpd.read_file(gpkg_path, layer='batiment_groupe_compile',
      where=f"code_commune_insee IN ('{codes_sql}')")
```
- **264 291 bâtiments** sur le dept 33, **234 116** après filtrage métropole + surface
- Le Codespace a ~8 Go de RAM : ne JAMAIS charger le gpkg entier sans filtre SQL

### DVF déjà intégrée dans la BDNB
Les colonnes `dvf_open_*` sont déjà dans `batiment_groupe_compile`.
Pas besoin de jointure spatiale séparée pour les prix.
Complétude : ~25% (feature optionnelle, pas primaire).

### Premier clustering réalisé (run_001 à run_003)
- Algorithme retenu : **KMeans k=15** (silhouette=0.350, DB=1.07, CH=52911)
- GMM : silhouette=0.125 (mauvais)
- Agglomerative : silhouette=0.324 (correct mais inférieur)
- Résultats sauvegardés dans `data/processed/clustered.geoparquet`
- Détail des runs dans `outputs/experiments.json`

### Les 15 clusters identifiés
- 0: Pavillon plain-pied (157m², 4m) — 10 890 bâtiments
- 1: Petit bâti non classé (40m², 3m, usage inconnu) — 46 823 bâtiments ⚠️ fourre-tout
- 2: Maison R+1 (101m², 6m) — 30 495
- 3: Pavillon standard (133m², 4m) — 56 137
- 4: Tertiaire moyen (1132m², 5m) — 1 278
- 5: Petit collectif (820m², 11m) — 2 830
- 6: Grand collectif (3226m², 13m) — 222
- 7: Collectif R+4/R+5 (135m², 15m) — 6 449
- 8: Individuel dense R+2 (99m², 9m) — 15 571
- 9: Individuel compact (94m², 5m) — 27 906
- 10: Grand équipement (4794m², 6m) — 720
- 11: Maison R+1 type (101m², 5m) — 25 335
- 12: Tour / IGH (704m², 23m) — 349
- 13: Grande maison (282m², 4m) — 1 211
- 14: Tertiaire courant (409m², 7m) — 7 900

### Scripts existants
- `first_run.py` : clustering complet (chargement → KMeans/GMM/Agglo → évaluation → sauvegarde)
- `generate_report_v2.py` : rapport PDF avec cartes, tableaux, règles
- `make_map_v2.py` : cartes statiques + Folium

### Points d'attention connus
- Le cluster 1 (46k bâtiments, 20%) est un fourre-tout à investiguer
- Les clusters 2, 9, 11 se ressemblent (individuel ~100m²) — à différencier
- HDBSCAN n'a pas encore été testé
- Moran's I (cohérence spatiale) non calculé
- Agglomerative nécessite un échantillon <10k (mémoire)
- Les features DPE (<20%) et DVF (<25%) sont exclues du clustering primaire

## Conventions de numérotation — OBLIGATOIRE

### Expériences (experiments.json)
- L'ID suit le format `run_XXX` avec numéro incrémental
- Lire experiments.json AVANT d'ajouter un run pour connaître le dernier numéro
- Ne JAMAIS écraser un run existant

### Rapports PDF
- Format : `clustering_bordeaux_vX.pdf` (v1, v2, v3...)
- Lire le dossier `outputs/reports/` AVANT de générer pour connaître le dernier numéro
- Chaque nouveau rapport incrémente la version

### Figures
- Préfixer par le numéro de version : `report_v3_carte.png`, `report_v3_zoom.png`
- Ne PAS écraser les figures des versions précédentes

### Scripts
- Garder les scripts précédents (first_run.py, generate_report_v2.py)
- Nommer les nouveaux : `clustering_run_v2.py`, `generate_report_v3.py`

## Cartes HTML interactives — IMPORTANT
- Ne JAMAIS mettre les 234k polygones dans une seule carte Folium/HTML
- Générer des cartes par zone avec les VRAIS polygones (pas des centroïdes)
- Simplifier les géométries avec `simplify(1.0)` avant export Folium (1m en Lambert-93)
- Zones de cartes à générer :
  - Métropole entière : simplifier + limiter à 30k polygones échantillonnés
  - Centre Bordeaux : bbox [410000, 6424000, 416000, 6430000]
  - Mériadeck : bbox [411000, 6425500, 413500, 6428000]
  - Caudéran / Le Bouscat : bbox [408500, 6427000, 411500, 6430000]
  - Bassins à flot : bbox [412000, 6428500, 414000, 6430500]
- Chaque carte : polygones colorés par cluster avec popup info + légende avec noms
- Reprojeter en EPSG:4326 avant export Folium
- Utiliser folium.GeoJson (pas CircleMarker) pour afficher les polygones
- Sauver dans `outputs/maps/` avec noms explicites
