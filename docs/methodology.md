# Méthodologie

## Approche générale

Clustering non supervisé de bâtiments à l'échelle de Bordeaux Métropole (28 communes).
Objectif : identifier ~15 typologies de bâtiments interprétables et spatialement cohérentes.

## Pipeline

1. **Chargement** : BDNB dept 33, filtrage sur codes INSEE métropole
2. **Audit de complétude** : ne retenir que les features renseignées à >70%
3. **Feature engineering** : encodage catégoriel, imputation médiane, normalisation StandardScaler
4. **Sélection** : suppression features corrélées (>0.85), VIF (>10), PCA optionnelle
5. **Clustering** : 4 algorithmes comparés (K-Means, HDBSCAN, GMM, hiérarchique)
6. **Évaluation** : silhouette, Davies-Bouldin, Calinski-Harabasz, stabilité bootstrap, Moran's I
7. **Cartographie** : cartes Folium interactives + figures statiques

## Choix du nombre de clusters

- Target initial : k=15 (hypothèse urbanistique)
- Validation par Elbow, Gap Statistic, Silhouette score sur la plage [8, 25]
- Si écart important entre k optimal et k=15 : les deux sont testés et comparés

## Enrichissement itératif

Chaque ajout de source est documenté ici avec :
- Source ajoutée
- Méthode de jointure
- Taux de couverture
- Impact sur les métriques (avant/après)

### Run initial — BDNB seule

- Date : TODO
- Features : TODO
- Meilleur algo : TODO
- Silhouette : TODO
- Moran's I : TODO
- Notes : TODO
