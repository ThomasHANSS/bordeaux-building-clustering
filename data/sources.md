# Sources de données

## BDNB — Base de Données Nationale des Bâtiments

- **URL** : https://www.data.gouv.fr/fr/datasets/base-de-donnees-nationale-des-batiments/
- **Périmètre** : Département 33 (Gironde)
- **Date de téléchargement** : mars 2026
- **Version** : 2025-07-a
- **Format** : GeoPackage (`data/raw/bdnb_33/gpkg/bdnb.gpkg`)
- **Table** : `batiment_groupe_compile` (277 colonnes, 264 291 bâtiments dept 33)
- **Après filtrage** : 234 116 bâtiments sur Bordeaux Métropole (28 communes)
- **Jointure** : `batiment_groupe_id` (interne BDNB), jointure spatiale pour les autres sources

## DVF — Demandes de Valeurs Foncières (à venir)

- **URL** : https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/
- **Jointure** : spatiale (bâtiment → parcelle DVF) ou par identifiant parcellaire
- **Features** : prix au m², nombre de transactions, date dernière transaction

## Fichiers fonciers (à venir)

- **URL** : https://datafoncier.cerema.fr/
- **Jointure** : identifiant parcellaire ou spatiale
- **Features** : type de propriétaire, surface parcellaire, nombre de locaux

## BD TOPO (à venir)

- **URL** : https://geoservices.ign.fr/bdtopo
- **Jointure** : spatiale
- **Features** : proximité routes, distance au centre, environnement bâti

## INSEE carroyées 200m — Filosofi 2019

- **URL** : https://www.insee.fr/fr/statistiques/7655475?sommaire=7655515
- **Fichier** : `Filosofi2019_carreaux_200m_shp.zip` → `carreaux_200m_met.shp`
- **Emplacement** : `data/raw/insee_carreaux/`
- **Date de téléchargement** : 2026-03-19
- **Millésime** : Filosofi 2019
- **Périmètre** : France métropolitaine, filtré sur bbox Bordeaux Métropole
- **Carreaux sur la zone** : 10 875
- **Jointure** : spatiale (centroïde bâtiment → carreau 200m), `predicate="within"`
- **Couverture** : 230 460 / 234 116 bâtiments (98.4%)
- **Script** : `enrich_insee.py`
- **Sortie** : `data/processed/bdnb_enrichi_insee.geoparquet`
- **Variables récupérées** :
  | Colonne shapefile | Colonne enrichie | Description |
  |---|---|---|
  | `ind` | `insee_population` | Population (individus) |
  | `men` | `insee_menages` | Nombre de ménages |
  | `men_pauv` | `insee_menages_pauvres` | Ménages sous seuil de pauvreté |
  | `men_prop` | `insee_menages_proprietaires` | Ménages propriétaires |
  | `ind_snv` | `insee_niveau_vie_sum` | Somme niveaux de vie winsorisés |
  | — (calculé) | `insee_niveau_vie_moyen` | Niveau de vie moyen (ind_snv / ind) |
  | `log_av45` | `insee_log_avant_1945` | Logements construits avant 1945 |
  | `log_45_70` | `insee_log_1945_1970` | Logements 1945-1970 |
  | `log_70_90` | `insee_log_1970_1990` | Logements 1970-1990 |
  | `log_ap90` | `insee_log_apres_1990` | Logements après 1990 |
  | `men_surf` | `insee_surface_moyenne` | Surface moyenne des logements |
  | `men_coll` | `insee_menages_collectif` | Ménages en logement collectif |
  | `men_mais` | `insee_menages_maison` | Ménages en maison |
- **Note** : les colonnes `Log_90_05` et `Log_ap05` demandées n'existent pas dans Filosofi 2019 ; la ventilation s'arrête à `log_ap90` (après 1990)
