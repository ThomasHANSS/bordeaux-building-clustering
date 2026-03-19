# Sources de données

## BDNB — Base de Données Nationale des Bâtiments

- **URL** : https://www.data.gouv.fr/fr/datasets/base-de-donnees-nationale-des-batiments/
- **Périmètre** : Département 33 (Gironde)
- **Date de téléchargement** : TODO
- **Version** : TODO
- **Format** : CSV / GeoJSON / GeoParquet
- **Jointure** : `batiment_groupe_id` (interne BDNB), jointure spatiale pour les autres sources
- **Colonnes clés** : TODO (compléter après notebook 01)
- **Taux de remplissage** : TODO (compléter après audit de complétude)

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

## INSEE carroyées 200m (à venir)

- **URL** : https://www.insee.fr/fr/statistiques/7655475
- **Jointure** : spatiale (bâtiment → carreau 200m)
- **Features** : revenu médian, densité de population, taux de logements vacants
