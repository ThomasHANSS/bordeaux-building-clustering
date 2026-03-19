# Dictionnaire des features

## Features projet (noms normalisés)

| Nom projet | Nom BDNB réel | Type | Unité | Description |
|------------|---------------|------|-------|-------------|
| surface_bat | TODO | num | m² | Surface au sol du bâtiment |
| hauteur_bat | TODO | num | m | Hauteur du bâtiment |
| nb_etages | TODO | num | - | Nombre de niveaux |
| annee_construction | TODO | num | année | Année de construction |
| surface_parcelle | TODO | num | m² | Surface de la parcelle |
| dpe_conso_energie | TODO | num | kWh/m²/an | Consommation énergétique (DPE) |
| usage_principal | TODO | cat | - | Usage principal (résidentiel, tertiaire…) |
| materiaux_murs | TODO | cat | - | Matériau des murs |
| type_chauffage | TODO | cat | - | Type de chauffage |

## Features enrichissement (à venir)

| Nom projet | Source | Type | Unité | Description |
|------------|--------|------|-------|-------------|
| prix_m2_median | DVF | num | €/m² | Prix médian au m² (transactions) |
| nb_transactions | DVF | num | - | Nombre de transactions sur la parcelle |
| revenu_median | INSEE | num | € | Revenu médian du carreau 200m |
| densite_pop | INSEE | num | hab/km² | Densité de population du carreau |
| distance_centre | BD TOPO | num | m | Distance au centre de Bordeaux |
| zone_plu | PLU | cat | - | Zone du PLU (U, AU, A, N…) |

## Notes

- Le mapping `nom projet → nom BDNB réel` est dans `config.yaml → features.column_mapping`
- Les noms réels sont à compléter après exploration du schéma BDNB (notebook 01)
- Toute nouvelle feature doit être ajoutée ici ET dans config.yaml
