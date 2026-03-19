"""Feature engineering : construction, encodage, normalisation.

Responsable de transformer les données brutes en features prêtes pour le clustering.
Sauvegarde le résultat en GeoParquet (conserve les géométries).
"""

import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, get_column_name, load_config, setup_project

logger = logging.getLogger(__name__)


def audit_completeness(gdf: gpd.GeoDataFrame, config: dict) -> pd.DataFrame:
    """Audit du taux de remplissage par colonne.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Données brutes.
    config : dict
        Configuration du projet.

    Returns
    -------
    pd.DataFrame
        Tableau avec colonnes [column, non_null_count, completeness_pct].
    """
    total = len(gdf)
    records = []
    for col in gdf.columns:
        if col == "geometry":
            continue
        non_null = gdf[col].notna().sum()
        records.append({
            "column": col,
            "non_null_count": non_null,
            "completeness_pct": round(non_null / total * 100, 1),
        })

    audit = pd.DataFrame(records).sort_values("completeness_pct", ascending=False)

    min_completeness = config["features"].get("min_completeness", 0.70) * 100
    n_above = (audit["completeness_pct"] >= min_completeness).sum()
    logger.info(
        "Audit complétude : %d/%d colonnes au-dessus de %.0f%%",
        n_above, len(audit), min_completeness,
    )
    return audit


def build_features(config: dict | None = None) -> gpd.GeoDataFrame:
    """Construit les features à partir des données nettoyées.

    Pipeline :
    1. Charger le GeoParquet brut
    2. Appliquer le column_mapping
    3. Filtrer par complétude
    4. Encoder les catégorielles
    5. Normaliser les numériques
    6. Sauvegarder en GeoParquet + scaler

    Parameters
    ----------
    config : dict, optional
        Configuration du projet.

    Returns
    -------
    gpd.GeoDataFrame
        Features prêtes pour le clustering, avec géométries.
    """
    if config is None:
        config = load_config()

    # Charger les données nettoyées
    input_path = PROJECT_ROOT / "data" / "processed" / "bdnb_metropole.geoparquet"
    logger.info("Chargement depuis %s", input_path)
    gdf = gpd.read_parquet(input_path)

    # Audit de complétude
    audit = audit_completeness(gdf, config)
    logger.info("Top 10 colonnes par complétude :\n%s", audit.head(10).to_string())

    # Appliquer le column mapping
    mapping = config["features"]["column_mapping"]
    num_features = config["features"]["numerical"]
    cat_features = config["features"]["categorical"]

    # Renommer les colonnes selon le mapping
    reverse_mapping = {}
    for project_name, real_name in mapping.items():
        if real_name != "TODO" and real_name in gdf.columns:
            reverse_mapping[real_name] = project_name

    gdf = gdf.rename(columns=reverse_mapping)

    # Filtrer par complétude
    min_completeness = config["features"].get("min_completeness", 0.70)
    available_num = [f for f in num_features if f in gdf.columns
                     and gdf[f].notna().mean() >= min_completeness]
    available_cat = [f for f in cat_features if f in gdf.columns
                     and gdf[f].notna().mean() >= min_completeness]

    logger.info("Features numériques retenues (%d/%d) : %s",
                len(available_num), len(num_features), available_num)
    logger.info("Features catégorielles retenues (%d/%d) : %s",
                len(available_cat), len(cat_features), available_cat)

    if not available_num:
        raise ValueError(
            "Aucune feature numérique ne passe le seuil de complétude. "
            "Vérifier le column_mapping et le seuil min_completeness."
        )

    # Filtrer les bâtiments par surface
    min_surface = config["features"].get("min_surface", 20)
    max_surface = config["features"].get("max_surface", 50000)
    if "surface_bat" in gdf.columns:
        before = len(gdf)
        gdf = gdf[
            (gdf["surface_bat"] >= min_surface) & (gdf["surface_bat"] <= max_surface)
        ]
        logger.info("Filtre surface [%d, %d] m² : %d → %d", min_surface, max_surface,
                     before, len(gdf))

    # Conserver géométries et uid
    geo_cols = ["uid", "geometry"]
    feature_cols = available_num + available_cat
    gdf_features = gdf[geo_cols + [c for c in feature_cols if c in gdf.columns]].copy()

    # Encoder les catégorielles
    max_card = config["features"]["encoding"]["onehot_max_cardinality"]
    for col in available_cat:
        if col not in gdf_features.columns:
            continue
        cardinality = gdf_features[col].nunique()
        if cardinality <= max_card:
            dummies = pd.get_dummies(gdf_features[col], prefix=col, dummy_na=False)
            gdf_features = pd.concat([gdf_features, dummies], axis=1)
            gdf_features = gdf_features.drop(columns=[col])
            logger.info("One-hot : %s (cardinalité %d)", col, cardinality)
        else:
            gdf_features[col] = gdf_features[col].astype("category").cat.codes
            logger.info("Ordinal : %s (cardinalité %d)", col, cardinality)

    # Imputation des NaN pour les numériques (médiane)
    for col in available_num:
        if col in gdf_features.columns:
            n_missing = gdf_features[col].isna().sum()
            if n_missing > 0:
                median_val = gdf_features[col].median()
                gdf_features[col] = gdf_features[col].fillna(median_val)
                logger.info("Imputation médiane %s : %d NaN → %.1f", col, n_missing, median_val)

    # Normalisation
    cols_to_scale = [c for c in gdf_features.columns if c not in geo_cols]
    scaler = StandardScaler()
    gdf_features[cols_to_scale] = scaler.fit_transform(gdf_features[cols_to_scale])

    # Sauvegarder le scaler
    scaler_path = PROJECT_ROOT / config["pipeline"]["scaler_file"]
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler sauvegardé : %s", scaler_path)

    # Sauvegarder en GeoParquet
    output_path = PROJECT_ROOT / config["pipeline"]["intermediate_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_features = gpd.GeoDataFrame(gdf_features, geometry="geometry")
    gdf_features.to_parquet(output_path)
    logger.info("Features sauvegardées : %s (%d lignes, %d colonnes)",
                output_path, len(gdf_features), len(gdf_features.columns))

    return gdf_features


if __name__ == "__main__":
    cfg = setup_project()
    gdf = build_features(cfg)
    print(f"Features construites : {len(gdf)} lignes, {len(gdf.columns)} colonnes")
    print(f"Colonnes : {list(gdf.columns)}")
