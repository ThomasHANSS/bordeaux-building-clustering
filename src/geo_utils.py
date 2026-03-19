"""Utilitaires géospatiaux : nettoyage, jointures, agrégation.

Ce module centralise TOUTE manipulation de géométries et de jointures spatiales.
Voir CLAUDE.md § "Jointures spatiales — Gestion des 4 risques".
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid

from src.config import load_config

logger = logging.getLogger(__name__)

TARGET_CRS = "EPSG:2154"


# ---------------------------------------------------------------------------
# Nettoyage des GeoDataFrames
# ---------------------------------------------------------------------------

def prepare_geodf(
    gdf: gpd.GeoDataFrame,
    id_col: str | None = None,
    config: dict | None = None,
) -> gpd.GeoDataFrame:
    """Nettoie, reprojette et valide un GeoDataFrame.

    Opérations :
    - Fallback CRS si absent (WGS84)
    - Reprojection en EPSG:2154
    - make_valid sur toutes les géométries
    - Explosion des MultiPolygon
    - Filtrage des micro-géométries et géométries vides
    - Création d'un uid unique post-explode

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Données géographiques brutes.
    id_col : str, optional
        Nom de la colonne identifiant source (pour traçabilité).
    config : dict, optional
        Configuration du projet. Si None, charge config.yaml.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame nettoyé, en EPSG:2154, avec uid unique.
    """
    if config is None:
        config = load_config()

    spatial_cfg = config.get("spatial", {})
    min_area = spatial_cfg.get("min_geometry_area", 1.0)
    fallback_crs = spatial_cfg.get("default_crs_fallback", "EPSG:4326")

    n_before = len(gdf)

    # CRS
    if gdf.crs is None:
        logger.warning("CRS absent — hypothèse %s", fallback_crs)
        gdf = gdf.set_crs(fallback_crs)
    gdf = gdf.to_crs(TARGET_CRS)

    # Géométries invalides
    gdf["geometry"] = gdf["geometry"].apply(make_valid)

    # Explosion MultiPolygon → Polygon
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Filtrage micro-géométries
    gdf = gdf[gdf.geometry.area > min_area]

    # Filtrage vides / nulles
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    gdf = gdf.reset_index(drop=True)

    # Identifiant unique post-explode
    gdf["uid"] = range(len(gdf))

    n_after = len(gdf)
    logger.info(
        "prepare_geodf : %d → %d entités (-%d), CRS=%s",
        n_before, n_after, n_before - n_after, gdf.crs,
    )
    if id_col and id_col in gdf.columns:
        logger.info(
            "  %d identifiants source uniques → %d géométries",
            gdf[id_col].nunique(), n_after,
        )

    return gdf


# ---------------------------------------------------------------------------
# Jointure spatiale
# ---------------------------------------------------------------------------

def spatial_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    predicate: str = "intersects",
    min_overlap: float | None = None,
    simplify_tolerance: float | None = None,
    use_chunked: bool = False,
    chunk_size: int = 10_000,
    config: dict | None = None,
) -> gpd.GeoDataFrame:
    """Jointure spatiale robuste avec nettoyage auto.

    Parameters
    ----------
    left : gpd.GeoDataFrame
        GeoDataFrame de gauche (ex: bâtiments).
    right : gpd.GeoDataFrame
        GeoDataFrame de droite (ex: parcelles DVF).
    predicate : str
        Prédicat spatial : 'intersects', 'within', 'contains'.
    min_overlap : float, optional
        Ratio minimum de recouvrement (0-1). Si défini, filtre les
        jointures dont le recouvrement est insuffisant.
    simplify_tolerance : float, optional
        Tolérance de simplification en mètres. Accélère le sjoin.
        Si None, lit depuis config.yaml.
    use_chunked : bool
        Forcer le traitement par chunks.
    chunk_size : int
        Taille des chunks si chunked.
    config : dict, optional
        Configuration du projet.

    Returns
    -------
    gpd.GeoDataFrame
        Résultat de la jointure.
    """
    if config is None:
        config = load_config()

    spatial_cfg = config.get("spatial", {})
    if simplify_tolerance is None:
        simplify_tolerance = spatial_cfg.get("simplify_tolerance")
    if not use_chunked:
        threshold = spatial_cfg.get("use_chunked_above", 100_000)
        use_chunked = len(left) > threshold
    if use_chunked:
        chunk_size = spatial_cfg.get("chunk_size_sjoin", chunk_size)

    # Nettoyage
    left = prepare_geodf(left, config=config)
    right = prepare_geodf(right, config=config)

    # Simplification pour performance
    if simplify_tolerance:
        left_simple = left.copy()
        left_simple["geometry"] = left_simple.geometry.simplify(
            simplify_tolerance, preserve_topology=True
        )
        right_simple = right.copy()
        right_simple["geometry"] = right_simple.geometry.simplify(
            simplify_tolerance, preserve_topology=True
        )
    else:
        left_simple = left
        right_simple = right

    # Forcer la construction de l'index spatial
    right_simple.sindex  # noqa: B018

    # Jointure (chunked ou directe)
    if use_chunked:
        logger.info(
            "sjoin chunked : %d × %d, chunks de %d",
            len(left_simple), len(right_simple), chunk_size,
        )
        results = []
        for start in range(0, len(left_simple), chunk_size):
            chunk = left_simple.iloc[start : start + chunk_size]
            partial = gpd.sjoin(chunk, right_simple, how="left", predicate=predicate)
            results.append(partial)
        joined = pd.concat(results, ignore_index=True)
    else:
        joined = gpd.sjoin(left_simple, right_simple, how="left", predicate=predicate)

    logger.info("sjoin terminé : %d résultats", len(joined))

    # Restaurer les géométries originales (non simplifiées)
    if simplify_tolerance:
        joined["geometry"] = left.geometry.loc[joined.index].values

    # Filtre de recouvrement
    if min_overlap is not None:
        joined = _filter_by_overlap(left, right, joined, min_overlap)

    return joined


# ---------------------------------------------------------------------------
# Calcul et filtrage du recouvrement
# ---------------------------------------------------------------------------

def _compute_overlap_area(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    joined: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Calcule la surface de recouvrement pour chaque paire.

    Parameters
    ----------
    left : gpd.GeoDataFrame
        GeoDataFrame de gauche (géométries originales).
    right : gpd.GeoDataFrame
        GeoDataFrame de droite (géométries originales).
    joined : gpd.GeoDataFrame
        Résultat du sjoin.

    Returns
    -------
    gpd.GeoDataFrame
        joined avec colonnes _overlap_area et _overlap_ratio ajoutées.
    """
    left_geom = left.geometry.loc[joined.index]
    right_idx = joined["index_right"].values
    right_geom = right.geometry.iloc[right_idx].values

    overlap_areas = []
    for a, b in zip(left_geom, right_geom):
        try:
            overlap_areas.append(a.intersection(b).area if a.intersects(b) else 0.0)
        except Exception:
            overlap_areas.append(0.0)

    joined["_overlap_area"] = overlap_areas
    left_area = left.geometry.area.loc[joined.index].values
    joined["_overlap_ratio"] = np.where(
        left_area > 0, joined["_overlap_area"] / left_area, 0.0
    )

    return joined


def _filter_by_overlap(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    joined: gpd.GeoDataFrame,
    min_overlap: float,
) -> gpd.GeoDataFrame:
    """Filtre les jointures dont le recouvrement est insuffisant.

    Parameters
    ----------
    left, right : gpd.GeoDataFrame
        GeoDataFrames originaux.
    joined : gpd.GeoDataFrame
        Résultat du sjoin.
    min_overlap : float
        Ratio minimum de recouvrement (0-1).

    Returns
    -------
    gpd.GeoDataFrame
        Jointures filtrées.
    """
    joined = _compute_overlap_area(left, right, joined)
    before = len(joined)
    joined = joined[joined["_overlap_ratio"] >= min_overlap]
    logger.info(
        "Filtre overlap >= %.2f : %d → %d lignes", min_overlap, before, len(joined)
    )
    return joined


# ---------------------------------------------------------------------------
# Agrégation des relations 1:N
# ---------------------------------------------------------------------------

def aggregate_multi_matches(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    value_cols: list[str],
    weight_col: str = "_overlap_area",
    strategy: str = "area_weighted",
) -> pd.DataFrame:
    """Résout les doublons 1:N après sjoin.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Résultat d'un sjoin avec doublons potentiels.
    id_col : str
        Colonne identifiant (ex: 'uid').
    value_cols : list[str]
        Colonnes à agréger.
    weight_col : str
        Colonne de poids pour l'agrégation pondérée.
    strategy : str
        'area_weighted', 'max_overlap', 'mean', 'majority'.

    Returns
    -------
    pd.DataFrame
        Données agrégées, une ligne par id_col.
    """
    if strategy == "mean":
        agg = gdf.groupby(id_col)[value_cols].mean()

    elif strategy == "max_overlap":
        agg = (
            gdf.sort_values(weight_col, ascending=False)
            .drop_duplicates(subset=id_col, keep="first")
            .set_index(id_col)[value_cols]
        )

    elif strategy == "area_weighted":
        result = {}
        for col in value_cols:
            weighted = gdf[col] * gdf[weight_col]
            sum_weights = gdf.groupby(id_col)[weight_col].sum()
            sum_weighted = weighted.groupby(gdf[id_col]).sum()
            result[col] = sum_weighted / sum_weights
        agg = pd.DataFrame(result)

    elif strategy == "majority":
        agg = gdf.groupby(id_col)[value_cols].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")

    logger.info("Agrégation %s : %d lignes → %d", strategy, len(gdf), len(agg))
    return agg.reset_index()


def smart_aggregate(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    weight_col: str = "_overlap_area",
) -> pd.DataFrame:
    """Agrège intelligemment selon le type de feature.

    Numériques → area_weighted. Catégorielles → majority.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Résultat d'un sjoin avec doublons.
    id_col : str
        Colonne identifiant.
    num_cols : list[str]
        Features numériques.
    cat_cols : list[str]
        Features catégorielles.
    weight_col : str
        Colonne de poids.

    Returns
    -------
    pd.DataFrame
        Données agrégées.
    """
    num_cols_present = [c for c in num_cols if c in gdf.columns]
    cat_cols_present = [c for c in cat_cols if c in gdf.columns]

    parts = []

    if num_cols_present:
        num_agg = aggregate_multi_matches(
            gdf, id_col, num_cols_present, weight_col, strategy="area_weighted"
        )
        parts.append(num_agg.set_index(id_col))

    if cat_cols_present:
        cat_agg = aggregate_multi_matches(
            gdf, id_col, cat_cols_present, weight_col, strategy="majority"
        )
        parts.append(cat_agg.set_index(id_col))

    if not parts:
        logger.warning("smart_aggregate : aucune colonne trouvée")
        return pd.DataFrame()

    result = pd.concat(parts, axis=1).reset_index()
    logger.info("smart_aggregate : %d lignes, %d colonnes", len(result), len(result.columns))
    return result


if __name__ == "__main__":
    from src.config import setup_project

    cfg = setup_project()
    print("geo_utils prêt")
    print(f"  CRS cible : {TARGET_CRS}")
    print(f"  Simplification : {cfg['spatial'].get('simplify_tolerance')} m")
    print(f"  Seuil chunking : {cfg['spatial'].get('use_chunked_above')}")
