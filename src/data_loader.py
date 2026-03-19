"""Chargement et nettoyage des données.

Responsable du chargement de la BDNB et des futures sources.
Retourne toujours un GeoDataFrame propre via geo_utils.prepare_geodf().
"""

import gc
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import PROJECT_ROOT, load_config, setup_project
from src.geo_utils import prepare_geodf

logger = logging.getLogger(__name__)


def load_bdnb(config: dict | None = None) -> gpd.GeoDataFrame:
    """Charge la BDNB pour Bordeaux Métropole.

    - Lecture par chunks pour gérer le volume
    - Filtrage sur les codes INSEE de la métropole
    - Nettoyage géométrique via prepare_geodf()
    - Sauvegarde en GeoParquet

    Parameters
    ----------
    config : dict, optional
        Configuration du projet. Si None, charge config.yaml.

    Returns
    -------
    gpd.GeoDataFrame
        Données BDNB nettoyées pour Bordeaux Métropole.
    """
    if config is None:
        config = load_config()

    codes_insee = set(config["zone"]["codes_insee"])
    chunk_size = config["spatial"]["chunk_size_read"]
    bdnb_path = config["features"].get("bdnb_path", "data/raw/bdnb_33/")
    bdnb_path = Path(bdnb_path)

    if not bdnb_path.exists():
        # Essayer avec le chemin relatif au projet
        bdnb_path = PROJECT_ROOT / bdnb_path

    logger.info("Chargement BDNB depuis %s", bdnb_path)
    logger.info("Communes cibles : %d codes INSEE", len(codes_insee))

    # Détecter le format des fichiers
    files = list(bdnb_path.glob("*.csv")) + list(bdnb_path.glob("*.parquet"))
    if not files:
        # Tenter de charger comme GeoJSON
        files = list(bdnb_path.glob("*.geojson"))

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier CSV, Parquet ou GeoJSON trouvé dans {bdnb_path}. "
            f"Télécharger la BDNB dept 33 et placer les fichiers dans ce dossier."
        )

    logger.info("Fichiers trouvés : %d", len(files))

    all_chunks = []

    for filepath in files:
        logger.info("Lecture de %s", filepath.name)

        if filepath.suffix == ".parquet":
            gdf = gpd.read_parquet(filepath)
            # Filtrer sur codes INSEE
            # Le nom de la colonne INSEE est à adapter selon le schéma réel
            insee_col = _detect_insee_column(gdf)
            if insee_col:
                gdf = gdf[gdf[insee_col].astype(str).isin(codes_insee)]
            all_chunks.append(gdf)

        elif filepath.suffix == ".csv":
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
                insee_col = _detect_insee_column(chunk)
                if insee_col:
                    chunk = chunk[chunk[insee_col].astype(str).isin(codes_insee)]
                if len(chunk) > 0:
                    all_chunks.append(chunk)
            gc.collect()

        elif filepath.suffix == ".geojson":
            gdf = gpd.read_file(filepath)
            insee_col = _detect_insee_column(gdf)
            if insee_col:
                gdf = gdf[gdf[insee_col].astype(str).isin(codes_insee)]
            all_chunks.append(gdf)

    if not all_chunks:
        raise ValueError("Aucune donnée trouvée pour les communes de la métropole.")

    # Concaténation
    logger.info("Concaténation de %d chunks...", len(all_chunks))

    if isinstance(all_chunks[0], gpd.GeoDataFrame):
        gdf = pd.concat(all_chunks, ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    else:
        df = pd.concat(all_chunks, ignore_index=True)
        # Convertir en GeoDataFrame si colonnes lat/lon détectées
        gdf = _to_geodataframe(df)

    del all_chunks
    gc.collect()

    # Vérifier la couverture par commune
    if "insee_col" in dir() and insee_col:
        _check_commune_coverage(gdf, insee_col, codes_insee)

    # Nettoyage géométrique
    gdf = prepare_geodf(gdf, config=config)

    # Sauvegarde
    output_path = PROJECT_ROOT / "data" / "processed" / "bdnb_metropole.geoparquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    logger.info("Sauvegardé : %s (%d lignes)", output_path, len(gdf))

    return gdf


def _detect_insee_column(df: pd.DataFrame) -> str | None:
    """Détecte la colonne contenant le code INSEE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à inspecter.

    Returns
    -------
    str or None
        Nom de la colonne INSEE, ou None si non trouvée.
    """
    candidates = [
        "code_insee", "code_commune_insee", "code_commune",
        "insee", "cog", "depcom", "code_departement_commune",
    ]
    for col in candidates:
        if col in df.columns:
            logger.debug("Colonne INSEE détectée : %s", col)
            return col

    # Recherche par pattern
    for col in df.columns:
        if "insee" in col.lower() or "commune" in col.lower():
            logger.debug("Colonne INSEE candidate : %s", col)
            return col

    logger.warning("Aucune colonne INSEE détectée parmi : %s", list(df.columns)[:20])
    return None


def _to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convertit un DataFrame en GeoDataFrame si possible.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec colonnes géographiques potentielles.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame avec géométrie.

    Raises
    ------
    ValueError
        Si aucune colonne géographique n'est trouvée.
    """
    # Colonnes géométrie WKT
    for col in ["geometry", "geom", "wkt", "the_geom"]:
        if col in df.columns:
            from shapely import wkt

            df["geometry"] = df[col].apply(wkt.loads)
            return gpd.GeoDataFrame(df, geometry="geometry")

    # Colonnes lat/lon
    lat_col = None
    lon_col = None
    for lat_candidate in ["latitude", "lat", "y"]:
        if lat_candidate in df.columns:
            lat_col = lat_candidate
            break
    for lon_candidate in ["longitude", "lon", "lng", "x"]:
        if lon_candidate in df.columns:
            lon_col = lon_candidate
            break

    if lat_col and lon_col:
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )

    raise ValueError(
        "Impossible de créer un GeoDataFrame : aucune colonne géométrique, "
        "WKT, ou lat/lon trouvée."
    )


def _check_commune_coverage(
    gdf: gpd.GeoDataFrame,
    insee_col: str,
    codes_insee: set,
) -> None:
    """Vérifie que toutes les communes de la métropole sont couvertes.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Données chargées.
    insee_col : str
        Nom de la colonne INSEE.
    codes_insee : set
        Codes INSEE attendus.
    """
    found = set(gdf[insee_col].astype(str).unique())
    missing = codes_insee - found
    if missing:
        logger.warning(
            "⚠️ %d communes sans données BDNB : %s",
            len(missing), sorted(missing),
        )
    else:
        logger.info("✓ Toutes les %d communes sont couvertes", len(codes_insee))

    # Stats par commune
    counts = gdf[insee_col].astype(str).value_counts()
    logger.info("Bâtiments par commune — min: %d, max: %d, médiane: %d",
                counts.min(), counts.max(), int(counts.median()))


if __name__ == "__main__":
    cfg = setup_project()
    try:
        gdf = load_bdnb(cfg)
        print(f"BDNB chargée : {len(gdf)} bâtiments")
        print(f"Colonnes : {list(gdf.columns)}")
        print(f"CRS : {gdf.crs}")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("Placez les données BDNB dans data/raw/bdnb_33/ puis relancez.")
