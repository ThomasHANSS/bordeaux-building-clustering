"""Analyse spatiale : autocorrélation, Moran's I.

Mesure la cohérence spatiale des clusters : des bâtiments proches
doivent-ils avoir le même type ?
"""

import logging

import geopandas as gpd
import numpy as np

from src.config import PROJECT_ROOT, load_config, setup_project

logger = logging.getLogger(__name__)


def compute_moran_i(
    gdf: gpd.GeoDataFrame,
    label_col: str = "cluster_label",
    k_neighbors: int = 8,
) -> dict:
    """Calcule le Moran's I sur les labels de cluster.

    Un Moran's I élevé (proche de 1) indique que les clusters sont
    spatialement cohérents (bâtiments proches = même cluster).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Données avec géométries et labels.
    label_col : str
        Colonne des labels de cluster.
    k_neighbors : int
        Nombre de voisins pour la matrice de poids.

    Returns
    -------
    dict
        moran_i, p_value, z_score.
    """
    try:
        from esda.moran import Moran
        from libpysal.weights import KNN
    except ImportError:
        logger.error("pysal (esda, libpysal) non installé. pip install esda libpysal")
        return {"moran_i": None, "p_value": None, "z_score": None}

    # Exclure le bruit
    mask = gdf[label_col] != -1
    gdf_clean = gdf[mask].copy()

    if len(gdf_clean) < 100:
        logger.warning("Trop peu de données pour Moran's I (%d)", len(gdf_clean))
        return {"moran_i": None, "p_value": None, "z_score": None}

    # Échantillonner si trop volumineux (Moran's I est coûteux en RAM)
    max_samples = 20000
    if len(gdf_clean) > max_samples:
        logger.info("Échantillonnage pour Moran's I : %d → %d", len(gdf_clean), max_samples)
        gdf_clean = gdf_clean.sample(n=max_samples, random_state=42)

    # Matrice de poids spatiaux (k plus proches voisins)
    logger.info("Construction matrice de poids KNN (k=%d)...", k_neighbors)
    w = KNN.from_dataframe(gdf_clean, k=k_neighbors)
    w.transform = "R"  # Row-standardize

    # Moran's I
    y = gdf_clean[label_col].values.astype(float)
    moran = Moran(y, w)

    result = {
        "moran_i": float(moran.I),
        "p_value": float(moran.p_sim),
        "z_score": float(moran.z_sim),
    }

    logger.info(
        "Moran's I = %.3f (p=%.4f, z=%.2f) — %s",
        result["moran_i"],
        result["p_value"],
        result["z_score"],
        "spatialement cohérent" if result["moran_i"] > 0.3 else "faible cohérence spatiale",
    )

    return result


if __name__ == "__main__":
    cfg = setup_project()
    results_path = PROJECT_ROOT / cfg["pipeline"]["results_file"]

    if results_path.exists():
        gdf = gpd.read_parquet(results_path)
        result = compute_moran_i(gdf)
        print(f"Moran's I : {result}")
    else:
        print("Aucun résultat de clustering trouvé. Lancer d'abord : make cluster")
