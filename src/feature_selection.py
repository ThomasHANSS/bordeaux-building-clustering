"""Sélection de features : corrélations, VIF, réduction de dimension.

Élimine les features redondantes et optionnellement réduit la dimension.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import PROJECT_ROOT, load_config, setup_project

logger = logging.getLogger(__name__)


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[str]]:
    """Supprime les features fortement corrélées.

    Parameters
    ----------
    df : pd.DataFrame
        Features numériques uniquement.
    threshold : float
        Seuil de corrélation au-delà duquel une feature est supprimée.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        DataFrame filtré et liste des features supprimées.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info(
        "Corrélation > %.2f : %d features supprimées : %s",
        threshold, len(to_drop), to_drop,
    )
    return df.drop(columns=to_drop), to_drop


def remove_high_vif(
    df: pd.DataFrame,
    threshold: float = 10.0,
    max_iterations: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    """Supprime itérativement les features avec un VIF trop élevé.

    Parameters
    ----------
    df : pd.DataFrame
        Features numériques uniquement.
    threshold : float
        Seuil VIF au-delà duquel supprimer.
    max_iterations : int
        Nombre maximum d'itérations.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        DataFrame filtré et liste des features supprimées.
    """
    dropped = []

    for iteration in range(max_iterations):
        if len(df.columns) < 2:
            break

        # Calcul VIF
        vif_data = pd.DataFrame({
            "feature": df.columns,
            "vif": [
                variance_inflation_factor(df.values, i)
                for i in range(len(df.columns))
            ],
        })

        max_vif = vif_data["vif"].max()
        if max_vif <= threshold:
            logger.info("VIF : toutes les features sous %.1f après %d itérations",
                        threshold, iteration)
            break

        worst = vif_data.loc[vif_data["vif"].idxmax()]
        logger.info("VIF itération %d : suppression de %s (VIF=%.1f)",
                     iteration + 1, worst["feature"], worst["vif"])
        dropped.append(worst["feature"])
        df = df.drop(columns=[worst["feature"]])

    logger.info("VIF : %d features supprimées au total : %s", len(dropped), dropped)
    return df, dropped


def apply_reduction(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, object]:
    """Applique PCA ou UMAP si activé dans la config.

    Parameters
    ----------
    df : pd.DataFrame
        Features numériques.
    config : dict
        Configuration du projet.

    Returns
    -------
    tuple[pd.DataFrame, object]
        DataFrame réduit et objet de réduction (PCA ou UMAP).
    """
    reduction_cfg = config["features"].get("reduction", {})
    if not reduction_cfg.get("enabled", False):
        logger.info("Réduction de dimension désactivée")
        return df, None

    method = reduction_cfg.get("method", "pca")
    n_components = min(reduction_cfg.get("n_components", 10), len(df.columns))

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=config["clustering"]["random_state"])
        reduced = reducer.fit_transform(df)
        explained = reducer.explained_variance_ratio_.cumsum()[-1]
        logger.info("PCA : %d composantes, %.1f%% variance expliquée", n_components, explained * 100)
        cols = [f"pc_{i+1}" for i in range(n_components)]

    elif method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=config["clustering"]["random_state"],
        )
        reduced = reducer.fit_transform(df)
        logger.info("UMAP : %d composantes", n_components)
        cols = [f"umap_{i+1}" for i in range(n_components)]

    else:
        raise ValueError(f"Méthode de réduction inconnue : {method}")

    return pd.DataFrame(reduced, columns=cols, index=df.index), reducer


def select_features(config: dict | None = None) -> gpd.GeoDataFrame:
    """Pipeline complet de sélection de features.

    1. Charger features_ready.geoparquet
    2. Supprimer les features corrélées
    3. Supprimer les features avec VIF élevé
    4. Optionnellement : PCA/UMAP
    5. Sauvegarder le résultat

    Parameters
    ----------
    config : dict, optional
        Configuration du projet.

    Returns
    -------
    gpd.GeoDataFrame
        Features sélectionnées avec géométries.
    """
    if config is None:
        config = load_config()

    # Charger
    input_path = PROJECT_ROOT / config["pipeline"]["intermediate_file"]
    logger.info("Chargement features depuis %s", input_path)
    gdf = gpd.read_parquet(input_path)

    # Séparer géométries et features
    geo_cols = ["uid", "geometry"]
    feature_cols = [c for c in gdf.columns if c not in geo_cols]
    df_features = gdf[feature_cols].copy()

    logger.info("Features en entrée : %d", len(feature_cols))

    # Corrélations
    corr_threshold = config["features"].get("correlation_threshold", 0.85)
    df_features, dropped_corr = remove_correlated_features(df_features, corr_threshold)

    # VIF
    vif_threshold = config["features"].get("vif_threshold", 10.0)
    df_features, dropped_vif = remove_high_vif(df_features, vif_threshold)

    # Réduction de dimension
    df_features, reducer = apply_reduction(df_features, config)

    logger.info("Features en sortie : %d", len(df_features.columns))

    # Reconstruire le GeoDataFrame
    result = gpd.GeoDataFrame(
        pd.concat([gdf[geo_cols].reset_index(drop=True),
                    df_features.reset_index(drop=True)], axis=1),
        geometry="geometry",
    )

    # Sauvegarder (écraser le fichier intermédiaire)
    output_path = PROJECT_ROOT / config["pipeline"]["intermediate_file"]
    result.to_parquet(output_path)
    logger.info("Features sélectionnées sauvegardées : %s", output_path)

    return result


if __name__ == "__main__":
    cfg = setup_project()
    gdf = select_features(cfg)
    print(f"Features sélectionnées : {len(gdf.columns) - 2} (+ uid + geometry)")
