"""Clustering : algorithmes multiples, détection de k optimal, sauvegarde.

Lit features_ready.geoparquet, applique les algorithmes définis dans config,
sauvegarde clustered.geoparquet et logge dans experiments.json.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.config import PROJECT_ROOT, load_config, setup_project

logger = logging.getLogger(__name__)


def find_optimal_k(
    X: np.ndarray,
    k_range: tuple[int, int],
    random_state: int = 42,
) -> dict:
    """Détermine le k optimal via Elbow et Silhouette.

    Parameters
    ----------
    X : np.ndarray
        Matrice de features.
    k_range : tuple[int, int]
        Plage de k à tester (min, max).

    Returns
    -------
    dict
        Résultats : k_elbow, k_silhouette, inertias, silhouettes.
    """
    k_values = range(k_range[0], k_range[1] + 1)
    inertias = []
    silhouettes = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=min(10000, len(X))))
        logger.info("k=%d : inertie=%.0f, silhouette=%.3f", k, km.inertia_, silhouettes[-1])

    # Elbow
    kneedle = KneeLocator(list(k_values), inertias, curve="convex", direction="decreasing")
    k_elbow = kneedle.elbow if kneedle.elbow else k_range[0]

    # Meilleur silhouette
    k_silhouette = list(k_values)[np.argmax(silhouettes)]

    logger.info("k optimal — Elbow: %d, Silhouette: %d", k_elbow, k_silhouette)

    return {
        "k_elbow": k_elbow,
        "k_silhouette": k_silhouette,
        "k_values": list(k_values),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def run_single_algorithm(
    X: np.ndarray,
    algo_name: str,
    params: dict,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Exécute un algorithme de clustering.

    Parameters
    ----------
    X : np.ndarray
        Matrice de features.
    algo_name : str
        Nom de l'algorithme.
    params : dict
        Paramètres spécifiques.
    n_clusters : int
        Nombre de clusters cible.
    random_state : int
        Graine aléatoire.

    Returns
    -------
    np.ndarray
        Labels de cluster.
    """
    logger.info("Clustering %s (k=%d, params=%s)", algo_name, n_clusters, params)

    if algo_name == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **params)
        labels = model.fit_predict(X)

    elif algo_name == "hdbscan":
        import hdbscan
        model = hdbscan.HDBSCAN(**params)
        labels = model.fit_predict(X)

    elif algo_name == "gaussian_mixture":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state, **params)
        labels = model.fit_predict(X)

    elif algo_name == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, **params)
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Algorithme inconnu : {algo_name}")

    n_unique = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    logger.info(
        "%s terminé : %d clusters, %d bruit (%.1f%%)",
        algo_name, n_unique, n_noise, n_noise / len(labels) * 100,
    )
    return labels


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Calcule les métriques d'évaluation.

    Parameters
    ----------
    X : np.ndarray
        Matrice de features.
    labels : np.ndarray
        Labels de cluster.

    Returns
    -------
    dict
        Métriques : silhouette, davies_bouldin, calinski_harabasz.
    """
    # Exclure le bruit pour les métriques
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        logger.warning("Pas assez de clusters pour les métriques")
        return {"silhouette": -1, "davies_bouldin": -1, "calinski_harabasz": -1}

    X_clean = X[mask]
    labels_clean = labels[mask]

    sample_size = min(10000, len(X_clean))
    metrics = {
        "silhouette": float(silhouette_score(X_clean, labels_clean, sample_size=sample_size)),
        "davies_bouldin": float(davies_bouldin_score(X_clean, labels_clean)),
        "calinski_harabasz": float(calinski_harabasz_score(X_clean, labels_clean)),
    }
    logger.info("Métriques : %s", metrics)
    return metrics


def log_experiment(
    config: dict,
    algo_name: str,
    params: dict,
    labels: np.ndarray,
    metrics: dict,
    feature_cols: list[str],
    notes: str = "",
) -> None:
    """Logge un run dans experiments.json.

    Parameters
    ----------
    config : dict
        Configuration du projet.
    algo_name : str
        Nom de l'algorithme.
    params : dict
        Paramètres utilisés.
    labels : np.ndarray
        Labels de cluster.
    metrics : dict
        Métriques calculées.
    feature_cols : list[str]
        Features utilisées.
    notes : str
        Notes libres.
    """
    exp_path = PROJECT_ROOT / config["pipeline"]["experiments_file"]
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    # Charger l'existant
    if exp_path.exists():
        with open(exp_path, "r", encoding="utf-8") as f:
            experiments = json.load(f)
    else:
        experiments = {"runs": []}

    run_id = f"run_{len(experiments['runs']) + 1:03d}"
    n_noise = int((labels == -1).sum())

    run = {
        "id": run_id,
        "date": datetime.now().isoformat(),
        "algorithm": algo_name,
        "params": params,
        "n_clusters": int(len(set(labels) - {-1})),
        "n_noise": n_noise,
        "noise_pct": round(n_noise / len(labels), 3),
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": len(labels),
        "metrics": metrics,
        "notes": notes,
    }
    experiments["runs"].append(run)

    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)

    logger.info("Expérience loggée : %s", run_id)


def run_clustering(config: dict | None = None) -> gpd.GeoDataFrame:
    """Pipeline complet de clustering.

    1. Charger features_ready.geoparquet
    2. Déterminer k optimal
    3. Lancer tous les algorithmes
    4. Évaluer et comparer
    5. Sauvegarder le meilleur dans clustered.geoparquet

    Parameters
    ----------
    config : dict, optional
        Configuration du projet.

    Returns
    -------
    gpd.GeoDataFrame
        Données avec labels du meilleur clustering.
    """
    if config is None:
        config = load_config()

    # Charger
    input_path = PROJECT_ROOT / config["pipeline"]["intermediate_file"]
    logger.info("Chargement features depuis %s", input_path)
    gdf = gpd.read_parquet(input_path)

    # Séparer features et géométries
    geo_cols = ["uid", "geometry"]
    feature_cols = [c for c in gdf.columns if c not in geo_cols]
    X = gdf[feature_cols].values

    random_state = config["clustering"]["random_state"]
    target_k = config["clustering"]["target_k"]
    k_range = tuple(config["clustering"]["k_range"])

    # Déterminer k optimal
    logger.info("=== Recherche de k optimal ===")
    k_analysis = find_optimal_k(X, k_range, random_state)

    if abs(k_analysis["k_elbow"] - target_k) > 3:
        logger.warning(
            "⚠️ k optimal (Elbow=%d, Silhouette=%d) très différent de target_k=%d. "
            "Les deux seront testés.",
            k_analysis["k_elbow"], k_analysis["k_silhouette"], target_k,
        )

    # Lancer tous les algorithmes
    results = {}
    for algo_cfg in config["clustering"]["algorithms"]:
        algo_name = algo_cfg["name"]
        params = algo_cfg.get("params", {})

        # Pour HDBSCAN, pas de n_clusters
        n_clusters = target_k if algo_name != "hdbscan" else 0

        labels = run_single_algorithm(X, algo_name, params, n_clusters, random_state)
        metrics = evaluate_clustering(X, labels)

        results[algo_name] = {"labels": labels, "metrics": metrics}
        log_experiment(config, algo_name, params, labels, metrics, feature_cols)

    # Trouver le meilleur (silhouette la plus élevée)
    best_algo = max(results, key=lambda k: results[k]["metrics"]["silhouette"])
    best_labels = results[best_algo]["labels"]
    best_metrics = results[best_algo]["metrics"]

    logger.info(
        "=== Meilleur algorithme : %s (silhouette=%.3f) ===",
        best_algo, best_metrics["silhouette"],
    )

    # Comparaison tabulaire
    comparison = pd.DataFrame({
        algo: res["metrics"] for algo, res in results.items()
    }).T
    logger.info("Comparaison :\n%s", comparison.to_string())

    # Ajouter les labels au GeoDataFrame
    gdf["cluster_label"] = best_labels
    gdf["cluster_algo"] = best_algo

    # Sauvegarder
    output_path = PROJECT_ROOT / config["pipeline"]["results_file"]
    gdf.to_parquet(output_path)
    logger.info("Résultats sauvegardés : %s", output_path)

    return gdf


if __name__ == "__main__":
    cfg = setup_project()
    gdf = run_clustering(cfg)
    print(f"Clustering terminé : {gdf['cluster_label'].nunique()} clusters")
    print(f"Algorithme retenu : {gdf['cluster_algo'].iloc[0]}")
