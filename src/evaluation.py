"""Évaluation des clusterings : stabilité, bootstrap, comparaison.

Complète les métriques de base calculées dans clustering.py avec
l'analyse de stabilité (bootstrap + ARI) et la cohérence spatiale.
"""

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import load_config, setup_project

logger = logging.getLogger(__name__)


def bootstrap_stability(
    X: np.ndarray,
    n_clusters: int,
    n_bootstrap: int = 50,
    sample_ratio: float = 0.8,
    random_state: int = 42,
) -> dict:
    """Mesure la stabilité du clustering par bootstrap.

    Exécute n_bootstrap runs sur des sous-échantillons et mesure
    la cohérence des assignations via l'Adjusted Rand Index (ARI).

    Parameters
    ----------
    X : np.ndarray
        Matrice de features.
    n_clusters : int
        Nombre de clusters.
    n_bootstrap : int
        Nombre d'itérations bootstrap.
    sample_ratio : float
        Proportion de l'échantillon pour chaque bootstrap.
    random_state : int
        Graine aléatoire.

    Returns
    -------
    dict
        ari_mean, ari_std, ari_scores.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    sample_size = int(n_samples * sample_ratio)

    # Référence : clustering sur l'ensemble complet
    ref_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    ref_labels = ref_model.fit_predict(X)

    ari_scores = []
    for i in range(n_bootstrap):
        idx = rng.choice(n_samples, size=sample_size, replace=True)
        X_sample = X[idx]

        model = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=10)
        sample_labels = model.fit_predict(X_sample)

        # Comparer avec la référence sur les mêmes indices
        ref_subset = ref_labels[idx]
        ari = adjusted_rand_score(ref_subset, sample_labels)
        ari_scores.append(ari)

    ari_mean = float(np.mean(ari_scores))
    ari_std = float(np.std(ari_scores))
    logger.info(
        "Bootstrap stabilité (n=%d) : ARI moyen=%.3f ± %.3f",
        n_bootstrap, ari_mean, ari_std,
    )

    return {
        "ari_mean": ari_mean,
        "ari_std": ari_std,
        "ari_scores": [float(s) for s in ari_scores],
    }


def compare_runs(experiments: dict) -> str:
    """Produit un tableau comparatif des runs.

    Parameters
    ----------
    experiments : dict
        Contenu de experiments.json.

    Returns
    -------
    str
        Tableau formaté pour le logging.
    """
    lines = [
        f"{'Run':<10} {'Algo':<20} {'k':>3} {'Silhouette':>10} "
        f"{'DB':>8} {'CH':>10} {'Bruit%':>7} {'Notes'}"
    ]
    lines.append("-" * 90)

    for run in experiments.get("runs", []):
        m = run.get("metrics", {})
        lines.append(
            f"{run['id']:<10} {run['algorithm']:<20} {run['n_clusters']:>3} "
            f"{m.get('silhouette', -1):>10.3f} "
            f"{m.get('davies_bouldin', -1):>8.2f} "
            f"{m.get('calinski_harabasz', -1):>10.0f} "
            f"{run.get('noise_pct', 0) * 100:>6.1f}% "
            f"{run.get('notes', '')}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.config import PROJECT_ROOT

    cfg = setup_project()
    exp_path = PROJECT_ROOT / cfg["pipeline"]["experiments_file"]

    if exp_path.exists():
        with open(exp_path) as f:
            experiments = json.load(f)
        print(compare_runs(experiments))
    else:
        print("Aucune expérience trouvée. Lancer d'abord : make cluster")
