"""Clustering v4 — bâti récent (10 ans ou moins, nb_logements < 4).

Filtre résidentiel individuel + annee_construction >= 2016.
KMeans k=10. Sauvegarde et log experiments.json.
"""

import geopandas as gpd
import numpy as np
import json
import logging
import os
import gc
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

RANDOM_STATE = 42
TARGET_K = 10
EXP_PATH = "outputs/experiments.json"
DATA_PATH = "data/processed/clustered_petit_residentiel.geoparquet"
OUTPUT_PATH = "data/processed/clustered_recent.geoparquet"
ANNEE_MIN = 2016  # 2026 - 10

FEATURE_COLS = [
    "surface_bat", "hauteur_mean", "hauteur_max", "nb_niveaux",
    "annee_construction", "nb_logements",
    "usage_principal_enc", "usage_foncier_enc", "mat_murs_enc",
    "nature_bdtopo_enc", "usage_bdtopo_enc",
]

POPUP_COLS = [
    "surface_bat", "hauteur_mean", "nb_niveaux",
    "annee_construction", "usage_principal", "mat_murs",
]


def name_cluster(sub: gpd.GeoDataFrame) -> str:
    """Nomme un cluster selon son profil médian."""
    s = sub["surface_bat"].median()
    h = sub["hauteur_mean"].median()
    a = sub["annee_construction"].median()
    mat = sub["mat_murs"].mode()
    m = str(mat.iloc[0]) if len(mat) > 0 else ""

    ml = m.lower()
    mat_s = ""
    if "pierre" in ml: mat_s = "pierre"
    elif "béton" in ml or "beton" in ml: mat_s = "béton"
    elif "brique" in ml: mat_s = "brique"
    elif "parpaing" in ml or "agglo" in ml: mat_s = "parpaing"
    elif "bois" in ml: mat_s = "bois"

    if s < 60:
        typ = "Petit bâti"
    elif h <= 4:
        typ = "Grande maison" if s > 200 else "Pavillon"
    elif h <= 7:
        typ = "Maison R+1"
    elif h <= 10:
        typ = "Individuel dense R+2"
    else:
        typ = "Individuel haut"

    parts = [typ, f"{s:.0f}m²", f"{h:.0f}m", f"{a:.0f}"]
    if mat_s:
        parts.append(mat_s)
    return " / ".join(parts)


def main() -> None:
    """Pipeline clustering v4."""
    os.makedirs("data/processed", exist_ok=True)

    # 1. Charger et filtrer
    logger.info("Chargement %s ...", DATA_PATH)
    gdf = gpd.read_parquet(DATA_PATH)
    logger.info("  %d bâtiments résidentiels individuels", len(gdf))

    n_before = len(gdf)
    gdf = gdf[gdf["annee_construction"] >= ANNEE_MIN].copy()
    logger.info("  Filtre année >= %d : -%d → %d", ANNEE_MIN, n_before - len(gdf), len(gdf))

    # 2. Features
    X = gdf[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("  Features : %d colonnes, %d lignes", X_scaled.shape[1], X_scaled.shape[0])

    # 3. KMeans k=10
    logger.info("KMeans k=%d ...", TARGET_K)
    km = KMeans(n_clusters=TARGET_K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    gdf["v4_label"] = labels

    # 4. Métriques
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    logger.info("  sil=%.4f  DB=%.4f  CH=%.1f", sil, db, ch)

    # 5. Nommer les clusters
    names = {}
    for cl in sorted(gdf["v4_label"].unique()):
        sub = gdf[gdf["v4_label"] == cl]
        names[cl] = name_cluster(sub)
        logger.info("  Cluster %2d : %s (%d bât.)", cl, names[cl], len(sub))

    # 6. Sauvegarder
    gdf["cluster_label"] = labels
    gdf["cluster_algo"] = "kmeans"
    gdf.to_parquet(OUTPUT_PATH)
    logger.info("Sauvegardé : %s (%d bâtiments)", OUTPUT_PATH, len(gdf))

    # 7. Log experiments
    with open(EXP_PATH) as f:
        experiments = json.load(f)

    next_id = len(experiments["runs"]) + 1
    experiments["runs"].append({
        "id": f"run_{next_id:03d}",
        "date": datetime.now().isoformat(),
        "algorithm": "kmeans",
        "params": {"n_clusters": TARGET_K},
        "n_clusters": TARGET_K,
        "n_noise": 0,
        "noise_pct": 0,
        "features_used": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "n_samples": len(gdf),
        "metrics": {
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 1),
        },
        "data_sources": ["bdnb"],
        "notes": f"v4 — bâti récent (>={ANNEE_MIN}), résidentiel individuel, nb_log<4, KMeans k={TARGET_K}",
    })

    with open(EXP_PATH, "w") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)
    logger.info("  run_%03d ajouté à %s", next_id, EXP_PATH)

    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
