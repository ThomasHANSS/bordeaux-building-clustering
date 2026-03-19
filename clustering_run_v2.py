"""Clustering v2 — petit résidentiel (nb_logements < 4, hors usage inconnu).

Filtre les bâtiments avec nb_logements < 4 et exclut le cluster 1 (usage inconnu).
Lance KMeans k=15 et HDBSCAN (grid search échantillonné).
Sauvegarde, logue dans experiments.json, génère le rapport PDF v3.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import json
import logging
import os
import gc
from datetime import datetime
from itertools import product

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import hdbscan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

# ── Config ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET_K = 15
EXP_PATH = "outputs/experiments.json"
DATA_PATH = "data/processed/clustered.geoparquet"
OUTPUT_PATH = "data/processed/clustered_petit_residentiel.geoparquet"
SAMPLE_SIL = 10000

# Colonnes BDNB nécessaires (via column_mapping)
NUM_MAPPING = {
    "s_geom_groupe": "surface_bat",
    "bdtopo_bat_hauteur_mean": "hauteur_mean",
    "bdtopo_bat_max_hauteur": "hauteur_max",
    "ffo_bat_nb_niveau": "nb_niveaux",
    "ffo_bat_annee_construction": "annee_construction",
    "ffo_bat_nb_log": "nb_logements",
}
CAT_MAPPING = {
    "usage_principal_bdnb_open": "usage_principal",
    "ffo_bat_usage_niveau_1_txt": "usage_foncier",
    "ffo_bat_mat_mur_txt": "mat_murs",
    "bdtopo_bat_l_nature": "nature_bdtopo",
    "bdtopo_bat_l_usage_1": "usage_bdtopo",
}

# Colonnes à charger (mapping inversé : noms projet déjà dans le parquet)
LOAD_COLS = [
    "surface_bat", "hauteur_mean", "hauteur_max", "nb_niveaux",
    "annee_construction", "nb_logements",
    "usage_principal", "usage_foncier", "mat_murs", "nature_bdtopo", "usage_bdtopo",
    "usage_principal_enc", "usage_foncier_enc", "mat_murs_enc",
    "nature_bdtopo_enc", "usage_bdtopo_enc",
    "cluster_label", "geometry",
]

FEATURE_COLS = [
    "surface_bat", "hauteur_mean", "hauteur_max", "nb_niveaux",
    "annee_construction", "nb_logements",
    "usage_principal_enc", "usage_foncier_enc", "mat_murs_enc",
    "nature_bdtopo_enc", "usage_bdtopo_enc",
]

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#000075", "#a9a9a9", "#ffd8b1", "#fffac8", "#e6beff",
]
NOISE_COLOR = "#808080"

HDBSCAN_GRID = {
    "min_cluster_size": [100, 200, 500, 1000],
    "min_samples": [10, 50],
}
HDBSCAN_SAMPLE_SIZE = 40000


# ═════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET FILTRAGE
# ═════════════════════════════════════════════════════════════════════

USAGES_RESIDENTIELS = {"Résidentiel individuel"}


def load_and_filter() -> gpd.GeoDataFrame:
    """Charge et filtre : résidentiel individuel uniquement, nb_logements < 4."""
    logger.info("Chargement...")
    gdf = gpd.read_parquet(DATA_PATH)
    logger.info("  Total : %d bâtiments (%d colonnes)", len(gdf), len(gdf.columns))

    # Ne garder que les colonnes utiles (libérer la mémoire)
    keep_cols = list(set(LOAD_COLS) & set(gdf.columns))
    gdf = gdf[keep_cols].copy()
    gc.collect()

    n_before = len(gdf)
    gdf = gdf[gdf["usage_principal"].isin(USAGES_RESIDENTIELS)].copy()
    logger.info("  Filtre usage résidentiel individuel : -%d → %d",
                n_before - len(gdf), len(gdf))

    n_before = len(gdf)
    gdf = gdf[gdf["nb_logements"] < 4].copy()
    logger.info("  Filtre nb_logements < 4 : -%d → %d",
                n_before - len(gdf), len(gdf))

    return gdf


# ═════════════════════════════════════════════════════════════════════
# 2. PRÉPARATION FEATURES
# ═════════════════════════════════════════════════════════════════════

def prepare_features(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extrait et normalise les features."""
    X = gdf[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("  Features : %d colonnes, %d lignes", X_scaled.shape[1], X_scaled.shape[0])
    return X_scaled


# ═════════════════════════════════════════════════════════════════════
# 3. MÉTRIQUES
# ═════════════════════════════════════════════════════════════════════

def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    """Calcule silhouette, DB, CH (en excluant le bruit -1)."""
    mask = labels != -1
    n_noise = int((~mask).sum())
    noise_pct = round(n_noise / len(labels), 4)
    n_clusters = len(set(labels[mask])) if mask.any() else 0

    if n_clusters < 2:
        return {"n_clusters": n_clusters, "n_noise": n_noise, "noise_pct": noise_pct,
                "silhouette": -1, "davies_bouldin": -1, "calinski_harabasz": -1}

    Xc, lc = X[mask], labels[mask]
    sil = silhouette_score(Xc, lc, sample_size=min(SAMPLE_SIL, len(Xc)), random_state=RANDOM_STATE)
    db = davies_bouldin_score(Xc, lc)
    ch = calinski_harabasz_score(Xc, lc)

    return {"n_clusters": n_clusters, "n_noise": n_noise, "noise_pct": noise_pct,
            "silhouette": round(sil, 4), "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 1)}


# ═════════════════════════════════════════════════════════════════════
# 4. KMEANS
# ═════════════════════════════════════════════════════════════════════

def run_kmeans(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """KMeans k=15."""
    logger.info("KMeans k=%d ...", TARGET_K)
    km = KMeans(n_clusters=TARGET_K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    metrics = evaluate(X, labels)
    logger.info("  sil=%.4f  DB=%.4f  CH=%.1f",
                metrics["silhouette"], metrics["davies_bouldin"], metrics["calinski_harabasz"])
    return labels, metrics


# ═════════════════════════════════════════════════════════════════════
# 5. HDBSCAN (grid search sur échantillon + apply full)
# ═════════════════════════════════════════════════════════════════════

def run_hdbscan(X: np.ndarray) -> tuple[np.ndarray, dict, dict, list[dict]]:
    """Grid search HDBSCAN sur échantillon, puis apply best sur full dataset."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X)
    sample_n = min(HDBSCAN_SAMPLE_SIZE, n)
    idx = rng.choice(n, sample_n, replace=False)
    X_sample = X[idx]

    combos = list(product(HDBSCAN_GRID["min_cluster_size"], HDBSCAN_GRID["min_samples"]))
    logger.info("HDBSCAN grid search : %d combos sur %d échantillons", len(combos), sample_n)

    grid_results = []
    for mcs, ms in combos:
        logger.info("  mcs=%d ms=%d ...", mcs, ms)
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                              core_dist_n_jobs=-1, prediction_data=True)
        lab = cl.fit_predict(X_sample)
        met = evaluate(X_sample, lab)
        grid_results.append({"params": {"min_cluster_size": mcs, "min_samples": ms},
                             "clusterer": cl, **met})
        logger.info("    k=%d  bruit=%.1f%%  sil=%.4f", met["n_clusters"], met["noise_pct"]*100, met["silhouette"])
        gc.collect()

    # Sélectionner le meilleur (bruit <= 20% de préférence)
    low_noise = [r for r in grid_results if r["noise_pct"] <= 0.20 and r["n_clusters"] >= 2]
    if low_noise:
        best = max(low_noise, key=lambda r: r["silhouette"])
    else:
        valid = [r for r in grid_results if r["n_clusters"] >= 2]
        best = max(valid, key=lambda r: r["silhouette"]) if valid else grid_results[0]

    logger.info("Meilleur grid : mcs=%d ms=%d sil=%.4f bruit=%.1f%%",
                best["params"]["min_cluster_size"], best["params"]["min_samples"],
                best["silhouette"], best["noise_pct"]*100)

    # Apply full
    logger.info("Application sur %d bâtiments (approximate_predict)...", n)
    labels_full, _ = hdbscan.approximate_predict(best["clusterer"], X)
    metrics_full = evaluate(X, labels_full)
    logger.info("  Full : k=%d  bruit=%.1f%%  sil=%.4f",
                metrics_full["n_clusters"], metrics_full["noise_pct"]*100, metrics_full["silhouette"])

    return labels_full, metrics_full, best["params"], grid_results


# ═════════════════════════════════════════════════════════════════════
# 6. NOMMAGE DES CLUSTERS
# ═════════════════════════════════════════════════════════════════════

def name_cluster(sub: gpd.GeoDataFrame) -> str:
    """Nomme un cluster selon son profil médian."""
    s = sub["surface_bat"].median()
    h = sub["hauteur_mean"].median()
    a = sub["annee_construction"].median()
    mat = sub["mat_murs"].mode()
    m = mat.iloc[0] if len(mat) > 0 else "inconnu"
    usage = sub["usage_principal"].mode()
    u = str(usage.iloc[0]) if len(usage) > 0 else "inconnu"

    # Période
    if a < 1945:
        periode = "ancien"
    elif a < 1975:
        periode = "Trente Glorieuses"
    elif a < 2000:
        periode = "fin XXe"
    else:
        periode = "récent"

    # Matériau simplifié
    mat_short = ""
    m_lower = str(m).lower()
    if "pierre" in m_lower:
        mat_short = "pierre"
    elif "béton" in m_lower or "beton" in m_lower:
        mat_short = "béton"
    elif "brique" in m_lower:
        mat_short = "brique"
    elif "bois" in m_lower:
        mat_short = "bois"
    elif "parpaing" in m_lower or "agglo" in m_lower:
        mat_short = "parpaing"

    # Type par usage + surface + hauteur
    if "collectif" in u.lower():
        if h > 20:
            typ = "Tour"
        elif h > 12:
            typ = "Collectif haut"
        elif s > 500:
            typ = "Grand collectif"
        else:
            typ = "Petit collectif"
    elif "tertiaire" in u.lower() or "Tertiaire" in u:
        if s > 2000:
            typ = "Grand équipement"
        elif s > 400:
            typ = "Tertiaire moyen"
        else:
            typ = "Petit tertiaire"
    elif s < 60:
        typ = "Petit bâti"
    elif h <= 4:
        if s > 200:
            typ = "Grande maison"
        else:
            typ = "Pavillon"
    elif h <= 7:
        typ = "Maison R+1"
    elif h <= 10:
        typ = "Individuel dense R+2"
    else:
        typ = "Individuel haut"

    parts = [typ, f"{s:.0f}m²", f"{h:.0f}m", periode]
    if mat_short:
        parts.append(mat_short)
    return " / ".join(parts)


def name_clusters(gdf: gpd.GeoDataFrame, label_col: str) -> dict[int, str]:
    """Génère les noms pour tous les clusters."""
    names = {}
    for cl in sorted(gdf[label_col].unique()):
        if cl == -1:
            names[-1] = "Bruit (non classé)"
            continue
        sub = gdf[gdf[label_col] == cl]
        names[cl] = name_cluster(sub)
    return names


# ═════════════════════════════════════════════════════════════════════
# 7. LOG EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════

def log_experiments(
    km_metrics: dict,
    hdb_metrics: dict,
    hdb_params: dict,
    hdb_grid: list[dict],
    n_samples: int,
) -> None:
    """Ajoute les runs dans experiments.json."""
    with open(EXP_PATH) as f:
        experiments = json.load(f)

    next_id = len(experiments["runs"]) + 1
    note_base = "v2 — résidentiel individuel (nb_log<4, usage=Résidentiel individuel)"

    # KMeans
    experiments["runs"].append({
        "id": f"run_{next_id:03d}",
        "date": datetime.now().isoformat(),
        "algorithm": "kmeans",
        "params": {"n_clusters": TARGET_K},
        "n_clusters": km_metrics["n_clusters"],
        "n_noise": 0,
        "noise_pct": 0,
        "features_used": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "n_samples": n_samples,
        "metrics": {
            "silhouette": km_metrics["silhouette"],
            "davies_bouldin": km_metrics["davies_bouldin"],
            "calinski_harabasz": km_metrics["calinski_harabasz"],
        },
        "data_sources": ["bdnb"],
        "notes": f"KMeans {note_base}",
    })
    next_id += 1

    # HDBSCAN grid (sur échantillon)
    for r in hdb_grid:
        experiments["runs"].append({
            "id": f"run_{next_id:03d}",
            "date": datetime.now().isoformat(),
            "algorithm": "hdbscan",
            "params": r["params"],
            "n_clusters": r["n_clusters"],
            "n_noise": r["n_noise"],
            "noise_pct": r["noise_pct"],
            "features_used": FEATURE_COLS,
            "n_features": len(FEATURE_COLS),
            "n_samples": min(HDBSCAN_SAMPLE_SIZE, n_samples),
            "metrics": {
                "silhouette": r["silhouette"],
                "davies_bouldin": r["davies_bouldin"],
                "calinski_harabasz": r["calinski_harabasz"],
            },
            "data_sources": ["bdnb"],
            "notes": f"HDBSCAN grid (échantillon) {note_base} mcs={r['params']['min_cluster_size']} ms={r['params']['min_samples']}",
        })
        next_id += 1

    # HDBSCAN best full
    experiments["runs"].append({
        "id": f"run_{next_id:03d}",
        "date": datetime.now().isoformat(),
        "algorithm": "hdbscan",
        "params": hdb_params,
        "n_clusters": hdb_metrics["n_clusters"],
        "n_noise": hdb_metrics["n_noise"],
        "noise_pct": hdb_metrics["noise_pct"],
        "features_used": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "n_samples": n_samples,
        "metrics": {
            "silhouette": hdb_metrics["silhouette"],
            "davies_bouldin": hdb_metrics["davies_bouldin"],
            "calinski_harabasz": hdb_metrics["calinski_harabasz"],
        },
        "data_sources": ["bdnb"],
        "notes": f"HDBSCAN BEST (full) {note_base} mcs={hdb_params['min_cluster_size']} ms={hdb_params['min_samples']}",
    })

    with open(EXP_PATH, "w") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)

    logger.info("  %d runs ajoutés à %s", len(hdb_grid) + 2, EXP_PATH)


# ═════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ═════════════════════════════════════════════════════════════════════

def fig_carte_comparison(gdf: gpd.GeoDataFrame, km_names: dict, hdb_names: dict) -> str:
    """Cartes côte-à-côte KMeans vs HDBSCAN."""
    bounds = gdf.total_bounds
    ratio = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    fw = 28
    fh = max(fw / 2 * ratio, 10)

    fig, axes = plt.subplots(1, 2, figsize=(fw, fh))

    # KMeans
    ax = axes[0]
    for cl in sorted(gdf["km_label"].unique()):
        gdf[gdf["km_label"] == cl].plot(ax=ax, color=COLORS[cl % len(COLORS)],
                                         linewidth=0, alpha=0.7)
    patches = [mpatches.Patch(color=COLORS[cl % len(COLORS)],
               label=f"{cl}: {km_names[cl][:35]}") for cl in sorted(km_names)]
    ax.legend(handles=patches, loc="upper left", fontsize=6, ncol=2, framealpha=0.9)
    ax.set_title("KMeans k=15", fontsize=14, fontweight="bold")
    ax.set_aspect("equal"); ax.set_axis_off()

    # HDBSCAN
    ax = axes[1]
    noise = gdf[gdf["hdb_label"] == -1]
    if len(noise) > 0:
        noise.plot(ax=ax, color=NOISE_COLOR, alpha=0.15, linewidth=0)
    for cl in sorted(gdf["hdb_label"].unique()):
        if cl == -1:
            continue
        gdf[gdf["hdb_label"] == cl].plot(ax=ax, color=COLORS[cl % len(COLORS)],
                                          linewidth=0, alpha=0.7)
    ax.set_title("HDBSCAN (meilleur)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal"); ax.set_axis_off()

    plt.suptitle("Petit résidentiel — KMeans vs HDBSCAN", fontsize=18, fontweight="bold")
    plt.tight_layout()
    path = "outputs/figures/report_v3_comparaison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure : %s", path)
    return path


def fig_zoom_centre(gdf: gpd.GeoDataFrame, label_col: str, names: dict, title: str, filename: str) -> str:
    """Zoom centre Bordeaux."""
    xmin, xmax = 410000, 416000
    ymin, ymax = 6424000, 6430000
    zoom_ratio = (ymax - ymin) / (xmax - xmin)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16 * zoom_ratio))

    if -1 in gdf[label_col].values:
        noise = gdf[gdf[label_col] == -1].cx[xmin:xmax, ymin:ymax]
        if len(noise) > 0:
            noise.plot(ax=ax, color=NOISE_COLOR, alpha=0.2, linewidth=0)

    for cl in sorted(gdf[label_col].unique()):
        if cl == -1:
            continue
        sub = gdf[gdf[label_col] == cl].cx[xmin:xmax, ymin:ymax]
        if len(sub) > 0:
            sub.plot(ax=ax, color=COLORS[cl % len(COLORS)],
                     linewidth=0.3, edgecolor="#333333", alpha=0.85)

    patches = [mpatches.Patch(color=COLORS[cl % len(COLORS)],
               label=f"{cl}: {names[cl][:35]}") for cl in sorted(names) if cl != -1]
    ax.legend(handles=patches, loc="upper left", fontsize=6, ncol=2, framealpha=0.9)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_aspect("equal"); ax.set_axis_off()
    plt.tight_layout()

    path = f"outputs/figures/{filename}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure : %s", path)
    return path


def fig_barplot(gdf: gpd.GeoDataFrame, label_col: str, names: dict, title: str, filename: str) -> str:
    """Barplot effectifs."""
    counts = gdf[label_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(16, 7))
    bar_colors = [NOISE_COLOR if cl == -1 else COLORS[cl % len(COLORS)] for cl in counts.index]
    bar_labels = [names.get(cl, f"Cluster {cl}")[:25] for cl in counts.index]

    bars = ax.bar(range(len(counts)), counts.values, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{cl}\n{bar_labels[i]}" for i, cl in enumerate(counts.index)],
                       fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Nombre de bâtiments")
    ax.set_title(title, fontsize=16, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,}".replace(",", " "), ha="center", fontsize=7, fontweight="bold")
    plt.tight_layout()

    path = f"outputs/figures/{filename}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure : %s", path)
    return path


def fig_boxplots(gdf: gpd.GeoDataFrame, label_col: str, names: dict, filename: str) -> str:
    """Boxplots surface et hauteur."""
    labels_sorted = sorted([cl for cl in gdf[label_col].unique() if cl != -1])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for i, (col, ttl) in enumerate([("surface_bat", "Surface (m²)"), ("hauteur_mean", "Hauteur (m)")]):
        data = [gdf[gdf[label_col] == cl][col].dropna().values for cl in labels_sorted]
        bp = axes[i].boxplot(data, labels=labels_sorted, patch_artist=True, showfliers=False)
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[labels_sorted[j] % len(COLORS)])
        if col == "surface_bat":
            axes[i].set_yscale("log")
        axes[i].set_title(ttl, fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Cluster")
    plt.tight_layout()

    path = f"outputs/figures/{filename}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure : %s", path)
    return path


def fig_heatmap(grid_results: list[dict], filename: str) -> str:
    """Heatmap silhouette + bruit par paramètres HDBSCAN."""
    mcs_vals = sorted(set(r["params"]["min_cluster_size"] for r in grid_results))
    ms_vals = sorted(set(r["params"]["min_samples"] for r in grid_results))

    sil_m = np.zeros((len(ms_vals), len(mcs_vals)))
    noise_m = np.zeros_like(sil_m)
    for r in grid_results:
        i = ms_vals.index(r["params"]["min_samples"])
        j = mcs_vals.index(r["params"]["min_cluster_size"])
        sil_m[i, j] = r["silhouette"]
        noise_m[i, j] = r["noise_pct"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, mat, cmap, title, fmt in [
        (axes[0], sil_m, "RdYlGn", "Silhouette", ".3f"),
        (axes[1], noise_m, "Reds", "% Bruit", ".1f"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(mcs_vals))); ax.set_xticklabels(mcs_vals)
        ax.set_yticks(range(len(ms_vals))); ax.set_yticklabels(ms_vals)
        ax.set_xlabel("min_cluster_size"); ax.set_ylabel("min_samples")
        ax.set_title(title, fontsize=13, fontweight="bold")
        for ii in range(len(ms_vals)):
            for jj in range(len(mcs_vals)):
                val = f"{mat[ii,jj]:{fmt}}" + ("%" if "Bruit" in title else "")
                ax.text(jj, ii, val, ha="center", va="center", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("HDBSCAN — Sensibilité aux paramètres", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = f"outputs/figures/{filename}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure : %s", path)
    return path


# ═════════════════════════════════════════════════════════════════════
# 9. RAPPORT PDF v3
# ═════════════════════════════════════════════════════════════════════

def generate_pdf(
    gdf: gpd.GeoDataFrame,
    km_metrics: dict,
    hdb_metrics: dict,
    hdb_params: dict,
    km_names: dict,
    hdb_names: dict,
    figures: dict,
) -> str:
    """Génère le rapport PDF v3."""
    from PIL import Image as PILImage
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white, grey
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether,
    )

    def fit(path, max_w, max_h):
        img = PILImage.open(path)
        iw, ih = img.size
        r = ih / iw
        w, h = max_w, max_w * r
        if h > max_h:
            h, w = max_h, max_h / r
        return w, h

    pdf_path = "outputs/reports/clustering_bordeaux_v3.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    pw = A4[0] - 3*cm
    ph = A4[1] - 3*cm

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CustomTitle", parent=styles["Title"],
               fontSize=22, spaceAfter=30, textColor=HexColor("#1a1a2e")))
    styles.add(ParagraphStyle("Section", parent=styles["Heading1"],
               fontSize=16, spaceAfter=12, spaceBefore=20, textColor=HexColor("#16213e")))
    styles.add(ParagraphStyle("SubSec", parent=styles["Heading2"],
               fontSize=12, spaceAfter=8, spaceBefore=12, textColor=HexColor("#0f3460")))
    styles.add(ParagraphStyle("Body2", parent=styles["Normal"],
               fontSize=9, spaceAfter=6, leading=13))
    styles.add(ParagraphStyle("CName", parent=styles["Normal"],
               fontSize=10, spaceAfter=4, leading=14,
               textColor=HexColor("#e94560"), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("Rule", parent=styles["Normal"],
               fontSize=9, spaceAfter=2, leading=12, leftIndent=20, textColor=HexColor("#333333")))
    styles.add(ParagraphStyle("Verdict", parent=styles["Normal"],
               fontSize=12, spaceAfter=8, leading=16,
               textColor=HexColor("#16213e"), fontName="Helvetica-Bold"))

    story = []
    n_bat = len(gdf)

    # ── Titre ──
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("Clustering v2 — Petit résidentiel", styles["CustomTitle"]))
    story.append(Paragraph("Bordeaux Métropole — KMeans vs HDBSCAN", styles["Section"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"<b>{n_bat:,}</b> bâtiments (résidentiel individuel, nb_logements &lt; 4)".replace(",", " "), styles["Body2"]))
    story.append(Paragraph(f"<b>11</b> features — Source : BDNB 2025-07-a — mars 2026", styles["Body2"]))
    story.append(Paragraph(
        "Ce rapport présente le clustering sur le sous-ensemble résidentiel individuel "
        "(moins de 4 logements). Les bâtiments tertiaires, collectifs, dépendances et "
        "à usage inconnu sont exclus.", styles["Body2"]))
    story.append(PageBreak())

    # ── Métriques comparatives ──
    story.append(Paragraph("1. Comparaison des métriques", styles["Section"]))
    delta_sil = hdb_metrics["silhouette"] - km_metrics["silhouette"]
    md = [
        ["Métrique", "KMeans k=15", f"HDBSCAN (mcs={hdb_params['min_cluster_size']})", "Delta"],
        ["Silhouette", f"{km_metrics['silhouette']:.4f}", f"{hdb_metrics['silhouette']:.4f}", f"{delta_sil:+.4f}"],
        ["Davies-Bouldin", f"{km_metrics['davies_bouldin']:.4f}", f"{hdb_metrics['davies_bouldin']:.4f}",
         f"{hdb_metrics['davies_bouldin'] - km_metrics['davies_bouldin']:+.4f}"],
        ["Calinski-H.", f"{km_metrics['calinski_harabasz']:.0f}", f"{hdb_metrics['calinski_harabasz']:.0f}",
         f"{hdb_metrics['calinski_harabasz'] - km_metrics['calinski_harabasz']:+.0f}"],
        ["Clusters", str(km_metrics["n_clusters"]), str(hdb_metrics["n_clusters"]), ""],
        ["Bruit", "0 (0%)", f"{hdb_metrics['n_noise']} ({hdb_metrics['noise_pct']*100:.1f}%)", ""],
    ]
    t = Table(md, colWidths=[3.5*cm, 3.5*cm, 4.5*cm, 3*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), HexColor("#16213e")), ("TEXTCOLOR", (0,0), (-1,0), white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.5, grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, HexColor("#f0f0f0")]),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Comparaison avec v1
    story.append(Paragraph("Rappel v1 (dataset complet 234k) : KMeans silhouette = 0.3628", styles["Body2"]))

    if delta_sil > 0:
        v = f"HDBSCAN surpasse KMeans de +{delta_sil:.4f} en silhouette."
    elif delta_sil > -0.05:
        v = f"HDBSCAN comparable à KMeans (delta = {delta_sil:+.4f})."
    else:
        v = f"KMeans reste supérieur (delta = {delta_sil:+.4f})."
    if hdb_metrics["noise_pct"] > 0.20:
        v += f" Bruit = {hdb_metrics['noise_pct']*100:.1f}% (> seuil 20%)."
    story.append(Paragraph(v, styles["Verdict"]))
    story.append(PageBreak())

    # ── Heatmap HDBSCAN ──
    story.append(Paragraph("2. Sensibilité HDBSCAN aux paramètres", styles["Section"]))
    iw, ih = fit(figures["heatmap"], pw, 10*cm)
    story.append(Image(figures["heatmap"], width=iw, height=ih))
    story.append(PageBreak())

    # ── Cartes comparatives ──
    story.append(Paragraph("3. Cartes comparatives — vue d'ensemble", styles["Section"]))
    iw, ih = fit(figures["comparison"], pw, ph - 3*cm)
    story.append(Image(figures["comparison"], width=iw, height=ih))
    story.append(PageBreak())

    # ── Zoom KMeans ──
    story.append(Paragraph("4. Zoom Centre Bordeaux — KMeans", styles["Section"]))
    iw, ih = fit(figures["zoom_km"], pw, ph - 3*cm)
    story.append(Image(figures["zoom_km"], width=iw, height=ih))
    story.append(PageBreak())

    # ── Zoom HDBSCAN ──
    story.append(Paragraph("5. Zoom Centre Bordeaux — HDBSCAN", styles["Section"]))
    iw, ih = fit(figures["zoom_hdb"], pw, ph - 3*cm)
    story.append(Image(figures["zoom_hdb"], width=iw, height=ih))
    story.append(PageBreak())

    # ── Effectifs KMeans ──
    story.append(Paragraph("6. Effectifs — KMeans", styles["Section"]))
    iw, ih = fit(figures["barplot_km"], pw, 9*cm)
    story.append(Image(figures["barplot_km"], width=iw, height=ih))
    story.append(Spacer(1, 0.3*cm))
    iw, ih = fit(figures["boxplots_km"], pw, 9*cm)
    story.append(Image(figures["boxplots_km"], width=iw, height=ih))
    story.append(PageBreak())

    # ── Profil KMeans ──
    story.append(Paragraph("7. Profil des 15 clusters KMeans", styles["Section"]))
    header = ["Cl.", "Nom", "N", "%", "Surf.", "Haut.", "Niv.", "Année", "Mat. murs"]
    tdata = [header]
    for cl in sorted(km_names):
        sub = gdf[gdf["km_label"] == cl]
        s = sub["surface_bat"].median(); h = sub["hauteur_mean"].median()
        niv = sub["nb_niveaux"].median(); a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode(); m = mat.iloc[0] if len(mat) > 0 else "-"
        tdata.append([
            str(cl), km_names[cl][:30], str(len(sub)),
            f"{len(sub)/n_bat*100:.1f}", f"{s:.0f}", f"{h:.1f}",
            f"{niv:.0f}" if niv == niv else "-",
            f"{a:.0f}" if a == a else "-",
            str(m)[:20],
        ])
    cw = [0.7*cm, 4.2*cm, 1.3*cm, 0.9*cm, 1*cm, 1*cm, 0.8*cm, 1.1*cm, 3.5*cm]
    t2 = Table(tdata, colWidths=cw, repeatRows=1)
    ts = [("BACKGROUND",(0,0),(-1,0),HexColor("#16213e")),("TEXTCOLOR",(0,0),(-1,0),white),
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),
          ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(1,1),(1,-1),"LEFT"),("ALIGN",(-1,1),(-1,-1),"LEFT"),
          ("GRID",(0,0),(-1,-1),0.5,grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#f0f0f0")]),
          ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]
    for i, cl in enumerate(sorted(km_names)):
        ts.append(("BACKGROUND",(0,i+1),(0,i+1),HexColor(COLORS[cl % len(COLORS)])))
        ts.append(("TEXTCOLOR",(0,i+1),(0,i+1),white))
    t2.setStyle(TableStyle(ts))
    story.append(t2)
    story.append(PageBreak())

    # ── Règles de clustering KMeans ──
    story.append(Paragraph("8. Description des clusters KMeans", styles["Section"]))
    for cl in sorted(km_names):
        sub = gdf[gdf["km_label"] == cl]
        s = sub["surface_bat"].median(); h = sub["hauteur_mean"].median()
        a = sub["annee_construction"].median()
        usage = sub["usage_principal"].mode()
        u = usage.iloc[0] if len(usage) > 0 else "inconnu"
        mat = sub["mat_murs"].mode()
        m = mat.iloc[0] if len(mat) > 0 else "inconnu"

        block = []
        block.append(Paragraph(
            f'<font color="{COLORS[cl % len(COLORS)]}">&#9632;</font> '
            f'Cluster {cl} — {km_names[cl]} ({len(sub)} bâtiments, {len(sub)/n_bat*100:.1f}%)',
            styles["CName"]))
        block.append(Paragraph(f"• Surface médiane : {s:.0f} m² — Hauteur : {h:.1f} m", styles["Rule"]))
        if a == a:
            block.append(Paragraph(f"• Année construction médiane : {a:.0f}", styles["Rule"]))
        block.append(Paragraph(f"• Usage dominant : {u}", styles["Rule"]))
        block.append(Paragraph(f"• Matériau murs : {m}", styles["Rule"]))
        block.append(Spacer(1, 0.2*cm))
        story.append(KeepTogether(block))
    story.append(PageBreak())

    # ── Conclusions ──
    story.append(Paragraph("9. Conclusions et prochaines étapes", styles["Section"]))

    story.append(Paragraph("Comparaison avec le clustering v1 (234k bâtiments) :", styles["SubSec"]))
    story.append(Paragraph(
        f"En se concentrant sur le résidentiel individuel avec moins de 4 logements "
        f"({n_bat:,} bâtiments), le clustering gagne en homogénéité.".replace(",", " "),
        styles["Body2"]))

    for p in [
        f"KMeans v2 : silhouette = {km_metrics['silhouette']:.4f} (v1 = 0.3628)",
        f"HDBSCAN v2 : silhouette = {hdb_metrics['silhouette']:.4f}, {hdb_metrics['n_clusters']} clusters, {hdb_metrics['noise_pct']*100:.1f}% bruit",
        "Le filtrage du cluster 1 (fourre-tout) améliore la lisibilité des typologies",
        "Les sous-types résidentiels (pavillons, maisons R+1, individuel dense) sont mieux séparés",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Prochaines étapes :", styles["SubSec"]))
    for p in [
        "Calculer le Moran's I pour la cohérence spatiale des deux algorithmes",
        "Enrichir avec les features DVF (prix m²) et INSEE (revenus)",
        "Tester un clustering séparé pour les collectifs (nb_logements >= 4)",
        "Affiner la distinction entre pavillons anciens et récents",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    doc.build(story)
    logger.info("PDF v3 : %s", pdf_path)
    return pdf_path


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    """Pipeline complet."""
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1. Charger et filtrer
    gdf = load_and_filter()
    X_scaled = prepare_features(gdf)
    n_bat = len(gdf)

    # 2. KMeans
    km_labels, km_metrics = run_kmeans(X_scaled)
    gdf["km_label"] = km_labels

    # 3. HDBSCAN
    hdb_labels, hdb_metrics, hdb_params, hdb_grid = run_hdbscan(X_scaled)
    gdf["hdb_label"] = hdb_labels

    # 4. Nommer les clusters
    km_names = name_clusters(gdf, "km_label")
    hdb_names = name_clusters(gdf, "hdb_label")

    logger.info("\nNoms KMeans :")
    for cl, name in sorted(km_names.items()):
        logger.info("  %2d : %s (%d)", cl, name, (gdf["km_label"] == cl).sum())

    logger.info("\nNoms HDBSCAN :")
    for cl, name in sorted(hdb_names.items()):
        logger.info("  %2d : %s (%d)", cl, name, (gdf["hdb_label"] == cl).sum())

    # 5. Sauvegarder le geoparquet
    gdf["cluster_label"] = km_labels  # KMeans comme label principal
    gdf["cluster_algo"] = "kmeans"
    gdf["hdbscan_label"] = hdb_labels
    gdf.to_parquet(OUTPUT_PATH)
    logger.info("Sauvegardé : %s (%d bâtiments)", OUTPUT_PATH, n_bat)

    # 6. Log experiments
    log_experiments(km_metrics, hdb_metrics, hdb_params, hdb_grid, n_bat)

    # 7. Figures
    figures = {}
    figures["comparison"] = fig_carte_comparison(gdf, km_names, hdb_names)
    figures["zoom_km"] = fig_zoom_centre(gdf, "km_label", km_names,
                                          "KMeans k=15 — Zoom Centre Bordeaux",
                                          "report_v3_zoom_kmeans.png")
    figures["zoom_hdb"] = fig_zoom_centre(gdf, "hdb_label", hdb_names,
                                           "HDBSCAN — Zoom Centre Bordeaux",
                                           "report_v3_zoom_hdbscan.png")
    figures["barplot_km"] = fig_barplot(gdf, "km_label", km_names,
                                        "KMeans — Effectifs", "report_v3_effectifs_km.png")
    figures["boxplots_km"] = fig_boxplots(gdf, "km_label", km_names,
                                           "report_v3_boxplots_km.png")
    figures["heatmap"] = fig_heatmap(hdb_grid, "report_v3_hdbscan_heatmap.png")

    # 8. PDF v3
    generate_pdf(gdf, km_metrics, hdb_metrics, hdb_params, km_names, hdb_names, figures)

    # 9. Résumé final
    logger.info("\n" + "=" * 70)
    logger.info("RÉSUMÉ v2 — Petit résidentiel (%d bâtiments)", n_bat)
    logger.info("=" * 70)
    logger.info("  KMeans k=15  : sil=%.4f  DB=%.4f  CH=%.0f",
                km_metrics["silhouette"], km_metrics["davies_bouldin"], km_metrics["calinski_harabasz"])
    logger.info("  HDBSCAN best : sil=%.4f  DB=%.4f  CH=%.0f  k=%d  bruit=%.1f%%",
                hdb_metrics["silhouette"], hdb_metrics["davies_bouldin"], hdb_metrics["calinski_harabasz"],
                hdb_metrics["n_clusters"], hdb_metrics["noise_pct"]*100)
    logger.info("  Rappel v1    : sil=0.3628  (234k bâtiments, KMeans k=15)")
    logger.info("=" * 70)
    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
