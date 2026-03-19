"""HDBSCAN clustering — grid search sur échantillon + application full dataset.

Stratégie mémoire :
1. Grid search HDBSCAN sur un échantillon de 50k bâtiments
2. Sélection du meilleur paramétrage (silhouette, bruit <= 20%)
3. Application du meilleur sur le dataset complet via approximate_predict
4. Évaluation, log experiments.json, figures comparatives, rapport v3 PDF
"""

import geopandas as gpd
import numpy as np
import json
import logging
import os
import gc
from datetime import datetime
from itertools import product

import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

# ── Config ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "surface_bat", "hauteur_mean", "hauteur_max", "nb_niveaux",
    "annee_construction", "nb_logements",
    "usage_principal_enc", "usage_foncier_enc", "mat_murs_enc",
    "nature_bdtopo_enc", "usage_bdtopo_enc",
]

HDBSCAN_GRID = {
    "min_cluster_size": [100, 200, 500, 1000],
    "min_samples": [10, 50],
}

GRID_SAMPLE_SIZE = 50000
SAMPLE_SIZE_SILHOUETTE = 10000
RANDOM_STATE = 42
EXP_PATH = "outputs/experiments.json"
DATA_PATH = "data/processed/clustered.geoparquet"

KMEANS_REF = {
    "algorithm": "kmeans",
    "n_clusters": 15,
    "noise_pct": 0,
    "silhouette": 0.3628,
    "davies_bouldin": 1.0689,
    "calinski_harabasz": 52910.6,
}

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#000075", "#a9a9a9", "#ffd8b1", "#fffac8", "#e6beff",
]


LOAD_COLS = FEATURE_COLS + ["cluster_label", "usage_principal", "geometry"]


def load_and_scale() -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Charge le geoparquet (colonnes utiles seulement) et retourne (gdf, X_scaled)."""
    logger.info("Chargement %s (colonnes sélectionnées)...", DATA_PATH)
    gdf = gpd.read_parquet(DATA_PATH, columns=LOAD_COLS)
    logger.info("  %d bâtiments, %d features, %.0f MB",
                len(gdf), len(FEATURE_COLS),
                gdf.memory_usage(deep=True).sum() / 1e6)

    X = gdf[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return gdf, X_scaled


def evaluate(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """Calcule les métriques sur les points non-bruit."""
    mask = labels != -1
    n_noise = int((~mask).sum())
    noise_pct = round(n_noise / len(labels), 4)
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2:
        return {
            "n_clusters": n_clusters, "n_noise": n_noise,
            "noise_pct": noise_pct, "silhouette": -1,
            "davies_bouldin": -1, "calinski_harabasz": -1,
        }

    X_clean = X_scaled[mask]
    labels_clean = labels[mask]

    sil = silhouette_score(
        X_clean, labels_clean,
        sample_size=min(SAMPLE_SIZE_SILHOUETTE, len(X_clean)),
        random_state=RANDOM_STATE,
    )
    db = davies_bouldin_score(X_clean, labels_clean)
    ch = calinski_harabasz_score(X_clean, labels_clean)

    return {
        "n_clusters": n_clusters, "n_noise": n_noise,
        "noise_pct": noise_pct, "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4), "calinski_harabasz": round(ch, 1),
    }


def run_hdbscan_grid(X_scaled: np.ndarray) -> list[dict]:
    """Grid search HDBSCAN sur un échantillon."""
    rng = np.random.RandomState(RANDOM_STATE)
    n_total = len(X_scaled)
    sample_size = min(GRID_SAMPLE_SIZE, n_total)

    idx_sample = rng.choice(n_total, sample_size, replace=False)
    X_sample = X_scaled[idx_sample]
    logger.info("Grid search HDBSCAN sur échantillon de %d (sur %d)", sample_size, n_total)

    combos = list(product(
        HDBSCAN_GRID["min_cluster_size"],
        HDBSCAN_GRID["min_samples"],
    ))
    logger.info("  %d combinaisons à tester", len(combos))

    results = []
    for mcs, ms in combos:
        logger.info("  mcs=%d, ms=%d ...", mcs, ms)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            core_dist_n_jobs=-1,
            prediction_data=True,
        )
        labels_sample = clusterer.fit_predict(X_sample)
        metrics = evaluate(X_sample, labels_sample)

        results.append({
            "params": {"min_cluster_size": mcs, "min_samples": ms},
            "labels_sample": labels_sample,
            "idx_sample": idx_sample,
            "clusterer": clusterer,
            **metrics,
        })

        logger.info(
            "    -> k=%d  bruit=%.1f%%  sil=%.3f  DB=%.2f  CH=%.0f",
            metrics["n_clusters"], metrics["noise_pct"] * 100,
            metrics["silhouette"], metrics["davies_bouldin"],
            metrics["calinski_harabasz"],
        )
        gc.collect()

    return results


def apply_best_full(
    X_scaled: np.ndarray,
    best_result: dict,
) -> tuple[np.ndarray, dict]:
    """Applique le meilleur HDBSCAN sur le dataset complet via approximate_predict."""
    clusterer = best_result["clusterer"]
    logger.info(
        "Application du meilleur (mcs=%d, ms=%d) sur %d bâtiments via approximate_predict...",
        best_result["params"]["min_cluster_size"],
        best_result["params"]["min_samples"],
        len(X_scaled),
    )

    # approximate_predict pour les points hors échantillon
    labels_full, strengths = hdbscan.approximate_predict(clusterer, X_scaled)
    metrics_full = evaluate(X_scaled, labels_full)

    logger.info(
        "  Résultat full : k=%d  bruit=%.1f%%  sil=%.3f  DB=%.2f  CH=%.0f",
        metrics_full["n_clusters"], metrics_full["noise_pct"] * 100,
        metrics_full["silhouette"], metrics_full["davies_bouldin"],
        metrics_full["calinski_harabasz"],
    )

    return labels_full, metrics_full


def log_experiments(grid_results: list[dict], full_metrics: dict, best_params: dict) -> None:
    """Ajoute les runs dans experiments.json."""
    with open(EXP_PATH) as f:
        experiments = json.load(f)

    next_id = len(experiments["runs"]) + 1

    # Tous les runs grid (sur échantillon)
    for r in grid_results:
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
            "n_samples": GRID_SAMPLE_SIZE,
            "metrics": {
                "silhouette": r["silhouette"],
                "davies_bouldin": r["davies_bouldin"],
                "calinski_harabasz": r["calinski_harabasz"],
            },
            "data_sources": ["bdnb"],
            "notes": f"HDBSCAN grid (échantillon {GRID_SAMPLE_SIZE}) mcs={r['params']['min_cluster_size']} ms={r['params']['min_samples']}",
        })
        next_id += 1

    # Run final sur dataset complet
    experiments["runs"].append({
        "id": f"run_{next_id:03d}",
        "date": datetime.now().isoformat(),
        "algorithm": "hdbscan",
        "params": best_params,
        "n_clusters": full_metrics["n_clusters"],
        "n_noise": full_metrics["n_noise"],
        "noise_pct": full_metrics["noise_pct"],
        "features_used": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "n_samples": 234116,
        "metrics": {
            "silhouette": full_metrics["silhouette"],
            "davies_bouldin": full_metrics["davies_bouldin"],
            "calinski_harabasz": full_metrics["calinski_harabasz"],
        },
        "data_sources": ["bdnb"],
        "notes": f"HDBSCAN BEST sur dataset complet (approximate_predict) mcs={best_params['min_cluster_size']} ms={best_params['min_samples']}",
    })

    with open(EXP_PATH, "w") as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)

    logger.info("  %d runs ajoutés à %s", len(grid_results) + 1, EXP_PATH)


def print_comparative_report(grid_results: list[dict], full_metrics: dict, best_params: dict) -> str:
    """Génère le rapport comparatif texte."""
    lines = []
    lines.append("=" * 95)
    lines.append("RAPPORT COMPARATIF — HDBSCAN vs KMeans k=15 (run_001)")
    lines.append("=" * 95)
    lines.append("")

    # Référence
    lines.append("RÉFÉRENCE : KMeans k=15 (run_001, meilleur v2)")
    lines.append(f"  Silhouette     : {KMEANS_REF['silhouette']:.4f}")
    lines.append(f"  Davies-Bouldin : {KMEANS_REF['davies_bouldin']:.4f}")
    lines.append(f"  Calinski-H.    : {KMEANS_REF['calinski_harabasz']:.1f}")
    lines.append(f"  Bruit          : 0%")
    lines.append(f"  Clusters       : 15")
    lines.append("")

    # Grid search
    header = f"{'mcs':>6} {'ms':>5} {'k':>4} {'bruit%':>7} {'silhouette':>11} {'DB':>8} {'CH':>10} {'vs KMeans':>10}"
    lines.append(f"GRID SEARCH HDBSCAN (échantillon {GRID_SAMPLE_SIZE})")
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    sorted_results = sorted(grid_results, key=lambda r: r["silhouette"], reverse=True)
    for r in sorted_results:
        delta = r["silhouette"] - KMEANS_REF["silhouette"]
        sign = "+" if delta >= 0 else ""
        flag = ""
        if r["noise_pct"] > 0.20:
            flag = " !! BRUIT"
        elif delta > 0:
            flag = " <-- MIEUX"

        lines.append(
            f"{r['params']['min_cluster_size']:>6} "
            f"{r['params']['min_samples']:>5} "
            f"{r['n_clusters']:>4} "
            f"{r['noise_pct']*100:>6.1f}% "
            f"{r['silhouette']:>11.4f} "
            f"{r['davies_bouldin']:>8.4f} "
            f"{r['calinski_harabasz']:>10.1f} "
            f"{sign}{delta:>9.4f}{flag}"
        )

    lines.append("-" * len(header))
    lines.append("")

    # Résultat full dataset
    lines.append("MEILLEUR HDBSCAN — DATASET COMPLET (234 116 bâtiments)")
    lines.append(f"  Paramètres     : min_cluster_size={best_params['min_cluster_size']}, min_samples={best_params['min_samples']}")
    lines.append(f"  Clusters       : {full_metrics['n_clusters']}")
    lines.append(f"  Bruit          : {full_metrics['noise_pct']*100:.1f}% ({full_metrics['n_noise']} bâtiments)")
    lines.append(f"  Silhouette     : {full_metrics['silhouette']:.4f}")
    lines.append(f"  Davies-Bouldin : {full_metrics['davies_bouldin']:.4f}")
    lines.append(f"  Calinski-H.    : {full_metrics['calinski_harabasz']:.1f}")
    lines.append("")

    delta = full_metrics["silhouette"] - KMEANS_REF["silhouette"]
    if delta > 0:
        verdict = f"HDBSCAN SURPASSE KMeans de +{delta:.4f} en silhouette"
    elif delta > -0.05:
        verdict = f"HDBSCAN comparable à KMeans (delta={delta:+.4f})"
    else:
        verdict = f"KMeans reste supérieur (delta={delta:+.4f})"
    lines.append(f"  >>> {verdict}")

    if full_metrics["noise_pct"] > 0.20:
        lines.append(f"  !!! ATTENTION : bruit > 20% — envisager réaffectation au cluster le plus proche")
    lines.append("")

    # Synthèse
    lines.append("SYNTHÈSE")
    lines.append("-" * 40)
    low_noise = [r for r in grid_results if r["noise_pct"] <= 0.20 and r["n_clusters"] >= 2]
    lines.append(f"  Configs avec bruit <= 20% : {len(low_noise)}/{len(grid_results)}")
    k_values = [r["n_clusters"] for r in grid_results if r["n_clusters"] >= 2]
    if k_values:
        lines.append(f"  Plage de k détectés : {min(k_values)} à {max(k_values)} (cible : ~15)")
    lines.append("")
    lines.append("=" * 95)

    report = "\n".join(lines)
    logger.info("\n%s", report)
    return report


def generate_comparison_figure(
    gdf: gpd.GeoDataFrame,
    hdbscan_labels: np.ndarray,
    full_metrics: dict,
) -> str:
    """Carte statique côte-à-côte KMeans vs HDBSCAN."""
    bounds = gdf.total_bounds
    data_w = bounds[2] - bounds[0]
    data_h = bounds[3] - bounds[1]
    ratio = data_h / data_w

    fig_w = 28
    fig_h = fig_w / 2 * ratio
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, max(fig_h, 10)))

    # KMeans
    ax = axes[0]
    for cl in sorted(gdf["cluster_label"].unique()):
        sub = gdf[gdf["cluster_label"] == cl]
        sub.plot(ax=ax, color=COLORS[cl % len(COLORS)], linewidth=0, alpha=0.7)
    ax.set_title(
        f"KMeans k=15 — sil={KMEANS_REF['silhouette']:.3f} — bruit=0%",
        fontsize=14, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.set_axis_off()

    # HDBSCAN
    ax = axes[1]
    gdf_h = gdf.copy()
    gdf_h["hdbscan_label"] = hdbscan_labels

    noise = gdf_h[gdf_h["hdbscan_label"] == -1]
    if len(noise) > 0:
        noise.plot(ax=ax, color="#808080", alpha=0.15, linewidth=0)

    for cl in sorted(gdf_h["hdbscan_label"].unique()):
        if cl == -1:
            continue
        sub = gdf_h[gdf_h["hdbscan_label"] == cl]
        sub.plot(ax=ax, color=COLORS[cl % len(COLORS)], linewidth=0, alpha=0.7)

    ax.set_title(
        f"HDBSCAN k={full_metrics['n_clusters']} — sil={full_metrics['silhouette']:.3f} "
        f"— bruit={full_metrics['noise_pct']*100:.1f}%",
        fontsize=14, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.suptitle(
        "Comparaison KMeans vs HDBSCAN — Bordeaux Métropole",
        fontsize=18, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    path = "outputs/figures/report_v3_comparaison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure comparative : %s", path)
    return path


def generate_zoom_hdbscan(gdf: gpd.GeoDataFrame, hdbscan_labels: np.ndarray) -> str:
    """Zoom centre Bordeaux pour HDBSCAN."""
    gdf_h = gdf.copy()
    gdf_h["hdbscan_label"] = hdbscan_labels

    xmin, xmax = 410000, 416000
    ymin, ymax = 6424000, 6430000
    zoom_ratio = (ymax - ymin) / (xmax - xmin)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16 * zoom_ratio))

    # Bruit
    noise = gdf_h[gdf_h["hdbscan_label"] == -1].cx[xmin:xmax, ymin:ymax]
    if len(noise) > 0:
        noise.plot(ax=ax, color="#808080", alpha=0.2, linewidth=0)

    for cl in sorted(gdf_h["hdbscan_label"].unique()):
        if cl == -1:
            continue
        sub = gdf_h[gdf_h["hdbscan_label"] == cl].cx[xmin:xmax, ymin:ymax]
        if len(sub) > 0:
            sub.plot(ax=ax, color=COLORS[cl % len(COLORS)], linewidth=0.3,
                     edgecolor="#333333", alpha=0.85)

    n_clusters = len(set(hdbscan_labels) - {-1})
    ax.set_title(f"HDBSCAN k={n_clusters} — Zoom Centre Bordeaux", fontsize=16, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()

    path = "outputs/figures/report_v3_zoom_hdbscan.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Zoom HDBSCAN : %s", path)
    return path


def generate_hdbscan_barplot(gdf: gpd.GeoDataFrame, hdbscan_labels: np.ndarray) -> str:
    """Barplot des effectifs HDBSCAN."""
    import pandas as pd
    labels_series = pd.Series(hdbscan_labels)
    counts = labels_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(16, 7))
    bar_colors = []
    bar_labels = []
    for cl in counts.index:
        if cl == -1:
            bar_colors.append("#808080")
            bar_labels.append("Bruit")
        else:
            bar_colors.append(COLORS[cl % len(COLORS)])
            bar_labels.append(f"Cluster {cl}")

    bars = ax.bar(range(len(counts)), counts.values, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Nombre de bâtiments", fontsize=12)
    ax.set_title("HDBSCAN — Effectifs par cluster", fontsize=16, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}".replace(",", " "), ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()

    path = "outputs/figures/report_v3_effectifs_hdbscan.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Barplot HDBSCAN : %s", path)
    return path


def generate_grid_heatmap(grid_results: list[dict]) -> str:
    """Heatmap silhouette par paramètres."""
    mcs_vals = sorted(set(r["params"]["min_cluster_size"] for r in grid_results))
    ms_vals = sorted(set(r["params"]["min_samples"] for r in grid_results))

    sil_matrix = np.zeros((len(ms_vals), len(mcs_vals)))
    noise_matrix = np.zeros_like(sil_matrix)

    for r in grid_results:
        i = ms_vals.index(r["params"]["min_samples"])
        j = mcs_vals.index(r["params"]["min_cluster_size"])
        sil_matrix[i, j] = r["silhouette"]
        noise_matrix[i, j] = r["noise_pct"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im1 = axes[0].imshow(sil_matrix, cmap="RdYlGn", aspect="auto")
    axes[0].set_xticks(range(len(mcs_vals)))
    axes[0].set_xticklabels(mcs_vals)
    axes[0].set_yticks(range(len(ms_vals)))
    axes[0].set_yticklabels(ms_vals)
    axes[0].set_xlabel("min_cluster_size")
    axes[0].set_ylabel("min_samples")
    axes[0].set_title("Silhouette Score", fontsize=13, fontweight="bold")
    for i in range(len(ms_vals)):
        for j in range(len(mcs_vals)):
            axes[0].text(j, i, f"{sil_matrix[i,j]:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
    fig.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(noise_matrix, cmap="Reds", aspect="auto")
    axes[1].set_xticks(range(len(mcs_vals)))
    axes[1].set_xticklabels(mcs_vals)
    axes[1].set_yticks(range(len(ms_vals)))
    axes[1].set_yticklabels(ms_vals)
    axes[1].set_xlabel("min_cluster_size")
    axes[1].set_ylabel("min_samples")
    axes[1].set_title("% Bruit", fontsize=13, fontweight="bold")
    for i in range(len(ms_vals)):
        for j in range(len(mcs_vals)):
            axes[1].text(j, i, f"{noise_matrix[i,j]:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.suptitle("HDBSCAN — Sensibilité aux paramètres", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = "outputs/figures/report_v3_hdbscan_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap paramètres : %s", path)
    return path


def generate_pdf_report(
    gdf: gpd.GeoDataFrame,
    hdbscan_labels: np.ndarray,
    full_metrics: dict,
    best_params: dict,
    grid_results: list[dict],
    report_text: str,
    fig_comparison: str,
    fig_zoom: str,
    fig_barplot: str,
    fig_heatmap: str,
) -> str:
    """Génère le rapport PDF v3 comparatif."""
    from PIL import Image as PILImage
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, white, grey
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether,
    )

    def fit_image(path: str, max_w: float, max_h: float) -> tuple[float, float]:
        img = PILImage.open(path)
        iw, ih = img.size
        ratio = ih / iw
        w = max_w
        h = max_w * ratio
        if h > max_h:
            h = max_h
            w = max_h / ratio
        return w, h

    pdf_path = "outputs/reports/clustering_bordeaux_v3.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    page_w = A4[0] - 3*cm
    page_h = A4[1] - 3*cm

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
               fontSize=11, spaceAfter=4, leading=14,
               textColor=HexColor("#e94560"), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("RuleStyle", parent=styles["Normal"],
               fontSize=9, spaceAfter=2, leading=12, leftIndent=20, textColor=HexColor("#333333")))
    styles.add(ParagraphStyle("Verdict", parent=styles["Normal"],
               fontSize=12, spaceAfter=8, leading=16,
               textColor=HexColor("#16213e"), fontName="Helvetica-Bold"))

    story = []

    # ── Page titre ──
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("Clustering des bâtiments — v3", styles["CustomTitle"]))
    story.append(Paragraph("Comparaison KMeans vs HDBSCAN", styles["Section"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"<b>234 116</b> bâtiments — Bordeaux Métropole (28 communes)", styles["Body2"]))
    story.append(Paragraph(f"<b>11</b> features (6 numériques + 5 catégorielles encodées)", styles["Body2"]))
    story.append(Paragraph("Source : BDNB millésime 2025-07-a — Date : mars 2026", styles["Body2"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Ce rapport compare le clustering KMeans (k=15, v2) "
                           "avec HDBSCAN testé sur plusieurs paramétrisations.", styles["Body2"]))
    story.append(PageBreak())

    # ── Tableau comparatif métriques ──
    story.append(Paragraph("1. Comparaison des métriques", styles["Section"]))

    md = [
        ["Métrique", "KMeans k=15", f"HDBSCAN (mcs={best_params['min_cluster_size']}, ms={best_params['min_samples']})", "Delta"],
        ["Silhouette", f"{KMEANS_REF['silhouette']:.4f}", f"{full_metrics['silhouette']:.4f}",
         f"{full_metrics['silhouette'] - KMEANS_REF['silhouette']:+.4f}"],
        ["Davies-Bouldin", f"{KMEANS_REF['davies_bouldin']:.4f}", f"{full_metrics['davies_bouldin']:.4f}",
         f"{full_metrics['davies_bouldin'] - KMEANS_REF['davies_bouldin']:+.4f}"],
        ["Calinski-Harabasz", f"{KMEANS_REF['calinski_harabasz']:.0f}", f"{full_metrics['calinski_harabasz']:.0f}",
         f"{full_metrics['calinski_harabasz'] - KMEANS_REF['calinski_harabasz']:+.0f}"],
        ["Clusters", "15", str(full_metrics["n_clusters"]), ""],
        ["Bruit", "0 (0%)", f"{full_metrics['n_noise']} ({full_metrics['noise_pct']*100:.1f}%)", ""],
    ]
    t = Table(md, colWidths=[3.5*cm, 3.5*cm, 5.5*cm, 2.5*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))

    # Verdict
    delta_sil = full_metrics["silhouette"] - KMEANS_REF["silhouette"]
    if delta_sil > 0:
        verdict = f"HDBSCAN surpasse KMeans de +{delta_sil:.4f} en silhouette."
    elif delta_sil > -0.05:
        verdict = f"HDBSCAN comparable à KMeans (delta silhouette = {delta_sil:+.4f})."
    else:
        verdict = f"KMeans reste supérieur (delta silhouette = {delta_sil:+.4f})."

    if full_metrics["noise_pct"] > 0.20:
        verdict += f" Attention : {full_metrics['noise_pct']*100:.1f}% de bruit (> seuil 20%)."

    story.append(Paragraph(verdict, styles["Verdict"]))
    story.append(PageBreak())

    # ── Heatmap sensibilité ──
    story.append(Paragraph("2. Sensibilité HDBSCAN aux paramètres", styles["Section"]))
    story.append(Paragraph(
        f"Grid search sur un échantillon de {GRID_SAMPLE_SIZE} bâtiments. "
        f"{len(grid_results)} combinaisons testées (min_cluster_size × min_samples).",
        styles["Body2"],
    ))
    iw, ih = fit_image(fig_heatmap, page_w, 10*cm)
    story.append(Image(fig_heatmap, width=iw, height=ih))
    story.append(PageBreak())

    # ── Cartes comparatives ──
    story.append(Paragraph("3. Cartes comparatives — vue d'ensemble", styles["Section"]))
    iw, ih = fit_image(fig_comparison, page_w, page_h - 3*cm)
    story.append(Image(fig_comparison, width=iw, height=ih))
    story.append(PageBreak())

    # ── Zoom HDBSCAN ──
    story.append(Paragraph("4. Zoom Centre Bordeaux — HDBSCAN", styles["Section"]))
    iw, ih = fit_image(fig_zoom, page_w, page_h - 3*cm)
    story.append(Image(fig_zoom, width=iw, height=ih))
    story.append(PageBreak())

    # ── Effectifs HDBSCAN ──
    story.append(Paragraph("5. Effectifs par cluster — HDBSCAN", styles["Section"]))
    iw, ih = fit_image(fig_barplot, page_w, 10*cm)
    story.append(Image(fig_barplot, width=iw, height=ih))
    story.append(Spacer(1, 0.5*cm))

    # Profil des clusters HDBSCAN
    story.append(Paragraph("6. Profil des clusters HDBSCAN", styles["Section"]))
    gdf_h = gdf.copy()
    gdf_h["hdbscan_label"] = hdbscan_labels

    profile_header = ["Cl.", "N", "%", "Surface", "Hauteur", "Niveaux", "Année", "Usage"]
    profile_data = [profile_header]

    for cl in sorted(gdf_h["hdbscan_label"].unique()):
        sub = gdf_h[gdf_h["hdbscan_label"] == cl]
        pct = len(sub) / len(gdf_h) * 100
        surf = sub["surface_bat"].median()
        haut = sub["hauteur_mean"].median()
        niv = sub["nb_niveaux"].median()
        annee = sub["annee_construction"].median()
        usage = sub["usage_principal"].mode()
        u = usage.iloc[0] if len(usage) > 0 else "inconnu"
        cl_str = "Bruit" if cl == -1 else str(cl)
        profile_data.append([
            cl_str, str(len(sub)), f"{pct:.1f}",
            f"{surf:.0f}", f"{haut:.1f}",
            f"{niv:.0f}" if niv == niv else "-",
            f"{annee:.0f}" if annee == annee else "-",
            str(u)[:25],
        ])

    pcw = [1*cm, 1.5*cm, 1*cm, 1.5*cm, 1.3*cm, 1.3*cm, 1.3*cm, 5.5*cm]
    pt = Table(profile_data, colWidths=pcw, repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (-1, 1), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(pt)
    story.append(PageBreak())

    # ── Conclusions ──
    story.append(Paragraph("7. Conclusions et recommandations", styles["Section"]))

    conclusions = []
    if delta_sil > 0:
        conclusions.append(
            f"HDBSCAN avec mcs={best_params['min_cluster_size']}, ms={best_params['min_samples']} "
            f"obtient une silhouette supérieure ({full_metrics['silhouette']:.4f} vs {KMEANS_REF['silhouette']:.4f})."
        )
    else:
        conclusions.append(
            f"KMeans k=15 conserve la meilleure silhouette ({KMEANS_REF['silhouette']:.4f} vs {full_metrics['silhouette']:.4f})."
        )

    conclusions.append(
        f"HDBSCAN détecte automatiquement {full_metrics['n_clusters']} clusters "
        f"(vs 15 imposés pour KMeans) avec {full_metrics['noise_pct']*100:.1f}% de bruit."
    )

    if full_metrics["noise_pct"] > 0.20:
        conclusions.append(
            "Le taux de bruit dépasse le seuil de 20%. "
            "Pour la cartographie finale, les points bruit seraient réaffectés au cluster le plus proche."
        )

    conclusions.append(
        "Avantage HDBSCAN : détection automatique du nombre de typologies et identification explicite des outliers."
    )
    conclusions.append(
        "Avantage KMeans : pas de bruit, effectifs plus équilibrés, silhouette stable."
    )

    for c in conclusions:
        story.append(Paragraph(f"• {c}", styles["Body2"]))

    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Pistes suivantes", styles["SubSec"]))
    for p in [
        "Calculer le Moran's I (cohérence spatiale) pour les deux algorithmes",
        "Tester HDBSCAN avec réaffectation du bruit pour comparaison équitable",
        "Enrichir avec features DVF/INSEE pour affiner les typologies",
        "Investiguer le cluster 1 KMeans (46k bâtiments non classés)",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    doc.build(story)
    logger.info("Rapport PDF v3 : %s", pdf_path)
    return pdf_path


def main() -> None:
    """Point d'entrée principal."""
    gdf, X_scaled = load_and_scale()

    # 1. Grid search sur échantillon
    grid_results = run_hdbscan_grid(X_scaled)

    # 2. Sélectionner le meilleur (bruit <= 20% de préférence)
    low_noise = [r for r in grid_results if r["noise_pct"] <= 0.20 and r["n_clusters"] >= 2]
    if low_noise:
        best_grid = max(low_noise, key=lambda r: r["silhouette"])
    else:
        valid = [r for r in grid_results if r["n_clusters"] >= 2]
        best_grid = max(valid, key=lambda r: r["silhouette"]) if valid else grid_results[0]

    best_params = best_grid["params"]
    logger.info("Meilleur grid : mcs=%d ms=%d sil=%.4f bruit=%.1f%%",
                best_params["min_cluster_size"], best_params["min_samples"],
                best_grid["silhouette"], best_grid["noise_pct"] * 100)

    # 3. Appliquer sur dataset complet
    hdbscan_labels, full_metrics = apply_best_full(X_scaled, best_grid)

    # 4. Log experiments
    log_experiments(grid_results, full_metrics, best_params)

    # 5. Rapport texte
    report_text = print_comparative_report(grid_results, full_metrics, best_params)
    os.makedirs("outputs/reports", exist_ok=True)
    report_txt_path = "outputs/reports/comparaison_hdbscan_v1.txt"
    with open(report_txt_path, "w") as f:
        f.write(report_text)
    logger.info("Rapport texte : %s", report_txt_path)

    # 6. Figures
    fig_comparison = generate_comparison_figure(gdf, hdbscan_labels, full_metrics)
    fig_zoom = generate_zoom_hdbscan(gdf, hdbscan_labels)
    fig_barplot = generate_hdbscan_barplot(gdf, hdbscan_labels)
    fig_heatmap = generate_grid_heatmap(grid_results)

    # 7. Profil clusters
    gdf_tmp = gdf.copy()
    gdf_tmp["_label"] = hdbscan_labels
    logger.info("\nProfil clusters HDBSCAN (dataset complet) :")
    for cl in sorted(gdf_tmp["_label"].unique()):
        sub = gdf_tmp[gdf_tmp["_label"] == cl]
        if cl == -1:
            logger.info("  Bruit    : n=%d (%.1f%%)", len(sub), len(sub)/len(gdf)*100)
            continue
        usage = "?"
        if "usage_principal" in sub.columns and not sub["usage_principal"].mode().empty:
            usage = sub["usage_principal"].mode().iloc[0]
        logger.info(
            "  Cluster %2d : n=%6d  surf=%6.0fm²  haut=%4.1fm  usage=%s",
            cl, len(sub), sub["surface_bat"].median(), sub["hauteur_mean"].median(), usage,
        )

    # 8. Rapport PDF v3
    generate_pdf_report(
        gdf, hdbscan_labels, full_metrics, best_params, grid_results,
        report_text, fig_comparison, fig_zoom, fig_barplot, fig_heatmap,
    )

    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
