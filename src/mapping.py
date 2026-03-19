"""Cartographie : cartes interactives Folium et figures statiques.

Produit les cartes de visualisation des clusters.
"""

import logging
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from src.config import PROJECT_ROOT, load_config, setup_project

logger = logging.getLogger(__name__)

# Palette colorblind-safe pour 15 clusters + bruit
CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]
NOISE_COLOR = "#808080"


def create_folium_map(
    gdf: gpd.GeoDataFrame,
    label_col: str = "cluster_label",
    config: dict | None = None,
) -> folium.Map:
    """Crée une carte Folium interactive des clusters.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Données avec géométries et labels.
    label_col : str
        Colonne des labels.
    config : dict, optional
        Configuration du projet.

    Returns
    -------
    folium.Map
        Carte interactive.
    """
    if config is None:
        config = load_config()

    # Reprojeter en WGS84 pour Folium
    crs_display = config["project"]["crs_display"]
    gdf_wgs = gdf.to_crs(crs_display)

    # Centre de la carte
    bounds = gdf_wgs.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # Un FeatureGroup par cluster
    unique_labels = sorted(gdf_wgs[label_col].unique())

    for label in unique_labels:
        subset = gdf_wgs[gdf_wgs[label_col] == label]

        if label == -1:
            color = NOISE_COLOR
            name = "Bruit (non classé)"
            opacity = 0.3
        else:
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            name = f"Cluster {label}"
            opacity = 0.7

        fg = folium.FeatureGroup(name=name, show=(label != -1))

        for _, row in subset.iterrows():
            # Simplifier pour la performance de la carte
            geom = row.geometry.simplify(0.0001)

            popup_html = f"""
            <b>Cluster {label}</b><br>
            UID: {row.get('uid', 'N/A')}<br>
            """

            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda x, c=color, o=opacity: {
                    "fillColor": c,
                    "color": c,
                    "weight": 0.5,
                    "fillOpacity": o,
                },
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(fg)

        fg.add_to(m)
        logger.info("Cluster %s : %d entités ajoutées", name, len(subset))

    folium.LayerControl(collapsed=False).add_to(m)

    return m


def create_static_map(
    gdf: gpd.GeoDataFrame,
    label_col: str = "cluster_label",
    title: str = "Clustering — Bordeaux Métropole",
) -> plt.Figure:
    """Crée une carte statique matplotlib des clusters.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Données avec géométries et labels.
    label_col : str
        Colonne des labels.
    title : str
        Titre de la carte.

    Returns
    -------
    plt.Figure
        Figure matplotlib.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))

    # Bruit d'abord (en fond)
    noise = gdf[gdf[label_col] == -1]
    if len(noise) > 0:
        noise.plot(ax=ax, color=NOISE_COLOR, alpha=0.2, linewidth=0)

    # Clusters
    clusters = gdf[gdf[label_col] != -1]
    unique_labels = sorted(clusters[label_col].unique())
    colors = {l: CLUSTER_COLORS[l % len(CLUSTER_COLORS)] for l in unique_labels}

    for label in unique_labels:
        subset = clusters[clusters[label_col] == label]
        subset.plot(ax=ax, color=colors[label], alpha=0.7, linewidth=0, label=f"Cluster {label}")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_axis_off()
    plt.tight_layout()

    return fig


def generate_maps(config: dict | None = None) -> None:
    """Génère toutes les cartes (Folium + statique).

    Parameters
    ----------
    config : dict, optional
        Configuration du projet.
    """
    if config is None:
        config = load_config()

    results_path = PROJECT_ROOT / config["pipeline"]["results_file"]
    if not results_path.exists():
        logger.error("Fichier résultats non trouvé : %s", results_path)
        return

    gdf = gpd.read_parquet(results_path)
    logger.info("Chargement : %d entités, %d clusters",
                len(gdf), gdf["cluster_label"].nunique())

    # Carte Folium
    logger.info("Génération carte Folium...")
    m = create_folium_map(gdf, config=config)
    folium_path = PROJECT_ROOT / "outputs" / "maps" / "clusters_interactive.html"
    folium_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(folium_path))
    logger.info("Carte interactive sauvegardée : %s", folium_path)

    # Carte statique
    logger.info("Génération carte statique...")
    fig = create_static_map(gdf)
    static_path = PROJECT_ROOT / "outputs" / "figures" / "clusters_static.png"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(static_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Carte statique sauvegardée : %s", static_path)


if __name__ == "__main__":
    cfg = setup_project()
    generate_maps(cfg)
