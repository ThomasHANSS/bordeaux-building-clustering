"""Cartes HTML interactives par zone avec vrais polygones.

Génère une carte Folium par zone avec GeoJson polygones colorés par cluster,
popup informatif et légende nommée. Géométries simplifiées (1m Lambert-93).
"""

import geopandas as gpd
import folium
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

# Palette colorblind-safe 15 clusters
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]

LABELS = {
    0: "Pavillon moyen (157m², 4m)",
    1: "Petit bâti / annexes (40m²)",
    2: "Individuel dense R+1 (101m²)",
    3: "Pavillon standard (133m², 4m)",
    4: "Tertiaire moyen (1132m²)",
    5: "Collectif moyen (820m², 11m)",
    6: "Grand collectif (3226m², 13m)",
    7: "Collectif haut R+4 (135m², 15m)",
    8: "Individuel R+2 (99m², 9m)",
    9: "Individuel compact (94m², 5m)",
    10: "Grand équipement (4794m²)",
    11: "Individuel type (101m², 5m)",
    12: "Tour / IGH (704m², 23m)",
    13: "Grande maison (282m², 4m)",
    14: "Tertiaire courant (409m², 7m)",
}

# Zones en EPSG:2154 (Lambert-93)
ZONES = {
    "centre_bordeaux": {
        "name": "Centre Bordeaux",
        "bbox": (410000, 6424000, 416000, 6430000),
        "zoom": 14,
    },
    "meriadeck": {
        "name": "Mériadeck",
        "bbox": (411000, 6425500, 413500, 6428000),
        "zoom": 15,
    },
    "cauderan_bouscat": {
        "name": "Caudéran / Le Bouscat",
        "bbox": (408500, 6427000, 411500, 6430000),
        "zoom": 14,
    },
    "bassins_a_flot": {
        "name": "Bassins à flot",
        "bbox": (412000, 6428500, 414000, 6430500),
        "zoom": 15,
    },
}


def make_legend_html(clusters_present: list[int]) -> str:
    """Génère le HTML de la légende pour les clusters présents."""
    items = ""
    for cl in sorted(clusters_present):
        color = COLORS[cl % len(COLORS)]
        label = LABELS.get(cl, f"Cluster {cl}")
        items += (
            f'<li><span style="background:{color};width:14px;height:14px;'
            f'display:inline-block;margin-right:6px;border:1px solid #333;'
            f'vertical-align:middle;"></span>{cl}: {label}</li>\n'
        )
    return f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3);
                max-height:60vh;overflow-y:auto;font-size:12px;">
        <b style="font-size:13px;">Clusters</b>
        <ul style="list-style:none;padding:4px 0;margin:0;">{items}</ul>
    </div>
    """


def create_zone_map(
    gdf_2154: gpd.GeoDataFrame,
    zone_key: str,
    output_dir: str = "outputs/maps",
) -> str:
    """Crée une carte Folium pour une zone avec polygones GeoJson.

    Parameters
    ----------
    gdf_2154 : gpd.GeoDataFrame
        Données en EPSG:2154 avec cluster_label et geometry.
    zone_key : str
        Clé de la zone dans ZONES.
    output_dir : str
        Dossier de sortie.

    Returns
    -------
    str
        Chemin du fichier HTML sauvegardé.
    """
    zone = ZONES[zone_key]
    xmin, ymin, xmax, ymax = zone["bbox"]

    # Filtrage spatial en Lambert-93
    gdf_zone = gdf_2154.cx[xmin:xmax, ymin:ymax].copy()
    n_buildings = len(gdf_zone)
    logger.info(f"  Zone '{zone['name']}' : {n_buildings} bâtiments")

    if n_buildings == 0:
        logger.warning(f"  Aucun bâtiment dans la zone {zone_key}, carte ignorée.")
        return ""

    # Simplifier les géométries (1m en Lambert-93)
    gdf_zone["geometry"] = gdf_zone.geometry.simplify(1.0)

    # Reprojeter en WGS84 pour Folium
    gdf_wgs = gdf_zone.to_crs("EPSG:4326")

    # Centre de la carte
    bounds = gdf_wgs.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=zone["zoom"], tiles="CartoDB positron")

    # Colonnes utiles pour les popups
    col_surface = "s_geom_groupe"
    col_hauteur = "bdtopo_bat_hauteur_mean"
    col_annee = "ffo_bat_annee_construction"
    col_usage = "usage_principal_bdnb_open"
    col_niveaux = "ffo_bat_nb_niveau"

    # Un FeatureGroup par cluster
    clusters_present = sorted(gdf_wgs["cluster_label"].unique())

    for cl in clusters_present:
        sub = gdf_wgs[gdf_wgs["cluster_label"] == cl]
        color = COLORS[cl % len(COLORS)]
        label = LABELS.get(cl, f"Cluster {cl}")
        fg = folium.FeatureGroup(name=f"{cl}: {label}")

        # Construire le GeoJSON avec propriétés pour popup
        features = []
        for _, row in sub.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            surface = row.get(col_surface, None)
            hauteur = row.get(col_hauteur, None)
            annee = row.get(col_annee, None)
            usage = row.get(col_usage, None)
            niveaux = row.get(col_niveaux, None)

            popup_lines = [f"<b>Cluster {cl}</b> — {label}"]
            if surface is not None and surface == surface:
                popup_lines.append(f"Surface : {surface:.0f} m²")
            if hauteur is not None and hauteur == hauteur:
                popup_lines.append(f"Hauteur : {hauteur:.1f} m")
            if niveaux is not None and niveaux == niveaux:
                popup_lines.append(f"Niveaux : {int(niveaux)}")
            if annee is not None and annee == annee and annee > 0:
                popup_lines.append(f"Année : {int(annee)}")
            if usage is not None and usage == usage and str(usage) != "nan":
                popup_lines.append(f"Usage : {usage}")

            popup_html = "<br>".join(popup_lines)

            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "color": c,
                    "weight": 0.5,
                    "fillOpacity": 0.7,
                },
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg)

        fg.add_to(m)

    # Légende HTML fixe
    legend_html = make_legend_html(clusters_present)
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    filename = f"clusters_v3_{zone_key}.html"
    filepath = os.path.join(output_dir, filename)
    m.save(filepath)
    logger.info(f"  Sauvegardée : {filepath} ({n_buildings} polygones)")

    return filepath


def main() -> None:
    """Point d'entrée : charge les données et génère les 4 cartes par zone."""
    logger.info("Chargement des données clustered...")
    gdf = gpd.read_parquet("data/processed/clustered.geoparquet")
    logger.info(f"  {len(gdf)} bâtiments, CRS={gdf.crs}")

    # S'assurer qu'on est en Lambert-93
    if gdf.crs is None or gdf.crs.to_epsg() != 2154:
        logger.info("  Reprojection en EPSG:2154...")
        gdf = gdf.to_crs("EPSG:2154")

    logger.info("Génération des cartes par zone...")
    generated = []
    for zone_key in ZONES:
        path = create_zone_map(gdf, zone_key)
        if path:
            generated.append(path)

    logger.info(f"Terminé : {len(generated)} cartes générées.")
    for p in generated:
        logger.info(f"  - {p}")


if __name__ == "__main__":
    main()
