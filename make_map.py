import geopandas as gpd
import folium
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger()

# Charger les résultats
logger.info("Chargement des résultats...")
gdf = gpd.read_parquet('data/processed/clustered.geoparquet')
gdf = gdf.to_crs("EPSG:4326")
logger.info(f"  {len(gdf)} bâtiments, {gdf['cluster_label'].nunique()} clusters")

# Palette
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]

LABELS = {
    0: "Pavillon moyen (157m²)",
    1: "Petit bâti / annexes (40m²)",
    2: "Individuel dense R+1 (101m²)",
    3: "Pavillon standard (133m²)",
    4: "Tertiaire moyen (1132m²)",
    5: "Collectif moyen (820m²)",
    6: "Grand collectif (3226m²)",
    7: "Collectif haut R+4 (135m²)",
    8: "Individuel R+2 (99m²)",
    9: "Individuel compact (94m²)",
    10: "Grand tertiaire (4794m²)",
    11: "Individuel type (101m²)",
    12: "Tour / IGH (704m², 23m)",
    13: "Grande maison (282m²)",
    14: "Tertiaire courant (409m²)",
}

# Centre sur Bordeaux
center = [44.8378, -0.5792]
m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

# Échantillonner pour la performance (carte trop lourde avec 234k polygones)
logger.info("Échantillonnage pour la carte (max 20k bâtiments)...")
sample = gdf.sample(n=min(20000, len(gdf)), random_state=42)

for cl in sorted(sample['cluster_label'].unique()):
    sub = sample[sample['cluster_label'] == cl]
    label = LABELS.get(cl, f"Cluster {cl}")
    color = COLORS[cl % len(COLORS)]
    
    fg = folium.FeatureGroup(name=f"{cl}: {label}", show=True)
    
    for _, row in sub.iterrows():
        geom = row.geometry.simplify(0.0001)
        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda x, c=color: {
                "fillColor": c, "color": c,
                "weight": 0.3, "fillOpacity": 0.6,
            },
        ).add_to(fg)
    
    fg.add_to(m)
    logger.info(f"  Cluster {cl:>2} ({label}): {len(sub)} bâtiments")

folium.LayerControl(collapsed=False).add_to(m)

os.makedirs("outputs/maps", exist_ok=True)
m.save("outputs/maps/clusters_v1.html")
logger.info("Carte sauvegardée : outputs/maps/clusters_v1.html")
logger.info("TERMINÉ ✓")
