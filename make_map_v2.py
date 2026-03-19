import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.plugins import FastMarkerCluster
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger()

# Charger
logger.info("Chargement...")
gdf = gpd.read_parquet('data/processed/clustered.geoparquet')
logger.info(f"  {len(gdf)} bâtiments")

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]

LABELS = {
    0: "Pavillon moyen (157m²)",
    1: "Petit bâti / annexes (40m²)",
    2: "Individuel dense R+1",
    3: "Pavillon standard (133m²)",
    4: "Tertiaire moyen (1132m²)",
    5: "Collectif moyen (820m²)",
    6: "Grand collectif (3226m²)",
    7: "Collectif haut R+4",
    8: "Individuel R+2 (99m²)",
    9: "Individuel compact (94m²)",
    10: "Grand tertiaire (4794m²)",
    11: "Individuel type (101m²)",
    12: "Tour / IGH (23m)",
    13: "Grande maison (282m²)",
    14: "Tertiaire courant (409m²)",
}

# ============================================================
# CARTE 1 : Statique matplotlib (tous les bâtiments)
# ============================================================
logger.info("Carte statique (tous les bâtiments)...")
fig, ax = plt.subplots(1, 1, figsize=(20, 20))

for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl]
    color = COLORS[cl % len(COLORS)]
    sub.plot(ax=ax, color=color, linewidth=0, alpha=0.7)

# Légende
patches = [mpatches.Patch(color=COLORS[i], label=f"{i}: {LABELS.get(i, '')}")
           for i in sorted(gdf['cluster_label'].unique())]
ax.legend(handles=patches, loc='upper left', fontsize=8, ncol=2,
          framealpha=0.9, title="Clusters", title_fontsize=10)

ax.set_title("Clustering bâtiments — Bordeaux Métropole (KMeans k=15)", fontsize=16)
ax.set_axis_off()
plt.tight_layout()

os.makedirs("outputs/figures", exist_ok=True)
fig.savefig("outputs/figures/clusters_v1.png", dpi=150, bbox_inches='tight')
plt.close()
logger.info("  Sauvegardée : outputs/figures/clusters_v1.png")

# ============================================================
# CARTE 2 : Zoom centre Bordeaux (validation visuelle)
# ============================================================
logger.info("Carte zoom centre Bordeaux...")
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

# Bbox centre Bordeaux (EPSG:2154)
xmin, xmax = 410000, 416000
ymin, ymax = 6424000, 6430000

for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl]
    sub = sub.cx[xmin:xmax, ymin:ymax]
    if len(sub) > 0:
        color = COLORS[cl % len(COLORS)]
        sub.plot(ax=ax, color=color, linewidth=0.2, edgecolor='black', alpha=0.8)

patches = [mpatches.Patch(color=COLORS[i], label=f"{i}: {LABELS.get(i, '')}")
           for i in sorted(gdf['cluster_label'].unique())]
ax.legend(handles=patches, loc='upper left', fontsize=7, ncol=2, framealpha=0.9)
ax.set_title("Zoom Centre Bordeaux — Validation visuelle", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
fig.savefig("outputs/figures/clusters_zoom_bordeaux.png", dpi=150, bbox_inches='tight')
plt.close()
logger.info("  Sauvegardée : outputs/figures/clusters_zoom_bordeaux.png")

# ============================================================
# CARTE 3 : Folium interactive avec centroïdes
# ============================================================
logger.info("Carte Folium interactive (centroïdes)...")
gdf_wgs = gdf.to_crs("EPSG:4326")
gdf_wgs['centroid'] = gdf_wgs.geometry.centroid
gdf_wgs['lat'] = gdf_wgs.centroid.y
gdf_wgs['lon'] = gdf_wgs.centroid.x

center = [44.8378, -0.5792]
m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

# Échantillon pour Folium (30k points)
sample = gdf_wgs.sample(n=min(30000, len(gdf_wgs)), random_state=42)

for cl in sorted(sample['cluster_label'].unique()):
    sub = sample[sample['cluster_label'] == cl]
    color = COLORS[cl % len(COLORS)]
    label = LABELS.get(cl, f"Cluster {cl}")
    
    fg = folium.FeatureGroup(name=f"{cl}: {label}")
    
    for _, row in sub.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=0,
        ).add_to(fg)
    
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

os.makedirs("outputs/maps", exist_ok=True)
m.save("outputs/maps/clusters_v2.html")
logger.info("  Sauvegardée : outputs/maps/clusters_v2.html")

logger.info("TERMINÉ ✓")
