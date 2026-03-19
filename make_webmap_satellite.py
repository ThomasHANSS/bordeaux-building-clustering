"""Génère la carte web interactive Leaflet avec fonds satellite.

Exporte un GeoJSON par cluster (géométries simplifiées 2m Lambert-93)
puis crée une page HTML Leaflet autonome dans outputs/maps/webmap_satellite/.
"""

import geopandas as gpd
import json
import logging
import os
import gc
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

DATA_PATH = "data/processed/clustered.geoparquet"
OUTPUT_DIR = "outputs/maps/webmap_satellite"
SIMPLIFY_TOLERANCE = 2.0  # mètres en Lambert-93

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]

CLUSTER_NAMES = {
    0: "Pavillon plain-pied (157m², 4m)",
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

# Colonnes à conserver pour les popups
POPUP_COLS = [
    "cluster_label", "surface_bat", "hauteur_mean", "nb_niveaux",
    "annee_construction", "usage_principal", "mat_murs",
    "libelle_adr_principale_ban", "geometry",
]


def export_geojson_per_cluster() -> dict:
    """Exporte un GeoJSON par cluster, retourne les métadonnées."""
    logger.info("Chargement %s ...", DATA_PATH)
    gdf = gpd.read_parquet(DATA_PATH)
    logger.info("  %d bâtiments chargés", len(gdf))

    # Ne garder que les colonnes utiles
    cols = [c for c in POPUP_COLS if c in gdf.columns]
    gdf = gdf[cols].copy()
    gc.collect()

    # Simplifier les géométries (2m en Lambert-93)
    logger.info("  Simplification des géométries (tolérance=%gm)...", SIMPLIFY_TOLERANCE)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE)

    # Reprojeter en WGS84
    logger.info("  Reprojection EPSG:4326...")
    gdf = gdf.to_crs("EPSG:4326")

    # Préparer les propriétés pour le GeoJSON (nettoyer NaN)
    for col in gdf.columns:
        if col == "geometry":
            continue
        if gdf[col].dtype in ("float64", "float32"):
            gdf[col] = gdf[col].where(gdf[col].notna(), None)
        gdf[col] = gdf[col].replace({np.nan: None, "nan": None})

    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
    cluster_meta = {}

    for cl in sorted(gdf["cluster_label"].unique()):
        sub = gdf[gdf["cluster_label"] == cl].copy()
        n = len(sub)
        name = CLUSTER_NAMES.get(cl, f"Cluster {cl}")
        color = COLORS[cl % len(COLORS)]

        # Convertir en GeoJSON
        fname = f"cluster_{cl}.geojson"
        fpath = os.path.join(OUTPUT_DIR, "data", fname)

        # Écrire avec json pour un meilleur contrôle de la taille
        geojson = json.loads(sub.to_json())

        # Arrondir les coordonnées à 6 décimales (~11cm) pour réduire la taille
        def round_coords(coords):
            if isinstance(coords[0], (list, tuple)):
                return [round_coords(c) for c in coords]
            return [round(coords[0], 6), round(coords[1], 6)]

        for feature in geojson["features"]:
            geom = feature["geometry"]
            if geom and "coordinates" in geom:
                geom["coordinates"] = round_coords(geom["coordinates"])

        with open(fpath, "w") as f:
            json.dump(geojson, f, separators=(",", ":"))

        fsize = os.path.getsize(fpath) / 1e6
        logger.info("  Cluster %2d : %6d bâtiments → %s (%.1f MB)", cl, n, fname, fsize)

        cluster_meta[cl] = {"name": name, "color": color, "count": n, "file": f"data/{fname}"}

    return cluster_meta


def generate_html(cluster_meta: dict) -> str:
    """Génère le fichier HTML Leaflet."""

    # Construire le JS pour les métadonnées des clusters
    clusters_js = json.dumps(
        {str(k): v for k, v in sorted(cluster_meta.items())},
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clustering bâtiments — Bordeaux Métropole</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
#map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: 1; }}

#panel {{
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: white; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    width: 320px; max-height: calc(100vh - 20px); overflow-y: auto;
    font-size: 13px;
}}
#panel-header {{
    padding: 12px 16px; background: #1a1a2e; color: white;
    border-radius: 8px 8px 0 0; font-weight: bold; font-size: 15px;
    cursor: pointer; user-select: none; display: flex; justify-content: space-between;
}}
#panel-header:hover {{ background: #16213e; }}
#panel-body {{ padding: 0; }}
.section {{ padding: 10px 14px; border-bottom: 1px solid #eee; }}
.section-title {{ font-weight: 600; font-size: 12px; color: #666; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px; }}

.basemap-option {{ display: flex; align-items: center; padding: 3px 0; cursor: pointer; }}
.basemap-option input {{ margin-right: 8px; }}
.basemap-option label {{ cursor: pointer; }}

.cluster-option {{ display: flex; align-items: center; padding: 2px 0; cursor: pointer; }}
.cluster-option input {{ margin-right: 6px; flex-shrink: 0; }}
.cluster-swatch {{
    width: 14px; height: 14px; border-radius: 3px; margin-right: 6px;
    border: 1px solid #333; flex-shrink: 0;
}}
.cluster-label {{ font-size: 12px; line-height: 1.3; }}
.cluster-count {{ color: #999; font-size: 11px; }}

.btn-row {{ display: flex; gap: 6px; margin-top: 8px; }}
.btn {{ padding: 4px 10px; border: 1px solid #ccc; border-radius: 4px;
    background: #f8f8f8; cursor: pointer; font-size: 11px; }}
.btn:hover {{ background: #e8e8e8; }}

#loading {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    z-index: 2000; background: rgba(0,0,0,0.8); color: white; padding: 20px 30px;
    border-radius: 8px; font-size: 16px;
}}
#progress {{ color: #42d4f4; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="loading">Chargement... <span id="progress">0/15</span></div>

<div id="panel">
    <div id="panel-header">
        <span>Clusters — Bordeaux Métropole</span>
        <span id="toggle-icon">&#9660;</span>
    </div>
    <div id="panel-body">
        <div class="section">
            <div class="section-title">Fond de carte</div>
            <div id="basemap-controls"></div>
        </div>
        <div class="section">
            <div class="section-title">Clusters (234 116 bâtiments)</div>
            <div class="btn-row">
                <button class="btn" onclick="toggleAll(true)">Tout afficher</button>
                <button class="btn" onclick="toggleAll(false)">Tout masquer</button>
            </div>
            <div id="cluster-controls" style="margin-top:8px;"></div>
        </div>
    </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const CLUSTERS = {clusters_js};

// ── Map ──
const map = L.map('map', {{ center: [44.8378, -0.5792], zoom: 12, zoomControl: true }});

// ── Basemaps ──
const basemaps = {{
    'Google Satellite': L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}', {{
        maxZoom: 21, attribution: '&copy; Google'
    }}),
    'Google Hybrid': L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}', {{
        maxZoom: 21, attribution: '&copy; Google'
    }}),
    'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
    }}),
    'CartoDB Positron': L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png', {{
        maxZoom: 20, attribution: '&copy; CartoDB', subdomains: 'abcd'
    }})
}};
let currentBasemap = basemaps['Google Hybrid'];
currentBasemap.addTo(map);

// ── Basemap controls ──
const bmContainer = document.getElementById('basemap-controls');
Object.keys(basemaps).forEach((name, i) => {{
    const div = document.createElement('div');
    div.className = 'basemap-option';
    const checked = name === 'Google Hybrid' ? 'checked' : '';
    div.innerHTML = `<input type="radio" name="basemap" id="bm_${{i}}" value="${{name}}" ${{checked}}>
        <label for="bm_${{i}}">${{name}}</label>`;
    div.querySelector('input').addEventListener('change', () => {{
        map.removeLayer(currentBasemap);
        currentBasemap = basemaps[name];
        currentBasemap.addTo(map);
    }});
    bmContainer.appendChild(div);
}});

// ── Cluster layers ──
const clusterLayers = {{}};

function formatPopup(props, clId) {{
    const meta = CLUSTERS[clId];
    let html = `<div style="font-family:sans-serif;font-size:13px;min-width:200px;">`;
    html += `<div style="background:${{meta.color}};color:white;padding:6px 10px;margin:-1px -1px 8px;border-radius:4px 4px 0 0;font-weight:bold;">`;
    html += `Cluster ${{clId}} — ${{meta.name}}</div>`;
    html += `<table style="width:100%;border-collapse:collapse;">`;

    const rows = [
        ['Surface', props.surface_bat != null ? props.surface_bat.toFixed(0) + ' m²' : '—'],
        ['Hauteur', props.hauteur_mean != null ? props.hauteur_mean.toFixed(1) + ' m' : '—'],
        ['Niveaux', props.nb_niveaux != null ? Math.round(props.nb_niveaux) : '—'],
        ['Année', props.annee_construction != null && props.annee_construction > 0 ? Math.round(props.annee_construction) : '—'],
        ['Usage', props.usage_principal || '—'],
        ['Matériau murs', props.mat_murs || '—'],
    ];
    if (props.libelle_adr_principale_ban) {{
        rows.unshift(['Adresse', props.libelle_adr_principale_ban]);
    }}
    rows.forEach(([k, v]) => {{
        html += `<tr><td style="padding:2px 6px;color:#666;white-space:nowrap;">${{k}}</td>
                     <td style="padding:2px 6px;font-weight:500;">${{v}}</td></tr>`;
    }});
    html += '</table></div>';
    return html;
}}

function styleCluster(color) {{
    return {{
        fillColor: color,
        color: '#000000',
        weight: 0.8,
        fillOpacity: 0.6,
    }};
}}

// ── Load clusters ──
let loaded = 0;
const total = Object.keys(CLUSTERS).length;
const progressEl = document.getElementById('progress');
const loadingEl = document.getElementById('loading');

async function loadCluster(clId) {{
    const meta = CLUSTERS[clId];
    try {{
        const resp = await fetch(meta.file);
        const geojson = await resp.json();

        const layer = L.geoJSON(geojson, {{
            style: () => styleCluster(meta.color),
            onEachFeature: (feature, layer) => {{
                layer.bindPopup(() => formatPopup(feature.properties, clId), {{ maxWidth: 300 }});
            }}
        }});

        clusterLayers[clId] = layer;
        layer.addTo(map);
    }} catch (e) {{
        console.error(`Erreur cluster ${{clId}}:`, e);
    }}

    loaded++;
    progressEl.textContent = `${{loaded}}/${{total}}`;
    if (loaded >= total) {{
        loadingEl.style.display = 'none';
    }}
}}

// Load clusters sequentially to avoid memory spikes
async function loadAll() {{
    const ids = Object.keys(CLUSTERS).sort((a, b) => a - b);
    // Load 3 at a time
    for (let i = 0; i < ids.length; i += 3) {{
        const batch = ids.slice(i, i + 3);
        await Promise.all(batch.map(id => loadCluster(id)));
    }}
}}

// ── Cluster controls ──
const ccContainer = document.getElementById('cluster-controls');
Object.keys(CLUSTERS).sort((a, b) => a - b).forEach(clId => {{
    const meta = CLUSTERS[clId];
    const div = document.createElement('div');
    div.className = 'cluster-option';
    div.innerHTML = `
        <input type="checkbox" id="cl_${{clId}}" checked>
        <div class="cluster-swatch" style="background:${{meta.color}};"></div>
        <div>
            <span class="cluster-label">${{clId}}: ${{meta.name}}</span><br>
            <span class="cluster-count">${{meta.count.toLocaleString('fr-FR')}} bâtiments</span>
        </div>`;
    div.querySelector('input').addEventListener('change', (e) => {{
        if (clusterLayers[clId]) {{
            if (e.target.checked) {{
                map.addLayer(clusterLayers[clId]);
            }} else {{
                map.removeLayer(clusterLayers[clId]);
            }}
        }}
    }});
    ccContainer.appendChild(div);
}});

function toggleAll(show) {{
    Object.keys(CLUSTERS).forEach(clId => {{
        const cb = document.getElementById('cl_' + clId);
        if (cb) cb.checked = show;
        if (clusterLayers[clId]) {{
            if (show) map.addLayer(clusterLayers[clId]);
            else map.removeLayer(clusterLayers[clId]);
        }}
    }});
}}

// ── Panel toggle ──
const panelBody = document.getElementById('panel-body');
const toggleIcon = document.getElementById('toggle-icon');
document.getElementById('panel-header').addEventListener('click', () => {{
    if (panelBody.style.display === 'none') {{
        panelBody.style.display = 'block';
        toggleIcon.innerHTML = '&#9660;';
    }} else {{
        panelBody.style.display = 'none';
        toggleIcon.innerHTML = '&#9654;';
    }}
}});

// Start loading
loadAll();
</script>
</body>
</html>"""

    html_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)

    logger.info("HTML : %s", html_path)
    return html_path


def main() -> None:
    """Point d'entrée : export GeoJSON + génération HTML."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cluster_meta = export_geojson_per_cluster()
    html_path = generate_html(cluster_meta)

    total_size = 0
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    logger.info("Dossier : %s (%.1f MB total)", OUTPUT_DIR, total_size / 1e6)
    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
