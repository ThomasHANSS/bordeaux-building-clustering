"""Génère la carte web interactive Leaflet avec fonds satellite.

Données : clustering v3 (résidentiel individuel, nb_logements < 4, 150k bâtiments).
Exporte un GeoJSON par cluster (géométries simplifiées 0.5m Lambert-93)
puis crée une page HTML Leaflet Canvas dans outputs/maps/webmap_satellite/.
"""

import geopandas as gpd
import json
import logging
import os
import gc
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

DATA_PATH = "data/processed/clustered_petit_residentiel.geoparquet"
LABEL_COL = "km_label"
OUTPUT_DIR = "outputs/maps/webmap_satellite"
SIMPLIFY_TOLERANCE = 0.5  # mètres en Lambert-93 (0.5m préserve la forme)

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]

# Colonnes pour les popups
POPUP_COLS = [
    "surface_bat", "hauteur_mean", "nb_niveaux",
    "annee_construction", "usage_principal", "mat_murs",
]


def compute_cluster_names(gdf: gpd.GeoDataFrame) -> dict[int, str]:
    """Nomme chaque cluster selon son profil médian."""
    names = {}
    for cl in sorted(gdf[LABEL_COL].unique()):
        sub = gdf[gdf[LABEL_COL] == cl]
        s = sub["surface_bat"].median()
        h = sub["hauteur_mean"].median()
        a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode()
        m = str(mat.iloc[0]) if len(mat) > 0 else ""

        # Période
        if a < 1945:
            per = "ancien"
        elif a < 1975:
            per = "1945-75"
        elif a < 2000:
            per = "fin XXe"
        else:
            per = "récent"

        # Matériau court
        ml = m.lower()
        mat_s = ""
        if "pierre" in ml: mat_s = "pierre"
        elif "béton" in ml or "beton" in ml: mat_s = "béton"
        elif "brique" in ml: mat_s = "brique"
        elif "parpaing" in ml or "agglo" in ml: mat_s = "parpaing"
        elif "bois" in ml: mat_s = "bois"

        # Type
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

        parts = [typ, f"{s:.0f}m²", f"{h:.0f}m", per]
        if mat_s:
            parts.append(mat_s)
        names[cl] = " / ".join(parts)

    return names


def export_geojson_per_cluster(gdf: gpd.GeoDataFrame, names: dict) -> dict:
    """Exporte un GeoJSON par cluster, retourne les métadonnées."""
    # Colonnes utiles seulement
    cols = [c for c in POPUP_COLS if c in gdf.columns] + [LABEL_COL, "geometry"]
    gdf = gdf[cols].copy()

    # Simplifier (0.5m en Lambert-93)
    logger.info("  Simplification (tolérance=%.1fm)...", SIMPLIFY_TOLERANCE)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE)

    # Reprojeter WGS84
    logger.info("  Reprojection EPSG:4326...")
    gdf = gdf.to_crs("EPSG:4326")

    # Nettoyer NaN
    for col in gdf.columns:
        if col == "geometry":
            continue
        if gdf[col].dtype in ("float64", "float32"):
            gdf[col] = gdf[col].where(gdf[col].notna(), None)
        gdf[col] = gdf[col].replace({np.nan: None, "nan": None})

    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
    cluster_meta = {}

    for cl in sorted(gdf[LABEL_COL].unique()):
        sub = gdf[gdf[LABEL_COL] == cl].copy()
        # Drop label col from properties (already known)
        sub = sub.drop(columns=[LABEL_COL])
        n = len(sub)
        name = names.get(cl, f"Cluster {cl}")
        color = COLORS[cl % len(COLORS)]

        fname = f"cluster_{cl}.geojson"
        fpath = os.path.join(OUTPUT_DIR, "data", fname)

        geojson = json.loads(sub.to_json())

        # Arrondir les coordonnées à 6 décimales
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
        logger.info("  Cluster %2d : %6d bât → %s (%.1f MB)", cl, n, fname, fsize)

        cluster_meta[cl] = {"name": name, "color": color, "count": n, "file": f"data/{fname}"}

    return cluster_meta


def generate_html(cluster_meta: dict, n_total: int) -> str:
    """Génère le fichier HTML Leaflet avec renderer Canvas."""

    clusters_js = json.dumps(
        {str(k): v for k, v in sorted(cluster_meta.items())},
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clustering résidentiel individuel — Bordeaux Métropole</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
#map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: 1; }}

#panel {{
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: white; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    width: 330px; max-height: calc(100vh - 20px); overflow-y: auto;
    font-size: 13px;
}}
#panel-header {{
    position: sticky; top: 0; z-index: 10;
    padding: 12px 16px; background: #1a1a2e; color: white;
    border-radius: 8px 8px 0 0; font-weight: bold; font-size: 14px;
    cursor: pointer; user-select: none; display: flex; justify-content: space-between;
}}
#panel-header:hover {{ background: #16213e; }}
#panel-body {{ padding: 0; }}
.section {{ padding: 10px 14px; border-bottom: 1px solid #eee; }}
.section-title {{ font-weight: 600; font-size: 11px; color: #888; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px; }}

.basemap-option {{ display: flex; align-items: center; padding: 3px 0; cursor: pointer; }}
.basemap-option input {{ margin-right: 8px; cursor: pointer; }}
.basemap-option label {{ cursor: pointer; font-size: 12px; }}

.cluster-option {{ display: flex; align-items: center; padding: 3px 0; cursor: pointer; }}
.cluster-option input {{ margin-right: 6px; flex-shrink: 0; cursor: pointer; }}
.cluster-swatch {{
    width: 14px; height: 14px; border-radius: 3px; margin-right: 6px;
    border: 1px solid #333; flex-shrink: 0;
}}
.cluster-label {{ font-size: 11px; line-height: 1.3; }}
.cluster-count {{ color: #999; font-size: 10px; }}

.btn-row {{ display: flex; gap: 6px; margin-top: 8px; }}
.btn {{ padding: 4px 10px; border: 1px solid #ccc; border-radius: 4px;
    background: #f8f8f8; cursor: pointer; font-size: 11px; }}
.btn:hover {{ background: #e8e8e8; }}

#loading {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    z-index: 2000; background: rgba(0,0,0,0.85); color: white; padding: 20px 30px;
    border-radius: 10px; font-size: 16px; text-align: center;
}}
#progress {{ color: #42d4f4; }}
#load-bar {{ width: 200px; height: 6px; background: #333; border-radius: 3px; margin-top: 10px; }}
#load-fill {{ height: 100%; width: 0%; background: #42d4f4; border-radius: 3px; transition: width 0.3s; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="loading">
    Chargement... <span id="progress">0/15</span>
    <div id="load-bar"><div id="load-fill"></div></div>
</div>

<div id="panel">
    <div id="panel-header">
        <span>Résidentiel individuel</span>
        <span id="toggle-icon">&#9660;</span>
    </div>
    <div id="panel-body">
        <div class="section">
            <div class="section-title">Fond de carte</div>
            <div id="basemap-controls"></div>
        </div>
        <div class="section">
            <div class="section-title">Clusters ({n_total:,} bâtiments)</div>
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

// ── Map with Canvas renderer for performance ──
const canvasRenderer = L.canvas({{ padding: 0.5, tolerance: 5 }});
const map = L.map('map', {{
    center: [44.8378, -0.5792],
    zoom: 12,
    zoomControl: true,
    preferCanvas: true,
    renderer: canvasRenderer
}});

// ── Basemaps ──
const basemaps = {{
    'Google Satellite': L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}', {{
        maxZoom: 21, attribution: '&copy; Google'
    }}),
    'Google Hybrid': L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}', {{
        maxZoom: 21, attribution: '&copy; Google'
    }}),
    'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19, attribution: '&copy; OpenStreetMap'
    }}),
    'CartoDB Positron': L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png', {{
        maxZoom: 20, attribution: '&copy; CartoDB', subdomains: 'abcd'
    }}),
    'Aucun (blanc)': null
}};
let currentBasemap = basemaps['Google Hybrid'];
currentBasemap.addTo(map);

// ── Basemap controls ──
const bmContainer = document.getElementById('basemap-controls');
const bmNames = Object.keys(basemaps);
bmNames.forEach((name, i) => {{
    const div = document.createElement('div');
    div.className = 'basemap-option';
    const checked = name === 'Google Hybrid' ? 'checked' : '';
    div.innerHTML = `<input type="radio" name="basemap" id="bm_${{i}}" value="${{name}}" ${{checked}}>
        <label for="bm_${{i}}">${{name}}</label>`;
    div.querySelector('input').addEventListener('change', () => {{
        if (currentBasemap) map.removeLayer(currentBasemap);
        currentBasemap = basemaps[name];
        if (currentBasemap) currentBasemap.addTo(map);
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
        ['Surface', props.surface_bat != null ? Number(props.surface_bat).toFixed(0) + ' m²' : '—'],
        ['Hauteur', props.hauteur_mean != null ? Number(props.hauteur_mean).toFixed(1) + ' m' : '—'],
        ['Niveaux', props.nb_niveaux != null ? Math.round(props.nb_niveaux) : '—'],
        ['Année', props.annee_construction != null && props.annee_construction > 0 ? Math.round(props.annee_construction) : '—'],
        ['Usage', props.usage_principal || '—'],
        ['Matériau murs', props.mat_murs || '—'],
    ];
    rows.forEach(([k, v]) => {{
        html += `<tr><td style="padding:2px 6px;color:#666;white-space:nowrap;">${{k}}</td>
                     <td style="padding:2px 6px;font-weight:500;">${{v}}</td></tr>`;
    }});
    html += '</table></div>';
    return html;
}}

// ── Load clusters ──
let loaded = 0;
const total = Object.keys(CLUSTERS).length;
const progressEl = document.getElementById('progress');
const loadingEl = document.getElementById('loading');
const loadFill = document.getElementById('load-fill');

async function loadCluster(clId) {{
    const meta = CLUSTERS[clId];
    try {{
        const resp = await fetch(meta.file);
        const geojson = await resp.json();

        const layer = L.geoJSON(geojson, {{
            renderer: canvasRenderer,
            style: () => ({{
                fillColor: meta.color,
                color: '#000000',
                weight: 0.8,
                fillOpacity: 0.6,
            }}),
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
    loadFill.style.width = `${{(loaded / total * 100).toFixed(0)}}%`;
    if (loaded >= total) {{
        setTimeout(() => {{ loadingEl.style.display = 'none'; }}, 400);
    }}
}}

async function loadAll() {{
    const ids = Object.keys(CLUSTERS).sort((a, b) => a - b);
    for (let i = 0; i < ids.length; i += 2) {{
        const batch = ids.slice(i, i + 2);
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
            <span class="cluster-count">${{meta.count.toLocaleString('fr-FR')}} bât.</span>
        </div>`;
    div.querySelector('input').addEventListener('change', (e) => {{
        if (clusterLayers[clId]) {{
            if (e.target.checked) map.addLayer(clusterLayers[clId]);
            else map.removeLayer(clusterLayers[clId]);
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
    const hidden = panelBody.style.display === 'none';
    panelBody.style.display = hidden ? 'block' : 'none';
    toggleIcon.innerHTML = hidden ? '&#9660;' : '&#9654;';
}});

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
    """Point d'entrée."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Chargement %s ...", DATA_PATH)
    gdf = gpd.read_parquet(DATA_PATH)
    n_total = len(gdf)
    logger.info("  %d bâtiments, CRS=%s", n_total, gdf.crs.to_epsg())

    # Noms dynamiques
    names = compute_cluster_names(gdf)
    for cl, name in sorted(names.items()):
        logger.info("  Cluster %2d : %s (%d)", cl, name, (gdf[LABEL_COL] == cl).sum())

    # Export GeoJSON
    cluster_meta = export_geojson_per_cluster(gdf, names)
    del gdf
    gc.collect()

    # HTML
    generate_html(cluster_meta, n_total)

    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(OUTPUT_DIR)
        for f in files
    )
    logger.info("Dossier : %s (%.1f MB)", OUTPUT_DIR, total_size / 1e6)
    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
