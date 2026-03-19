"""Associe les clusters aux parcelles cadastrales pour cartographie parcellaire.

Étapes :
1. Charge le parcellaire BDNB (307k parcelles métropole)
2. Joint spatialement parcelles ↔ bâtiments clusterisés (V3 + V4)
3. Calcule des indicateurs parcellaires (CES, nb bâtiments, etc.)
4. Sauvegarde en GeoParquet
5. Génère une carte web interactive avec parcelles colorées par cluster

Usage :
    python make_parcellaire.py
"""

import gc
import json
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.validation import make_valid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
GPKG_PATH = "data/raw/bdnb_33/gpkg/bdnb.gpkg"
OUTPUT_DIR = "outputs/maps/webmap_parcelles"
SIMPLIFY_TOLERANCE = 1.0  # mètres (Lambert-93)

VERSIONS = {
    "v3": {
        "title": "Clustering v3 — Résidentiel individuel (parcelles)",
        "subtitle": "150k bâtiments → parcelles, KMeans k=15",
        "data_path": "data/processed/clustered_petit_residentiel.geoparquet",
        "label_col": "km_label",
        "pdf": "reports/clustering_bordeaux_v3.pdf",
    },
    "v4": {
        "title": "Clustering v4 — Bâti récent (parcelles)",
        "subtitle": "7k bâtiments (≥2016) → parcelles, KMeans k=10",
        "data_path": "data/processed/clustered_recent.geoparquet",
        "label_col": "v4_label",
        "pdf": "reports/clustering_bordeaux_v4.pdf",
    },
}

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#000075", "#a9a9a9", "#ffd8b1", "#fffac8", "#e6beff",
]


def load_config() -> dict:
    """Charge la configuration YAML."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_parcelles(config: dict) -> gpd.GeoDataFrame:
    """Charge les parcelles BDNB pour les communes de la métropole.

    Parameters
    ----------
    config : dict
        Configuration du projet.

    Returns
    -------
    gpd.GeoDataFrame
        Parcelles avec geometry, parcelle_id, code_commune_insee, s_geom_parcelle.
    """
    codes = config["zone"]["codes_insee"]
    codes_sql = "','".join(codes)
    where = f"code_commune_insee IN ('{codes_sql}')"

    logger.info("Chargement des parcelles (filtre : %d communes)...", len(codes))
    gdf = gpd.read_file(
        GPKG_PATH,
        layer="parcelle",
        where=where,
    )
    logger.info("  %d parcelles chargées", len(gdf))

    # Renommer la colonne géométrie si nécessaire
    geom_col = gdf.geometry.name
    if geom_col != "geometry":
        gdf = gdf.rename_geometry("geometry")

    # Valider les géométries
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )

    # Supprimer les géométries nulles/vides
    mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.warning("  %d parcelles sans géométrie supprimées", n_dropped)
    gdf = gdf[mask].copy()

    # S'assurer du CRS
    if gdf.crs is None:
        logger.warning("  CRS absent → hypothèse WGS84")
        gdf = gdf.set_crs("EPSG:4326").to_crs("EPSG:2154")
    elif gdf.crs.to_epsg() != 2154:
        gdf = gdf.to_crs("EPSG:2154")

    logger.info("  Parcelles prêtes : %d (CRS=%s)", len(gdf), gdf.crs.to_epsg())
    return gdf


def join_parcelles_buildings(
    parcelles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    label_col: str,
    version_key: str,
) -> pd.DataFrame:
    """Joint spatialement les parcelles aux bâtiments clusterisés.

    Retourne un mapping index_parcelle → (cluster, nb_batiments, emprise, ces)
    sans modifier le GeoDataFrame parcelles.

    Pour chaque parcelle :
    - 1 seul bâtiment → prend son cluster
    - Plusieurs bâtiments → cluster du bâtiment avec la plus grande emprise
    - Aucun bâtiment → cluster = -1

    Parameters
    ----------
    parcelles : gpd.GeoDataFrame
        Parcelles cadastrales (index utilisé comme clé).
    buildings : gpd.GeoDataFrame
        Bâtiments avec cluster label.
    label_col : str
        Nom de la colonne de cluster dans buildings.
    version_key : str
        Identifiant de la version (v3/v4).

    Returns
    -------
    pd.DataFrame
        Mapping avec colonnes cluster_{vkey}, nb_batiments_{vkey},
        emprise_batie_{vkey}, ces_{vkey}. Index = index des parcelles.
    """
    logger.info("[%s] Jointure spatiale parcelles ↔ bâtiments...", version_key)

    # Préparer les bâtiments : garder les colonnes utiles + calculer l'emprise
    bld = buildings[[label_col, "surface_bat", "geometry"]].copy()
    bld["bat_area"] = bld.geometry.area
    bld = bld.reset_index(drop=True)

    # Pré-calculer les bornes des bâtiments une seule fois
    bld_bounds = bld.geometry.bounds

    # Jointure spatiale par chunks de communes pour limiter la mémoire
    communes = parcelles["code_commune_insee"].unique()
    logger.info("  %d communes à traiter", len(communes))

    # Résultats partiels : DataFrames indexés par l'index original des parcelles
    mapping_parts = []

    for i, commune in enumerate(communes):
        parc_commune = parcelles[parcelles["code_commune_insee"] == commune]
        if len(parc_commune) == 0:
            continue

        # Filtrer les bâtiments dans le bbox de la commune (accélération)
        bounds = parc_commune.total_bounds  # minx, miny, maxx, maxy
        margin = 50  # mètres de marge
        bld_mask = (
            (bld_bounds["minx"] <= bounds[2] + margin)
            & (bld_bounds["maxx"] >= bounds[0] - margin)
            & (bld_bounds["miny"] <= bounds[3] + margin)
            & (bld_bounds["maxy"] >= bounds[1] - margin)
        )
        bld_commune = bld[bld_mask]

        if len(bld_commune) == 0:
            # Aucun bâtiment dans cette commune — toutes à -1
            part = pd.DataFrame(
                {
                    f"cluster_{version_key}": -1,
                    f"nb_batiments_{version_key}": 0,
                    f"emprise_batie_{version_key}": 0.0,
                },
                index=parc_commune.index,
            )
            mapping_parts.append(part)
            continue

        # Jointure spatiale — l'index gauche = index original des parcelles
        joined = gpd.sjoin(
            parc_commune[["geometry"]],
            bld_commune[[label_col, "bat_area", "geometry"]],
            how="left",
            predicate="intersects",
        )

        # Séparer les parcelles avec/sans match
        has_match = joined["index_right"].notna()
        matched = joined[has_match]

        if len(matched) > 0:
            # Cluster dominant : bâtiment avec la plus grande bat_area par parcelle
            # L'index du sjoin left = l'index de la parcelle (peut avoir des doublons)
            idx_max = matched.groupby(matched.index)["bat_area"].idxmax()
            dominant_cluster = matched.loc[idx_max, label_col]
            # Le groupby .index produit un index unique par parcelle
            dominant_cluster.index = matched.groupby(matched.index).ngroup().loc[idx_max].values
            # Reconstruire proprement avec le bon index
            dominant_series = matched.loc[idx_max, label_col]
            # Ensure index is unique (idxmax returns unique per group)
            dominant_series.index = [
                matched.index[matched.index == idx][0]
                for idx in matched.groupby(matched.index).ngroups
            ] if False else dominant_series.index  # placeholder, use simpler approach

            # Approche vectorisée simple
            grp = matched.groupby(matched.index)
            # Pour chaque parcelle, trouver le bâtiment avec la plus grande emprise
            best_bat_area = grp["bat_area"].transform("max")
            is_best = matched["bat_area"] == best_bat_area
            # En cas d'égalité, garder le premier
            best_rows = matched[is_best].groupby(level=0).first()

            cluster_series = best_rows[label_col].astype(int)
            nb_bat_series = grp["bat_area"].count()
            emprise_series = grp["bat_area"].sum()

            part = pd.DataFrame(
                {
                    f"cluster_{version_key}": cluster_series,
                    f"nb_batiments_{version_key}": nb_bat_series,
                    f"emprise_batie_{version_key}": emprise_series,
                },
            )
        else:
            part = pd.DataFrame(
                columns=[f"cluster_{version_key}",
                         f"nb_batiments_{version_key}",
                         f"emprise_batie_{version_key}"],
            )

        # Ajouter les parcelles sans match (cluster = -1)
        missing_idx = parc_commune.index.difference(part.index)
        if len(missing_idx) > 0:
            missing_part = pd.DataFrame(
                {
                    f"cluster_{version_key}": -1,
                    f"nb_batiments_{version_key}": 0,
                    f"emprise_batie_{version_key}": 0.0,
                },
                index=missing_idx,
            )
            part = pd.concat([part, missing_part])

        mapping_parts.append(part)

        if (i + 1) % 5 == 0:
            logger.info("  Communes traitées : %d/%d", i + 1, len(communes))

    logger.info("  Communes traitées : %d/%d", len(communes), len(communes))

    # Concaténer tous les mappings
    mapping_df = pd.concat(mapping_parts)
    mapping_df[f"cluster_{version_key}"] = mapping_df[f"cluster_{version_key}"].astype(int)
    mapping_df[f"nb_batiments_{version_key}"] = mapping_df[f"nb_batiments_{version_key}"].astype(int)

    # CES réel
    surface_parcelle = parcelles["s_geom_parcelle"].fillna(parcelles.geometry.area)
    mapping_df[f"ces_{version_key}"] = np.where(
        surface_parcelle.loc[mapping_df.index] > 0,
        mapping_df[f"emprise_batie_{version_key}"]
        / surface_parcelle.loc[mapping_df.index],
        0.0,
    )

    n_matched = (mapping_df[f"cluster_{version_key}"] >= 0).sum()
    n_unmatched = (mapping_df[f"cluster_{version_key}"] == -1).sum()
    logger.info(
        "[%s] Résultat : %d parcelles, %d avec cluster (%.1f%%), %d sans",
        version_key, len(mapping_df), n_matched,
        100 * n_matched / len(mapping_df), n_unmatched,
    )

    return mapping_df


def compute_cluster_names(
    gdf: gpd.GeoDataFrame, label_col: str
) -> dict[int, str]:
    """Nomme chaque cluster selon son profil médian."""
    names = {}
    for cl in sorted(gdf[label_col].unique()):
        if cl < 0:
            names[cl] = "Non classé"
            continue
        sub = gdf[gdf[label_col] == cl]
        s = sub["surface_bat"].median()
        h = sub["hauteur_mean"].median()
        a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode()
        m = str(mat.iloc[0]) if len(mat) > 0 else ""

        if a < 1945:
            per = "ancien"
        elif a < 1975:
            per = "1945-75"
        elif a < 2000:
            per = "fin XXe"
        else:
            per = str(int(a))

        ml = m.lower()
        mat_s = ""
        if "pierre" in ml:
            mat_s = "pierre"
        elif "béton" in ml or "beton" in ml:
            mat_s = "béton"
        elif "brique" in ml:
            mat_s = "brique"
        elif "parpaing" in ml or "agglo" in ml:
            mat_s = "parpaing"
        elif "bois" in ml:
            mat_s = "bois"

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


def build_parcellaire() -> tuple[gpd.GeoDataFrame, dict]:
    """Pipeline complet : charge parcelles, joint les deux clusterings, sauvegarde.

    Returns
    -------
    tuple[gpd.GeoDataFrame, dict]
        (Parcelles enrichies avec clusters V3 et V4, noms des clusters par version)
    """
    config = load_config()

    # ── Étape 1 : Charger les parcelles ──
    parcelles = load_parcelles(config)

    # Sauvegarder le parcellaire brut
    out_brut = "data/processed/parcellaire_metropole.geoparquet"
    parcelles.to_parquet(out_brut)
    logger.info("Parcellaire brut sauvegardé : %s (%.1f MB)",
                out_brut, os.path.getsize(out_brut) / 1e6)

    # Calculer surface_parcelle une seule fois
    parcelles["surface_parcelle"] = parcelles["s_geom_parcelle"].fillna(
        parcelles.geometry.area
    )

    # ── Étape 2-3 : Jointure pour chaque version ──
    all_cluster_names = {}

    for vkey, vcfg in VERSIONS.items():
        logger.info("=== %s ===", vcfg["title"])

        # Charger les bâtiments clusterisés
        bld = gpd.read_parquet(vcfg["data_path"])
        label_col = vcfg["label_col"]
        logger.info("  %d bâtiments chargés", len(bld))

        # Calculer les noms de clusters AVANT la jointure
        cluster_names = compute_cluster_names(bld, label_col)
        all_cluster_names[vkey] = cluster_names
        logger.info("  Noms des clusters :")
        for cl, name in sorted(cluster_names.items()):
            n = (bld[label_col] == cl).sum()
            logger.info("    %2d: %s (%d bât.)", cl, name, n)

        # Jointure spatiale → retourne un mapping (index → colonnes)
        mapping = join_parcelles_buildings(
            parcelles, bld, label_col, vkey,
        )

        # Joindre le mapping aux parcelles (par index, pas de duplication)
        for col in mapping.columns:
            parcelles[col] = mapping[col]

        # Ajouter le nom du cluster dominant
        parcelles[f"cluster_nom_{vkey}"] = parcelles[f"cluster_{vkey}"].map(
            cluster_names
        ).fillna("Non classé")

        del bld
        gc.collect()

    # ── Étape 4 : Sauvegarder ──
    out_path = "data/processed/parcellaire_clustered.geoparquet"

    # Sélectionner les colonnes finales
    cols = [
        "parcelle_id", "code_commune_insee", "geometry",
        "surface_parcelle",
    ]
    for vkey in VERSIONS:
        cols.extend([
            f"cluster_{vkey}", f"cluster_nom_{vkey}",
            f"nb_batiments_{vkey}", f"emprise_batie_{vkey}", f"ces_{vkey}",
        ])

    parcelles_out = parcelles[[c for c in cols if c in parcelles.columns]].copy()
    parcelles_out.to_parquet(out_path)
    logger.info(
        "Parcellaire clusterisé sauvegardé : %s (%.1f MB)",
        out_path, os.path.getsize(out_path) / 1e6,
    )

    # Stats finales
    for vkey in VERSIONS:
        col = f"cluster_{vkey}"
        if col in parcelles_out.columns:
            n_cls = (parcelles_out[col] >= 0).sum()
            logger.info(
                "[%s] %d parcelles classées / %d total (%.1f%%)",
                vkey, n_cls, len(parcelles_out),
                100 * n_cls / len(parcelles_out),
            )
            logger.info(
                "[%s] Distribution :\n%s",
                vkey, parcelles_out[col].value_counts().sort_index().to_string(),
            )

    return parcelles_out, all_cluster_names


# ── Carte web interactive ──────────────────────────────────────────────────


def export_parcelles_geojson(
    parcelles: gpd.GeoDataFrame,
    version_key: str,
    cluster_names: dict[int, str],
) -> tuple[dict, int]:
    """Exporte les GeoJSON des parcelles par cluster pour une version.

    Parameters
    ----------
    parcelles : gpd.GeoDataFrame
        Parcelles enrichies.
    version_key : str
        Identifiant de version (v3/v4).
    cluster_names : dict
        Mapping cluster_id → nom.

    Returns
    -------
    tuple[dict, int]
        (cluster_meta, n_total parcelles classées)
    """
    cluster_col = f"cluster_{version_key}"
    nom_col = f"cluster_nom_{version_key}"
    ces_col = f"ces_{version_key}"

    logger.info("[%s] Export GeoJSON parcelles...", version_key)

    # Simplifier les géométries pour la carte
    gdf = parcelles.copy()
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE)

    # Reprojeter en WGS84
    gdf = gdf.to_crs("EPSG:4326")

    # Colonnes pour les popups
    nb_bat_col = f"nb_batiments_{version_key}"
    emprise_col = f"emprise_batie_{version_key}"
    popup_cols = ["parcelle_id", "surface_parcelle"]
    if nb_bat_col in gdf.columns:
        popup_cols.append(nb_bat_col)
    if emprise_col in gdf.columns:
        popup_cols.append(emprise_col)
    if ces_col in gdf.columns:
        popup_cols.append(ces_col)

    # Nettoyer les NaN
    for col in gdf.columns:
        if col == "geometry":
            continue
        if gdf[col].dtype in ("float64", "float32"):
            gdf[col] = gdf[col].where(gdf[col].notna(), None)

    data_dir = os.path.join(OUTPUT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    cluster_meta = {}

    def round_coords(coords):
        if isinstance(coords[0], (list, tuple)):
            return [round_coords(c) for c in coords]
        return [round(coords[0], 6), round(coords[1], 6)]

    # Ne pas exporter les parcelles non classées (-1) si elles sont trop nombreuses
    # (elles alourdissent la carte sans apport visuel)
    n_unclassified = (gdf[cluster_col] == -1).sum()
    if n_unclassified > 50000:
        logger.info("  %d parcelles non classées → omises de la carte (trop nombreuses)",
                     n_unclassified)
        clusters_to_export = sorted([c for c in gdf[cluster_col].unique() if c >= 0])
    else:
        clusters_to_export = sorted(gdf[cluster_col].unique())

    for cl in clusters_to_export:
        sub = gdf[gdf[cluster_col] == cl].copy()
        n = len(sub)
        name = cluster_names.get(cl, "Non classé")
        color = "#d3d3d3" if cl == -1 else COLORS[cl % len(COLORS)]

        # Colonnes à exporter
        export_cols = [c for c in popup_cols if c in sub.columns] + ["geometry"]
        sub = sub[export_cols]

        fname = f"{version_key}_parcelles_cluster_{cl}.geojson"
        fpath = os.path.join(data_dir, fname)

        geojson = json.loads(sub.to_json())
        for feature in geojson["features"]:
            geom = feature["geometry"]
            if geom and "coordinates" in geom:
                geom["coordinates"] = round_coords(geom["coordinates"])

        with open(fpath, "w") as f:
            json.dump(geojson, f, separators=(",", ":"))

        fsize = os.path.getsize(fpath) / 1e6
        logger.info(
            "  Cluster %3d : %6d parcelles → %s (%.1f MB)",
            cl, n, fname, fsize,
        )
        cluster_meta[int(cl)] = {
            "name": name,
            "color": color,
            "count": n,
            "file": f"data/{fname}",
        }

    n_total = len(gdf[gdf[cluster_col] >= 0])
    return cluster_meta, n_total


def generate_html(all_versions: dict) -> str:
    """Génère le HTML Leaflet avec onglets multi-version (parcelles).

    Parameters
    ----------
    all_versions : dict
        Métadonnées par version (title, subtitle, clusters, etc.).

    Returns
    -------
    str
        Chemin du fichier HTML généré.
    """
    versions_js = json.dumps(all_versions, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clustering parcellaire — Bordeaux Métropole</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
#map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; z-index: 1; background: #ffffff; }}

#panel {{
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: white; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    width: 360px; max-height: calc(100vh - 20px); overflow-y: auto;
    font-size: 13px;
}}
#panel-header {{
    position: sticky; top: 0; z-index: 10;
    padding: 10px 16px; background: #2d5016; color: white;
    border-radius: 8px 8px 0 0; font-weight: bold; font-size: 14px;
    cursor: pointer; user-select: none; display: flex; justify-content: space-between;
}}
#panel-header:hover {{ background: #3a6b1e; }}
#panel-body {{ padding: 0; }}
.section {{ padding: 10px 14px; border-bottom: 1px solid #eee; }}
.section-title {{ font-weight: 600; font-size: 11px; color: #888; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 8px; }}

#tabs {{
    display: flex; border-bottom: 2px solid #eee; background: #fafafa;
}}
.tab {{
    flex: 1; padding: 10px 8px; text-align: center; cursor: pointer;
    font-size: 12px; font-weight: 600; color: #666; border-bottom: 3px solid transparent;
    transition: all 0.2s;
}}
.tab:hover {{ background: #f0f0f0; }}
.tab.active {{ color: #2d5016; border-bottom-color: #4363d8; background: white; }}
#version-subtitle {{ padding: 6px 14px; font-size: 11px; color: #888; background: #f8f8f8;
    border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
#version-subtitle a {{ color: #4363d8; text-decoration: none; font-weight: 600; font-size: 11px; }}
#version-subtitle a:hover {{ text-decoration: underline; }}

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
    Chargement des parcelles... <span id="progress">0/0</span>
    <div id="load-bar"><div id="load-fill"></div></div>
</div>

<div id="panel">
    <div id="panel-header">
        <span>Parcellaire — Bordeaux Métropole</span>
        <span id="toggle-icon">&#9660;</span>
    </div>
    <div id="panel-body">
        <div id="tabs"></div>
        <div id="version-subtitle"></div>
        <div class="section">
            <div class="section-title">Fond de carte</div>
            <div id="basemap-controls"></div>
        </div>
        <div class="section">
            <div class="section-title" id="clusters-title">Clusters</div>
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
const VERSIONS = {versions_js};

const canvasRenderer = L.canvas({{ padding: 0.5, tolerance: 5 }});
const map = L.map('map', {{
    center: [44.8378, -0.5792], zoom: 12,
    preferCanvas: true, renderer: canvasRenderer
}});

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
let currentBasemap = basemaps['CartoDB Positron'];
currentBasemap.addTo(map);

const bmContainer = document.getElementById('basemap-controls');
Object.keys(basemaps).forEach((name, i) => {{
    const div = document.createElement('div');
    div.className = 'basemap-option';
    const checked = name === 'CartoDB Positron' ? 'checked' : '';
    div.innerHTML = `<input type="radio" name="basemap" id="bm_${{i}}" value="${{name}}" ${{checked}}>
        <label for="bm_${{i}}">${{name}}</label>`;
    div.querySelector('input').addEventListener('change', () => {{
        if (currentBasemap) map.removeLayer(currentBasemap);
        currentBasemap = basemaps[name];
        if (currentBasemap) currentBasemap.addTo(map);
    }});
    bmContainer.appendChild(div);
}});

let activeVersion = null;
const loadedLayers = {{}};
const loadedData = {{}};

function formatPopup(props, clId, vKey) {{
    const meta = VERSIONS[vKey].clusters[clId];
    const isUnclassified = clId == -1;
    const headerColor = isUnclassified ? '#999' : meta.color;
    const headerText = isUnclassified ? 'Parcelle non classée' : `Cluster ${{clId}} — ${{meta.name}}`;

    let html = `<div style="font-family:sans-serif;font-size:13px;min-width:220px;">`;
    html += `<div style="background:${{headerColor}};color:white;padding:6px 10px;margin:-1px -1px 8px;border-radius:4px 4px 0 0;font-weight:bold;">`;
    html += `${{headerText}}</div>`;
    html += `<table style="width:100%;border-collapse:collapse;">`;

    const cesKey = `ces_${{vKey}}`;
    const nbBatKey = `nb_batiments_${{vKey}}`;
    const empriseKey = `emprise_batie_${{vKey}}`;
    const rows = [
        ['Parcelle', props.parcelle_id || '—'],
        ['Surface parcelle', props.surface_parcelle != null ? Number(props.surface_parcelle).toFixed(0) + ' m²' : '—'],
        ['Nb bâtiments', props[nbBatKey] != null ? props[nbBatKey] : '—'],
        ['Emprise bâtie', props[empriseKey] != null ? Number(props[empriseKey]).toFixed(0) + ' m²' : '—'],
        ['CES', props[cesKey] != null ? (Number(props[cesKey]) * 100).toFixed(1) + ' %' : '—'],
    ];
    rows.forEach(([k, v]) => {{
        html += `<tr><td style="padding:2px 6px;color:#666;white-space:nowrap;">${{k}}</td>
                     <td style="padding:2px 6px;font-weight:500;">${{v}}</td></tr>`;
    }});
    html += '</table></div>';
    return html;
}}

async function loadVersion(vKey) {{
    if (loadedData[vKey]) return;

    const version = VERSIONS[vKey];
    const clusters = version.clusters;
    const ids = Object.keys(clusters).sort((a, b) => a - b);
    const total = ids.length;
    let loaded = 0;

    const loadingEl = document.getElementById('loading');
    const progressEl = document.getElementById('progress');
    const loadFill = document.getElementById('load-fill');
    loadingEl.style.display = 'block';
    progressEl.textContent = `0/${{total}}`;
    loadFill.style.width = '0%';

    loadedLayers[vKey] = {{}};

    for (let i = 0; i < ids.length; i += 2) {{
        const batch = ids.slice(i, i + 2);
        await Promise.all(batch.map(async (clId) => {{
            const meta = clusters[clId];
            try {{
                const resp = await fetch(meta.file);
                const geojson = await resp.json();
                const isUnclassified = clId == -1;
                const layer = L.geoJSON(geojson, {{
                    renderer: canvasRenderer,
                    style: () => ({{
                        fillColor: meta.color,
                        color: '#333333',
                        weight: 0.4,
                        fillOpacity: isUnclassified ? 0.3 : 0.85,
                    }}),
                    onEachFeature: (feature, layer) => {{
                        layer.bindPopup(() => formatPopup(feature.properties, clId, vKey), {{ maxWidth: 320 }});
                    }}
                }});
                loadedLayers[vKey][clId] = layer;
            }} catch (e) {{
                console.error(`Erreur ${{vKey}} cluster ${{clId}}:`, e);
            }}
            loaded++;
            progressEl.textContent = `${{loaded}}/${{total}}`;
            loadFill.style.width = `${{(loaded / total * 100).toFixed(0)}}%`;
        }}));
    }}

    loadedData[vKey] = true;
    setTimeout(() => {{ loadingEl.style.display = 'none'; }}, 300);
}}

async function switchVersion(vKey) {{
    if (activeVersion && loadedLayers[activeVersion]) {{
        Object.values(loadedLayers[activeVersion]).forEach(layer => map.removeLayer(layer));
    }}

    activeVersion = vKey;

    document.querySelectorAll('.tab').forEach(t => {{
        t.classList.toggle('active', t.dataset.version === vKey);
    }});

    const subtitleEl = document.getElementById('version-subtitle');
    const pdfLink = VERSIONS[vKey].pdf
        ? `<a href="${{VERSIONS[vKey].pdf}}" target="_blank">&#128196; Rapport PDF</a>`
        : '';
    subtitleEl.innerHTML = `<span>${{VERSIONS[vKey].subtitle}}</span>${{pdfLink}}`;

    await loadVersion(vKey);

    const clusters = VERSIONS[vKey].clusters;
    const ids = Object.keys(clusters).sort((a, b) => a - b);
    const n_total = ids.reduce((sum, id) => sum + clusters[id].count, 0);

    document.getElementById('clusters-title').textContent =
        `Parcelles (${{n_total.toLocaleString('fr-FR')}})`;

    const ccContainer = document.getElementById('cluster-controls');
    ccContainer.innerHTML = '';

    ids.forEach(clId => {{
        const meta = clusters[clId];
        const layer = loadedLayers[vKey][clId];
        if (layer) layer.addTo(map);

        const isUnclassified = clId == -1;
        const displayName = isUnclassified ? 'Non classé (sans bâtiment)' : `${{clId}}: ${{meta.name}}`;

        const div = document.createElement('div');
        div.className = 'cluster-option';
        div.innerHTML = `
            <input type="checkbox" id="cl_${{vKey}}_${{clId}}" checked>
            <div class="cluster-swatch" style="background:${{meta.color}};${{isUnclassified ? 'opacity:0.4;' : ''}}"></div>
            <div>
                <span class="cluster-label">${{displayName}}</span><br>
                <span class="cluster-count">${{meta.count.toLocaleString('fr-FR')}} parcelles</span>
            </div>`;
        div.querySelector('input').addEventListener('change', (e) => {{
            if (layer) {{
                if (e.target.checked) map.addLayer(layer);
                else map.removeLayer(layer);
            }}
        }});
        ccContainer.appendChild(div);
    }});
}}

function toggleAll(show) {{
    if (!activeVersion || !loadedLayers[activeVersion]) return;
    const clusters = VERSIONS[activeVersion].clusters;
    Object.keys(clusters).forEach(clId => {{
        const cb = document.getElementById(`cl_${{activeVersion}}_${{clId}}`);
        if (cb) cb.checked = show;
        const layer = loadedLayers[activeVersion][clId];
        if (layer) {{
            if (show) map.addLayer(layer);
            else map.removeLayer(layer);
        }}
    }});
}}

const tabsContainer = document.getElementById('tabs');
Object.keys(VERSIONS).forEach(vKey => {{
    const tab = document.createElement('div');
    tab.className = 'tab';
    tab.dataset.version = vKey;
    tab.textContent = VERSIONS[vKey].title;
    tab.addEventListener('click', () => switchVersion(vKey));
    tabsContainer.appendChild(tab);
}});

const panelBody = document.getElementById('panel-body');
const toggleIcon = document.getElementById('toggle-icon');
document.getElementById('panel-header').addEventListener('click', () => {{
    const hidden = panelBody.style.display === 'none';
    panelBody.style.display = hidden ? 'block' : 'none';
    toggleIcon.innerHTML = hidden ? '&#9660;' : '&#9654;';
}});

switchVersion(Object.keys(VERSIONS)[0]);
</script>
</body>
</html>"""

    html_path = os.path.join(OUTPUT_DIR, "index.html")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html)
    logger.info("HTML parcellaire : %s", html_path)
    return html_path


def generate_map(
    parcelles: gpd.GeoDataFrame,
    all_cluster_names: dict[str, dict[int, str]],
) -> str:
    """Génère la carte web interactive des parcelles.

    Parameters
    ----------
    parcelles : gpd.GeoDataFrame
        Parcelles enrichies avec clusters V3 et V4.
    all_cluster_names : dict
        Mapping version → {cluster_id → nom}.

    Returns
    -------
    str
        Chemin du fichier HTML.
    """
    # Nettoyer les anciens GeoJSON
    data_dir = os.path.join(OUTPUT_DIR, "data")
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".geojson"):
                os.remove(os.path.join(data_dir, f))

    all_versions_meta = {}

    for vkey, vcfg in VERSIONS.items():
        cluster_names = all_cluster_names[vkey]
        cluster_meta, n_total = export_parcelles_geojson(
            parcelles, vkey, cluster_names,
        )
        all_versions_meta[vkey] = {
            "title": vcfg["title"],
            "subtitle": vcfg["subtitle"],
            "n_total": n_total,
            "pdf": vcfg.get("pdf", ""),
            "clusters": cluster_meta,
        }

    html_path = generate_html(all_versions_meta)

    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(OUTPUT_DIR)
        for f in files
    )
    logger.info("Dossier carte : %s (%.1f MB)", OUTPUT_DIR, total_size / 1e6)
    return html_path


def main() -> None:
    """Pipeline complet : parcellaire + carte."""
    parcelles, all_cluster_names = build_parcellaire()

    logger.info("=== Génération de la carte web ===")
    html_path = generate_map(parcelles, all_cluster_names)

    logger.info("TERMINÉ — carte : %s", html_path)


if __name__ == "__main__":
    main()
