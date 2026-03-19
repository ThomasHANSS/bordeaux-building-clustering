"""Enrichissement BDNB avec données INSEE carroyées 200m (Filosofi 2019).

Jointure spatiale : chaque bâtiment hérite des caractéristiques
du carreau 200m dans lequel se trouve son centroïde.
"""

import geopandas as gpd
import logging
import os
import gc
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

BDNB_PATH = "data/processed/clustered.geoparquet"
INSEE_SHP = "data/raw/insee_carreaux/carreaux_200m_met.shp"
OUTPUT_PATH = "data/processed/bdnb_enrichi_insee.geoparquet"

# Bbox BDNB Bordeaux Métropole + marge 1km
BBOX = box(394000, 6410000, 428000, 6445000)

# Variables INSEE à récupérer (noms en minuscules dans le shapefile)
INSEE_COLS = [
    "ind",        # Population (individus)
    "men",        # Ménages
    "men_pauv",   # Ménages pauvres
    "men_prop",   # Ménages propriétaires
    "ind_snv",    # Somme des niveaux de vie winsorisés
    "log_av45",   # Logements construits avant 1945
    "log_45_70",  # Logements 1945-1970
    "log_70_90",  # Logements 1970-1990
    "log_ap90",   # Logements après 1990
    "men_surf",   # Surface moyenne des logements
    "men_coll",   # Ménages en collectif
    "men_mais",   # Ménages en maison
]

# Renommage pour clarté dans le dataset final
INSEE_RENAME = {
    "ind": "insee_population",
    "men": "insee_menages",
    "men_pauv": "insee_menages_pauvres",
    "men_prop": "insee_menages_proprietaires",
    "ind_snv": "insee_niveau_vie_sum",
    "log_av45": "insee_log_avant_1945",
    "log_45_70": "insee_log_1945_1970",
    "log_70_90": "insee_log_1970_1990",
    "log_ap90": "insee_log_apres_1990",
    "men_surf": "insee_surface_moyenne",
    "men_coll": "insee_menages_collectif",
    "men_mais": "insee_menages_maison",
}


def main() -> None:
    """Pipeline d'enrichissement."""
    os.makedirs("data/processed", exist_ok=True)

    # 1. Charger les carreaux INSEE (filtré bbox)
    logger.info("Chargement carreaux INSEE (bbox Bordeaux Métropole)...")
    insee = gpd.read_file(INSEE_SHP, bbox=BBOX)
    logger.info("  %d carreaux chargés, CRS=%s", len(insee), insee.crs.to_epsg())

    # Ne garder que les colonnes utiles + geometry
    keep = [c for c in INSEE_COLS if c in insee.columns] + ["geometry"]
    missing = [c for c in INSEE_COLS if c not in insee.columns]
    if missing:
        logger.warning("  Colonnes absentes du shapefile : %s", missing)
    insee = insee[keep].copy()

    # Calculer le niveau de vie moyen par carreau (ind_snv / ind)
    if "ind_snv" in insee.columns and "ind" in insee.columns:
        insee["niveau_vie_moyen"] = (insee["ind_snv"] / insee["ind"]).round(0)
        INSEE_RENAME["niveau_vie_moyen"] = "insee_niveau_vie_moyen"

    logger.info("  Variables INSEE : %s", [c for c in insee.columns if c != "geometry"])

    # 2. Charger les bâtiments BDNB
    logger.info("Chargement bâtiments BDNB...")
    bdnb = gpd.read_parquet(BDNB_PATH)
    n_bdnb = len(bdnb)
    logger.info("  %d bâtiments, CRS=%s", n_bdnb, bdnb.crs.to_epsg())

    # Vérifier CRS identique
    if bdnb.crs.to_epsg() != insee.crs.to_epsg():
        logger.info("  Reprojection INSEE vers %s...", bdnb.crs)
        insee = insee.to_crs(bdnb.crs)

    # 3. Jointure spatiale par centroïde du bâtiment
    logger.info("Jointure spatiale (centroïde bâtiment → carreau 200m)...")

    # Sauvegarder la géométrie originale
    bdnb["_geom_orig"] = bdnb.geometry

    # Remplacer temporairement par le centroïde pour la jointure
    bdnb.geometry = bdnb.geometry.centroid

    # sjoin : chaque bâtiment prend les valeurs du carreau qui contient son centroïde
    enrichi = gpd.sjoin(bdnb, insee, how="left", predicate="within")

    # Restaurer la géométrie polygone
    enrichi.geometry = enrichi["_geom_orig"]
    enrichi = enrichi.drop(columns=["_geom_orig", "index_right"], errors="ignore")

    # Stats de couverture
    n_matched = enrichi[INSEE_COLS[0]].notna().sum() if INSEE_COLS[0] in enrichi.columns else 0
    coverage = n_matched / n_bdnb * 100
    logger.info("  Couverture : %d/%d bâtiments (%.1f%%)", n_matched, n_bdnb, coverage)

    # Gérer les doublons (si un centroïde tombe sur la frontière de 2 carreaux)
    if len(enrichi) > n_bdnb:
        logger.info("  Doublons sjoin : %d → dédoublonnage (garder le premier)...", len(enrichi) - n_bdnb)
        enrichi = enrichi[~enrichi.index.duplicated(keep="first")]
        logger.info("  Après dédoublonnage : %d", len(enrichi))

    # 4. Renommer les colonnes INSEE
    rename_map = {k: v for k, v in INSEE_RENAME.items() if k in enrichi.columns}
    enrichi = enrichi.rename(columns=rename_map)

    # 5. Stats résumé
    logger.info("\nStatistiques des variables INSEE enrichies :")
    for old, new in sorted(rename_map.items(), key=lambda x: x[1]):
        col = new
        if col in enrichi.columns:
            notna = enrichi[col].notna().sum()
            logger.info("  %-35s : %d valeurs (%.1f%%)  médiane=%.1f",
                        col, notna, notna/n_bdnb*100,
                        enrichi[col].median() if notna > 0 else 0)

    # 6. Sauvegarder
    enrichi.to_parquet(OUTPUT_PATH)
    fsize = os.path.getsize(OUTPUT_PATH) / 1e6
    logger.info("\nSauvegardé : %s (%d bâtiments, %d colonnes, %.0f MB)",
                OUTPUT_PATH, len(enrichi), len(enrichi.columns), fsize)
    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
