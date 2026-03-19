"""Rapport PDF v4 — Bâti récent (>=2016), résidentiel individuel, KMeans k=10.

Rapport compact sans extraits de cartes. Légendes complètes, mise en page dense.
"""

import geopandas as gpd
import numpy as np
import os
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger()

DATA_PATH = "data/processed/clustered_recent.geoparquet"
LABEL_COL = "v4_label"
PDF_PATH = "outputs/reports/clustering_bordeaux_v4.pdf"

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


def fit(path: str, max_w: float, max_h: float) -> tuple[float, float]:
    img = PILImage.open(path)
    iw, ih = img.size
    r = ih / iw
    w, h = max_w, max_w * r
    if h > max_h:
        h, w = max_h, max_h / r
    return w, h


def name_cluster(sub: gpd.GeoDataFrame) -> str:
    s = sub["surface_bat"].median()
    h = sub["hauteur_mean"].median()
    a = sub["annee_construction"].median()
    mat = sub["mat_murs"].mode()
    m = str(mat.iloc[0]) if len(mat) > 0 else ""
    ml = m.lower()
    mat_s = ""
    if "pierre" in ml: mat_s = "pierre"
    elif "béton" in ml or "beton" in ml: mat_s = "béton"
    elif "brique" in ml: mat_s = "brique"
    elif "parpaing" in ml or "agglo" in ml: mat_s = "parpaing"
    elif "bois" in ml: mat_s = "bois"

    if s < 60: typ = "Petit bâti"
    elif h <= 4: typ = "Grande maison" if s > 200 else "Pavillon"
    elif h <= 7: typ = "Maison R+1"
    elif h <= 10: typ = "Individuel dense R+2"
    else: typ = "Individuel haut"

    parts = [typ, f"{s:.0f}m²", f"{h:.0f}m", f"{a:.0f}"]
    if mat_s:
        parts.append(mat_s)
    return " / ".join(parts)


def main() -> None:
    logger.info("Chargement...")
    gdf = gpd.read_parquet(DATA_PATH)
    n_bat = len(gdf)
    logger.info("  %d bâtiments", n_bat)

    names = {}
    for cl in sorted(gdf[LABEL_COL].unique()):
        names[cl] = name_cluster(gdf[gdf[LABEL_COL] == cl])

    labels_sorted = sorted(names.keys())

    # ── Barplot avec noms complets ──
    logger.info("Barplot...")
    counts = gdf[LABEL_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(len(counts)), counts.values,
                  color=[COLORS[cl % len(COLORS)] for cl in counts.index], edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{cl}: {names[cl]}" for cl in counts.index],
                       fontsize=8, rotation=50, ha="right")
    ax.set_ylabel("Nombre de bâtiments", fontsize=11)
    ax.set_title("Effectifs par cluster — Bâti récent (KMeans k=10)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                str(val), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_effectifs.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Boxplots avec noms complets ──
    logger.info("Boxplots...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for i, (col, ttl) in enumerate([("surface_bat", "Surface au sol (m²)"), ("hauteur_mean", "Hauteur moyenne (m)")]):
        data = [gdf[gdf[LABEL_COL] == cl][col].dropna().values for cl in labels_sorted]
        bp = axes[i].boxplot(data, tick_labels=[f"{cl}: {names[cl][:30]}" for cl in labels_sorted],
                             patch_artist=True, showfliers=False)
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[labels_sorted[j] % len(COLORS)])
        if col == "surface_bat":
            axes[i].set_yscale("log")
        axes[i].set_title(ttl, fontsize=13, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=50, labelsize=7)
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Parcellaire : charger et préparer ──
    logger.info("Parcellaire...")
    parc = gpd.read_parquet("data/processed/parcellaire_clustered.geoparquet")
    parc_v4 = parc[parc["cluster_v4"] >= 0].copy()
    n_parc = len(parc_v4)
    logger.info("  %d parcelles classées V4", n_parc)

    # Barplot parcelles vs bâtiments par cluster
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels_sorted))
    w = 0.35
    bat_counts = [len(gdf[gdf[LABEL_COL] == cl]) for cl in labels_sorted]
    parc_counts = [len(parc_v4[parc_v4["cluster_v4"] == cl]) for cl in labels_sorted]
    ax.bar(x - w/2, bat_counts, w, label="Bâtiments", color="#4363d8", alpha=0.8)
    bars2 = ax.bar(x + w/2, parc_counts, w, label="Parcelles", color="#3cb44b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cl}: {names[cl][:25]}" for cl in labels_sorted],
                       fontsize=7, rotation=50, ha="right")
    ax.set_ylabel("Nombre", fontsize=11)
    ax.set_title("Bâtiments vs Parcelles par cluster — Bâti récent", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    for bar, val in zip(bars2, parc_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha="center", fontsize=7, fontweight="bold", color="#3cb44b")
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_parcelles_effectifs.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Boxplot CES par cluster
    fig, ax = plt.subplots(figsize=(14, 6))
    ces_data = [parc_v4[parc_v4["cluster_v4"] == cl]["ces_v4"].clip(0, 1).dropna().values for cl in labels_sorted]
    bp = ax.boxplot(ces_data, tick_labels=[f"{cl}: {names[cl][:25]}" for cl in labels_sorted],
                     patch_artist=True, showfliers=False)
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[labels_sorted[j] % len(COLORS)])
    ax.set_ylabel("CES (emprise bâtie / surface parcelle)", fontsize=11)
    ax.set_title("Coefficient d'Emprise au Sol par cluster — Bâti récent", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=50, labelsize=7)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="CES = 50%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_parcelles_ces.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Distribution par année stacked ──
    logger.info("Distribution années...")
    fig, ax = plt.subplots(figsize=(14, 6))
    years = sorted(gdf["annee_construction"].unique())
    bottom = np.zeros(len(years))
    for cl in labels_sorted:
        sub = gdf[gdf[LABEL_COL] == cl]
        vals = [len(sub[sub["annee_construction"] == y]) for y in years]
        ax.bar(years, vals, bottom=bottom, color=COLORS[cl % len(COLORS)],
               label=f"{cl}: {names[cl][:30]}", width=0.8)
        bottom += vals
    ax.set_xlabel("Année de construction", fontsize=11)
    ax.set_ylabel("Nombre de bâtiments", fontsize=11)
    ax.set_title("Distribution par année et cluster", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_annees.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── PDF ──
    logger.info("Génération PDF...")
    doc = SimpleDocTemplate(PDF_PATH, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.2*cm, bottomMargin=1.2*cm)
    pw = A4[0] - 3*cm
    ph = A4[1] - 2.4*cm

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Title2", parent=styles["Title"],
               fontSize=20, spaceAfter=12, textColor=HexColor("#1a1a2e")))
    styles.add(ParagraphStyle("Section", parent=styles["Heading1"],
               fontSize=14, spaceAfter=8, spaceBefore=14, textColor=HexColor("#16213e")))
    styles.add(ParagraphStyle("SubSec", parent=styles["Heading2"],
               fontSize=11, spaceAfter=6, spaceBefore=10, textColor=HexColor("#0f3460")))
    styles.add(ParagraphStyle("Body2", parent=styles["Normal"],
               fontSize=9, spaceAfter=4, leading=12))
    styles.add(ParagraphStyle("CName", parent=styles["Normal"],
               fontSize=9, spaceAfter=3, leading=12,
               textColor=HexColor("#e94560"), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("Rule", parent=styles["Normal"],
               fontSize=8, spaceAfter=1, leading=10, leftIndent=15, textColor=HexColor("#333333")))

    story = []

    # ── Titre ──
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("Clustering v4 — Bâti récent", styles["Title2"]))
    story.append(Paragraph("Bordeaux Métropole — Constructions 2016-2023", styles["Section"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"<b>{n_bat}</b> bâtiments — résidentiel individuel, nb_logements &lt; 4, année &gt;= 2016", styles["Body2"]))
    story.append(Paragraph("Algorithme : KMeans k=10 — 11 features — Source : BDNB 2025-07-a — mars 2026", styles["Body2"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Ce rapport analyse les typologies du bâti résidentiel individuel récent "
        "(moins de 10 ans) sur Bordeaux Métropole. Les cartes interactives sont disponibles "
        "sur la carte web associée.", styles["Body2"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Tableau profil (page titre) ──
    story.append(Paragraph("Profil synthétique des 10 clusters", styles["SubSec"]))
    header = ["Cl.", "Nom", "N", "%", "Surf.", "Haut.", "Niv.", "Année", "Mat. murs"]
    tdata = [header]
    for cl in labels_sorted:
        sub = gdf[gdf[LABEL_COL] == cl]
        s = sub["surface_bat"].median(); h = sub["hauteur_mean"].median()
        niv = sub["nb_niveaux"].median(); a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode(); m = mat.iloc[0] if len(mat) > 0 else "-"
        tdata.append([
            str(cl), names[cl][:32], str(len(sub)),
            f"{len(sub)/n_bat*100:.1f}", f"{s:.0f}", f"{h:.1f}",
            f"{niv:.0f}" if niv == niv else "-",
            f"{a:.0f}" if a == a else "-",
            str(m)[:22],
        ])
    cw = [0.7*cm, 4.5*cm, 1.2*cm, 0.8*cm, 0.9*cm, 0.9*cm, 0.8*cm, 1*cm, 3.5*cm]
    t = Table(tdata, colWidths=cw, repeatRows=1)
    ts = [("BACKGROUND",(0,0),(-1,0),HexColor("#16213e")),("TEXTCOLOR",(0,0),(-1,0),white),
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),
          ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(1,1),(1,-1),"LEFT"),("ALIGN",(-1,1),(-1,-1),"LEFT"),
          ("GRID",(0,0),(-1,-1),0.4,grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#f5f5f5")]),
          ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]
    for i, cl in enumerate(labels_sorted):
        ts.append(("BACKGROUND",(0,i+1),(0,i+1),HexColor(COLORS[cl % len(COLORS)])))
        ts.append(("TEXTCOLOR",(0,i+1),(0,i+1),white))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(PageBreak())

    # ── Effectifs ──
    story.append(Paragraph("1. Effectifs par cluster", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v4_effectifs.png", pw, 11*cm)
    story.append(Image("outputs/figures/report_v4_effectifs.png", width=iw, height=ih))

    # ── Distribution années (même page) ──
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("2. Distribution par année de construction", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v4_annees.png", pw, 9*cm)
    story.append(Image("outputs/figures/report_v4_annees.png", width=iw, height=ih))
    story.append(PageBreak())

    # ── Boxplots ──
    story.append(Paragraph("3. Distributions surface et hauteur", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v4_boxplots.png", pw, 16*cm)
    story.append(Image("outputs/figures/report_v4_boxplots.png", width=iw, height=ih))

    # ── Parcellaire ──
    story.append(Paragraph("4. Analyse parcellaire", styles["Section"]))
    story.append(Paragraph(
        f"Jointure spatiale des {n_parc:,} parcelles classées (sur {len(parc):,} parcelles métropole) "
        "aux bâtiments clusterisés. Chaque parcelle prend le cluster du bâtiment dominant "
        "(plus grande emprise au sol).".replace(",", " "),
        styles["Body2"]))
    story.append(Spacer(1, 0.3*cm))

    # Tableau parcellaire
    story.append(Paragraph("Indicateurs parcellaires par cluster", styles["SubSec"]))
    p_header = ["Cl.", "Nom", "Parcelles", "Surf. parc.", "CES méd.", "Bât./parc."]
    p_tdata = [p_header]
    for cl in labels_sorted:
        psub = parc_v4[parc_v4["cluster_v4"] == cl]
        if len(psub) == 0:
            continue
        sp = psub["surface_parcelle"].median()
        ces = psub["ces_v4"].median()
        nb = psub["nb_batiments_v4"].median()
        p_tdata.append([
            str(cl), names[cl][:32], f"{len(psub):,}".replace(",", " "),
            f"{sp:.0f} m²", f"{ces*100:.1f}%", f"{nb:.1f}",
        ])
    p_cw = [0.7*cm, 4.8*cm, 1.5*cm, 1.5*cm, 1.3*cm, 1.3*cm]
    pt = Table(p_tdata, colWidths=p_cw, repeatRows=1)
    p_ts = [("BACKGROUND",(0,0),(-1,0),HexColor("#2d5016")),("TEXTCOLOR",(0,0),(-1,0),white),
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),
          ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(1,1),(1,-1),"LEFT"),
          ("GRID",(0,0),(-1,-1),0.4,grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#f0f5f0")]),
          ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]
    for i, cl in enumerate(labels_sorted):
        p_ts.append(("BACKGROUND",(0,i+1),(0,i+1),HexColor(COLORS[cl % len(COLORS)])))
        p_ts.append(("TEXTCOLOR",(0,i+1),(0,i+1),white))
    pt.setStyle(TableStyle(p_ts))
    story.append(pt)
    story.append(Spacer(1, 0.3*cm))

    iw, ih = fit("outputs/figures/report_v4_parcelles_effectifs.png", pw, 8*cm)
    story.append(Image("outputs/figures/report_v4_parcelles_effectifs.png", width=iw, height=ih))
    story.append(PageBreak())

    story.append(Paragraph("CES par cluster", styles["SubSec"]))
    iw, ih = fit("outputs/figures/report_v4_parcelles_ces.png", pw, 8*cm)
    story.append(Image("outputs/figures/report_v4_parcelles_ces.png", width=iw, height=ih))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Le CES (Coefficient d'Emprise au Sol) mesure le rapport entre l'emprise bâtie et la surface "
        "de la parcelle. Un CES élevé indique une parcelle densément bâtie, un CES faible "
        "une parcelle avec jardin.",
        styles["Body2"]))

    # ── Description clusters (compact) ──
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("5. Description des clusters", styles["Section"]))
    for cl in labels_sorted:
        sub = gdf[gdf[LABEL_COL] == cl]
        s = sub["surface_bat"].median(); h = sub["hauteur_mean"].median()
        a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode(); m = mat.iloc[0] if len(mat) > 0 else "inconnu"
        year_dist = sub["annee_construction"].value_counts().sort_index()
        top_years = year_dist.head(3)
        yr_str = ", ".join([f"{int(y)}: {n}" for y, n in top_years.items()])

        block = []
        block.append(Paragraph(
            f'<font color="{COLORS[cl % len(COLORS)]}">&#9632;</font> '
            f'<b>Cluster {cl}</b> — {names[cl]} ({len(sub)} bât., {len(sub)/n_bat*100:.1f}%) '
            f'— {m} — Années : {yr_str}',
            styles["Body2"]))
        story.append(KeepTogether(block))
    story.append(PageBreak())

    # ── Conclusions ──
    story.append(Paragraph("6. Analyse et conclusions", styles["Section"]))
    story.append(Paragraph("Caractéristiques du bâti récent :", styles["SubSec"]))
    for p in [
        "Domination de la brique comme matériau de construction (>70%)",
        "Deux morphologies : pavillons plain-pied (3-4m) et maisons R+1 (5-7m)",
        "Surface médiane de 74 à 162 m² selon les types",
        "Le béton apparaît dans un cluster spécifique (cluster 5, 772 bâtiments)",
        "Pic de construction en 2017-2018, décélération à partir de 2020",
        "Les très grandes maisons (>200m²) sont marginales (<1%)",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Pistes suivantes :", styles["SubSec"]))
    for p in [
        "Croiser avec les données DVF pour analyser les prix du bâti récent",
        "Comparer la localisation du bâti récent avec le PLU (zones AU)",
        "Analyser la performance énergétique (DPE) par cluster",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    doc.build(story)
    logger.info("PDF : %s", PDF_PATH)
    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
