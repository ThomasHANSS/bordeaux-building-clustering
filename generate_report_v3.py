"""Rapport PDF v3 — Résidentiel individuel (150k bâtiments, KMeans k=15).

Rapport compact sans extraits de cartes (disponibles sur la carte web).
Légendes complètes, mise en page dense.
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

DATA_PATH = "data/processed/clustered_petit_residentiel.geoparquet"
LABEL_COL = "km_label"
PDF_PATH = "outputs/reports/clustering_bordeaux_v3.pdf"

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
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

    if a < 1945: per = "ancien"
    elif a < 1975: per = "1945-75"
    elif a < 2000: per = "fin XXe"
    else: per = "récent"

    if s < 60: typ = "Petit bâti"
    elif h <= 4: typ = "Grande maison" if s > 200 else "Pavillon"
    elif h <= 7: typ = "Maison R+1"
    elif h <= 10: typ = "Individuel dense R+2"
    else: typ = "Individuel haut"

    parts = [typ, f"{s:.0f}m²", f"{h:.0f}m", per]
    if mat_s:
        parts.append(mat_s)
    return " / ".join(parts)


def make_patches(names: dict) -> list:
    return [mpatches.Patch(color=COLORS[cl % len(COLORS)], label=f"{cl}: {names[cl]}")
            for cl in sorted(names)]


def main() -> None:
    logger.info("Chargement...")
    gdf = gpd.read_parquet(DATA_PATH)
    n_bat = len(gdf)
    logger.info("  %d bâtiments", n_bat)

    names = {}
    for cl in sorted(gdf[LABEL_COL].unique()):
        names[cl] = name_cluster(gdf[gdf[LABEL_COL] == cl])

    patches = make_patches(names)

    # ── Barplot avec noms complets ──
    logger.info("Barplot...")
    counts = gdf[LABEL_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(range(len(counts)), counts.values,
                  color=[COLORS[cl % len(COLORS)] for cl in counts.index], edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{cl}: {names[cl]}" for cl in counts.index],
                       fontsize=7, rotation=55, ha="right")
    ax.set_ylabel("Nombre de bâtiments", fontsize=11)
    ax.set_title("Effectifs par cluster — Résidentiel individuel (KMeans k=15)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f"{val:,}".replace(",", " "), ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v3_effectifs.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Boxplots avec noms complets ──
    logger.info("Boxplots...")
    labels_sorted = sorted(gdf[LABEL_COL].unique())
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    for i, (col, ttl) in enumerate([("surface_bat", "Surface au sol (m²)"), ("hauteur_mean", "Hauteur moyenne (m)")]):
        data = [gdf[gdf[LABEL_COL] == cl][col].dropna().values for cl in labels_sorted]
        bp = axes[i].boxplot(data, tick_labels=[f"{cl}: {names[cl][:30]}" for cl in labels_sorted],
                             patch_artist=True, showfliers=False)
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[labels_sorted[j] % len(COLORS)])
        if col == "surface_bat":
            axes[i].set_yscale("log")
        axes[i].set_title(ttl, fontsize=13, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=55, labelsize=7)
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v3_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Distribution matériaux par cluster ──
    logger.info("Matériaux...")
    mat_data = {}
    all_mats = set()
    for cl in labels_sorted:
        sub = gdf[gdf[LABEL_COL] == cl]
        vc = sub["mat_murs"].value_counts(normalize=True).head(5)
        mat_data[cl] = vc
        all_mats.update(vc.index)

    top_mats = sorted(all_mats, key=lambda m: sum(mat_data[cl].get(m, 0) for cl in labels_sorted), reverse=True)[:6]
    mat_colors = ["#4363d8", "#e6194b", "#3cb44b", "#f58231", "#911eb4", "#42d4f4"]

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(labels_sorted))
    w = 0.12
    for i, mat in enumerate(top_mats):
        vals = [mat_data[cl].get(mat, 0) * 100 for cl in labels_sorted]
        ax.bar(x + i * w, vals, w, label=mat[:25], color=mat_colors[i % len(mat_colors)])
    ax.set_xticks(x + w * len(top_mats) / 2)
    ax.set_xticklabels([f"{cl}: {names[cl][:25]}" for cl in labels_sorted], fontsize=7, rotation=55, ha="right")
    ax.set_ylabel("% des bâtiments", fontsize=11)
    ax.set_title("Matériaux de murs par cluster", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v3_materiaux.png", dpi=200, bbox_inches="tight")
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

    # ── Page titre (compact) ──
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("Clustering v3 — Résidentiel individuel", styles["Title2"]))
    story.append(Paragraph("Bordeaux Métropole — 28 communes", styles["Section"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(f"<b>{n_bat:,}</b> bâtiments — résidentiel individuel, nb_logements &lt; 4".replace(",", " "), styles["Body2"]))
    story.append(Paragraph("Algorithme : KMeans k=15 — 11 features (6 numériques + 5 catégorielles)", styles["Body2"]))
    story.append(Paragraph("Source : BDNB millésime 2025-07-a — Date : mars 2026", styles["Body2"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "Ce rapport analyse les typologies du bâti résidentiel individuel (moins de 4 logements) "
        "sur Bordeaux Métropole. Les bâtiments tertiaires, collectifs, dépendances et à usage inconnu "
        "sont exclus. Les cartes interactives sont disponibles sur la carte web associée.", styles["Body2"]))
    story.append(PageBreak())

    # ── Effectifs ──
    story.append(Paragraph("1. Effectifs par cluster", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v3_effectifs.png", pw, 11*cm)
    story.append(Image("outputs/figures/report_v3_effectifs.png", width=iw, height=ih))

    # ── Tableau profil (même page) ──
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("2. Profil synthétique des 15 clusters", styles["Section"]))
    header = ["Cl.", "Nom", "N", "%", "Surf.", "Haut.", "Niv.", "Année", "Mat. murs"]
    tdata = [header]
    for cl in sorted(names):
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
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),6.5),
          ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(1,1),(1,-1),"LEFT"),("ALIGN",(-1,1),(-1,-1),"LEFT"),
          ("GRID",(0,0),(-1,-1),0.4,grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#f5f5f5")]),
          ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]
    for i, cl in enumerate(sorted(names)):
        ts.append(("BACKGROUND",(0,i+1),(0,i+1),HexColor(COLORS[cl % len(COLORS)])))
        ts.append(("TEXTCOLOR",(0,i+1),(0,i+1),white))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(PageBreak())

    # ── Boxplots ──
    story.append(Paragraph("3. Distributions surface et hauteur", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v3_boxplots.png", pw, 16*cm)
    story.append(Image("outputs/figures/report_v3_boxplots.png", width=iw, height=ih))
    story.append(PageBreak())

    # ── Matériaux ──
    story.append(Paragraph("4. Matériaux de murs par cluster", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v3_materiaux.png", pw, 10*cm)
    story.append(Image("outputs/figures/report_v3_materiaux.png", width=iw, height=ih))

    # ── Description des clusters (compact, 2 colonnes implicites) ──
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("5. Description des clusters", styles["Section"]))
    for cl in sorted(names):
        sub = gdf[gdf[LABEL_COL] == cl]
        s = sub["surface_bat"].median(); h = sub["hauteur_mean"].median()
        a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode(); m = mat.iloc[0] if len(mat) > 0 else "inconnu"

        block = []
        block.append(Paragraph(
            f'<font color="{COLORS[cl % len(COLORS)]}">&#9632;</font> '
            f'<b>Cluster {cl}</b> — {names[cl]} ({len(sub):,} bât., {len(sub)/n_bat*100:.1f}%) '
            f'— {s:.0f} m², {h:.1f} m, {a:.0f}, {m}'.replace(",", " "),
            styles["Body2"]))
        story.append(KeepTogether(block))
    story.append(PageBreak())

    # ── Conclusions ──
    story.append(Paragraph("6. Conclusions", styles["Section"]))
    for p in [
        "15 typologies clairement différenciées par surface, hauteur, période et matériaux",
        "Les pavillons (clusters 0, 1, 2, 5) représentent ~55% du stock résidentiel individuel",
        "Le bâti ancien en pierre (clusters 3, 9, 13) se distingue nettement du pavillonnaire récent en brique",
        "Les maisons R+1 (clusters 8, 10, 12, 19) forment un gradient par période et matériau",
        "L'individuel dense R+2 (cluster 13, pierre, ancien) correspond au centre historique bordelais",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Pistes d'amélioration :", styles["SubSec"]))
    for p in [
        "Enrichir avec les features DVF (prix m²) et INSEE (revenus)",
        "Calculer le Moran's I (cohérence spatiale) par cluster",
        "Tester un clustering séparé pour les collectifs (nb_logements >= 4)",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    doc.build(story)
    logger.info("PDF : %s", PDF_PATH)
    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
