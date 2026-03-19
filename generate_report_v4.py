"""Rapport PDF v4 — Bâti récent (>=2016), résidentiel individuel, KMeans k=10."""

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

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


def fit(path, max_w, max_h):
    img = PILImage.open(path)
    iw, ih = img.size
    r = ih / iw
    w, h = max_w, max_w * r
    if h > max_h:
        h, w = max_h, max_h / r
    return w, h


def name_cluster(sub):
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


def main():
    logger.info("Chargement...")
    gdf = gpd.read_parquet(DATA_PATH)
    n_bat = len(gdf)
    logger.info("  %d bâtiments", n_bat)

    # Noms
    names = {}
    for cl in sorted(gdf[LABEL_COL].unique()):
        names[cl] = name_cluster(gdf[gdf[LABEL_COL] == cl])

    # ── Figures ──

    # Carte ensemble
    logger.info("Carte d'ensemble...")
    bounds = gdf.total_bounds
    ratio = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    fw = 16
    fig, ax = plt.subplots(1, 1, figsize=(fw, fw * ratio))
    for cl in sorted(gdf[LABEL_COL].unique()):
        gdf[gdf[LABEL_COL] == cl].plot(ax=ax, color=COLORS[cl % len(COLORS)], linewidth=0, alpha=0.8)
    patches = [mpatches.Patch(color=COLORS[cl % len(COLORS)], label=f"{cl}: {names[cl]}")
               for cl in sorted(names)]
    ax.legend(handles=patches, loc="upper left", fontsize=8, ncol=2, framealpha=0.95,
              title="Typologies bâti récent", title_fontsize=10)
    ax.set_title(f"Clustering v4 — Bâti récent (>= 2016)\n{n_bat} bâtiments, KMeans k=10",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_aspect("equal"); ax.set_axis_off()
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_carte.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Zoom centre
    logger.info("Zoom centre Bordeaux...")
    xmin, xmax, ymin, ymax = 410000, 416000, 6424000, 6430000
    zr = (ymax - ymin) / (xmax - xmin)
    fig, ax = plt.subplots(1, 1, figsize=(16, 16 * zr))
    for cl in sorted(gdf[LABEL_COL].unique()):
        sub = gdf[gdf[LABEL_COL] == cl].cx[xmin:xmax, ymin:ymax]
        if len(sub) > 0:
            sub.plot(ax=ax, color=COLORS[cl % len(COLORS)], linewidth=0.3, edgecolor="#333", alpha=0.85)
    ax.legend(handles=patches, loc="upper left", fontsize=7, ncol=2, framealpha=0.95)
    ax.set_title("Zoom Centre Bordeaux — Bâti récent", fontsize=16, fontweight="bold", pad=15)
    ax.set_aspect("equal"); ax.set_axis_off()
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_zoom.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Barplot
    logger.info("Barplot...")
    counts = gdf[LABEL_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS[i % len(COLORS)] for i in counts.index], edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{i}\n{names[i][:25]}" for i in counts.index], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Nombre de bâtiments")
    ax.set_title("Effectifs par cluster — Bâti récent", fontsize=16, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(val), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_effectifs.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Boxplots
    logger.info("Boxplots...")
    labels_sorted = sorted(gdf[LABEL_COL].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (col, ttl) in enumerate([("surface_bat", "Surface (m²)"), ("hauteur_mean", "Hauteur (m)")]):
        data = [gdf[gdf[LABEL_COL] == cl][col].dropna().values for cl in labels_sorted]
        bp = axes[i].boxplot(data, tick_labels=labels_sorted, patch_artist=True, showfliers=False)
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[labels_sorted[j] % len(COLORS)])
        if col == "surface_bat":
            axes[i].set_yscale("log")
        axes[i].set_title(ttl, fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Cluster")
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Distribution par année
    logger.info("Distribution années...")
    fig, ax = plt.subplots(figsize=(14, 5))
    years = sorted(gdf["annee_construction"].unique())
    bottom = np.zeros(len(years))
    for cl in labels_sorted:
        sub = gdf[gdf[LABEL_COL] == cl]
        vals = [len(sub[sub["annee_construction"] == y]) for y in years]
        ax.bar(years, vals, bottom=bottom, color=COLORS[cl % len(COLORS)], label=f"{cl}", width=0.8)
        bottom += vals
    ax.set_xlabel("Année de construction"); ax.set_ylabel("Nombre de bâtiments")
    ax.set_title("Distribution par année et cluster", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=5, title="Cluster")
    plt.tight_layout()
    fig.savefig("outputs/figures/report_v4_annees.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── PDF ──
    logger.info("Génération PDF...")
    pdf_path = "outputs/reports/clustering_bordeaux_v4.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    pw = A4[0] - 3*cm
    ph = A4[1] - 3*cm

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CustomTitle", parent=styles["Title"],
               fontSize=22, spaceAfter=30, textColor=HexColor("#1a1a2e")))
    styles.add(ParagraphStyle("Section", parent=styles["Heading1"],
               fontSize=16, spaceAfter=12, spaceBefore=20, textColor=HexColor("#16213e")))
    styles.add(ParagraphStyle("SubSec", parent=styles["Heading2"],
               fontSize=12, spaceAfter=8, spaceBefore=12, textColor=HexColor("#0f3460")))
    styles.add(ParagraphStyle("Body2", parent=styles["Normal"],
               fontSize=9, spaceAfter=6, leading=13))
    styles.add(ParagraphStyle("CName", parent=styles["Normal"],
               fontSize=10, spaceAfter=4, leading=14,
               textColor=HexColor("#e94560"), fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("Rule", parent=styles["Normal"],
               fontSize=9, spaceAfter=2, leading=12, leftIndent=20, textColor=HexColor("#333333")))

    story = []

    # Titre
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("Clustering v4 — Bâti récent", styles["CustomTitle"]))
    story.append(Paragraph("Bordeaux Métropole — Constructions 2016-2023", styles["Section"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"<b>{n_bat}</b> bâtiments (résidentiel individuel, nb_logements &lt; 4, année &gt;= 2016)", styles["Body2"]))
    story.append(Paragraph("Algorithme : KMeans k=10 — 11 features — Source : BDNB 2025-07-a", styles["Body2"]))
    story.append(Paragraph("Date : mars 2026", styles["Body2"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "Ce rapport analyse les typologies du bâti résidentiel individuel récent "
        "(moins de 10 ans) sur Bordeaux Métropole. Le clustering en 10 types permet "
        "d'identifier les profils de construction contemporaine.", styles["Body2"]))
    story.append(PageBreak())

    # Carte
    story.append(Paragraph("1. Vue d'ensemble", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v4_carte.png", pw, ph - 3*cm)
    story.append(Image("outputs/figures/report_v4_carte.png", width=iw, height=ih))
    story.append(PageBreak())

    # Zoom
    story.append(Paragraph("2. Zoom Centre Bordeaux", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v4_zoom.png", pw, ph - 3*cm)
    story.append(Image("outputs/figures/report_v4_zoom.png", width=iw, height=ih))
    story.append(PageBreak())

    # Effectifs + boxplots
    story.append(Paragraph("3. Effectifs et distributions", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v4_effectifs.png", pw, 9*cm)
    story.append(Image("outputs/figures/report_v4_effectifs.png", width=iw, height=ih))
    story.append(Spacer(1, 0.3*cm))
    iw, ih = fit("outputs/figures/report_v4_boxplots.png", pw, 8*cm)
    story.append(Image("outputs/figures/report_v4_boxplots.png", width=iw, height=ih))
    story.append(PageBreak())

    # Distribution années
    story.append(Paragraph("4. Distribution par année de construction", styles["Section"]))
    iw, ih = fit("outputs/figures/report_v4_annees.png", pw, 8*cm)
    story.append(Image("outputs/figures/report_v4_annees.png", width=iw, height=ih))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "La production de logements individuels récents montre un pic en 2017-2018, "
        "suivi d'un fléchissement post-2020 (effet COVID et hausse des coûts).",
        styles["Body2"]))
    story.append(PageBreak())

    # Tableau profil
    story.append(Paragraph("5. Profil des 10 clusters", styles["Section"]))
    header = ["Cl.", "Nom", "N", "%", "Surf.", "Haut.", "Niv.", "Année", "Mat. murs"]
    tdata = [header]
    for cl in sorted(names):
        sub = gdf[gdf[LABEL_COL] == cl]
        s = sub["surface_bat"].median(); h = sub["hauteur_mean"].median()
        niv = sub["nb_niveaux"].median(); a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode(); m = mat.iloc[0] if len(mat) > 0 else "-"
        tdata.append([
            str(cl), names[cl][:30], str(len(sub)),
            f"{len(sub)/n_bat*100:.1f}", f"{s:.0f}", f"{h:.1f}",
            f"{niv:.0f}" if niv == niv else "-",
            f"{a:.0f}" if a == a else "-",
            str(m)[:20],
        ])
    cw = [0.7*cm, 4.5*cm, 1.2*cm, 0.9*cm, 1*cm, 1*cm, 0.8*cm, 1.1*cm, 3.5*cm]
    t = Table(tdata, colWidths=cw, repeatRows=1)
    ts = [("BACKGROUND",(0,0),(-1,0),HexColor("#16213e")),("TEXTCOLOR",(0,0),(-1,0),white),
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),
          ("ALIGN",(0,0),(-1,-1),"CENTER"),("ALIGN",(1,1),(1,-1),"LEFT"),("ALIGN",(-1,1),(-1,-1),"LEFT"),
          ("GRID",(0,0),(-1,-1),0.5,grey),("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#f0f0f0")]),
          ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]
    for i, cl in enumerate(sorted(names)):
        ts.append(("BACKGROUND",(0,i+1),(0,i+1),HexColor(COLORS[cl % len(COLORS)])))
        ts.append(("TEXTCOLOR",(0,i+1),(0,i+1),white))
    t.setStyle(TableStyle(ts))
    story.append(t)
    story.append(PageBreak())

    # Description clusters
    story.append(Paragraph("6. Description des clusters", styles["Section"]))
    for cl in sorted(names):
        sub = gdf[gdf[LABEL_COL] == cl]
        s = sub["surface_bat"].median(); h = sub["hauteur_mean"].median()
        a = sub["annee_construction"].median()
        mat = sub["mat_murs"].mode(); m = mat.iloc[0] if len(mat) > 0 else "inconnu"
        usage_f = sub["usage_foncier"].mode()
        uf = usage_f.iloc[0] if len(usage_f) > 0 else "inconnu"

        block = []
        block.append(Paragraph(
            f'<font color="{COLORS[cl % len(COLORS)]}">&#9632;</font> '
            f'Cluster {cl} — {names[cl]} ({len(sub)} bâtiments, {len(sub)/n_bat*100:.1f}%)',
            styles["CName"]))
        block.append(Paragraph(f"• Surface médiane : {s:.0f} m² — Hauteur : {h:.1f} m", styles["Rule"]))
        if a == a:
            block.append(Paragraph(f"• Année construction médiane : {a:.0f}", styles["Rule"]))
        block.append(Paragraph(f"• Matériau murs : {m}", styles["Rule"]))
        block.append(Paragraph(f"• Usage foncier : {uf}", styles["Rule"]))

        # Distribution années dans ce cluster
        year_dist = sub["annee_construction"].value_counts().sort_index()
        top_years = year_dist.head(3)
        yr_str = ", ".join([f"{int(y)}: {n}" for y, n in top_years.items()])
        block.append(Paragraph(f"• Années principales : {yr_str}", styles["Rule"]))
        block.append(Spacer(1, 0.2*cm))
        story.append(KeepTogether(block))
    story.append(PageBreak())

    # Conclusions
    story.append(Paragraph("7. Analyse et conclusions", styles["Section"]))
    story.append(Paragraph("Caractéristiques du bâti récent :", styles["SubSec"]))
    for p in [
        "Domination de la brique comme matériau de construction (>70%)",
        "Deux morphologies principales : pavillons plain-pied (3-4m) et maisons R+1 (5-7m)",
        "Surface médiane de 74 à 162 m² selon les types",
        "Le béton apparaît dans un cluster spécifique (cluster 5, 772 bâtiments)",
        f"Pic de construction en 2017-2018, décélération à partir de 2020",
        "Les très grandes maisons (>200m²) sont marginales (<1%)",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Pistes suivantes :", styles["SubSec"]))
    for p in [
        "Croiser avec les données DVF pour analyser les prix du bâti récent",
        "Comparer la localisation du bâti récent avec le PLU (zones AU)",
        "Analyser la performance énergétique (DPE) par cluster",
        "Étudier la densification des quartiers pavillonnaires existants",
    ]:
        story.append(Paragraph(f"• {p}", styles["Body2"]))

    doc.build(story)
    logger.info("PDF : %s", pdf_path)
    logger.info("TERMINÉ")


if __name__ == "__main__":
    main()
