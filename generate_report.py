"""Rapport PDF — Analyse des règles de clustering."""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, grey
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)

print("Chargement des données...")
gdf = gpd.read_parquet('data/processed/clustered.geoparquet')

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
]

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# FIGURES
print("Carte d'ensemble...")
fig, ax = plt.subplots(1, 1, figsize=(14, 14))
for cl in sorted(gdf['cluster_label'].unique()):
    gdf[gdf['cluster_label'] == cl].plot(ax=ax, color=COLORS[cl], linewidth=0, alpha=0.7)
patches_legend = [mpatches.Patch(color=COLORS[i], label=f"Cluster {i}") for i in range(15)]
ax.legend(handles=patches_legend, loc='upper left', fontsize=7, ncol=3, framealpha=0.9)
ax.set_title("Clustering — Bordeaux Métropole (KMeans k=15)", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
fig.savefig("outputs/figures/report_carte.png", dpi=120, bbox_inches='tight')
plt.close()

print("Zoom centre Bordeaux...")
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl].cx[410000:416000, 6424000:6430000]
    if len(sub) > 0:
        sub.plot(ax=ax, color=COLORS[cl], linewidth=0.2, edgecolor='black', alpha=0.8)
ax.set_title("Zoom Centre Bordeaux", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
fig.savefig("outputs/figures/report_zoom.png", dpi=120, bbox_inches='tight')
plt.close()

print("Barplot effectifs...")
fig, ax = plt.subplots(figsize=(12, 5))
counts = gdf['cluster_label'].value_counts().sort_index()
bars = ax.bar(counts.index, counts.values, color=[COLORS[i] for i in counts.index])
ax.set_xlabel("Cluster"); ax.set_ylabel("Nombre de bâtiments")
ax.set_title("Effectifs par cluster")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500, f"{val}", ha='center', fontsize=8)
plt.tight_layout()
fig.savefig("outputs/figures/report_effectifs.png", dpi=120, bbox_inches='tight')
plt.close()

print("Boxplots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (col, title) in enumerate([('surface_bat', 'Surface (m²)'), ('hauteur_mean', 'Hauteur (m)')]):
    data = [gdf[gdf['cluster_label']==cl][col].dropna().values for cl in range(15)]
    bp = axes[i].boxplot(data, labels=range(15), patch_artist=True, showfliers=False)
    for j, patch in enumerate(bp['boxes']): patch.set_facecolor(COLORS[j])
    if col == 'surface_bat': axes[i].set_yscale('log')
    axes[i].set_title(title); axes[i].set_xlabel("Cluster")
plt.tight_layout()
fig.savefig("outputs/figures/report_boxplots.png", dpi=120, bbox_inches='tight')
plt.close()

# ANALYSE
print("Analyse des clusters...")
descs = {}
for cl in range(15):
    sub = gdf[gdf['cluster_label'] == cl]
    s = sub['surface_bat'].median()
    h = sub['hauteur_mean'].median()
    n = sub['nb_niveaux'].median()
    a = sub['annee_construction'].median()
    nl = sub['nb_logements'].median()
    usage = sub['usage_principal'].mode()
    u = usage.iloc[0] if len(usage) > 0 else "inconnu"

    if u == 'inconnu' and s < 60: nom = "Petit bâti non classé (annexe, garage)"
    elif 'individuel' in str(u).lower() and s < 60: nom = "Petit bâti résidentiel"
    elif 'individuel' in str(u).lower() and h <= 4 and s < 160: nom = "Pavillon de plain-pied"
    elif 'individuel' in str(u).lower() and h <= 4: nom = "Grande maison / propriété"
    elif 'individuel' in str(u).lower() and h <= 7: nom = "Maison R+1"
    elif 'individuel' in str(u).lower() and h <= 10: nom = "Individuel dense / R+2"
    elif 'collectif' in str(u).lower() and h <= 12: nom = "Petit collectif"
    elif 'collectif' in str(u).lower() and h <= 20: nom = "Collectif intermédiaire R+4/R+5"
    elif 'collectif' in str(u).lower(): nom = "Tour / IGH"
    elif 'Tertiaire' in str(u) and s > 2000: nom = "Grand équipement tertiaire"
    elif 'Tertiaire' in str(u): nom = "Bâtiment tertiaire"
    else: nom = f"Type mixte"

    rules = []
    if s < 60: rules.append(f"très petite surface ({s:.0f} m²)")
    elif s < 120: rules.append(f"petite surface ({s:.0f} m²)")
    elif s < 200: rules.append(f"surface moyenne ({s:.0f} m²)")
    elif s < 500: rules.append(f"grande surface ({s:.0f} m²)")
    elif s < 2000: rules.append(f"très grande surface ({s:.0f} m²)")
    else: rules.append(f"surface exceptionnelle ({s:.0f} m²)")

    if h <= 4: rules.append("bâtiment bas (RDC)")
    elif h <= 7: rules.append(f"hauteur moyenne ({h:.0f} m, R+1/R+2)")
    elif h <= 12: rules.append(f"bâtiment élevé ({h:.0f} m)")
    elif h <= 20: rules.append(f"bâtiment haut ({h:.0f} m)")
    else: rules.append(f"tour ({h:.0f} m)")

    rules.append(f"usage dominant : {u}")
    if not np.isnan(nl) and nl > 0: rules.append(f"{nl:.0f} logement(s) médian")
    if not np.isnan(a):
        if a < 1945: rules.append("avant 1945")
        elif a < 1975: rules.append(f"Trente Glorieuses ({a:.0f})")
        elif a < 2000: rules.append(f"fin XXe ({a:.0f})")
        else: rules.append(f"récent ({a:.0f})")

    descs[cl] = {'nom': nom, 'n': len(sub), 'pct': len(sub)/len(gdf)*100,
                 'surface': s, 'hauteur': h, 'niveaux': n, 'annee': a,
                 'logements': nl, 'usage': u, 'regles': rules,
                 'usage_detail': sub['usage_principal'].value_counts(normalize=True).head(3),
                 'mat_murs': sub['mat_murs'].value_counts(normalize=True).head(2) if 'mat_murs' in sub.columns else None}

# PDF
print("Génération du PDF...")
pdf_path = "outputs/reports/clustering_bordeaux_v1.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=22, spaceAfter=30, textColor=HexColor("#1a1a2e")))
styles.add(ParagraphStyle('Section', parent=styles['Heading1'], fontSize=16, spaceAfter=12, spaceBefore=20, textColor=HexColor("#16213e")))
styles.add(ParagraphStyle('SubSec', parent=styles['Heading2'], fontSize=12, spaceAfter=8, spaceBefore=12, textColor=HexColor("#0f3460")))
styles.add(ParagraphStyle('Body2', parent=styles['Normal'], fontSize=9, spaceAfter=6, leading=13))
styles.add(ParagraphStyle('CName', parent=styles['Normal'], fontSize=11, spaceAfter=4, leading=14, textColor=HexColor("#e94560"), fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('RuleStyle', parent=styles['Normal'], fontSize=9, spaceAfter=2, leading=12, leftIndent=20, textColor=HexColor("#333333")))

story = []

# Titre
story.append(Spacer(1, 3*cm))
story.append(Paragraph("Clustering des bâtiments", styles['CustomTitle']))
story.append(Paragraph("Bordeaux Métropole — 28 communes", styles['Section']))
story.append(Spacer(1, 1*cm))
story.append(Paragraph(f"<b>{len(gdf)}</b> bâtiments analysés — <b>15</b> typologies identifiées", styles['Body2']))
story.append(Paragraph("Algorithme : KMeans (k=15) — Silhouette : 0.350", styles['Body2']))
story.append(Paragraph("Source : BDNB millésime 2025-07-a — Date : mars 2026", styles['Body2']))
story.append(PageBreak())

# Cartes
story.append(Paragraph("1. Vue d'ensemble", styles['Section']))
story.append(Image("outputs/figures/report_carte.png", width=16*cm, height=16*cm))
story.append(PageBreak())
story.append(Paragraph("2. Zoom centre Bordeaux", styles['Section']))
story.append(Image("outputs/figures/report_zoom.png", width=15*cm, height=15*cm))
story.append(PageBreak())
story.append(Paragraph("3. Effectifs et distributions", styles['Section']))
story.append(Image("outputs/figures/report_effectifs.png", width=16*cm, height=6.5*cm))
story.append(Spacer(1, 0.5*cm))
story.append(Image("outputs/figures/report_boxplots.png", width=16*cm, height=6*cm))
story.append(PageBreak())

# Tableau synthétique
story.append(Paragraph("4. Profil synthétique des 15 clusters", styles['Section']))
header = ['Cl.', 'Nom', 'N', '%', 'Surf.', 'Haut.', 'Niv.', 'Année', 'Log.', 'Usage']
tdata = [header]
for cl, d in descs.items():
    tdata.append([str(cl), d['nom'][:28], str(d['n']),
        f"{d['pct']:.1f}", f"{d['surface']:.0f}", f"{d['hauteur']:.0f}",
        f"{d['niveaux']:.0f}" if not np.isnan(d['niveaux']) else "-",
        f"{d['annee']:.0f}" if not np.isnan(d['annee']) else "-",
        f"{d['logements']:.0f}" if not np.isnan(d['logements']) else "-",
        d['usage'][:20]])

cw = [0.6*cm, 4*cm, 1.5*cm, 0.8*cm, 1.2*cm, 1*cm, 0.8*cm, 1.2*cm, 0.8*cm, 3.2*cm]
t = Table(tdata, colWidths=cw, repeatRows=1)
ts = [('BACKGROUND',(0,0),(-1,0),HexColor("#16213e")),('TEXTCOLOR',(0,0),(-1,0),white),
      ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7),
      ('ALIGN',(0,0),(-1,-1),'CENTER'),('ALIGN',(1,1),(1,-1),'LEFT'),('ALIGN',(-1,1),(-1,-1),'LEFT'),
      ('GRID',(0,0),(-1,-1),0.5,grey),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor("#f0f0f0")]),
      ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]
for i, cl in enumerate(sorted(descs.keys())):
    ts.append(('BACKGROUND',(0,i+1),(0,i+1),HexColor(COLORS[cl])))
    ts.append(('TEXTCOLOR',(0,i+1),(0,i+1),white))
t.setStyle(TableStyle(ts))
story.append(t)
story.append(PageBreak())

# Règles détaillées
story.append(Paragraph("5. Règles de clustering en langage courant", styles['Section']))
story.append(Paragraph("Chaque cluster est décrit par les critères qui le distinguent des autres.", styles['Body2']))
story.append(Spacer(1, 0.3*cm))

for cl in range(15):
    d = descs[cl]
    block = []
    block.append(Paragraph(
        f'<font color="{COLORS[cl]}">&#9632;</font> Cluster {cl} — {d["nom"]} ({d["n"]} bâtiments, {d["pct"]:.1f}%)',
        styles['CName']))
    for rule in d['regles']:
        block.append(Paragraph(f"• {rule}", styles['RuleStyle']))
    u_str = " / ".join([f"{k}: {v:.0%}" for k,v in d['usage_detail'].items()])
    block.append(Paragraph(f"<i>Usages : {u_str}</i>", styles['RuleStyle']))
    if d['mat_murs'] is not None and len(d['mat_murs']) > 0:
        m_str = " / ".join([f"{k}: {v:.0%}" for k,v in d['mat_murs'].items()])
        block.append(Paragraph(f"<i>Murs : {m_str}</i>", styles['RuleStyle']))
    block.append(Spacer(1, 0.3*cm))
    story.append(KeepTogether(block))

story.append(PageBreak())

# Clusters proches
story.append(Paragraph("6. Distinction des clusters proches", styles['Section']))
story.append(Paragraph("Résidentiel individuel — 4 sous-types", styles['SubSec']))
for cl in [2,3,9,11]:
    d = descs[cl]
    a_str = f", année {d['annee']:.0f}" if not np.isnan(d['annee']) else ""
    story.append(Paragraph(f"• <b>Cluster {cl}</b> ({d['nom']}) : {d['surface']:.0f} m², {d['hauteur']:.0f} m{a_str}", styles['RuleStyle']))
story.append(Spacer(1,0.3*cm))

story.append(Paragraph("Résidentiel collectif — gradient", styles['SubSec']))
for cl in [5,6,7,12]:
    d = descs[cl]
    nl_str = f", {d['logements']:.0f} log." if not np.isnan(d['logements']) else ""
    story.append(Paragraph(f"• <b>Cluster {cl}</b> ({d['nom']}) : {d['surface']:.0f} m², {d['hauteur']:.0f} m{nl_str}", styles['RuleStyle']))
story.append(Spacer(1,0.3*cm))

story.append(Paragraph("Tertiaire — gradient", styles['SubSec']))
for cl in [4,10,14]:
    d = descs[cl]
    story.append(Paragraph(f"• <b>Cluster {cl}</b> ({d['nom']}) : {d['surface']:.0f} m², {d['hauteur']:.0f} m", styles['RuleStyle']))
story.append(PageBreak())

# Métriques
story.append(Paragraph("7. Métriques de qualité", styles['Section']))
md = [['Métrique','Valeur','Interprétation'],
      ['Silhouette','0.350','Correct — clusters distincts avec chevauchement modéré'],
      ['Davies-Bouldin','1.07','Bon — clusters compacts'],
      ['Calinski-Harabasz','52 911','Élevé — bonne séparation'],
      ['Bâtiments',str(len(gdf)),'234k bâtiments sur 28 communes'],
      ['Features','11','6 numériques + 5 catégorielles encodées']]
t2 = Table(md, colWidths=[3.5*cm,2.5*cm,9.5*cm], repeatRows=1)
t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor("#16213e")),('TEXTCOLOR',(0,0),(-1,0),white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),
    ('GRID',(0,0),(-1,-1),0.5,grey),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor("#f0f0f0")]),
    ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
story.append(t2)
story.append(Spacer(1,1*cm))

story.append(Paragraph("8. Pistes d'amélioration", styles['Section']))
for p in ["Investiguer le cluster 1 (46k bâtiments non classés)",
          "Tester HDBSCAN pour détection automatique du nombre de clusters",
          "Intégrer les features DVF (prix au m²) déjà dans la BDNB",
          "Calculer le Moran's I (cohérence spatiale)",
          "Affiner les sous-types résidentiels avec des features supplémentaires",
          "Ajouter les données INSEE carroyées (revenus, densité)"]:
    story.append(Paragraph(f"• {p}", styles['Body2']))

doc.build(story)
print(f"\nPDF : {pdf_path}")
print("TERMINÉ ✓")
