import geopandas as gpd
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

gdf = gpd.read_parquet('data/processed/clustered.geoparquet')

print("=" * 80)
print("ANALYSE DES RÈGLES DE CLUSTERING")
print("=" * 80)

# 1. Profil statistique complet par cluster
num_cols = ['surface_bat', 'hauteur_mean', 'hauteur_max', 'nb_niveaux', 
            'annee_construction', 'nb_logements']
cat_cols = ['usage_principal', 'usage_foncier', 'mat_murs', 'nature_bdtopo', 'usage_bdtopo']

print("\n" + "=" * 80)
print("1. PROFIL NUMÉRIQUE (médianes par cluster)")
print("=" * 80)

profiles = []
for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl]
    row = {'cluster': cl, 'n_bat': len(sub), 'pct': f"{len(sub)/len(gdf)*100:.1f}%"}
    for col in num_cols:
        if col in sub.columns:
            row[col] = sub[col].median()
    profiles.append(row)

df_prof = pd.DataFrame(profiles)
print(df_prof.to_string(index=False))

print("\n" + "=" * 80)
print("2. USAGE DOMINANT par cluster")
print("=" * 80)

for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl]
    usage = sub['usage_principal'].value_counts(normalize=True).head(3)
    usage_str = " | ".join([f"{k}: {v:.0%}" for k, v in usage.items()])
    print(f"  Cluster {cl:>2} (n={len(sub):>6}) : {usage_str}")

print("\n" + "=" * 80)
print("3. MATÉRIAUX DOMINANTS par cluster")
print("=" * 80)

for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl]
    murs = sub['mat_murs'].value_counts(normalize=True).head(2)
    murs_str = " | ".join([f"{k}: {v:.0%}" for k, v in murs.items()])
    print(f"  Cluster {cl:>2} : {murs_str}")

print("\n" + "=" * 80)
print("4. RÈGLES SIMPLIFIÉES (arbre de décision lisible)")
print("=" * 80)

for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl]
    s = sub['surface_bat'].median()
    h = sub['hauteur_mean'].median()
    n = sub['nb_niveaux'].median()
    a = sub['annee_construction'].median()
    nl = sub['nb_logements'].median()
    usage = sub['usage_principal'].mode()
    usage_top = usage.iloc[0] if len(usage) > 0 else "?"
    n_bat = len(sub)
    
    # Déterminer les critères discriminants
    rules = []
    
    # Surface
    if s < 60:
        rules.append(f"très petite surface ({s:.0f}m²)")
    elif s < 120:
        rules.append(f"petite surface ({s:.0f}m²)")
    elif s < 200:
        rules.append(f"surface moyenne ({s:.0f}m²)")
    elif s < 500:
        rules.append(f"grande surface ({s:.0f}m²)")
    elif s < 2000:
        rules.append(f"très grande surface ({s:.0f}m²)")
    else:
        rules.append(f"surface exceptionnelle ({s:.0f}m²)")
    
    # Hauteur
    if h <= 4:
        rules.append("bas (≤4m, RDC)")
    elif h <= 7:
        rules.append(f"moyen ({h:.0f}m, R+1/R+2)")
    elif h <= 12:
        rules.append(f"élevé ({h:.0f}m, R+2/R+3)")
    elif h <= 20:
        rules.append(f"haut ({h:.0f}m, R+4/R+5)")
    else:
        rules.append(f"très haut ({h:.0f}m, tour/IGH)")
    
    # Usage
    rules.append(f"usage: {usage_top}")
    
    # Logements
    if not np.isnan(nl):
        if nl == 0:
            rules.append("0 logement")
        elif nl <= 2:
            rules.append(f"{nl:.0f} logement(s)")
        elif nl <= 10:
            rules.append(f"{nl:.0f} logements")
        else:
            rules.append(f"{nl:.0f}+ logements")
    
    # Année
    if not np.isnan(a):
        if a < 1945:
            rules.append(f"ancien (avant 1945)")
        elif a < 1975:
            rules.append(f"30 glorieuses ({a:.0f})")
        elif a < 2000:
            rules.append(f"fin XXe ({a:.0f})")
        else:
            rules.append(f"récent ({a:.0f})")
    
    print(f"\n  Cluster {cl:>2} — {n_bat:>6} bâtiments ({n_bat/len(gdf)*100:.1f}%)")
    print(f"    → {' + '.join(rules)}")

print("\n" + "=" * 80)
print("5. CE QUI DISTINGUE LES CLUSTERS PROCHES")
print("=" * 80)

# Comparer les clusters résidentiels individuels (2, 3, 9, 11)
print("\n  RÉSIDENTIEL INDIVIDUEL — pourquoi 4 clusters distincts ?")
for cl in [2, 3, 9, 11]:
    sub = gdf[gdf['cluster_label'] == cl]
    print(f"    Cluster {cl}: surface={sub['surface_bat'].median():.0f}m² "
          f"hauteur={sub['hauteur_mean'].median():.1f}m "
          f"niveaux={sub['nb_niveaux'].median():.0f} "
          f"année={sub['annee_construction'].median():.0f} "
          f"logements={sub['nb_logements'].median():.0f}")

# Comparer les clusters collectifs (5, 6, 7, 12)
print("\n  RÉSIDENTIEL COLLECTIF — gradient de taille")
for cl in [5, 6, 7, 12]:
    sub = gdf[gdf['cluster_label'] == cl]
    print(f"    Cluster {cl}: surface={sub['surface_bat'].median():.0f}m² "
          f"hauteur={sub['hauteur_mean'].median():.1f}m "
          f"niveaux={sub['nb_niveaux'].median():.0f} "
          f"année={sub['annee_construction'].median():.0f} "
          f"logements={sub['nb_logements'].median():.0f}")

# Comparer tertiaire (4, 10, 14)
print("\n  TERTIAIRE — gradient de taille")
for cl in [4, 10, 14]:
    sub = gdf[gdf['cluster_label'] == cl]
    print(f"    Cluster {cl}: surface={sub['surface_bat'].median():.0f}m² "
          f"hauteur={sub['hauteur_mean'].median():.1f}m "
          f"année={sub['annee_construction'].median():.0f}")

# Cluster 1 — le fourre-tout
print("\n  CLUSTER 1 — les 'inconnus' (46k bâtiments)")
sub = gdf[gdf['cluster_label'] == 1]
print(f"    Surface: Q25={sub['surface_bat'].quantile(0.25):.0f} "
      f"médiane={sub['surface_bat'].median():.0f} "
      f"Q75={sub['surface_bat'].quantile(0.75):.0f}")
print(f"    Hauteur: médiane={sub['hauteur_mean'].median():.1f}m")
print(f"    Année construction connue: {sub['annee_construction'].notna().mean():.0%}")
print(f"    Nb logements connu: {sub['nb_logements'].notna().mean():.0%}")
print(f"    Usage principal:")
for usage, pct in sub['usage_principal'].value_counts(normalize=True).head(5).items():
    print(f"      {usage}: {pct:.0%}")

