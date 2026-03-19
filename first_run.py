import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
import json
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger()

# Config
cfg = yaml.safe_load(open('config.yaml'))
codes = cfg['zone']['codes_insee']
codes_sql = "','".join(codes)

# 1. Charger les données filtrées
logger.info("Chargement BDNB Bordeaux Métropole...")
gdf = gpd.read_file(
    cfg['data']['gpkg_path'],
    layer=cfg['data']['layer'],
    where=f"code_commune_insee IN ('{codes_sql}')"
)
logger.info(f"  {len(gdf)} bâtiments chargés")

# 2. Features numériques
num_mapping = {
    's_geom_groupe': 'surface_bat',
    'bdtopo_bat_hauteur_mean': 'hauteur_mean',
    'bdtopo_bat_max_hauteur': 'hauteur_max',
    'ffo_bat_nb_niveau': 'nb_niveaux',
    'ffo_bat_annee_construction': 'annee_construction',
    'ffo_bat_nb_log': 'nb_logements',
}
cat_mapping = {
    'usage_principal_bdnb_open': 'usage_principal',
    'ffo_bat_usage_niveau_1_txt': 'usage_foncier',
    'ffo_bat_mat_mur_txt': 'mat_murs',
    'bdtopo_bat_l_nature': 'nature_bdtopo',
    'bdtopo_bat_l_usage_1': 'usage_bdtopo',
}

# Renommer
for real, projet in {**num_mapping, **cat_mapping}.items():
    if real in gdf.columns:
        gdf[projet] = gdf[real]

# 3. Filtrer surface
gdf = gdf[(gdf['surface_bat'] >= 20) & (gdf['surface_bat'] <= 10000)]
logger.info(f"  {len(gdf)} après filtre surface")

# 4. Traiter les "nan" strings
for col in cat_mapping.values():
    if col in gdf.columns:
        gdf[col] = gdf[col].replace('nan', np.nan)

# 5. Ne garder que les lignes avec au moins les features numériques principales
key_features = ['surface_bat', 'hauteur_mean']
gdf = gdf.dropna(subset=key_features)
logger.info(f"  {len(gdf)} après dropna sur features clés")

# 6. Préparer les features
num_cols = [c for c in num_mapping.values() if c in gdf.columns]
cat_cols = [c for c in cat_mapping.values() if c in gdf.columns]

# Imputer les numériques par la médiane
for col in num_cols:
    gdf[col] = gdf[col].fillna(gdf[col].median())

# Encoder les catégorielles
for col in cat_cols:
    gdf[col] = gdf[col].fillna('inconnu')
    le = LabelEncoder()
    gdf[col + '_enc'] = le.fit_transform(gdf[col].astype(str))

feature_cols = num_cols + [c + '_enc' for c in cat_cols]
X = gdf[feature_cols].values

# 7. Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logger.info(f"  Features : {len(feature_cols)} colonnes, {X_scaled.shape[0]} lignes")

# 8. Clustering — 3 algorithmes
results = {}
target_k = cfg['clustering']['target_k']

# K-Means
logger.info("KMeans k=%d ...", target_k)
km = KMeans(n_clusters=target_k, random_state=42, n_init=10)
labels_km = km.fit_predict(X_scaled)
results['kmeans'] = labels_km

# GMM
logger.info("GaussianMixture k=%d ...", target_k)
gmm = GaussianMixture(n_components=target_k, random_state=42, covariance_type='full')
labels_gmm = gmm.fit_predict(X_scaled)
results['gmm'] = labels_gmm

# Agglomerative — échantillonné si trop gros
if len(X_scaled) > 10000:
    logger.info("Agglomerative sur échantillon 50k...")
    idx = np.random.RandomState(42).choice(len(X_scaled), 10000, replace=False)
    X_sample = X_scaled[idx]
    agg = AgglomerativeClustering(n_clusters=target_k, linkage='ward')
    labels_agg_sample = agg.fit_predict(X_sample)
    # Réaffecter les autres au centroïde le plus proche
    from sklearn.neighbors import NearestCentroid
    nc = NearestCentroid()
    nc.fit(X_sample, labels_agg_sample)
    labels_agg = nc.predict(X_scaled)
    results['agglomerative'] = labels_agg
else:
    agg = AgglomerativeClustering(n_clusters=target_k, linkage='ward')
    results['agglomerative'] = agg.fit_predict(X_scaled)

# 9. Évaluer
logger.info("\n" + "=" * 70)
logger.info("RÉSULTATS")
logger.info("=" * 70)

sample_size = min(10000, len(X_scaled))
best_algo = None
best_sil = -1

for algo, labels in results.items():
    sil = silhouette_score(X_scaled, labels, sample_size=sample_size)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    logger.info(f"  {algo:<18} silhouette={sil:.3f}  DB={db:.2f}  CH={ch:.0f}")
    if sil > best_sil:
        best_sil = sil
        best_algo = algo

logger.info(f"\n  >>> Meilleur : {best_algo} (silhouette={best_sil:.3f})")

# 10. Sauvegarder le meilleur
gdf['cluster_label'] = results[best_algo]
gdf['cluster_algo'] = best_algo

# Stats par cluster
logger.info("\nProfil des clusters :")
for cl in sorted(gdf['cluster_label'].unique()):
    sub = gdf[gdf['cluster_label'] == cl]
    logger.info(
        f"  Cluster {cl:>2} : n={len(sub):>6}  "
        f"surf={sub['surface_bat'].median():>6.0f}m²  "
        f"haut={sub['hauteur_mean'].median():>4.1f}m  "
        f"usage={sub['usage_principal'].mode().iloc[0] if not sub['usage_principal'].mode().empty else '?'}"
    )

# 11. Sauvegarder
import os; os.makedirs("data/processed", exist_ok=True)
output = "data/processed/clustered.geoparquet"
gdf.to_parquet(output)
logger.info(f"\nSauvegardé : {output}")

# 12. Log experiment
exp_path = 'outputs/experiments.json'
with open(exp_path) as f:
    experiments = json.load(f)

for algo, labels in results.items():
    sil = silhouette_score(X_scaled, labels, sample_size=sample_size)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    experiments['runs'].append({
        'id': f'run_{len(experiments["runs"])+1:03d}',
        'date': datetime.now().isoformat(),
        'algorithm': algo,
        'params': {'n_clusters': target_k},
        'n_clusters': target_k,
        'n_noise': 0,
        'noise_pct': 0,
        'features_used': feature_cols,
        'n_features': len(feature_cols),
        'n_samples': len(labels),
        'metrics': {
            'silhouette': round(sil, 4),
            'davies_bouldin': round(db, 4),
            'calinski_harabasz': round(ch, 1)
        },
        'data_sources': ['bdnb'],
        'notes': 'Premier run — BDNB seule'
    })

with open(exp_path, 'w') as f:
    json.dump(experiments, f, indent=2, ensure_ascii=False)

logger.info(f"Expériences loggées : {exp_path}")
logger.info("TERMINÉ ✓")
