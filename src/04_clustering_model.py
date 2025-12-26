"""
FraudGuard AI - Modelo de Clustering
Identificar padrões e grupos de comportamento nas transações
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*22 + "FRAUDGUARD AI - MODELO DE CLUSTERING")
print("="*80)

# ============================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================
print("\n📂 1. CARREGANDO DADOS...")
print("-"*80)

try:
    df = pd.read_csv('datasets/fraud/creditcard_processed.csv')
except:
    df = pd.read_csv('datasets/fraud/creditcard.csv')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

print(f"✅ Dados carregados: {len(df):,} transações")

# Preparar features (sem a classe)
X = df.drop(['Class', 'Amount', 'Time'], axis=1, errors='ignore')
y_true = df['Class']  # Para análise posterior

print(f"   Features: {X.shape[1]}")
print(f"   Fraudes reais: {y_true.sum():,} ({y_true.sum()/len(y_true)*100:.2f}%)")

# ============================================
# 2. AMOSTRAGEM (PARA VELOCIDADE)
# ============================================
print("\n🎲 2. AMOSTRAGEM DE DADOS")
print("-"*80)

# Usar uma amostra para clustering (K-Means é lento com muitos dados)
# Manter TODAS as fraudes + amostra das legítimas
fraud_data = df[df['Class'] == 1]
legit_data = df[df['Class'] == 0].sample(n=10000, random_state=42)

df_sample = pd.concat([fraud_data, legit_data]).sample(frac=1, random_state=42)  # Shuffle
X_sample = df_sample.drop(['Class', 'Amount', 'Time'], axis=1, errors='ignore')
y_sample = df_sample['Class']

print(f"✅ Amostra criada:")
print(f"   Total: {len(X_sample):,} transações")
print(f"   Fraudes: {y_sample.sum():,} ({y_sample.sum()/len(y_sample)*100:.2f}%)")
print(f"   Legítimas: {(y_sample==0).sum():,}")

# Normalizar dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

print(f"\n✅ Dados normalizados (StandardScaler)")

# ============================================
# 3. REDUÇÃO DE DIMENSIONALIDADE (PCA)
# ============================================
print("\n🔬 3. REDUÇÃO DE DIMENSIONALIDADE (PCA)")
print("-"*80)

# PCA para visualização e análise
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"✅ PCA aplicado:")
print(f"   2D: Variância explicada = {pca_2d.explained_variance_ratio_.sum()*100:.2f}%")
print(f"   3D: Variância explicada = {pca_3d.explained_variance_ratio_.sum()*100:.2f}%")

# Visualização PCA com classes reais
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA 2D - Por classe real
scatter1 = axes[0].scatter(X_pca_2d[y_sample==0, 0], X_pca_2d[y_sample==0, 1], 
                          c='green', alpha=0.3, s=10, label='Legítima')
scatter2 = axes[0].scatter(X_pca_2d[y_sample==1, 0], X_pca_2d[y_sample==1, 1], 
                          c='red', alpha=0.7, s=20, label='Fraude')
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('PCA 2D - Classes Reais', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PCA 2D - Densidade
axes[1].hexbin(X_pca_2d[:, 0], X_pca_2d[:, 1], gridsize=50, cmap='YlOrRd')
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA 2D - Densidade', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/clustering/pca_analysis.png', dpi=300, bbox_inches='tight')
print("\n💾 Gráfico salvo: models/clustering/pca_analysis.png")
plt.close()

# ============================================
# 4. DETERMINAR NÚMERO ÓTIMO DE CLUSTERS
# ============================================
print("\n📊 4. DETERMINANDO NÚMERO ÓTIMO DE CLUSTERS")
print("-"*80)

# Método do Cotovelo (Elbow Method)
print("\n🔍 Testando K-Means com diferentes números de clusters...")

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    print(f"   K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_scores[-1]:.4f}")

# Visualizar método do cotovelo
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico de Inertia
axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('Método do Cotovelo', fontweight='bold', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(k_range)

# Gráfico de Silhouette
axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por K', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(k_range)

plt.tight_layout()
plt.savefig('models/clustering/elbow_method.png', dpi=300, bbox_inches='tight')
print("\n💾 Gráfico salvo: models/clustering/elbow_method.png")
plt.close()

# Escolher melhor K (maior silhouette)
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n🎯 Melhor K sugerido: {best_k} (Silhouette Score: {max(silhouette_scores):.4f})")

# ============================================
# 5. TREINAR MODELOS DE CLUSTERING
# ============================================
print("\n🤖 5. TREINANDO MODELOS DE CLUSTERING")
print("-"*80)

results = {}

# === MODELO 1: K-MEANS (K ÓTIMO) ===
print(f"\n🎯 Treinando K-Means (K={best_k})...")
start_time = time.time()

kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
labels_kmeans_best = kmeans_best.fit_predict(X_scaled)
kmeans_time = time.time() - start_time

print(f"✅ K-Means treinado em {kmeans_time:.2f}s")

# Métricas
results[f'K-Means (K={best_k})'] = {
    'model': kmeans_best,
    'labels': labels_kmeans_best,
    'silhouette': silhouette_score(X_scaled, labels_kmeans_best),
    'davies_bouldin': davies_bouldin_score(X_scaled, labels_kmeans_best),
    'calinski_harabasz': calinski_harabasz_score(X_scaled, labels_kmeans_best),
    'time': kmeans_time
}

# === MODELO 2: K-MEANS (K=5) ===
print(f"\n🎯 Treinando K-Means (K=5) para comparação...")
start_time = time.time()

kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=300)
labels_kmeans_5 = kmeans_5.fit_predict(X_scaled)
kmeans_5_time = time.time() - start_time

print(f"✅ K-Means (K=5) treinado em {kmeans_5_time:.2f}s")

results['K-Means (K=5)'] = {
    'model': kmeans_5,
    'labels': labels_kmeans_5,
    'silhouette': silhouette_score(X_scaled, labels_kmeans_5),
    'davies_bouldin': davies_bouldin_score(X_scaled, labels_kmeans_5),
    'calinski_harabasz': calinski_harabasz_score(X_scaled, labels_kmeans_5),
    'time': kmeans_5_time
}

# === MODELO 3: DBSCAN ===
print(f"\n🔍 Treinando DBSCAN...")
start_time = time.time()

dbscan = DBSCAN(eps=3, min_samples=50, n_jobs=-1)
labels_dbscan = dbscan.fit_predict(X_scaled)
dbscan_time = time.time() - start_time

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"✅ DBSCAN treinado em {dbscan_time:.2f}s")
print(f"   Clusters encontrados: {n_clusters_dbscan}")
print(f"   Pontos de ruído: {n_noise}")

# Métricas (apenas se houver mais de 1 cluster)
if n_clusters_dbscan > 1:
    # Remover ruído para cálculo de métricas
    mask = labels_dbscan != -1
    X_no_noise = X_scaled[mask]
    labels_no_noise = labels_dbscan[mask]
    
    results['DBSCAN'] = {
        'model': dbscan,
        'labels': labels_dbscan,
        'n_clusters': n_clusters_dbscan,
        'n_noise': n_noise,
        'silhouette': silhouette_score(X_no_noise, labels_no_noise) if len(set(labels_no_noise)) > 1 else 0,
        'davies_bouldin': davies_bouldin_score(X_no_noise, labels_no_noise) if len(set(labels_no_noise)) > 1 else 999,
        'calinski_harabasz': calinski_harabasz_score(X_no_noise, labels_no_noise) if len(set(labels_no_noise)) > 1 else 0,
        'time': dbscan_time
    }

# ============================================
# 6. COMPARAR MODELOS
# ============================================
print("\n📊 6. COMPARAÇÃO DE MODELOS")
print("-"*80)

print("\n" + "="*95)
print(f"{'MODELO':<20} {'SILHOUETTE':>12} {'DAVIES-BOULDIN':>16} {'CALINSKI-H':>14} {'TEMPO':>10}")
print("="*95)

for model_name, metrics in results.items():
    print(f"{model_name:<20} "
          f"{metrics['silhouette']:>12.4f} "
          f"{metrics['davies_bouldin']:>16.4f} "
          f"{metrics['calinski_harabasz']:>14.2f} "
          f"{metrics['time']:>9.2f}s")

print("="*95)

# Melhor modelo (maior silhouette)
best_model_name = max(results, key=lambda x: results[x]['silhouette'])
best_labels = results[best_model_name]['labels']

print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"   Silhouette Score: {results[best_model_name]['silhouette']:.4f}")

# ============================================
# 7. ANÁLISE DOS CLUSTERS
# ============================================
print(f"\n🔍 7. ANÁLISE DOS CLUSTERS - {best_model_name}")
print("-"*80)

# Adicionar labels ao dataframe
df_sample['Cluster'] = best_labels

# Análise por cluster
print("\n📊 Distribuição por Cluster:\n")
print(f"{'Cluster':<10} {'Total':>10} {'Fraudes':>10} {'% Fraude':>12} {'Tamanho Médio':>15}")
print("-"*60)

for cluster in sorted(df_sample['Cluster'].unique()):
    if cluster == -1:  # Ruído (DBSCAN)
        continue
    
    cluster_data = df_sample[df_sample['Cluster'] == cluster]
    total = len(cluster_data)
    fraudes = cluster_data['Class'].sum()
    pct_fraude = (fraudes / total * 100) if total > 0 else 0
    
    print(f"{cluster:<10} {total:>10} {fraudes:>10} {pct_fraude:>11.2f}% {cluster_data['Class'].mean():>14.4f}")

# Identificar cluster mais fraudulento
cluster_fraud_rates = {}
for cluster in df_sample['Cluster'].unique():
    if cluster == -1:
        continue
    cluster_data = df_sample[df_sample['Cluster'] == cluster]
    fraud_rate = cluster_data['Class'].sum() / len(cluster_data)
    cluster_fraud_rates[cluster] = fraud_rate

if cluster_fraud_rates:
    most_fraudulent = max(cluster_fraud_rates, key=cluster_fraud_rates.get)
    print(f"\n🚨 CLUSTER MAIS FRAUDULENTO: Cluster {most_fraudulent} ({cluster_fraud_rates[most_fraudulent]*100:.2f}% fraude)")

# ============================================
# 8. VISUALIZAÇÕES
# ============================================
print("\n📈 8. GERANDO VISUALIZAÇÕES")
print("-"*80)

# === GRÁFICO 1: Clusters em PCA 2D ===
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Clusters (cores por cluster)
scatter = axes[0, 0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                            c=best_labels, cmap='viridis', alpha=0.6, s=15)
axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0, 0].set_title(f'Clusters - {best_model_name}', fontweight='bold', fontsize=14)
plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
axes[0, 0].grid(True, alpha=0.3)

# Classes reais
axes[0, 1].scatter(X_pca_2d[y_sample==0, 0], X_pca_2d[y_sample==0, 1], 
                  c='green', alpha=0.3, s=10, label='Legítima')
axes[0, 1].scatter(X_pca_2d[y_sample==1, 0], X_pca_2d[y_sample==1, 1], 
                  c='red', alpha=0.7, s=20, label='Fraude')
axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0, 1].set_title('Classes Reais', fontweight='bold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Distribuição de clusters
cluster_counts = pd.Series(best_labels).value_counts().sort_index()
cluster_counts.plot(kind='bar', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Quantidade')
axes[1, 0].set_title('Tamanho dos Clusters', fontweight='bold', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Taxa de fraude por cluster
fraud_rates = []
cluster_ids = []
for cluster in sorted(df_sample['Cluster'].unique()):
    if cluster == -1:
        continue
    cluster_data = df_sample[df_sample['Cluster'] == cluster]
    fraud_rate = cluster_data['Class'].sum() / len(cluster_data) * 100
    fraud_rates.append(fraud_rate)
    cluster_ids.append(cluster)

axes[1, 1].bar(cluster_ids, fraud_rates, color='coral')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('% de Fraudes')
axes[1, 1].set_title('Taxa de Fraude por Cluster', fontweight='bold', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_xticks(cluster_ids)

plt.tight_layout()
plt.savefig('models/clustering/cluster_analysis.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/clustering/cluster_analysis.png")
plt.close()

# === GRÁFICO 2: Clusters 3D ===
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 7))

# 3D - Clusters
ax1 = fig.add_subplot(121, projection='3d')
scatter = ax1.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=best_labels, cmap='viridis', alpha=0.6, s=10)
ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax1.set_title(f'Clusters 3D - {best_model_name}', fontweight='bold', fontsize=14)
plt.colorbar(scatter, ax=ax1, label='Cluster', shrink=0.5)

# 3D - Classes reais
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_pca_3d[y_sample==0, 0], X_pca_3d[y_sample==0, 1], X_pca_3d[y_sample==0, 2],
           c='green', alpha=0.3, s=5, label='Legítima')
ax2.scatter(X_pca_3d[y_sample==1, 0], X_pca_3d[y_sample==1, 1], X_pca_3d[y_sample==1, 2],
           c='red', alpha=0.7, s=15, label='Fraude')
ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax2.set_title('Classes Reais 3D', fontweight='bold', fontsize=14)
ax2.legend()

plt.tight_layout()
plt.savefig('models/clustering/cluster_3d.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/clustering/cluster_3d.png")
plt.close()

# ============================================
# 9. SALVAR MODELO
# ============================================
print(f"\n💾 9. SALVANDO MODELO E OBJETOS")
print("-"*80)

# Salvar modelo de clustering
model_path = 'models/clustering/pattern_analyzer.pkl'
joblib.dump(results[best_model_name]['model'], model_path)
print(f"✅ Modelo salvo: {model_path}")

# Salvar scaler
scaler_path = 'models/clustering/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler salvo: {scaler_path}")

# Salvar PCA
pca_path = 'models/clustering/pca.pkl'
joblib.dump(pca_2d, pca_path)
print(f"✅ PCA salvo: {pca_path}")

# Salvar informações
import json
cluster_info = {}
for cluster in sorted(df_sample['Cluster'].unique()):
    if cluster == -1:
        continue
    cluster_data = df_sample[df_sample['Cluster'] == cluster]
    cluster_info[int(cluster)] = {
        'size': int(len(cluster_data)),
        'fraud_count': int(cluster_data['Class'].sum()),
        'fraud_rate': float(cluster_data['Class'].sum() / len(cluster_data)),
        'risk_level': 'HIGH' if (cluster_data['Class'].sum() / len(cluster_data)) > 0.1 else 
                     'MEDIUM' if (cluster_data['Class'].sum() / len(cluster_data)) > 0.01 else 'LOW'
    }

model_info = {
    'model_name': best_model_name,
    'metrics': {
        'silhouette_score': results[best_model_name]['silhouette'],
        'davies_bouldin_score': results[best_model_name]['davies_bouldin'],
        'calinski_harabasz_score': results[best_model_name]['calinski_harabasz']
    },
    'n_clusters': len(cluster_info),
    'cluster_info': cluster_info,
    'feature_names': X.columns.tolist()
}

with open('models/clustering/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)
print(f"✅ Informações salvas: models/clustering/model_info.json")

# ============================================
# 10. RESUMO FINAL
# ============================================
print("\n" + "="*80)
print(" "*25 + "RESUMO FINAL - CLUSTERING")
print("="*80)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    RESULTADOS DO CLUSTERING                           ║
╠══════════════════════════════════════════════════════════════════════╣
║ 🏆 MELHOR MODELO: {best_model_name:<48}       ║
║                                                                       ║
║ 📊 MÉTRICAS DE QUALIDADE:                                            ║
║    • Silhouette Score:        {results[best_model_name]['silhouette']:>6.4f} (quanto maior, melhor)       ║
║    • Davies-Bouldin Score:    {results[best_model_name]['davies_bouldin']:>6.4f} (quanto menor, melhor)       ║
║    • Calinski-Harabasz Score: {results[best_model_name]['calinski_harabasz']:>8.2f} (quanto maior, melhor)     ║
║                                                                       ║
║ 🎯 CLUSTERS IDENTIFICADOS: {len(cluster_info):>2}                                        ║
║                                                                       ║
║ ⏱️  TEMPO DE TREINAMENTO: {results[best_model_name]['time']:>6.2f}s                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n💡 INTERPRETAÇÃO DOS CLUSTERS:")
print("-"*80)
for cluster_id, info in cluster_info.items():
    risk_emoji = '🚨' if info['risk_level'] == 'HIGH' else '⚠️' if info['risk_level'] == 'MEDIUM' else '✅'
    print(f"\n{risk_emoji} CLUSTER {cluster_id} - Risco {info['risk_level']}:")
    print(f"   • Tamanho: {info['size']:,} transações")
    print(f"   • Fraudes: {info['fraud_count']:,} ({info['fraud_rate']*100:.2f}%)")
    print(f"   • Recomendação: {'Bloquear automaticamente' if info['risk_level']=='HIGH' else 'Revisar manualmente' if info['risk_level']=='MEDIUM' else 'Processar normalmente'}")

print("\n💡 USO PRÁTICO:")
print("-"*80)
print("""
Os clusters identificam PADRÕES DE COMPORTAMENTO:

• Clusters com alta taxa de fraude (>10%): Padrões suspeitos
• Clusters com baixa taxa (<1%): Comportamento normal
• Clusters médios (1-10%): Requerem análise adicional

🎯 INTEGRAÇÃO COM OUTROS MODELOS:
   1. Classificação diz: "É fraude ou não?"
   2. Regressão diz: "Qual o score de risco (0-100)?"
   3. Clustering diz: "A que padrão de comportamento pertence?"

🎯 PRÓXIMO PASSO: Modelo de Visão Computacional (OCR de dígitos)
""")

print("="*80)
print("✅ MODELO DE CLUSTERING CONCLUÍDO COM SUCESSO!")
print("="*80)
print("\n🚀 Execute agora: python src/05_computer_vision.py")
print("="*80)