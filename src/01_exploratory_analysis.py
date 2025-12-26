"""
FraudGuard AI - Análise Exploratória de Dados
Análise completa do dataset de fraudes bancárias
"""

import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*20 + "FRAUDGUARD AI - ANÁLISE EXPLORATÓRIA")
print("="*80)

# ============================================
# 1. CARREGAR DADOS
# ============================================
print("\n📂 1. CARREGANDO DADOS...")
print("-"*80)

try:
    df = pd.read_csv('datasets/fraud/creditcard.csv')
    print(f"✅ Dataset carregado com sucesso!")
    print(f"   - Total de transações: {len(df):,}")
    print(f"   - Total de features: {len(df.columns)}")
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    exit(1)

# ============================================
# 2. INFORMAÇÕES GERAIS
# ============================================
print("\n📊 2. INFORMAÇÕES GERAIS DO DATASET")
print("-"*80)

print("\n🔹 Primeiras 5 linhas:")
print(df.head())

print("\n🔹 Informações das colunas:")
print(df.info())

print("\n🔹 Estatísticas descritivas:")
print(df.describe())

print("\n🔹 Valores nulos:")
print(df.isnull().sum())

# ============================================
# 3. ANÁLISE DE FRAUDES
# ============================================
print("\n🎯 3. ANÁLISE DE FRAUDES")
print("-"*80)

fraud_count = df['Class'].value_counts()
fraud_percentage = df['Class'].value_counts(normalize=True) * 100

print(f"\n🔹 Distribuição de Classes:")
print(f"   Legítimas (0): {fraud_count[0]:,} ({fraud_percentage[0]:.2f}%)")
print(f"   Fraudes (1):   {fraud_count[1]:,} ({fraud_percentage[1]:.2f}%)")
print(f"\n   ⚠️  Dataset ALTAMENTE DESBALANCEADO!")
print(f"   Razão: 1 fraude para cada {fraud_count[0]//fraud_count[1]} transações legítimas")

# Gráfico de pizza
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
colors = ['#2ecc71', '#e74c3c']
explode = (0, 0.1)
plt.pie(fraud_count, labels=['Legítima', 'Fraude'], autopct='%1.2f%%', 
        colors=colors, explode=explode, shadow=True, startangle=90)
plt.title('Distribuição de Classes', fontsize=14, fontweight='bold')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Class', palette=colors)
plt.title('Contagem de Transações', fontsize=14, fontweight='bold')
plt.xlabel('Classe (0=Legítima, 1=Fraude)')
plt.ylabel('Quantidade')
plt.xticks([0, 1], ['Legítima', 'Fraude'])

plt.tight_layout()
plt.savefig('datasets/fraud/01_class_distribution.png', dpi=300, bbox_inches='tight')
print("\n💾 Gráfico salvo: datasets/fraud/01_class_distribution.png")
plt.close()

# ============================================
# 4. ANÁLISE DO VALOR DAS TRANSAÇÕES
# ============================================
print("\n💰 4. ANÁLISE DO VALOR DAS TRANSAÇÕES")
print("-"*80)

print("\n🔹 Estatísticas do valor (Amount):")
print(df['Amount'].describe())

print(f"\n🔹 Comparação Legítimas vs Fraudes:")
print(f"   Valor médio - Legítimas: ${df[df['Class']==0]['Amount'].mean():.2f}")
print(f"   Valor médio - Fraudes:   ${df[df['Class']==1]['Amount'].mean():.2f}")
print(f"   Valor máximo - Legítimas: ${df[df['Class']==0]['Amount'].max():.2f}")
print(f"   Valor máximo - Fraudes:   ${df[df['Class']==1]['Amount'].max():.2f}")

# Gráficos de distribuição de valores
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histograma geral
axes[0, 0].hist(df['Amount'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribuição de Valores - Todas as Transações', fontweight='bold')
axes[0, 0].set_xlabel('Valor ($)')
axes[0, 0].set_ylabel('Frequência')
axes[0, 0].set_yscale('log')

# Boxplot por classe
df.boxplot(column='Amount', by='Class', ax=axes[0, 1])
axes[0, 1].set_title('Distribuição de Valores por Classe', fontweight='bold')
axes[0, 1].set_xlabel('Classe (0=Legítima, 1=Fraude)')
axes[0, 1].set_ylabel('Valor ($)')
plt.sca(axes[0, 1])
plt.xticks([1, 2], ['Legítima', 'Fraude'])

# Histograma legítimas
axes[1, 0].hist(df[df['Class']==0]['Amount'], bins=50, color='#2ecc71', 
                edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribuição - Transações Legítimas', fontweight='bold')
axes[1, 0].set_xlabel('Valor ($)')
axes[1, 0].set_ylabel('Frequência')
axes[1, 0].set_yscale('log')

# Histograma fraudes
axes[1, 1].hist(df[df['Class']==1]['Amount'], bins=50, color='#e74c3c', 
                edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Distribuição - Fraudes', fontweight='bold')
axes[1, 1].set_xlabel('Valor ($)')
axes[1, 1].set_ylabel('Frequência')

plt.tight_layout()
plt.savefig('datasets/fraud/02_amount_distribution.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: datasets/fraud/02_amount_distribution.png")
plt.close()

# ============================================
# 5. ANÁLISE TEMPORAL
# ============================================
print("\n⏰ 5. ANÁLISE TEMPORAL")
print("-"*80)

print("\n🔹 Estatísticas de Tempo (Time):")
print(df['Time'].describe())

# Converter tempo em horas
df['Hour'] = (df['Time'] / 3600) % 24

print(f"\n🔹 Transações por período:")
fraud_by_hour = df.groupby('Hour')['Class'].agg(['sum', 'count'])
fraud_by_hour.columns = ['Fraudes', 'Total']
fraud_by_hour['Taxa'] = (fraud_by_hour['Fraudes'] / fraud_by_hour['Total'] * 100)

print(fraud_by_hour)

# Gráfico temporal
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Transações ao longo do tempo
axes[0].plot(df[df['Class']==0]['Time'], df[df['Class']==0]['Amount'], 
             'g.', alpha=0.1, label='Legítima', markersize=1)
axes[0].plot(df[df['Class']==1]['Time'], df[df['Class']==1]['Amount'], 
             'r.', alpha=0.5, label='Fraude', markersize=3)
axes[0].set_title('Transações ao Longo do Tempo', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Tempo (segundos)')
axes[0].set_ylabel('Valor ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Fraudes por hora
hour_counts = df.groupby(['Hour', 'Class']).size().unstack(fill_value=0)
hour_counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], width=0.8)
axes[1].set_title('Distribuição de Transações por Hora', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Hora do Dia')
axes[1].set_ylabel('Número de Transações')
axes[1].legend(['Legítima', 'Fraude'])
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('datasets/fraud/03_temporal_analysis.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: datasets/fraud/03_temporal_analysis.png")
plt.close()

# ============================================
# 6. ANÁLISE DAS FEATURES V1-V28
# ============================================
print("\n🔬 6. ANÁLISE DAS FEATURES V1-V28 (PCA)")
print("-"*80)

# Selecionar apenas as features V
v_features = [col for col in df.columns if col.startswith('V')]

print(f"\n🔹 Total de features PCA: {len(v_features)}")

# Correlação média com a classe
correlations = df[v_features + ['Class']].corr()['Class'].drop('Class').abs().sort_values(ascending=False)

print(f"\n🔹 Top 10 features mais correlacionadas com Fraude:")
for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
    print(f"   {i:2d}. {feature}: {corr:.4f}")

# Heatmap das correlações
plt.figure(figsize=(12, 10))
correlation_matrix = df[v_features[:14]].corr()  # Primeiras 14 features
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Mapa de Correlação - Features V1-V14', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('datasets/fraud/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n💾 Gráfico salvo: datasets/fraud/04_correlation_heatmap.png")
plt.close()

# Distribuição das top features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(correlations.head(6).index):
    # Legítimas
    axes[i].hist(df[df['Class']==0][feature], bins=50, alpha=0.5, 
                 color='#2ecc71', label='Legítima', density=True)
    # Fraudes
    axes[i].hist(df[df['Class']==1][feature], bins=50, alpha=0.5, 
                 color='#e74c3c', label='Fraude', density=True)
    axes[i].set_title(f'{feature} (corr: {correlations[feature]:.3f})', fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('datasets/fraud/05_top_features.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: datasets/fraud/05_top_features.png")
plt.close()

# ============================================
# 7. PREPARAÇÃO DOS DADOS PARA MODELAGEM
# ============================================
print("\n🔧 7. PREPARAÇÃO DOS DADOS")
print("-"*80)

# Normalizar Amount e Time
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

print("✅ Features 'Amount' e 'Time' normalizadas")

# Separar features e target
X = df.drop(['Class', 'Amount', 'Time', 'Hour'], axis=1)
y = df['Class']

print(f"\n🔹 Shape final dos dados:")
print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# Salvar dados processados
df.to_csv('datasets/fraud/creditcard_processed.csv', index=False)
print("\n💾 Dados processados salvos: datasets/fraud/creditcard_processed.csv")

# ============================================
# 8. RESUMO ESTATÍSTICO
# ============================================
print("\n📈 8. RESUMO ESTATÍSTICO FINAL")
print("-"*80)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    RESUMO DO DATASET                          ║
╠══════════════════════════════════════════════════════════════╣
║ Total de transações:        {len(df):>10,}                      ║
║ Transações legítimas:       {fraud_count[0]:>10,} ({fraud_percentage[0]:>5.2f}%)      ║
║ Transações fraudulentas:    {fraud_count[1]:>10,} ({fraud_percentage[1]:>5.2f}%)       ║
║                                                              ║
║ Valor médio - Legítimas:    ${df[df['Class']==0]['Amount'].mean():>10,.2f}              ║
║ Valor médio - Fraudes:      ${df[df['Class']==1]['Amount'].mean():>10,.2f}              ║
║                                                              ║
║ Features totais:            {len(X.columns):>10}                          ║
║ Features PCA (V1-V28):      {len(v_features):>10}                          ║
║                                                              ║
║ Top 3 features correlacionadas:                             ║
║   1. {correlations.index[0]:>10}: {correlations.iloc[0]:>6.4f}                          ║
║   2. {correlations.index[1]:>10}: {correlations.iloc[1]:>6.4f}                          ║
║   3. {correlations.index[2]:>10}: {correlations.iloc[2]:>6.4f}                          ║
╚══════════════════════════════════════════════════════════════╝
""")

# ============================================
# 9. CONCLUSÕES E PRÓXIMOS PASSOS
# ============================================
print("\n💡 9. CONCLUSÕES E INSIGHTS")
print("-"*80)

print("""
🔍 INSIGHTS PRINCIPAIS:

1. DESBALANCEAMENTO EXTREMO:
   ⚠️  Apenas 0.17% das transações são fraudes
   → Precisaremos usar técnicas de balanceamento (SMOTE, undersampling)
   → Métricas como Precision, Recall e F1-Score são mais importantes que Accuracy

2. CARACTERÍSTICAS DAS FRAUDES:
   💰 Fraudes tendem a ter valores médios MENORES que transações legítimas
   ⏰ Padrões temporais podem indicar horários de maior risco
   
3. FEATURES MAIS RELEVANTES:
   📊 V14, V4, V11, V12 e V10 são as mais correlacionadas com fraude
   → Essas features serão críticas para nossos modelos

4. NORMALIZAÇÃO NECESSÁRIA:
   🔧 Amount e Time possuem escalas muito diferentes das features V
   → Já normalizamos essas variáveis para melhorar o desempenho dos modelos

🎯 PRÓXIMOS PASSOS:

   ✅ Análise Exploratória: CONCLUÍDA
   → Modelo de Classificação (Random Forest, XGBoost, SVM)
   → Modelo de Regressão (Score de Risco)
   → Modelo de Clustering (Padrões de fraude)
   → Modelo de Visão Computacional (OCR de dígitos)
   → Integração e Deploy Web
""")

print("\n" + "="*80)
print(" "*25 + "ANÁLISE CONCLUÍDA COM SUCESSO! ✅")
print("="*80)
print("\n📂 Arquivos gerados:")
print("   • datasets/fraud/creditcard_processed.csv")
print("   • datasets/fraud/01_class_distribution.png")
print("   • datasets/fraud/02_amount_distribution.png")
print("   • datasets/fraud/03_temporal_analysis.png")
print("   • datasets/fraud/04_correlation_heatmap.png")
print("   • datasets/fraud/05_top_features.png")
print("\n🚀 Execute agora: python src/02_classification_model.py")
print("="*80)