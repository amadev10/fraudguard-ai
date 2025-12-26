"""
FraudGuard AI - Modelo de Regressão
Calcular Score de Risco (0-100) para cada transação
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*22 + "FRAUDGUARD AI - MODELO DE REGRESSÃO")
print("="*80)

# ============================================
# 1. CARREGAR DADOS E PREPARAR TARGET
# ============================================
print("\n📂 1. CARREGANDO E PREPARANDO DADOS...")
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

# Preparar features
X = df.drop(['Class', 'Amount', 'Time'], axis=1, errors='ignore')

# CRIAR TARGET: Score de Risco (0-100)
# Vamos usar a probabilidade do modelo de classificação + features
print("\n🎯 Criando Score de Risco...")

# Carregar modelo de classificação treinado
clf_model = joblib.load('models/classification/fraud_classifier.pkl')

# Calcular probabilidades de fraude
fraud_probabilities = clf_model.predict_proba(X)[:, 1]

# Criar score base (0-100)
risk_score_base = fraud_probabilities * 100

# Adicionar fatores de risco baseados em features
# Exemplo: valores muito altos ou muito baixos aumentam risco
if 'Amount_scaled' in df.columns:
    amount_factor = np.abs(df['Amount_scaled']) * 5  # Valores extremos = mais risco
else:
    amount_factor = 0

# Score final (0-100)
y = np.clip(risk_score_base + amount_factor, 0, 100)

print(f"✅ Score de Risco criado!")
print(f"   • Média: {y.mean():.2f}")
print(f"   • Mediana: {np.median(y):.2f}")
print(f"   • Min: {y.min():.2f}, Max: {y.max():.2f}")
print(f"   • Desvio padrão: {y.std():.2f}")

# Distribuição por classe
print(f"\n📊 Score por classe:")
print(f"   • Legítimas (Class=0): {y[df['Class']==0].mean():.2f} ± {y[df['Class']==0].std():.2f}")
print(f"   • Fraudes (Class=1):   {y[df['Class']==1].mean():.2f} ± {y[df['Class']==1].std():.2f}")

# ============================================
# 2. DIVIDIR DADOS
# ============================================
print("\n🔀 2. DIVIDINDO DADOS EM TREINO E TESTE")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Dados divididos:")
print(f"   Treino: {len(X_train):,} amostras")
print(f"   Teste:  {len(X_test):,} amostras")

# ============================================
# 3. TREINAR MODELOS DE REGRESSÃO
# ============================================
print("\n🤖 3. TREINANDO MODELOS DE REGRESSÃO")
print("-"*80)

models = {}
results = {}

# === MODELO 1: RANDOM FOREST REGRESSOR ===
print("\n🌲 Treinando Random Forest Regressor...")
start_time = time.time()

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"✅ Random Forest treinado em {rf_time:.2f}s")

# Predições
y_pred_rf = rf_reg.predict(X_test)
y_pred_rf = np.clip(y_pred_rf, 0, 100)  # Garantir 0-100

# Métricas
results['Random Forest'] = {
    'model': rf_reg,
    'predictions': y_pred_rf,
    'mse': mean_squared_error(y_test, y_pred_rf),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'r2': r2_score(y_test, y_pred_rf),
    'time': rf_time
}

# === MODELO 2: GRADIENT BOOSTING ===
print("\n⚡ Treinando Gradient Boosting Regressor...")
start_time = time.time()

gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)

gb_reg.fit(X_train, y_train)
gb_time = time.time() - start_time

print(f"✅ Gradient Boosting treinado em {gb_time:.2f}s")

# Predições
y_pred_gb = gb_reg.predict(X_test)
y_pred_gb = np.clip(y_pred_gb, 0, 100)

# Métricas
results['Gradient Boosting'] = {
    'model': gb_reg,
    'predictions': y_pred_gb,
    'mse': mean_squared_error(y_test, y_pred_gb),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'mae': mean_absolute_error(y_test, y_pred_gb),
    'r2': r2_score(y_test, y_pred_gb),
    'time': gb_time
}

# === MODELO 3: RIDGE REGRESSION ===
print("\n📊 Treinando Ridge Regression...")
start_time = time.time()

ridge_reg = Ridge(alpha=1.0, random_state=42)

ridge_reg.fit(X_train, y_train)
ridge_time = time.time() - start_time

print(f"✅ Ridge Regression treinado em {ridge_time:.2f}s")

# Predições
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_ridge = np.clip(y_pred_ridge, 0, 100)

# Métricas
results['Ridge Regression'] = {
    'model': ridge_reg,
    'predictions': y_pred_ridge,
    'mse': mean_squared_error(y_test, y_pred_ridge),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'mae': mean_absolute_error(y_test, y_pred_ridge),
    'r2': r2_score(y_test, y_pred_ridge),
    'time': ridge_time
}

# ============================================
# 4. COMPARAR MODELOS
# ============================================
print("\n📊 4. COMPARAÇÃO DE MODELOS")
print("-"*80)

print("\n" + "="*95)
print(f"{'MODELO':<20} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'R² SCORE':>12} {'TEMPO':>10}")
print("="*95)

for model_name, metrics in results.items():
    print(f"{model_name:<20} "
          f"{metrics['mse']:>12.4f} "
          f"{metrics['rmse']:>12.4f} "
          f"{metrics['mae']:>12.4f} "
          f"{metrics['r2']:>12.4f} "
          f"{metrics['time']:>9.2f}s")

print("="*95)

# Melhor modelo (menor RMSE)
best_model_name = min(results, key=lambda x: results[x]['rmse'])
best_model = results[best_model_name]['model']

print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"   • RMSE: {results[best_model_name]['rmse']:.4f}")
print(f"   • MAE: {results[best_model_name]['mae']:.4f}")
print(f"   • R²: {results[best_model_name]['r2']:.4f}")

# ============================================
# 5. ANÁLISE DE ERROS
# ============================================
print(f"\n🔍 5. ANÁLISE DE ERROS - {best_model_name}")
print("-"*80)

y_pred_best = results[best_model_name]['predictions']
errors = y_test - y_pred_best

print(f"\n📊 Estatísticas dos Erros:")
print(f"   • Erro médio: {errors.mean():.4f}")
print(f"   • Erro absoluto médio: {np.abs(errors).mean():.4f}")
print(f"   • Erro máximo: {np.abs(errors).max():.4f}")
print(f"   • Desvio padrão: {errors.std():.4f}")

# Percentis de erro
print(f"\n📈 Distribuição de Erros Absolutos:")
for percentil in [50, 75, 90, 95, 99]:
    error_val = np.percentile(np.abs(errors), percentil)
    print(f"   • {percentil}º percentil: {error_val:.2f} pontos")

# ============================================
# 6. VISUALIZAÇÕES
# ============================================
print("\n📈 6. GERANDO VISUALIZAÇÕES")
print("-"*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# === GRÁFICO 1: Comparação de Modelos ===
metrics_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'RMSE': [r['rmse'] for r in results.values()],
    'MAE': [r['mae'] for r in results.values()],
    'R²': [r['r2'] for r in results.values()],
})

metrics_df.plot(x='Modelo', y=['RMSE', 'MAE'], kind='bar', ax=axes[0, 0], rot=15)
axes[0, 0].set_title('Comparação de Erros', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Erro')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# === GRÁFICO 2: R² Score ===
metrics_df.plot(x='Modelo', y='R²', kind='bar', ax=axes[0, 1], 
                rot=15, color='coral', legend=False)
axes[0, 1].set_title('R² Score', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('R²')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1)

# === GRÁFICO 3: Predito vs Real ===
axes[0, 2].scatter(y_test, y_pred_best, alpha=0.5, s=10)
axes[0, 2].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfeito')
axes[0, 2].set_xlabel('Score Real')
axes[0, 2].set_ylabel('Score Predito')
axes[0, 2].set_title(f'Predito vs Real - {best_model_name}', fontweight='bold', fontsize=12)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xlim([0, 100])
axes[0, 2].set_ylim([0, 100])

# === GRÁFICO 4: Distribuição de Erros ===
axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Erro (Real - Predito)')
axes[1, 0].set_ylabel('Frequência')
axes[1, 0].set_title('Distribuição de Erros', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# === GRÁFICO 5: Erro Absoluto ===
abs_errors = np.abs(errors)
axes[1, 1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1, 1].set_xlabel('Erro Absoluto')
axes[1, 1].set_ylabel('Frequência')
axes[1, 1].set_title('Distribuição de Erros Absolutos', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# === GRÁFICO 6: Resíduos ===
axes[1, 2].scatter(y_pred_best, errors, alpha=0.5, s=10)
axes[1, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Score Predito')
axes[1, 2].set_ylabel('Resíduo (Real - Predito)')
axes[1, 2].set_title('Gráfico de Resíduos', fontweight='bold', fontsize=12)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/regression/regression_analysis.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/regression/regression_analysis.png")
plt.close()

# === GRÁFICO 7: Score por Categoria de Risco ===
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Categorizar scores
def categorize_risk(score):
    if score < 20:
        return 'Muito Baixo'
    elif score < 40:
        return 'Baixo'
    elif score < 60:
        return 'Médio'
    elif score < 80:
        return 'Alto'
    else:
        return 'Muito Alto'

risk_categories_real = [categorize_risk(s) for s in y_test]
risk_categories_pred = [categorize_risk(s) for s in y_pred_best]

# Distribuição real
pd.Series(risk_categories_real).value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color='steelblue', rot=45)
axes[0].set_title('Distribuição Real de Risco', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Quantidade')
axes[0].grid(True, alpha=0.3, axis='y')

# Distribuição predita
pd.Series(risk_categories_pred).value_counts().sort_index().plot(
    kind='bar', ax=axes[1], color='coral', rot=45)
axes[1].set_title('Distribuição Predita de Risco', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Quantidade')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/regression/risk_distribution.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/regression/risk_distribution.png")
plt.close()

# ============================================
# 7. SALVAR MODELO
# ============================================
print(f"\n💾 7. SALVANDO MELHOR MODELO")
print("-"*80)

# Salvar modelo
model_path = 'models/regression/risk_predictor.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Modelo salvo: {model_path}")

# Salvar scaler de scores (para normalização)
score_info = {
    'min': float(y.min()),
    'max': float(y.max()),
    'mean': float(y.mean()),
    'std': float(y.std())
}

import json
model_info = {
    'model_name': best_model_name,
    'metrics': {
        'rmse': results[best_model_name]['rmse'],
        'mae': results[best_model_name]['mae'],
        'r2_score': results[best_model_name]['r2']
    },
    'score_info': score_info,
    'feature_names': X.columns.tolist()
}

with open('models/regression/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)
print(f"✅ Informações salvas: models/regression/model_info.json")

# ============================================
# 8. EXEMPLOS DE PREDIÇÕES
# ============================================
print(f"\n🧪 8. EXEMPLOS DE PREDIÇÕES")
print("-"*80)

# Selecionar amostras de diferentes níveis de risco
sample_indices = []
for category in ['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto']:
    category_indices = [i for i, cat in enumerate(risk_categories_real) if cat == category]
    if category_indices:
        sample_indices.append(np.random.choice(category_indices))

print("\n📋 Exemplos de Scores de Risco:\n")
print(f"{'Score Real':<12} {'Score Predito':<15} {'Erro':<10} {'Categoria':<15}")
print("-"*60)

for idx in sample_indices:
    real = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
    pred = y_pred_best[idx]
    error = abs(real - pred)
    category = categorize_risk(real)
    
    print(f"{real:>6.2f}       {pred:>6.2f}          {error:>6.2f}     {category:<15}")

# ============================================
# 9. INTERPRETAÇÃO DOS SCORES
# ============================================
print("\n" + "="*80)
print(" "*25 + "INTERPRETAÇÃO DOS SCORES")
print("="*80)

print("""
📊 CATEGORIAS DE RISCO:

   0-20:   🟢 MUITO BAIXO  - Transação segura, processar normalmente
  20-40:   🟡 BAIXO        - Monitorar, baixa chance de fraude
  40-60:   🟠 MÉDIO        - Atenção redobrada, verificar padrões
  60-80:   🔴 ALTO         - Provável fraude, revisar manualmente
  80-100:  🚨 MUITO ALTO   - Bloqueio imediato recomendado

💡 USO PRÁTICO:

   • Score < 40:  Aprovar automaticamente
   • Score 40-60: Solicitar verificação adicional (2FA, SMS)
   • Score > 60:  Bloquear e avisar cliente
   • Score > 80:  Bloquear + investigação de segurança
""")

# ============================================
# 10. RESUMO FINAL
# ============================================
print("\n" + "="*80)
print(" "*25 + "RESUMO FINAL - REGRESSÃO")
print("="*80)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    RESULTADOS DO MODELO DE REGRESSÃO                  ║
╠══════════════════════════════════════════════════════════════════════╣
║ 🏆 MELHOR MODELO: {best_model_name:<48}       ║
║                                                                       ║
║ 📊 MÉTRICAS DE DESEMPENHO:                                           ║
║    • RMSE (Root Mean Squared Error): {results[best_model_name]['rmse']:>6.2f} pontos                   ║
║    • MAE (Mean Absolute Error):      {results[best_model_name]['mae']:>6.2f} pontos                   ║
║    • R² Score:                       {results[best_model_name]['r2']:>6.2f} ({results[best_model_name]['r2']*100:>5.1f}% variância explicada) ║
║                                                                       ║
║ 📈 ANÁLISE DE ERROS:                                                 ║
║    • Erro médio absoluto: {np.abs(errors).mean():>6.2f} pontos                              ║
║    • 50% dos erros < {np.percentile(np.abs(errors), 50):>6.2f} pontos                              ║
║    • 95% dos erros < {np.percentile(np.abs(errors), 95):>6.2f} pontos                              ║
║                                                                       ║
║ ⏱️  TEMPO DE TREINAMENTO: {results[best_model_name]['time']:>6.2f}s                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n💡 O QUE SIGNIFICA:")
print("-"*80)
print(f"""
• RMSE ({results[best_model_name]['rmse']:.2f}): Em média, o erro é de ±{results[best_model_name]['rmse']:.1f} pontos no score
• MAE ({results[best_model_name]['mae']:.2f}): Erro absoluto médio de {results[best_model_name]['mae']:.1f} pontos
• R² ({results[best_model_name]['r2']:.2f}): O modelo explica {results[best_model_name]['r2']*100:.1f}% da variação nos scores

🎯 INTEGRAÇÃO COM CLASSIFICAÇÃO:
   O modelo de regressão complementa o classificador fornecendo
   um score granular (0-100) ao invés de apenas Fraude/Legítima.
   Isso permite decisões mais nuanceadas!

🎯 PRÓXIMO PASSO: Modelo de Clustering (Identificar padrões de fraude)
""")

print("="*80)
print("✅ MODELO DE REGRESSÃO CONCLUÍDO COM SUCESSO!")
print("="*80)
print("\n🚀 Execute agora: python src/04_clustering_model.py")
print("="*80)