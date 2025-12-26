"""
FraudGuard AI - Modelo de Classificação
Detectar transações fraudulentas vs legítimas
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*20 + "FRAUDGUARD AI - MODELO DE CLASSIFICAÇÃO")
print("="*80)

# ============================================
# 1. CARREGAR DADOS PROCESSADOS
# ============================================
print("\n📂 1. CARREGANDO DADOS PROCESSADOS...")
print("-"*80)

try:
    df = pd.read_csv('datasets/fraud/creditcard_processed.csv')
    print(f"✅ Dados carregados: {len(df):,} transações")
except:
    print("⚠️  Arquivo processado não encontrado. Carregando original...")
    df = pd.read_csv('datasets/fraud/creditcard.csv')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Preparar features e target
X = df.drop(['Class', 'Amount', 'Time'], axis=1, errors='ignore')
y = df['Class']

print(f"   Features: {X.shape[1]}")
print(f"   Amostras: {len(X):,}")
print(f"   Fraudes: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")

# ============================================
# 2. DIVIDIR DADOS (TREINO/TESTE)
# ============================================
print("\n🔀 2. DIVIDINDO DADOS EM TREINO E TESTE")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Dados divididos:")
print(f"   Treino: {len(X_train):,} amostras ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Teste:  {len(X_test):,} amostras ({len(X_test)/len(X)*100:.1f}%)")
print(f"\n   Fraudes no treino: {y_train.sum():,}")
print(f"   Fraudes no teste:  {y_test.sum():,}")

# ============================================
# 3. BALANCEAMENTO COM SMOTE
# ============================================
print("\n⚖️  3. BALANCEAMENTO DE CLASSES (SMOTE)")
print("-"*80)

print("🔹 Antes do balanceamento:")
print(f"   Classe 0 (Legítima): {(y_train==0).sum():,}")
print(f"   Classe 1 (Fraude):   {(y_train==1).sum():,}")
print(f"   Razão: 1:{(y_train==0).sum()//(y_train==1).sum()}")

# Aplicar SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 50% de fraudes
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\n🔹 Depois do balanceamento:")
print(f"   Classe 0 (Legítima): {(y_train_balanced==0).sum():,}")
print(f"   Classe 1 (Fraude):   {(y_train_balanced==1).sum():,}")
print(f"   Razão: 1:{(y_train_balanced==0).sum()//(y_train_balanced==1).sum()}")

# ============================================
# 4. TREINAR MODELOS
# ============================================
print("\n🤖 4. TREINANDO MODELOS DE CLASSIFICAÇÃO")
print("-"*80)

models = {}
results = {}

# === MODELO 1: RANDOM FOREST ===
print("\n🌲 Treinando Random Forest...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train_balanced, y_train_balanced)
rf_time = time.time() - start_time

print(f"✅ Random Forest treinado em {rf_time:.2f}s")

# Predições
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Métricas
results['Random Forest'] = {
    'model': rf_model,
    'predictions': y_pred_rf,
    'probabilities': y_pred_proba_rf,
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
    'time': rf_time
}

# === MODELO 2: LOGISTIC REGRESSION ===
print("\n📊 Treinando Logistic Regression...")
start_time = time.time()

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='lbfgs'
)

lr_model.fit(X_train_balanced, y_train_balanced)
lr_time = time.time() - start_time

print(f"✅ Logistic Regression treinado em {lr_time:.2f}s")

# Predições
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Métricas
results['Logistic Regression'] = {
    'model': lr_model,
    'predictions': y_pred_lr,
    'probabilities': y_pred_proba_lr,
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lr),
    'time': lr_time
}

# === MODELO 3: SVM ===
print("\n🎯 Treinando SVM (amostra reduzida)...")
print("   ⚠️  SVM é lento, usando apenas 20% dos dados de treino...")
start_time = time.time()

# Usar apenas uma amostra para SVM (muito lento)
sample_size = int(len(X_train_balanced) * 0.2)
X_train_sample = X_train_balanced[:sample_size]
y_train_sample = y_train_balanced[:sample_size]

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42,
    probability=True,
    class_weight='balanced'
)

svm_model.fit(X_train_sample, y_train_sample)
svm_time = time.time() - start_time

print(f"✅ SVM treinado em {svm_time:.2f}s")

# Predições
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]

# Métricas
results['SVM'] = {
    'model': svm_model,
    'predictions': y_pred_svm,
    'probabilities': y_pred_proba_svm,
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'f1': f1_score(y_test, y_pred_svm),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_svm),
    'time': svm_time
}

# ============================================
# 5. COMPARAR MODELOS
# ============================================
print("\n📊 5. COMPARAÇÃO DE MODELOS")
print("-"*80)

# Tabela de resultados
print("\n" + "="*90)
print(f"{'MODELO':<20} {'ACCURACY':>10} {'PRECISION':>10} {'RECALL':>10} {'F1-SCORE':>10} {'ROC-AUC':>10} {'TEMPO':>8}")
print("="*90)

for model_name, metrics in results.items():
    print(f"{model_name:<20} "
          f"{metrics['accuracy']:>10.4f} "
          f"{metrics['precision']:>10.4f} "
          f"{metrics['recall']:>10.4f} "
          f"{metrics['f1']:>10.4f} "
          f"{metrics['roc_auc']:>10.4f} "
          f"{metrics['time']:>7.2f}s")

print("="*90)

# Identificar melhor modelo
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\n🏆 MELHOR MODELO: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")

# ============================================
# 6. ANÁLISE DETALHADA DO MELHOR MODELO
# ============================================
print(f"\n🔍 6. ANÁLISE DETALHADA - {best_model_name}")
print("-"*80)

y_pred_best = results[best_model_name]['predictions']
y_proba_best = results[best_model_name]['probabilities']

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_best)
print("\n📊 Matriz de Confusão:")
print(f"\n                  Previsto")
print(f"               Legítima  Fraude")
print(f"Real Legítima    {cm[0,0]:>6}   {cm[0,1]:>6}")
print(f"     Fraude      {cm[1,0]:>6}   {cm[1,1]:>6}")

tn, fp, fn, tp = cm.ravel()
print(f"\n   True Negatives (TN):  {tn:>6} - Legítimas corretamente identificadas")
print(f"   False Positives (FP): {fp:>6} - Legítimas marcadas como fraude")
print(f"   False Negatives (FN): {fn:>6} - Fraudes NÃO detectadas ⚠️")
print(f"   True Positives (TP):  {tp:>6} - Fraudes corretamente detectadas ✅")

# Classification Report
print(f"\n📋 Relatório de Classificação:")
print(classification_report(y_test, y_pred_best, 
                           target_names=['Legítima', 'Fraude'],
                           digits=4))

# ============================================
# 7. VISUALIZAÇÕES
# ============================================
print("\n📈 7. GERANDO VISUALIZAÇÕES")
print("-"*80)

# === GRÁFICO 1: Comparação de Modelos ===
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Métricas
metrics_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()],
})

metrics_df.plot(x='Modelo', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                kind='bar', ax=axes[0, 0], rot=15)
axes[0, 0].set_title('Comparação de Métricas', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

# ROC-AUC
roc_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'ROC-AUC': [r['roc_auc'] for r in results.values()],
})
roc_df.plot(x='Modelo', y='ROC-AUC', kind='bar', ax=axes[0, 1], 
            rot=15, color='coral', legend=False)
axes[0, 1].set_title('ROC-AUC Score', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Score')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# Matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Legítima', 'Fraude'],
            yticklabels=['Legítima', 'Fraude'])
axes[1, 0].set_title(f'Matriz de Confusão - {best_model_name}', 
                     fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Real')
axes[1, 0].set_xlabel('Previsto')

# Curva ROC
for model_name, metrics in results.items():
    fpr, tpr, _ = roc_curve(y_test, metrics['probabilities'])
    axes[1, 1].plot(fpr, tpr, label=f"{model_name} (AUC={metrics['roc_auc']:.3f})", 
                    linewidth=2)

axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[1, 1].set_title('Curva ROC', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/classification/model_comparison.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/classification/model_comparison.png")
plt.close()

# === GRÁFICO 2: Feature Importance (Random Forest) ===
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'], 
             color='steelblue')
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importância')
    plt.title('Top 15 Features Mais Importantes', fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('models/classification/feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico salvo: models/classification/feature_importance.png")
    plt.close()

# === GRÁFICO 3: Precision-Recall Curve ===
plt.figure(figsize=(10, 6))

for model_name, metrics in results.items():
    precision, recall, _ = precision_recall_curve(y_test, metrics['probabilities'])
    plt.plot(recall, precision, label=f"{model_name}", linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall', fontweight='bold', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/classification/precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/classification/precision_recall_curve.png")
plt.close()

# ============================================
# 8. SALVAR MELHOR MODELO
# ============================================
print(f"\n💾 8. SALVANDO MELHOR MODELO")
print("-"*80)

# Salvar modelo
model_path = 'models/classification/fraud_classifier.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Modelo salvo: {model_path}")

# Salvar informações adicionais
model_info = {
    'model_name': best_model_name,
    'metrics': {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1_score': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc']
    },
    'confusion_matrix': cm.tolist(),
    'feature_names': X.columns.tolist()
}

import json
with open('models/classification/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)
print(f"✅ Informações salvas: models/classification/model_info.json")

# ============================================
# 9. TESTAR PREDIÇÕES
# ============================================
print(f"\n🧪 9. TESTE DE PREDIÇÕES")
print("-"*80)

# Pegar algumas amostras aleatórias
n_samples = 5
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

print("\n📋 Exemplos de predições:\n")
print(f"{'Real':<10} {'Previsto':<10} {'Probabilidade':<15} {'Resultado':<15}")
print("-"*50)

for idx in sample_indices:
    real = 'Fraude' if y_test.iloc[idx] == 1 else 'Legítima'
    pred = 'Fraude' if y_pred_best[idx] == 1 else 'Legítima'
    proba = y_proba_best[idx]
    correct = '✅ Correto' if y_test.iloc[idx] == y_pred_best[idx] else '❌ Erro'
    
    print(f"{real:<10} {pred:<10} {proba:>6.2%}{'':>8} {correct:<15}")

# ============================================
# 10. RESUMO FINAL
# ============================================
print("\n" + "="*80)
print(" "*25 + "RESUMO FINAL - CLASSIFICAÇÃO")
print("="*80)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    RESULTADOS DO TREINAMENTO                          ║
╠══════════════════════════════════════════════════════════════════════╣
║ 🏆 MELHOR MODELO: {best_model_name:<45}       ║
║                                                                       ║
║ 📊 MÉTRICAS DE DESEMPENHO:                                           ║
║    • Accuracy:  {results[best_model_name]['accuracy']:>6.2%}                                               ║
║    • Precision: {results[best_model_name]['precision']:>6.2%}  (Das fraudes detectadas, quantas eram reais) ║
║    • Recall:    {results[best_model_name]['recall']:>6.2%}  (De todas fraudes, quantas detectamos)    ║
║    • F1-Score:  {results[best_model_name]['f1']:>6.2%}  (Média harmônica)                        ║
║    • ROC-AUC:   {results[best_model_name]['roc_auc']:>6.2%}  (Capacidade de discriminação)            ║
║                                                                       ║
║ 🎯 RESULTADOS NO TESTE:                                              ║
║    • Fraudes detectadas:     {tp:>4} de {tp+fn:>4} ({tp/(tp+fn)*100:>5.1f}%)                    ║
║    • Fraudes não detectadas: {fn:>4} ({fn/(tp+fn)*100:>5.1f}%)                             ║
║    • Falsos positivos:       {fp:>4}                                      ║
║                                                                       ║
║ ⏱️  TEMPO DE TREINAMENTO: {results[best_model_name]['time']:>6.2f}s                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n💡 INTERPRETAÇÃO:")
print("-"*80)
print(f"""
• PRECISION ({results[best_model_name]['precision']:.2%}): Quando o modelo diz "é fraude", está certo {results[best_model_name]['precision']:.0%} das vezes
• RECALL ({results[best_model_name]['recall']:.2%}): O modelo detecta {results[best_model_name]['recall']:.0%} de todas as fraudes reais
• F1-SCORE ({results[best_model_name]['f1']:.2%}): Equilíbrio entre precision e recall

⚠️  IMPORTANTE:
   - False Negatives ({fn}): Fraudes que passaram despercebidas - CRÍTICO!
   - False Positives ({fp}): Clientes legítimos bloqueados - Ruim para UX

🎯 PRÓXIMO PASSO: Modelo de Regressão (Calcular Score de Risco 0-100)
""")

print("="*80)
print("✅ MODELO DE CLASSIFICAÇÃO CONCLUÍDO COM SUCESSO!")
print("="*80)
print("\n🚀 Execute agora: python src/03_regression_model.py")
print("="*80)