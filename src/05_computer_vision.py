"""
FraudGuard AI - Modelo de Visão Computacional
Reconhecimento de Dígitos Manuscritos (OCR) usando SVM
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" "*18 + "FRAUDGUARD AI - VISÃO COMPUTACIONAL (OCR)")
print("="*80)

# ============================================
# 1. CARREGAR DATASET MNIST
# ============================================
print("\n📂 1. CARREGANDO DATASET MNIST...")
print("-"*80)

try:
    X_train = np.load('datasets/mnist/x_train.npy')
    y_train = np.load('datasets/mnist/y_train.npy')
    X_test = np.load('datasets/mnist/x_test.npy')
    y_test = np.load('datasets/mnist/y_test.npy')
    
    print(f"✅ MNIST carregado com sucesso!")
    print(f"   • Treino: {len(X_train):,} imagens")
    print(f"   • Teste:  {len(X_test):,} imagens")
    print(f"   • Shape: {X_train.shape}")
    
except Exception as e:
    print(f"❌ Erro ao carregar MNIST: {e}")
    print("\n⚠️  Execute: python download_datasets.py")
    exit(1)

# ============================================
# 2. EXPLORAR DADOS
# ============================================
print("\n🔍 2. EXPLORANDO DADOS")
print("-"*80)

print(f"\n📊 Informações do dataset:")
print(f"   • Pixels por imagem: {X_train.shape[1]}")
print(f"   • Dimensão original: 28x28 pixels")
print(f"   • Classes: {len(np.unique(y_train))} dígitos (0-9)")

# Contar amostras por classe
print(f"\n📊 Distribuição por classe (treino):")
for digit in range(10):
    count = (y_train == digit).sum()
    print(f"   Dígito {digit}: {count:>6,} amostras ({count/len(y_train)*100:.1f}%)")

# Visualizar exemplos
fig, axes = plt.subplots(5, 10, figsize=(15, 8))
fig.suptitle('Exemplos do Dataset MNIST', fontsize=16, fontweight='bold')

for i in range(50):
    ax = axes[i // 10, i % 10]
    # Reshape de 784 para 28x28
    img = X_train[i].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'{y_train[i]}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('models/vision/mnist_examples.png', dpi=300, bbox_inches='tight')
print("\n💾 Gráfico salvo: models/vision/mnist_examples.png")
plt.close()

# ============================================
# 3. PRÉ-PROCESSAMENTO
# ============================================
print("\n🔧 3. PRÉ-PROCESSAMENTO DOS DADOS")
print("-"*80)

# Normalizar pixels (0-255 -> 0-1)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

print(f"✅ Dados normalizados:")
print(f"   • Antes: [0, 255]")
print(f"   • Depois: [0, 1]")
print(f"   • Min: {X_train_scaled.min():.3f}, Max: {X_train_scaled.max():.3f}")

# Usar apenas uma amostra para velocidade (10% dos dados)
print(f"\n🎲 Usando amostra de treino (para velocidade):")
sample_size = int(len(X_train_scaled) * 0.1)  # 10%
indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_train_sample = X_train_scaled[indices]
y_train_sample = y_train[indices]

print(f"   • Treino original: {len(X_train_scaled):,}")
print(f"   • Treino amostrado: {len(X_train_sample):,} ({len(X_train_sample)/len(X_train_scaled)*100:.0f}%)")
print(f"   • Teste (completo): {len(X_test_scaled):,}")

# ============================================
# 4. TREINAR MODELOS
# ============================================
print("\n🤖 4. TREINANDO MODELOS DE CLASSIFICAÇÃO")
print("-"*80)

results = {}

# === MODELO 1: SVM (RBF Kernel) ===
print("\n🎯 Treinando SVM (RBF)...")
print("   ⚠️  Isso pode demorar alguns minutos...")
start_time = time.time()

svm_model = SVC(
    kernel='rbf',
    C=5.0,
    gamma='scale',
    random_state=42,
    cache_size=1000
)

svm_model.fit(X_train_sample, y_train_sample)
svm_time = time.time() - start_time

print(f"✅ SVM treinado em {svm_time:.2f}s")

# Predições
print("   Testando no conjunto de teste...")
y_pred_svm = svm_model.predict(X_test_scaled)

results['SVM (RBF)'] = {
    'model': svm_model,
    'predictions': y_pred_svm,
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm, average='weighted'),
    'recall': recall_score(y_test, y_pred_svm, average='weighted'),
    'f1': f1_score(y_test, y_pred_svm, average='weighted'),
    'time': svm_time
}

# === MODELO 2: RANDOM FOREST ===
print("\n🌲 Treinando Random Forest...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_sample, y_train_sample)
rf_time = time.time() - start_time

print(f"✅ Random Forest treinado em {rf_time:.2f}s")

# Predições
y_pred_rf = rf_model.predict(X_test_scaled)

results['Random Forest'] = {
    'model': rf_model,
    'predictions': y_pred_rf,
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf, average='weighted'),
    'recall': recall_score(y_test, y_pred_rf, average='weighted'),
    'f1': f1_score(y_test, y_pred_rf, average='weighted'),
    'time': rf_time
}

# === MODELO 3: K-NEAREST NEIGHBORS ===
print("\n📍 Treinando K-Nearest Neighbors...")
start_time = time.time()

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    n_jobs=-1
)

knn_model.fit(X_train_sample, y_train_sample)
knn_time = time.time() - start_time

print(f"✅ KNN treinado em {knn_time:.2f}s")

# Predições
y_pred_knn = knn_model.predict(X_test_scaled)

results['KNN'] = {
    'model': knn_model,
    'predictions': y_pred_knn,
    'accuracy': accuracy_score(y_test, y_pred_knn),
    'precision': precision_score(y_test, y_pred_knn, average='weighted'),
    'recall': recall_score(y_test, y_pred_knn, average='weighted'),
    'f1': f1_score(y_test, y_pred_knn, average='weighted'),
    'time': knn_time
}

# ============================================
# 5. COMPARAR MODELOS
# ============================================
print("\n📊 5. COMPARAÇÃO DE MODELOS")
print("-"*80)

print("\n" + "="*85)
print(f"{'MODELO':<20} {'ACCURACY':>12} {'PRECISION':>12} {'RECALL':>12} {'F1-SCORE':>12} {'TEMPO':>8}")
print("="*85)

for model_name, metrics in results.items():
    print(f"{model_name:<20} "
          f"{metrics['accuracy']:>12.4f} "
          f"{metrics['precision']:>12.4f} "
          f"{metrics['recall']:>12.4f} "
          f"{metrics['f1']:>12.4f} "
          f"{metrics['time']:>7.2f}s")

print("="*85)

# Melhor modelo
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['predictions']

print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")

# ============================================
# 6. ANÁLISE DETALHADA
# ============================================
print(f"\n🔍 6. ANÁLISE DETALHADA - {best_model_name}")
print("-"*80)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_best)

print("\n📊 Matriz de Confusão:")
print(f"\n{'':>8}", end='')
for i in range(10):
    print(f"{i:>6}", end='')
print()

for i in range(10):
    print(f"Real {i}:", end='')
    for j in range(10):
        print(f"{cm[i,j]:>6}", end='')
    print()

# Acurácia por dígito
print(f"\n📈 Acurácia por Dígito:")
for digit in range(10):
    digit_mask = y_test == digit
    digit_acc = accuracy_score(y_test[digit_mask], y_pred_best[digit_mask])
    correct = (y_pred_best[digit_mask] == digit).sum()
    total = digit_mask.sum()
    print(f"   Dígito {digit}: {digit_acc*100:>6.2f}% ({correct:>4}/{total:>4} corretos)")

# Classification Report
print(f"\n📋 Relatório de Classificação:")
print(classification_report(y_test, y_pred_best, digits=4))

# ============================================
# 7. VISUALIZAÇÕES
# ============================================
print("\n📈 7. GERANDO VISUALIZAÇÕES")
print("-"*80)

# === GRÁFICO 1: Comparação de Modelos ===
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Métricas
import pandas as pd
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
axes[0, 0].set_ylim([0.8, 1.0])

# Tempo de treinamento
time_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Tempo (s)': [r['time'] for r in results.values()]
})
time_df.plot(x='Modelo', y='Tempo (s)', kind='bar', ax=axes[0, 1], 
            rot=15, color='coral', legend=False)
axes[0, 1].set_title('Tempo de Treinamento', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Segundos')
axes[0, 1].grid(True, alpha=0.3)

# Matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
            cbar_kws={'label': 'Quantidade'})
axes[1, 0].set_xlabel('Dígito Predito')
axes[1, 0].set_ylabel('Dígito Real')
axes[1, 0].set_title(f'Matriz de Confusão - {best_model_name}', 
                     fontweight='bold', fontsize=12)

# Acurácia por dígito
accuracies = []
for digit in range(10):
    digit_mask = y_test == digit
    digit_acc = accuracy_score(y_test[digit_mask], y_pred_best[digit_mask])
    accuracies.append(digit_acc * 100)

axes[1, 1].bar(range(10), accuracies, color='steelblue')
axes[1, 1].set_xlabel('Dígito')
axes[1, 1].set_ylabel('Acurácia (%)')
axes[1, 1].set_title('Acurácia por Dígito', fontweight='bold', fontsize=12)
axes[1, 1].set_xticks(range(10))
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([90, 100])

plt.tight_layout()
plt.savefig('models/vision/model_comparison.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/vision/model_comparison.png")
plt.close()

# === GRÁFICO 2: Predições Corretas e Erradas ===
fig, axes = plt.subplots(4, 10, figsize=(20, 9))
fig.suptitle('Exemplos de Predições - Corretas (verde) e Erradas (vermelho)', 
             fontsize=16, fontweight='bold')

# 20 corretas
correct_indices = np.where(y_pred_best == y_test)[0]
correct_sample = np.random.choice(correct_indices, 20, replace=False)

for i, idx in enumerate(correct_sample):
    ax = axes[i // 10, i % 10]
    img = X_test[idx].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Real:{y_test[idx]}\nPred:{y_pred_best[idx]}', 
                fontsize=9, color='green', fontweight='bold')
    ax.axis('off')

# 20 erradas
wrong_indices = np.where(y_pred_best != y_test)[0]
if len(wrong_indices) >= 20:
    wrong_sample = np.random.choice(wrong_indices, 20, replace=False)
    
    for i, idx in enumerate(wrong_sample):
        ax = axes[2 + i // 10, i % 10]
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Real:{y_test[idx]}\nPred:{y_pred_best[idx]}', 
                    fontsize=9, color='red', fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig('models/vision/predictions_examples.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/vision/predictions_examples.png")
plt.close()

# === GRÁFICO 3: Matriz de Confusão Normalizada ===
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
            cbar_kws={'label': 'Taxa de Acerto'})
plt.xlabel('Dígito Predito', fontsize=12)
plt.ylabel('Dígito Real', fontsize=12)
plt.title(f'Matriz de Confusão Normalizada - {best_model_name}', 
         fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('models/vision/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print("💾 Gráfico salvo: models/vision/confusion_matrix_normalized.png")
plt.close()

# ============================================
# 8. SALVAR MODELO
# ============================================
print(f"\n💾 8. SALVANDO MELHOR MODELO")
print("-"*80)

# Salvar modelo
model_path = 'models/vision/digit_recognizer.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Modelo salvo: {model_path}")

# Salvar informações
import json
model_info = {
    'model_name': best_model_name,
    'metrics': {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1_score': results[best_model_name]['f1']
    },
    'confusion_matrix': cm.tolist(),
    'per_digit_accuracy': {
        str(digit): float(accuracies[digit]/100) for digit in range(10)
    },
    'training_samples': len(X_train_sample),
    'test_samples': len(X_test)
}

with open('models/vision/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)
print(f"✅ Informações salvas: models/vision/model_info.json")

# ============================================
# 9. TESTE DE PREDIÇÃO
# ============================================
print(f"\n🧪 9. TESTE DE PREDIÇÃO EM TEMPO REAL")
print("-"*80)

# Função para reconhecer dígito
def recognize_digit(image_data):
    """
    Reconhece dígito de uma imagem
    image_data: array 28x28 ou 784 elementos, valores 0-255
    """
    # Normalizar
    if image_data.max() > 1:
        image_data = image_data / 255.0
    
    # Garantir formato correto
    if len(image_data.shape) == 2:
        image_data = image_data.reshape(1, -1)
    elif len(image_data.shape) == 1:
        image_data = image_data.reshape(1, -1)
    
    # Predição
    digit = best_model.predict(image_data)[0]
    
    return int(digit)

# Testar com alguns exemplos
print("\n📋 Exemplos de Reconhecimento:\n")
print(f"{'Imagem':<10} {'Real':<8} {'Predito':<10} {'Resultado':<12}")
print("-"*50)

n_examples = 10
example_indices = np.random.choice(len(X_test), n_examples, replace=False)

for idx in example_indices:
    real_digit = y_test[idx]
    pred_digit = recognize_digit(X_test[idx])
    result = '✅ Correto' if real_digit == pred_digit else '❌ Erro'
    
    print(f"#{idx:<9} {real_digit:<8} {pred_digit:<10} {result:<12}")

# Calcular taxa de acerto nestes exemplos
correct = sum(1 for idx in example_indices if y_test[idx] == recognize_digit(X_test[idx]))
print(f"\nTaxa de acerto: {correct}/{n_examples} ({correct/n_examples*100:.1f}%)")

# ============================================
# 10. RESUMO FINAL
# ============================================
print("\n" + "="*80)
print(" "*20 + "RESUMO FINAL - VISÃO COMPUTACIONAL")
print("="*80)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    RESULTADOS DO MODELO DE VISÃO                      ║
╠══════════════════════════════════════════════════════════════════════╣
║ 🏆 MELHOR MODELO: {best_model_name:<48}       ║
║                                                                       ║
║ 📊 MÉTRICAS DE DESEMPENHO:                                           ║
║    • Accuracy:  {results[best_model_name]['accuracy']:>6.2%} - {int(results[best_model_name]['accuracy']*len(y_test)):>5,}/{len(y_test):>5,} corretos                   ║
║    • Precision: {results[best_model_name]['precision']:>6.2%} - Confiança nas predições              ║
║    • Recall:    {results[best_model_name]['recall']:>6.2%} - Taxa de detecção                      ║
║    • F1-Score:  {results[best_model_name]['f1']:>6.2%} - Balanço geral                         ║
║                                                                       ║
║ 🎯 ANÁLISE DE ERROS:                                                 ║
║    • Total de erros: {(y_pred_best != y_test).sum():>5,} ({(y_pred_best != y_test).sum()/len(y_test)*100:>5.2f}%)                            ║
║    • Melhor dígito:  {accuracies.index(max(accuracies)):>1} ({max(accuracies):>5.1f}% acurácia)                      ║
║    • Pior dígito:    {accuracies.index(min(accuracies)):>1} ({min(accuracies):>5.1f}% acurácia)                      ║
║                                                                       ║
║ ⏱️  TEMPO DE TREINAMENTO: {results[best_model_name]['time']:>6.2f}s                                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n💡 APLICAÇÃO NO FRAUDGUARD:")
print("-"*80)
print("""
Este modelo de OCR será usado para:

🔢 VALIDAÇÃO DE CHEQUES:
   • Ler valores manuscritos em cheques digitalizados
   • Comparar com valores digitados pelo usuário
   • Detectar discrepâncias (possível fraude)

📄 VERIFICAÇÃO DE DOCUMENTOS:
   • Extrair números de documentos (RG, CPF, etc)
   • Validar códigos de segurança
   • Automatizar entrada de dados

💳 PROCESSAMENTO DE CARTÕES:
   • Ler números de cartão em imagens
   • Validar CVV e datas de validade

🎯 INTEGRAÇÃO COMPLETA:
   1. Classificação → É fraude? (Sim/Não)
   2. Regressão → Score de risco (0-100)
   3. Clustering → Padrão de comportamento
   4. Visão → Validar documentos digitalizados

🎯 PRÓXIMO PASSO: Criar a Aplicação Web!
""")

print("="*80)
print("✅ TODOS OS MODELOS FORAM TREINADOS COM SUCESSO!")
print("="*80)
print("\n🎉 Modelos salvos:")
print("   • models/classification/fraud_classifier.pkl")
print("   • models/regression/risk_predictor.pkl")
print("   • models/clustering/pattern_analyzer.pkl")
print("   • models/vision/digit_recognizer.pkl")
print("\n🚀 PRÓXIMO PASSO: Criar aplicação web!")
print("   Execute: python web/app.py")
print("="*80)