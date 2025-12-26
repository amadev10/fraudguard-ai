import os
import numpy as np
from sklearn.datasets import fetch_openml
import pandas as pd

print("="*60)
print("📥 FRAUDGUARD AI - Download de Datasets")
print("="*60)

# Criar pastas se não existirem
os.makedirs('datasets/fraud', exist_ok=True)
os.makedirs('datasets/mnist', exist_ok=True)

# ============================================
# 1. BAIXAR MNIST
# ============================================
print("\n1️⃣  Baixando dataset MNIST (dígitos manuscritos)...")
print("   Isso pode demorar alguns minutos...")

try:
    # Baixar MNIST do OpenML
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.values
    y = mnist.target.values
    
    # Converter para numpy arrays
    X = X.astype('float32')
    y = y.astype('int')
    
    # Dividir em treino e teste (60000 treino, 10000 teste)
    x_train = X[:60000]
    y_train = y[:60000]
    x_test = X[60000:]
    y_test = y[60000:]
    
    # Salvar
    np.save('datasets/mnist/x_train.npy', x_train)
    np.save('datasets/mnist/y_train.npy', y_train)
    np.save('datasets/mnist/x_test.npy', x_test)
    np.save('datasets/mnist/y_test.npy', y_test)
    
    print(f"   ✅ MNIST baixado com sucesso!")
    print(f"      - Treino: {len(x_train)} imagens")
    print(f"      - Teste: {len(x_test)} imagens")
    print(f"      - Tamanho: {x_train.shape[1]} pixels (28x28)")
    
except Exception as e:
    print(f"   ❌ Erro ao baixar MNIST: {e}")
    print("   Tente baixar manualmente depois.")

# ============================================
# 2. DATASET DE FRAUDE
# ============================================
print("\n2️⃣  Dataset de Fraude Bancária:")
print("   ⚠️  Este dataset precisa ser baixado MANUALMENTE")
print()
print("   📋 INSTRUÇÕES:")
print("   1. Acesse: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
print("   2. Crie uma conta no Kaggle (grátis)")
print("   3. Clique em 'Download' (arquivo ZIP)")
print("   4. Extraia o arquivo 'creditcard.csv'")
print("   5. Coloque em: datasets/fraud/creditcard.csv")
print()

fraud_path = 'datasets/fraud/creditcard.csv'
if os.path.exists(fraud_path):
    df = pd.read_csv(fraud_path)
    print(f"   ✅ Dataset de fraude já existe!")
    print(f"      - Transações: {len(df)}")
    print(f"      - Features: {len(df.columns)}")
else:
    print("   ⏳ Dataset ainda não baixado. Siga as instruções acima.")

print("\n" + "="*60)
print("🏁 CONCLUSÃO")
print("="*60)

# Verificação final
mnist_ok = all(os.path.exists(f'datasets/mnist/{f}') for f in 
               ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy'])
fraud_ok = os.path.exists(fraud_path)

if mnist_ok and fraud_ok:
    print("✅ Todos os datasets estão prontos!")
    print("   Você pode começar o desenvolvimento! 🚀")
elif mnist_ok:
    print("✅ MNIST pronto!")
    print("⚠️  Falta baixar o dataset de fraude do Kaggle")
elif fraud_ok:
    print("✅ Dataset de fraude pronto!")
    print("⚠️  Falta baixar o MNIST (execute este script novamente)")
else:
    print("⚠️  Baixe os datasets seguindo as instruções acima")

print("="*60)