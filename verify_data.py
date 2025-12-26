import os
import pandas as pd
import numpy as np

print("🔍 Verificando datasets...\n")

# Verificar dataset de fraude
fraud_path = 'datasets/fraud/creditcard.csv'
if os.path.exists(fraud_path):
    df = pd.read_csv(fraud_path)
    print(f"✅ Dataset de Fraude encontrado!")
    print(f"   - Linhas: {len(df):,}")
    print(f"   - Colunas: {len(df.columns)}")
    print(f"   - Fraudes: {df['Class'].sum():,} ({df['Class'].sum()/len(df)*100:.2f}%)")
    print(f"   - Legítimas: {(df['Class']==0).sum():,} ({(df['Class']==0).sum()/len(df)*100:.2f}%)")
    print(f"   - Tamanho: {os.path.getsize(fraud_path) / (1024*1024):.2f} MB")
else:
    print("❌ Dataset de fraude NÃO encontrado!")
    print("   Baixe de: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")

print()

# Verificar MNIST
mnist_files = [
    'datasets/mnist/x_train.npy',
    'datasets/mnist/y_train.npy',
    'datasets/mnist/x_test.npy',
    'datasets/mnist/y_test.npy'
]

mnist_ok = all(os.path.exists(f) for f in mnist_files)
if mnist_ok:
    x_train = np.load('datasets/mnist/x_train.npy')
    y_train = np.load('datasets/mnist/y_train.npy')
    x_test = np.load('datasets/mnist/x_test.npy')
    
    print(f"✅ Dataset MNIST encontrado!")
    print(f"   - Imagens de treino: {len(x_train):,}")
    print(f"   - Imagens de teste: {len(x_test):,}")
    
    # Verificar se está achatado (784) ou em matriz (28x28)
    if len(x_train.shape) == 2:
        print(f"   - Formato: {x_train.shape[1]} pixels (28x28 achatado)")
    else:
        print(f"   - Tamanho da imagem: {x_train.shape[1]}x{x_train.shape[2]}")
    
    print(f"   - Classes únicas: {len(np.unique(y_train))} dígitos (0-9)")
else:
    print("❌ Dataset MNIST NÃO encontrado!")
    print("   Execute: python download_datasets.py")

print("\n" + "="*60)
if os.path.exists(fraud_path) and mnist_ok:
    print("🎉 TODOS OS DATASETS ESTÃO PRONTOS!")
    print("="*60)
    print("\n✨ Você pode começar o desenvolvimento agora! ✨")
    print("\n📚 Próximo passo: Análise Exploratória de Dados (EDA)")
else:
    print("⚠️  Alguns datasets estão faltando. Veja acima.")
    print("="*60)