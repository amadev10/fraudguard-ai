import os

# Caminhos dos datasets
FRAUD_DATA_PATH = 'datasets/fraud/creditcard.csv'
MNIST_TRAIN_X = 'datasets/mnist/x_train.npy'
MNIST_TRAIN_Y = 'datasets/mnist/y_train.npy'
MNIST_TEST_X = 'datasets/mnist/x_test.npy'
MNIST_TEST_Y = 'datasets/mnist/y_test.npy'

# Caminhos dos modelos salvos
MODELS_DIR = 'models'
CLASSIFICATION_MODEL = os.path.join(MODELS_DIR, 'classification', 'fraud_classifier.pkl')
REGRESSION_MODEL = os.path.join(MODELS_DIR, 'regression', 'risk_predictor.pkl')
CLUSTERING_MODEL = os.path.join(MODELS_DIR, 'clustering', 'pattern_analyzer.pkl')
VISION_MODEL = os.path.join(MODELS_DIR, 'vision', 'digit_recognizer.pkl')  # Mudou de .h5 para .pkl

# Criar pastas se não existirem
os.makedirs(os.path.join(MODELS_DIR, 'classification'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'regression'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'clustering'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'vision'), exist_ok=True)

# Configurações de treino
RANDOM_STATE = 42
TEST_SIZE = 0.2