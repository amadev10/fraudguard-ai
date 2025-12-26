"""
FraudGuard AI - Backend Flask - VERSÃO FINAL
API para detecção de fraudes e OCR com análise em lote
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import json
from PIL import Image
import io
import base64
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============================================
# CARREGAR MODELOS TREINADOS
# ============================================
print("🔄 Carregando modelos...")

try:
    clf_model = joblib.load('models/classification/fraud_classifier.pkl')
    with open('models/classification/model_info.json', 'r') as f:
        clf_info = json.load(f)
    print("✅ Modelo de Classificação carregado")
except:
    clf_model = None
    clf_info = None
    print("⚠️  Modelo de Classificação não encontrado")

try:
    reg_model = joblib.load('models/regression/risk_predictor.pkl')
    with open('models/regression/model_info.json', 'r') as f:
        reg_info = json.load(f)
    print("✅ Modelo de Regressão carregado")
except:
    reg_model = None
    reg_info = None
    print("⚠️  Modelo de Regressão não encontrado")

try:
    cluster_model = joblib.load('models/clustering/pattern_analyzer.pkl')
    cluster_scaler = joblib.load('models/clustering/scaler.pkl')
    with open('models/clustering/model_info.json', 'r') as f:
        cluster_info = json.load(f)
    print("✅ Modelo de Clustering carregado")
except:
    cluster_model = None
    cluster_scaler = None
    cluster_info = None
    print("⚠️  Modelo de Clustering não encontrado")

try:
    vision_model = joblib.load('models/vision/digit_recognizer.pkl')
    with open('models/vision/model_info.json', 'r') as f:
        vision_info = json.load(f)
    print("✅ Modelo de Visão Computacional carregado")
except:
    vision_model = None
    vision_info = None
    print("⚠️  Modelo de Visão Computacional não encontrado")

print("✅ Servidor pronto!\n")

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def prepare_transaction_features(data):
    """Preparar features de transação para os modelos"""
    features = []
    
    # V1-V28 (features PCA)
    for i in range(1, 29):
        features.append(data.get(f'V{i}', 0.0))
    
    # Amount_scaled e Time_scaled
    features.append(data.get('Amount_scaled', 0.0))
    features.append(data.get('Time_scaled', 0.0))
    
    return np.array(features).reshape(1, -1)

def categorize_risk(score):
    """Categorizar score de risco"""
    if score < 20:
        return 'Muito Baixo', '🟢', 'success'
    elif score < 40:
        return 'Baixo', '🟡', 'warning'
    elif score < 60:
        return 'Médio', '🟠', 'info'
    elif score < 80:
        return 'Alto', '🔴', 'danger'
    else:
        return 'Muito Alto', '🚨', 'danger'

def preprocess_digit_image(image_data):
    """Preprocessar imagem de dígito para o modelo OCR"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('L')
        img_array = np.array(img)
        
        # Inverter cores (MNIST tem fundo preto)
        img_array = 255 - img_array
        
        # Encontrar bounding box
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            img_array = img_array[rmin:rmax+1, cmin:cmax+1]
        
        # Padding para manter aspecto quadrado
        h, w = img_array.shape
        size = max(h, w)
        squared = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        squared[y_offset:y_offset+h, x_offset:x_offset+w] = img_array
        
        # Redimensionar para 28x28
        img_pil = Image.fromarray(squared)
        img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_final = np.array(img_resized)
        
        # Normalizar (0-1)
        img_final = img_final / 255.0
        img_flat = img_final.flatten().reshape(1, -1)
        
        return img_flat, img_final
    
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_single_transaction(data):
    """Analisar uma única transação"""
    features = prepare_transaction_features(data)
    
    result = {
        'classification': None,
        'regression': None,
        'clustering': None
    }
    
    # Classificação
    if clf_model:
        prediction = clf_model.predict(features)[0]
        probability = clf_model.predict_proba(features)[0]
        
        fraud_prob = float(probability[1])
        legit_prob = float(probability[0])
        final_prediction = 1 if fraud_prob > 0.5 else 0
        
        result['classification'] = {
            'prediction': 'Fraude' if final_prediction == 1 else 'Legítima',
            'class': int(final_prediction),
            'probability': fraud_prob if final_prediction == 1 else legit_prob,
            'fraud_probability': fraud_prob,
            'legit_probability': legit_prob,
            'model_name': clf_info['model_name'] if clf_info else 'Unknown',
            'accuracy': clf_info['metrics']['accuracy'] if clf_info else 0
        }
    
    # Regressão
    if reg_model:
        risk_score = reg_model.predict(features)[0]
        risk_score = np.clip(risk_score, 0, 100)
        category, emoji, css_class = categorize_risk(risk_score)
        
        result['regression'] = {
            'risk_score': float(risk_score),
            'risk_category': category,
            'risk_emoji': emoji,
            'risk_class': css_class,
            'model_name': reg_info['model_name'] if reg_info else 'Unknown',
            'rmse': reg_info['metrics']['rmse'] if reg_info else 0
        }
    
    # Clustering
    if cluster_model and cluster_scaler:
        features_scaled = cluster_scaler.transform(features)
        cluster = cluster_model.predict(features_scaled)[0]
        cluster_data = cluster_info['cluster_info'].get(str(cluster), {})
        
        result['clustering'] = {
            'cluster': int(cluster),
            'cluster_size': cluster_data.get('size', 0),
            'fraud_rate': cluster_data.get('fraud_rate', 0) * 100,
            'risk_level': cluster_data.get('risk_level', 'UNKNOWN'),
            'model_name': cluster_info['model_name'] if cluster_info else 'Unknown'
        }
    
    return result

# ============================================
# ROTAS
# ============================================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/analyze_transaction', methods=['POST'])
def analyze_transaction():
    """Analisar transação individual"""
    try:
        data = request.json
        result = analyze_single_transaction(data)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/analyze_batch', methods=['POST'])
def analyze_batch():
    """Analisar múltiplas transações de um CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        
        # Ler CSV
        df = pd.read_csv(file)
        
        if len(df) == 0:
            return jsonify({'success': False, 'error': 'CSV vazio'}), 400
        
        if len(df) > 1000:
            return jsonify({'success': False, 'error': 'Máximo de 1000 transações por vez'}), 400
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Preparar dados
                data = row.to_dict()
                
                # Analisar
                analysis = analyze_single_transaction(data)
                
                results.append({
                    'index': int(idx),
                    'amount': float(data.get('Amount', data.get('amount', 0))),
                    'prediction': analysis['classification']['prediction'] if analysis['classification'] else 'N/A',
                    'fraud_probability': analysis['classification']['fraud_probability'] if analysis['classification'] else 0,
                    'risk_score': analysis['regression']['risk_score'] if analysis['regression'] else 0,
                    'cluster': analysis['clustering']['cluster'] if analysis['clustering'] else -1
                })
            except Exception as e:
                results.append({
                    'index': int(idx),
                    'error': str(e)
                })
        
        # Estatísticas
        valid_results = [r for r in results if 'error' not in r]
        frauds = len([r for r in valid_results if r['prediction'] == 'Fraude'])
        
        stats = {
            'total': len(results),
            'processed': len(valid_results),
            'errors': len(results) - len(valid_results),
            'frauds_detected': frauds,
            'fraud_rate': (frauds / len(valid_results) * 100) if valid_results else 0,
            'avg_risk_score': sum(r['risk_score'] for r in valid_results) / len(valid_results) if valid_results else 0
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/recognize_digit', methods=['POST'])
def recognize_digit():
    """Reconhecer dígito manuscrito"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Nenhuma imagem fornecida'}), 400
        
        img_flat, img_array = preprocess_digit_image(image_data)
        
        if img_flat is None:
            return jsonify({'success': False, 'error': 'Erro ao processar imagem'}), 400
        
        if vision_model:
            digit = vision_model.predict(img_flat)[0]
            
            confidence = 0.0
            if hasattr(vision_model, 'predict_proba'):
                proba = vision_model.predict_proba(img_flat)[0]
                confidence = float(np.max(proba))
            elif hasattr(vision_model, 'decision_function'):
                decision = vision_model.decision_function(img_flat)
                confidence = float(np.max(decision) / np.sum(np.abs(decision)))
            else:
                confidence = 0.95
            
            return jsonify({
                'success': True,
                'digit': int(digit),
                'confidence': confidence,
                'model_name': vision_info['model_name'] if vision_info else 'Unknown',
                'accuracy': vision_info['metrics']['accuracy'] if vision_info else 0
            })
        else:
            return jsonify({'success': False, 'error': 'Modelo de visão não carregado'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/models_info', methods=['GET'])
def models_info():
    """Retornar informações sobre os modelos carregados"""
    return jsonify({
        'classification': {
            'loaded': clf_model is not None,
            'info': clf_info
        },
        'regression': {
            'loaded': reg_model is not None,
            'info': reg_info
        },
        'clustering': {
            'loaded': cluster_model is not None,
            'info': cluster_info
        },
        'vision': {
            'loaded': vision_model is not None,
            'info': vision_info
        }
    })

@app.route('/api/validate_check', methods=['POST'])
def validate_check():
    """Validar cheque - comparar valor digitado com valor no cheque"""
    try:
        data = request.json
        image_data = data.get('image')
        declared_amount = float(data.get('amount', 0))
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Nenhuma imagem fornecida'}), 400
        
        # Preprocessar imagem
        img_flat, img_array = preprocess_digit_image(image_data)
        
        if img_flat is None:
            return jsonify({'success': False, 'error': 'Erro ao processar imagem'}), 400
        
        if not vision_model:
            return jsonify({'success': False, 'error': 'Modelo OCR não carregado'}), 500
        
        # Reconhecer dígitos múltiplos
        # Aqui vamos fazer uma abordagem simplificada: reconhecer o dígito principal
        # Em produção real, você segmentaria a imagem em múltiplos dígitos
        
        digit = vision_model.predict(img_flat)[0]
        
        # Confiança
        confidence = 0.0
        if hasattr(vision_model, 'predict_proba'):
            proba = vision_model.predict_proba(img_flat)[0]
            confidence = float(np.max(proba))
        elif hasattr(vision_model, 'decision_function'):
            decision = vision_model.decision_function(img_flat)
            confidence = float(np.max(decision) / np.sum(np.abs(decision)))
        else:
            confidence = 0.95
        
        # Para demonstração, vamos simular a leitura de um valor
        # Formato: reconhecer o dígito e multiplicar por 1000 (simulando leitura de milhares)
        read_amount = float(digit) * 1000
        
        # Comparar valores
        is_fraud = abs(read_amount - declared_amount) > 10  # Tolerância de 10 MT
        
        result = {
            'success': True,
            'digit_read': int(digit),
            'amount_read': float(read_amount),
            'amount_declared': float(declared_amount),
            'is_fraud': is_fraud,
            'difference': float(abs(read_amount - declared_amount)),
            'confidence': confidence,
            'message': 'Valores correspondem!' if not is_fraud else 'ALERTA: Valores não correspondem!'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/test', methods=['GET'])
def test():
    """Testar se API está funcionando"""
    return jsonify({
        'status': 'online',
        'message': 'FraudGuard AI API está funcionando!',
        'version': '2.0',
        'models_loaded': {
            'classification': clf_model is not None,
            'regression': reg_model is not None,
            'clustering': cluster_model is not None,
            'vision': vision_model is not None
        }
    })

# ============================================
# EXECUTAR SERVIDOR
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" "*15 + "FRAUDGUARD AI - SERVER v2.0")
    print("="*60)
    print("\n🌐 Servidor rodando em: http://localhost:5000")
    print("📊 Modelos carregados:")
    print(f"   • Classificação: {'✅' if clf_model else '❌'}")
    print(f"   • Regressão: {'✅' if reg_model else '❌'}")
    print(f"   • Clustering: {'✅' if cluster_model else '❌'}")
    print(f"   • Visão Computacional: {'✅' if vision_model else '❌'}")
    print("\n💡 Acesse http://localhost:5000 no navegador")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)