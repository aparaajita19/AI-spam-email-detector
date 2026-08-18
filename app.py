import os
import sys
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)

# Load model and vectorizer
MODEL_PATH = os.path.join(MODELS_DIR, 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("Model and vectorizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model files: {e}")
    else:
        print("Warning: Model files not found. Run src/spam_detector.py to train.")

load_artifacts()


def analyze_keywords(text: str, top_n: int = 6):
    """Identify top TF-IDF words and risk indicators from the text."""
    if not vectorizer or not model:
        return []
    
    try:
        # Transform single document
        tfidf_vec = vectorizer.transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        # Non-zero entries
        row = tfidf_vec.tocoo()
        if row.nnz == 0:
            return []
        
        # Sort words by TF-IDF score
        sorted_indices = row.col[np.argsort(row.data)[::-1]]
        top_words = []
        
        # Get spam class index
        classes = list(model.classes_)
        spam_idx = classes.index('spam') if 'spam' in classes else 1
        
        # Check feature log prob for spam if available
        feature_log_prob = getattr(model, 'feature_log_prob_', None)
        
        for idx in sorted_indices[:top_n]:
            word = feature_names[idx]
            tfidf_weight = float(tfidf_vec[0, idx])
            
            # Estimate risk level if log prob is available
            risk_level = 'neutral'
            if feature_log_prob is not None:
                spam_score = feature_log_prob[spam_idx][idx]
                ham_score = feature_log_prob[1 - spam_idx][idx]
                if spam_score > ham_score:
                    risk_level = 'high' if (spam_score - ham_score) > 1.5 else 'medium'
                else:
                    risk_level = 'low'
            
            top_words.append({
                'word': word,
                'weight': round(tfidf_weight, 4),
                'risk': risk_level
            })
            
        return top_words
    except Exception as e:
        print(f"Keyword analysis error: {e}")
        return []


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and vectorizer is not None,
        'model_type': 'Multinomial Naive Bayes + TF-IDF',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        load_artifacts()
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not initialized. Please train the model first.'
            }), 503

    data = request.get_json(silent=True) or {}
    text = data.get('text') or data.get('email') or data.get('content') or ''
    text = str(text).strip()

    if not text:
        return jsonify({
            'error': 'No email text provided. Please enter content to analyze.'
        }), 400

    try:
        # Transform and predict
        features = vectorizer.transform([text])
        raw_pred = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Determine label and probabilities
        classes = list(model.classes_)
        spam_idx = classes.index('spam') if 'spam' in classes else 1
        ham_idx = 1 - spam_idx

        spam_prob = float(probabilities[spam_idx]) * 100
        ham_prob = float(probabilities[ham_idx]) * 100

        if isinstance(raw_pred, str):
            label = raw_pred.capitalize()
        else:
            label = 'Spam' if raw_pred == 1 else 'Ham'

        confidence = spam_prob if label == 'Spam' else ham_prob

        # Keyword analysis
        keywords = analyze_keywords(text)

        # Quick text metrics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        uppercase_chars = sum(1 for c in text if c.isupper())
        uppercase_ratio = round((uppercase_chars / char_count) * 100, 1) if char_count > 0 else 0
        has_suspicious_symbols = any(sym in text for sym in ['$', '€', '£', '!!!', '***', '100%'])

        response_payload = {
            'text': text,
            'label': label,
            'is_spam': label == 'Spam',
            'confidence': round(confidence, 2),
            'spam_probability': round(spam_prob, 2),
            'ham_probability': round(ham_prob, 2),
            'keywords': keywords,
            'stats': {
                'word_count': word_count,
                'char_count': char_count,
                'uppercase_ratio': uppercase_ratio,
                'has_suspicious_symbols': has_suspicious_symbols
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting AI Spam Email Detector server at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
