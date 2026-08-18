import os
import sys
import joblib

# Paths for saved model/vectorizer
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')

# Load trained model and vectorizer
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("Error: Model or vectorizer file not found. Please run 'src/spam_detector.py' first.")
    sys.exit(1)

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


def predict_email(text: str) -> dict:
    """Predict whether an email text is Spam or Ham with confidence score."""
    text_features = vectorizer.transform([text])
    raw_prediction = model.predict(text_features)[0]
    probabilities = model.predict_proba(text_features)[0]

    # Normalize label string
    if isinstance(raw_prediction, str):
        label = raw_prediction.capitalize()
        spam_index = list(model.classes_).index('spam') if 'spam' in model.classes_ else 1
    else:
        label = 'Spam' if raw_prediction == 1 else 'Ham'
        spam_index = 1

    spam_confidence = probabilities[spam_index] * 100
    confidence = spam_confidence if label == 'Spam' else (100 - spam_confidence)

    result = {
        'text': text,
        'label': label,
        'confidence': confidence,
        'spam_probability': spam_confidence,
    }

    print(f"Email: {text}")
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%, Spam Probability: {spam_confidence:.2f}%)\n")
    return result


if __name__ == '__main__':
    # If text provided via command line arguments, predict on that text
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predict_email(input_text)
    else:
        # Default sample emails
        sample_email1 = "Congratulations! You've won a $1000 Amazon gift card. Click here to claim."
        predict_email(sample_email1)

        sample_email2 = "Hi, I have attached the project report for your review. Thanks."
        predict_email(sample_email2)
