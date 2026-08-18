# AI Spam Email Detector & Web Interface

An AI/Machine Learning email spam and phishing classifier with a modern web dashboard, powered by TF-IDF natural language processing and Multinomial Naive Bayes classification.

---

## 🌟 Features
- **Machine Learning Core**: High-accuracy TF-IDF vectorizer + Multinomial Naive Bayes classifier trained on thousands of email examples.
- **Modern Web Dashboard**: Glassmorphism UI with real-time risk assessment, dual probability breakdown bars, trigger keyword inspector, and diagnostic telemetry.
- **Quick Preset Samples**: Test instant scenarios (Amazon Gift Card scam, Bank security phishing, Crypto giveaway, Work report, Team meeting).
- **Session Scan History**: Keep track of analyzed emails with instant reload and inspection.
- **Dual Mode**: Use via Web UI (`app.py`), Command Line CLI (`src/predict_spam.py`), or as an importable Python module.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to: **[http://127.0.0.1:5001](http://127.0.0.1:5001)**

---

### 3. Command Line (CLI) Usage

#### Run with default test samples:
```bash
python src/predict_spam.py
```

#### Run with custom email text:
```bash
python src/predict_spam.py "Congratulations! You have won $1000 cash. Click here to claim your reward."
```

#### Use as a Python module:
```python
from src.predict_spam import predict_email

result = predict_email("Hi team, please find attached the quarterly project review.")
print(result["label"])        # 'Ham' or 'Spam'
print(result["confidence"])   # e.g. 96.63%
```

---

### 4. Retrain the Model (Optional)
To retrain the model on [data/emails.csv](file:///Users/mac/Abish/AI%20spam/AI-spam-email-detector/data/emails.csv):
```bash
python src/spam_detector.py
```

---

## 📁 Project Architecture
```
AI-spam-email-detector/
├── app.py                     # Flask REST API & Web Server
├── data/
│   └── emails.csv             # Training dataset
├── models/
│   ├── spam_model.pkl         # Trained Naive Bayes model
│   └── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
├── src/
│   ├── spam_detector.py       # Model training & evaluation pipeline
│   └── predict_spam.py        # Prediction script & CLI interface
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphism dark mode design system
│   ├── js/
│   │   └── app.js             # Interactive client-side application logic
│   └── index.html             # Web dashboard interface
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md                  # Documentation
```
