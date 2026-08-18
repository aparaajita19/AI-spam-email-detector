import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define the path to your data file
data_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'emails.csv'
)

# Try different encodings for CSV reading
try:
    # First try utf-8 (preferred)
    data = pd.read_csv(data_path, encoding='utf-8')

except UnicodeDecodeError:
    try:
        # Fallback to latin1 if utf-8 fails
        data = pd.read_csv(data_path, encoding='latin1')

    except Exception as e:
        print("Failed to load CSV file. Check that it is a true CSV and not an Excel file.")
        print("Error details:", e)
        exit(1)


# Data checks
if 'text' not in data.columns or 'label' not in data.columns:
    print("CSV columns incorrect! Ensure your CSV has 'text' and 'label'.")
    exit(1)


# Clean text data
X = data['text'].fillna('').astype(str).str.strip()
y = data['label']


# Remove empty email entries
valid_rows = X.str.len() > 0
X = X[valid_rows]
y = y[valid_rows]


# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Initialize and train classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# Predict and evaluate
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(
    "Classification Report:\n",
    classification_report(y_test, y_pred)
)


# Save model and vectorizer
models_dir = os.path.join(
    os.path.dirname(__file__),
    '..',
    'models'
)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


model_path = os.path.join(
    models_dir,
    'spam_model.pkl'
)

vectorizer_path = os.path.join(
    models_dir,
    'tfidf_vectorizer.pkl'
)


joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)


print("Model and vectorizer saved successfully.")