from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__, static_folder='frontend')
CORS(app)

# -----------------------------
# 1. LOAD & PREPARE MODEL
# -----------------------------
print("⏳ Loading dataset and training model...")

df = pd.read_csv(r"C:\Users\rjhar\OneDrive\Desktop\Backend hackathon\advanced_hc3_3class_small.csv")

df = df[df['label'] != 'mixed']
df['label'] = df['label'].apply(
    lambda x: 'AI-generated' if x != 'human' else 'Human-written'
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

print("✅ Model ready! Server starting...")

# -----------------------------
# 2. PREDICT FUNCTION
# -----------------------------
def predict_text(text):
    if not text.strip():
        return None, 0, "No input provided"

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    prob = float(max(probs))

    if prob >= 0.80:
        decision = "Acceptable"
    elif prob >= 0.60:
        decision = "Needs Review"
    else:
        decision = "Uncertain"

    return pred, prob, decision

# -----------------------------
# 3. ROUTES
# -----------------------------

# Serve frontend pages
@app.route('/')
def home():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    pred, prob, decision = predict_text(text)

    return jsonify({
        'prediction': pred,
        'confidence': round(prob * 100, 2),
        'decision': decision,
        'word_count': len(text.split())
    })

# -----------------------------
# 4. RUN SERVER
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)