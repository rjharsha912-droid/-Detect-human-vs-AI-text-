import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. LOAD DATASET
# -----------------------------

try:
    df = pd.read_csv(r"C:\Users\rjhar\OneDrive\Desktop\Backend hackathon\advanced_hc3_3class_small.csv")
    print("✅ Dataset loaded. Shape:", df.shape)
except FileNotFoundError:
    print("❌ Error: Dataset file not found! Please check the path.")
    exit()

# -----------------------------
# 2. CLEAN LABELS
# -----------------------------
df = df[df['label'] != 'mixed']
df['label'] = df['label'].apply(
    lambda x: 'AI-generated' if x != 'human' else 'Human-written'
)

# -----------------------------
# 3. TEXT PREPROCESSING 
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# -----------------------------
# 5. TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 6. MODEL TRAINING
# -----------------------------
print("⏳ Training Model... Please wait.")
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)
print("✅ Training Completed!")

# -----------------------------
# 7. PREDICTION FUNCTION
# -----------------------------
def predict_text(text):
    if not text.strip():
        return None, 0, "No input provided"
        
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred = model.predict(vec)[0]
    # Predict probability for both classes
    probs = model.predict_proba(vec)[0]
    # Get the probability of the predicted class
    prob = max(probs)

    # Decision Layer
    if prob >= 0.80:
        decision = "✅ Acceptable (High certainty)"
    elif prob >= 0.60:
        decision = "⚠ Needs Review (Moderate certainty)"
    else:
        decision = "⚠ Uncertain / Mixed pattern"

    return pred, prob, decision

# -----------------------------
# 8. LIVE INPUT INTERFACE
# -----------------------------
print("\n" + "="*30)
print("   AI vs HUMAN TEXT DETECTOR")
print("="*30)

while True:
    print("\n------------------------------")
    user_input = input("Enter text to predict (or type 'exit' to stop): ")

    if user_input.lower() == 'exit':
        print("Exiting... Goodbye!")
        break

    pred, prob, decision = predict_text(user_input)

    if pred:
        print("\n--- RESULT ---")
        print(f"Prediction: {pred}")
        print(f"Confidence: {round(prob * 100, 2)}%")
        print(f"Status    : {decision}")