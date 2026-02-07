import joblib
from src.preprocessing import stemming

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)



def predict_sentiment(txt):
    clean = stemming(txt)
    x = vectorizer.transform([clean])
    probab = model.predict_proba(x)[0]

    if probab[1] > probab[0]:
        return {"sentiment": "positive", "confidence": round(float(probab[1]), 3)}
    else:
        return {"sentiment": "negative", "confidence": round(float(probab[0]), 3)}
