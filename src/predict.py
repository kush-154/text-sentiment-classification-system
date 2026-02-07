import joblib
from preprocessing import stemming

vectorizer = joblib.load("../../models/vectorizer.pkl")
model = joblib.load("../../models/logistic_model.pkl")


def predict_sentiment(txt):
    clean = stemming(txt)
    x = vectorizer.transform([clean])
    probab = model.predict_proba(x)[0]

    if probab[1] > probab[0]:
        return {"sentiment": "positive", "confidence": round(float(probab[1]), 3)}
    else:
        return {"sentiment": "negative", "confidence": round(float(probab[0]), 3)}
