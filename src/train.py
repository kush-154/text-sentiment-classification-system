import pandas as pd
from preprocessing import stemming
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import nltk
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv(
    r"twitter_sentiment_analysis\src\training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=["target", "id", "date", "flag", "user", "text"],
    header=None,
)

df = df.drop(columns=["id", "date", "flag", "user"])
df["text"] = df["text"].progress_apply(stemming)
df["target"] = df["target"].replace(4, 1)

x = df["text"]
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train_tfidf, y_train)

import os

os.makedirs("../../models", exist_ok=True)
joblib.dump(vectorizer, "../../models/vectorizer.pkl")
joblib.dump(lr, "../../models/logistic_model.pkl")
