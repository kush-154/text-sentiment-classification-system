import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_sentiment

app = FastAPI(title="Text Sentiment Analysis API")

class Tweet(BaseModel):
    text: str

@app.post("/predict")
def predict(tweet: Tweet):
    return predict_sentiment(tweet.text)

@app.get("/")
def root():
    return {"message": "Text Sentiment Analysis API", "status": "running"}
