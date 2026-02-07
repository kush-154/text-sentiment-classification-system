# Text Sentiment Classification System

## Overview

This project is an end-to-end **Text Sentiment Classification System** that predicts the sentiment of short text inputs as **positive or negative**.  
The system is trained on a large-scale dataset of **1.6 million labeled text samples** and is deployed with an interactive **Streamlit web interface** for real-time inference.

The focus of this project is not only model accuracy, but also clean NLP preprocessing, scalable design, and deployment readiness.

---

## Problem Statement

Sentiment analysis on short, informal text is challenging due to:
- Noisy language
- Slang and abbreviations
- Lack of context
- Weak supervision in labels

This project aims to build a robust classical NLP pipeline that can handle large volumes of short text efficiently and provide reliable sentiment predictions.

---

## Dataset

- **Dataset**: Sentiment140  
- **Size**: 1.6 million text samples  
- **Labels**:
  - `0` → Negative sentiment  
  - `4` → Positive sentiment (mapped to binary `1`)  
- **Characteristics**:
  - Short text
  - Informal language
  - Automatically labeled data

The dataset was chosen to simulate real-world sentiment classification challenges at scale.

---

## NLP Pipeline

The text processing pipeline includes:

1. Text normalization (lowercasing)
2. Removal of URLs and non-alphabetic characters
3. Tokenization
4. Stopword removal
5. Lemmatization using NLTK
6. Feature extraction using TF-IDF vectorization

The same preprocessing logic is used consistently during training and inference.

---

## Model

- **Algorithm**: Logistic Regression  
- **Feature Representation**: TF-IDF  
- **Reason for choice**:
  - Efficient on large sparse datasets
  - Fast inference
  - Strong baseline for text classification

### Performance

- **Accuracy**: ~77.5%

Additional metrics such as precision, recall, and F1-score were analyzed to understand class-wise performance.

---
## Model Comparison

Multiple classical machine learning models were evaluated to compare performance on large-scale short-text sentiment classification.

| Model                | Feature Type | Accuracy (%) | Remarks |
|---------------------|--------------|--------------|---------|
| Logistic Regression | TF-IDF       | 77.5         | Best balance of accuracy and inference speed |
| Multinomial Naive Bayes | TF-IDF   | 74.2         | Fast training, weaker performance on noisy text |
| Linear SVM          | TF-IDF       | 78.8         | Slightly better accuracy, higher computational cost |


Logistic Regression was selected for deployment due to its stable performance, faster inference, and suitability for large sparse feature spaces.


## Deployment

The trained model and vectorizer are saved and reused for inference.

The system is deployed as:
- An interactive Streamlit web application
- Accepts raw text input
- Returns predicted sentiment along with confidence score

This allows non-technical users to easily test the model in real time.

---

## Project Structure

```
twitter_sentiment_analysis/
│
├── src/
│   ├── preprocessing.py          # Text preprocessing functions
│   ├── train.py                   # Model training script
│   └── predict.py                 # Prediction logic
│
├── models/
│   ├── vectorizer.pkl             # Saved TF-IDF vectorizer
│   └── logistic_model.pkl         # Saved trained model
│
├── ui/
│   └── ui.py                      # Streamlit web interface
│
├── app/
│   └── app.py                     # FastAPI REST API
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```
