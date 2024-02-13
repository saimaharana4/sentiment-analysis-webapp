import sys
import os
import streamlit as st
import pandas as pd
import joblib
from src.data_preprocessing import preprocess_text

# Add the parent directory of 'src' to sys.path to make it discoverable
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)


# Example of constructing an absolute path
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
model_path = os.path.join(model_dir, 'sentiment_model.pkl')

# print("Loading model from:", model_path) # to get the path stored for model

# Load model and vectorizer
model = joblib.load(r"d:\sentiment-analysis-webapp\models\sentiment_model.pkl")
vectorizer = joblib.load(r"d:\sentiment-analysis-webapp\models\sentiment_model.pkl")

# Assuming these are the correct paths to your saved model and vectorizer
model_path = r"d:\sentiment-analysis-webapp\models\sentiment_model.pkl"
vectorizer_path = r"d:\sentiment-analysis-webapp\models\tfidf_vectorizer.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
st.title('Sentiment Analysis of Tweets')

user_input = st.text_area("Enter Tweet Text")

if st.button('Predict Sentiment'):
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    prediction = model.predict(input_vector)
    sentiment = 'Positive' if prediction == "positive" else 'Negative' if prediction == "negative" else 'Neutral'
    st.write('Sentiment:', sentiment)