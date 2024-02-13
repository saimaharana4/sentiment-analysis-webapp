import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

# Load dataset
train_df = pd.read_csv(r"D:\sentiment-analysis-webapp\data\train.csv",encoding='ISO-8859-1')
test_df = pd.read_csv(r"D:\sentiment-analysis-webapp\data\test.csv",encoding='ISO-8859-1')

# Preprocess data
def preprocess_text(text):
    tokenizer = TweetTokenizer()
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))

    tokens = tokenizer.tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(stemmed)

# Ensure text columns are of type string
train_df['text'] = train_df['text'].astype(str)
test_df['text'] = test_df['text'].astype(str)

# Now apply your preprocessing function
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectors = vectorizer.fit_transform(train_df['cleaned_text'])

# Model
model = LogisticRegression()
model.fit(X_train_vectors, train_df['sentiment'])

X_test_vectors = vectorizer.transform(test_df['cleaned_text'])
predictions = model.predict(X_test_vectors)


st.title('Sentiment Analysis of Tweets')

user_input = st.text_area("Enter Tweet Text")



if st.button('Predict Sentiment'):
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    prediction = model.predict(input_vector)
    st.write("prediction:", prediction)
    sentiment = 'Positive' if prediction == "positive" else 'Negative' if prediction == "negative" else 'Neutral'
    st.write('Sentiment:', sentiment)