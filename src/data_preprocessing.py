import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

def preprocess_text(text):
    tokenizer = TweetTokenizer()
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(stemmed)

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='ISO-8859-1')
    test_df = pd.read_csv(test_path, encoding='ISO-8859-1')
    train_df['text'] = train_df['text'].astype(str)
    test_df['text'] = test_df['text'].astype(str)
    train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
    test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)
    return train_df, test_df