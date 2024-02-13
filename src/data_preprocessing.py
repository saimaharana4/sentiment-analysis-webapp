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