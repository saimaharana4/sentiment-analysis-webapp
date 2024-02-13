from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_preprocessing import *

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectors = vectorizer.fit_transform(train_df['cleaned_text'])

# Model
model = LogisticRegression()
model.fit(X_train_vectors, train_df['sentiment'])