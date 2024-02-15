from data_preprocessing import load_and_preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

train_df, test_df = load_and_preprocess_data(r"data\train.csv", r"data\test.csv")

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vectors = vectorizer.fit_transform(train_df['cleaned_text'])

model = LogisticRegression(C = 1.0, penalty = 'l1', solver= 'liblinear')
model.fit(X_train_vectors, train_df['sentiment'])

# Save the trained model to model/ location
model_path = r'd:\sentiment-analysis-webapp\models\sentiment_model.pkl'
vectorizer_path = r"d:\sentiment-analysis-webapp\models\tfidf_vectorizer.pkl"
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)