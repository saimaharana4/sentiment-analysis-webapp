

X_test_vectors = vectorizer.transform(test_df['cleaned_text'])
predictions = model.predict(X_test_vectors)

# Evaluation
print("Accuracy:", accuracy_score(test_df['sentiment'], predictions))