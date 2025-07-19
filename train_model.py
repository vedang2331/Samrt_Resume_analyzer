import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv('resume_data.csv')

# Features and labels
X = data['resume_text']
y = data['label']

# Vectorize text data
cv = CountVectorizer()
X_vectorized = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
pickle.dump(model, open('model/nlp_model.pkl', 'wb'))
pickle.dump(cv, open('model/cv.pkl', 'wb'))
