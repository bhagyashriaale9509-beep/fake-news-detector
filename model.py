import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Combine title + text (VERY IMPORTANT)
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Features & labels
X = data["content"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ FINAL MODEL TRAINED!")