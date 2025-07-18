# train/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dataset
df = pd.read_csv('../data/hospital_recommendation_data.csv')

# Features and target
X = df.drop(['hospital_id', 'score'], axis=1)  # hospital_id is just metadata
y = df['score']  # 1 = suitable, 0 = not suitable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate (optional)
accuracy = clf.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# Save model
model_path = '../model/hospital_recommender.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"✅ Model saved to {model_path}")
