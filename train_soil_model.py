import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib, os

df = pd.read_csv("data/crop_recommendation.csv")
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/crop_recommender.pkl")
# REMOVE this line (has emoji):
print(f"Model accuracy: {model.score(X_test, y_test)*100:.2f}%")

# REPLACE with this (no emoji):
print("Model accuracy: %.2f%%" % (model.score(X_test, y_test) * 100))
print("Model saved to models/crop_recommender.pkl")