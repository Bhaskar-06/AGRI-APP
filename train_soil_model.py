import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")   # capital C — matches actual filename
MODEL_SAVE = os.path.join(BASE_DIR, "models", "crop_recommender.pkl")

df = pd.read_csv(CSV_PATH)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(model, MODEL_SAVE)

accuracy = model.score(X_test, y_test) * 100
print("Model accuracy: %.2f%%" % accuracy)
print(f"Model saved to {MODEL_SAVE}")