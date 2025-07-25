# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

df = pd.read_csv("data/gesture_data.csv")
X = df.iloc[:, 1:].values
y = LabelEncoder().fit_transform(df.iloc[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump((model, LabelEncoder().fit(df.iloc[:, 0])), f)

print("Model trained and saved.")
