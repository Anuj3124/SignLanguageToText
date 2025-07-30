import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

df = pd.read_csv("data/gesture_data.csv")

# Extract features
X = df.iloc[:, 1:].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df.iloc[:, 0])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train
model = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
train_accuracy = model.score(X_train, y_train)  
test_accuracy = model.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

# Save model
os.makedirs("models", exist_ok=True)
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump((model, label_encoder), f)

print("Model trained and saved.")
