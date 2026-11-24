# Smart Crop Advisor - Made by me
# Soil data diye best crop suggest kore

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Data load
crop_data = pd.read_csv("crop_data.csv")

X = crop_data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = crop_data["label"]

# 2. Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model train
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# 4. Notun soil data diye prediction
# Example: N=90, P=42, K=43, temp=20, humidity=82, ph=6.5, rainfall=202
sample = [[90, 42, 43, 20.8, 82, 6.5, 202]]
result = model.predict(sample)
print("Best Crop for this soil:", result[0])
