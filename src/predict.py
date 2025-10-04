import joblib
import pandas as pd

# Load trained model artifacts
artifact = joblib.load("models/anomaly_model.pkl")
model = artifact["model"]
scaler = artifact["scaler"]
encoders = artifact["encoders"]
feature_cols = artifact["feature_columns"]

# ✅ Load a real sample row from the test set
df_test = pd.read_csv("data/UNSW_NB15_testing-set.csv")

# Pick the first row (drop labels)
sample = df_test.drop(columns=["attack_cat", "label"]).iloc[[1012]].copy()

# Apply encoders for categorical columns
for c, mapping in encoders.items():
    if c in sample.columns:
        sample[c] = sample[c].astype(str).map(mapping).fillna(-1).astype(int)

# Ensure column order matches training
sample = sample[feature_cols]

# Scale features
sample_scaled = scaler.transform(sample)

# Predict
pred = model.predict(sample_scaled)[0]
print("✅ Prediction:", pred)
