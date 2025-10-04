import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

RANDOM_STATE = 42

def fit_label_encoders(df, cat_cols):
    encoders = {}
    for c in cat_cols:
        vals = df[c].astype(str).fillna("NA").unique().tolist()
        mapping = {v: i for i, v in enumerate(vals)}
        encoders[c] = mapping
        df[c] = df[c].astype(str).map(mapping).fillna(-1).astype(int)
    return df, encoders

def transform_with_encoders(df, encoders):
    for c, mapping in encoders.items():
        df[c] = df[c].astype(str).map(mapping).fillna(-1).astype(int)
    return df

def main(sample_frac=None):
    os.makedirs("models", exist_ok=True)
    print("📥 Loading training data...")
    df = pd.read_csv("data/UNSW_NB15_training-set.csv")
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Using sample_frac={sample_frac}, new shape: {df.shape}")

    # Basic cleanup
    df.fillna(0, inplace=True)

    y = df["attack_cat"].astype(str)
    X = df.drop(columns=["attack_cat", "label"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print("Categorical columns:", cat_cols)

    #fit encoders on train
    X_enc, encoders = fit_label_encoders(X.copy(), cat_cols)

    #scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    #train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("🤖 Training RandomForest (this may take a few minutes)...")
    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    model.fit(X_tr, y_tr)


    y_pred = model.predict(X_val)
    print("📊 Validation classification report:")
    print(classification_report(y_val, y_pred))

    artifact = {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": X_enc.columns.tolist()
    }
    joblib.dump(artifact, "models/anomaly_model.pkl")
    print("✅ Saved model artifact to models/anomaly_model.pkl")

if __name__ == "__main__":
    main(sample_frac=None)  #sample_frac=0.2 to train on 20% of the data