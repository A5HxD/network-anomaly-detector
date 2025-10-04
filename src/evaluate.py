import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def transform_with_encoders(df, encoders):
    for c, mapping in encoders.items():
        df[c] = df[c].astype(str).map(mapping).fillna(-1).astype(int)
    return df

def main():
    print("📥 Loading test data...")
    df_test = pd.read_csv("data/UNSW_NB15_testing-set.csv")
    df_test.fillna(0, inplace=True)

    y_test = df_test["attack_cat"].astype(str)
    X_test = df_test.drop(columns=["attack_cat", "label"])

    artifact = joblib.load("models/anomaly_model.pkl")
    model = artifact["model"]
    scaler = artifact["scaler"]
    encoders = artifact["encoders"]
    feature_cols = artifact["feature_columns"]

    # Apply encoders
    X_test_enc = transform_with_encoders(X_test.copy(), encoders)

    #Align columns if necessary
    X_test_enc = X_test_enc[feature_cols]

    #Scale
    X_test_scaled = scaler.transform(X_test_enc)

    #Predict
    y_pred = model.predict(X_test_scaled)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    #confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix (test set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()