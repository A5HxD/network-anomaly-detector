import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Load trained model artifact
# ---------------------------
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("models/anomaly_model.pkl")
    return artifact

artifact = load_artifacts()
model = artifact["model"]
scaler = artifact["scaler"]
encoders = artifact["encoders"]
feature_cols = artifact["feature_columns"]

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="🚨 Network Anomaly Dashboard", layout="wide")

st.title("🚨 Network Traffic Anomaly Detection Dashboard (UNSW-NB15)")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Explorer", "🤖 Model Evaluation", "🔮 Predict Single Sample"])

# ---------------------------
# PAGE 1: Data Explorer
# ---------------------------
if page == "📊 Data Explorer":
    st.header("📊 Data Explorer")

    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv("data/UNSW_NB15_training-set.csv")

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.write("Shape:", df.shape)

    if "attack_cat" in df.columns:
        st.subheader("Attack Category Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(y=df["attack_cat"], order=df["attack_cat"].value_counts().index, ax=ax)
        st.pyplot(fig)

# ---------------------------
# PAGE 2: Model Evaluation
# ---------------------------
elif page == "🤖 Model Evaluation":
    st.header("🤖 Model Evaluation")

    df_test = pd.read_csv("data/UNSW_NB15_testing-set.csv")
    y_test = df_test["attack_cat"]
    X_test = df_test.drop(columns=["attack_cat", "label"])

    # Encode categorical
    for c, mapping in encoders.items():
        if c in X_test.columns:
            X_test[c] = X_test[c].astype(str).map(mapping).fillna(-1).astype(int)

    X_test = X_test[feature_cols]
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    st.pyplot(fig)

# ---------------------------
# PAGE 3: Predict Single Sample
# ---------------------------
elif page == "🔮 Predict Single Sample":
    st.header("🔮 Predict Single Sample")

    st.write("Fill in feature values for one record:")

    inputs = {}
    for col in feature_cols:
        if col in encoders:  # categorical
            options = list(encoders[col].keys())
            inputs[col] = st.selectbox(f"{col}", options)
        else:  # numeric
            inputs[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Predict"):
        df = pd.DataFrame([inputs])

        for c, mapping in encoders.items():
            df[c] = df[c].astype(str).map(mapping).fillna(-1).astype(int)

        df = df[feature_cols]
        df_scaled = scaler.transform(df)

        pred = model.predict(df_scaled)[0]
        st.success(f"✅ Predicted Attack Category: **{pred}**")