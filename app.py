# app.py
# FIXED STREAMLIT DASHBOARD

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")

# -----------------------------------
# CHECK FILES
# -----------------------------------
if not os.path.exists("models/model_results.csv"):
    st.error("model_results.csv not found. Run training file first.")
    st.stop()

if not os.path.exists("models/best_model.pkl"):
    st.error("best_model.pkl not found. Run training file first.")
    st.stop()

if not os.path.exists("models/scaler.pkl"):
    st.error("scaler.pkl not found. Run training file first.")
    st.stop()

# -----------------------------------
# LOAD FILES
# -----------------------------------
results = pd.read_csv("models/model_results.csv")
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------------------
# MODEL TABLE
# -----------------------------------
st.subheader("📊 Model Performance Table")
st.dataframe(results, use_container_width=True)

# -----------------------------------
# BEST MODEL
# -----------------------------------
best = results.sort_values(
    by=["F1 Score", "ROC AUC"],
    ascending=False
).iloc[0]

st.success(f"🏆 Best Model: {best['Model']}")

# -----------------------------------
# ACCURACY CHART
# -----------------------------------
st.subheader("📈 Accuracy Comparison")

fig1, ax1 = plt.subplots(figsize=(10, 4))

ax1.bar(results["Model"], results["Accuracy"])
ax1.set_ylabel("Accuracy")
ax1.set_xticklabels(results["Model"], rotation=45, ha="right")

st.pyplot(fig1)

# -----------------------------------
# ROC AUC CHART
# -----------------------------------
st.subheader("📈 ROC AUC Comparison")

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.bar(results["Model"], results["ROC AUC"])
ax2.set_ylabel("ROC AUC")
ax2.set_xticklabels(results["Model"], rotation=45, ha="right")

st.pyplot(fig2)

# -----------------------------------
# CSV PREDICTION
# -----------------------------------
st.subheader("📂 Upload CSV for Fraud Prediction")

uploaded = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded is not None:

    try:
        df = pd.read_csv(uploaded)

        st.write("Uploaded Data")
        st.dataframe(df.head())

        # Remove Class column if exists
        df = df.drop("Class", axis=1, errors="ignore")

        # Match training columns
        df = df[model.feature_names_in_]

        # Scale data
        scaled_data = scaler.transform(df)

        # Predict
        pred = model.predict(scaled_data)
        prob = model.predict_proba(scaled_data)[:, 1]

        # Add results
        result_df = df.copy()
        result_df["Prediction"] = pred
        result_df["Fraud Probability"] = prob.round(4)

        fraud = int((pred == 1).sum())
        genuine = int((pred == 0).sum())

        col1, col2 = st.columns(2)

        col1.metric("Fraud", fraud)
        col2.metric("Genuine", genuine)

        st.subheader("Prediction Output")
        st.dataframe(result_df.head(100))

        # Download
        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption("Built with Streamlit + Machine Learning")