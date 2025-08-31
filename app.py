import streamlit as st
import pandas as pd
import numpy as np
from src.data_preprocessing import basic_cleaning, train_val_test_split, scale_numeric
from src.modeling import train_random_forest, evaluate, save_artifacts
from src.feature_engineering import add_time_lags, add_rolling_features
from src.visualization import plot_target_distribution, correlation_heatmap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Climate Change Modeling", layout="wide")

st.title("🌍 Climate Change Modeling — Baseline App")
st.write("Upload a CSV to run quick EDA and a baseline Random Forest model.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20))

    df = basic_cleaning(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found. Please upload a dataset with numeric targets.")
    else:
        target = st.selectbox("Select target column", options=numeric_cols, index=0)
        with st.expander("EDA"):
            st.write("Target distribution")
            fig = plt.figure()
            try:
                plot_target_distribution(df, target)
                st.pyplot(fig)
            except Exception:
                st.write("Could not plot distribution.")
            st.write("Correlation heatmap (top related to target)")
            fig2 = plt.figure()
            try:
                correlation_heatmap(df, top_k=20, target=target)
                st.pyplot(fig2)
            except Exception:
                st.write("Could not compute correlation heatmap.")

        # Optional time features
        use_time = st.checkbox("Add lag & rolling features based on target (for time-like data).")
        if use_time:
            df = add_time_lags(df, target)
            df = add_rolling_features(df, target)

        # Drop non-numeric before modeling (simple baseline)
        model_df = df.select_dtypes(include=[np.number]).dropna()
        if model_df.shape[0] < 50:
            st.warning("Not enough rows after cleaning to train a model (need ~50+).")
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(model_df, target)
            Xtr, Xv, Xte, scaler = scale_numeric(X_train, X_val, X_test)

            result = train_random_forest(Xtr, y_train, Xv, y_val)
            st.subheader("Validation Metrics")
            st.json(result["val_metrics"])

            # Final evaluation on test
            test_metrics = evaluate(result["model"], Xte, y_test)
            st.subheader("Test Metrics")
            st.json(test_metrics)

            if st.button("Save Artifacts"):
                save_artifacts(result["model"], scaler, out_dir="artifacts", prefix="rf")
                st.success("Saved to artifacts/")
else:
    st.info("Awaiting CSV upload…")
