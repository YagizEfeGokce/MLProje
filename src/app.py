import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config
st.set_page_config(page_title="University Rank Predictor", layout="wide")

st.title("🎓 University Global Rank Predictor")
st.markdown("Predict the **Score** and **Ranking Category** of a university based on its performance metrics.")

# Load Data (Cache for performance)
# Load Data (Cache for performance)
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("data/cwurData.csv")
    except:
        st.error("Data file not found. Please ensure 'data/cwurData.csv' exists.")
        return None, None, None, None, None, None

    # Cleaning needed for slider ranges (same as before)
    numeric_cols = [
            'quality_of_education', 'alumni_employment', 'quality_of_faculty', 
            'publications', 'influence', 'citations', 'broad_impact', 'patents', 'score'
        ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    feature_cols = [
        'quality_of_education', 'alumni_employment', 'quality_of_faculty',
        'publications', 'influence', 'citations', 'broad_impact', 'patents'
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Load Model & Scaler
    import joblib
    import os
    
    model_path = "results/xgboost_model.pkl"
    scaler_path = "results/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        clf_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        st.error("Model files not found! Please run 'src/main.py' locally first to generate models.")
        return None, None, None, None, None, None
        
    # We only have classifier now (per user request to save XGBoost)
    # If regressor is needed, it should be saved similarly. For now, we optionally disable regression or re-train quick regressor.
    # To keep it fast for web, we will focus on the main Classifier (XGBoost).
    
    return None, clf_model, scaler, feature_cols, df, None

reg_model, clf_model, scaler, feature_cols, df, _ = load_and_prep_data()

if clf_model is not None:
    # Sidebar: User Input
    st.sidebar.header("Input University Metrics")
    
    user_input = {}
    for col in feature_cols:
        # Get min/max for range
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        
        user_input[col] = st.sidebar.slider(f"{col.replace('_', ' ').title()}", min_val, max_val, mean_val)

    # Predict
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    
    # predicted_score = reg_model.predict(input_scaled)[0]
    predicted_category = clf_model.predict(input_scaled)[0]
    
    cat_map = {0: "Elite (Top 100)", 1: "High (101-500)", 2: "Average (>500)"}

    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Score")
        st.info("Score prediction is disabled in web mode to optimize performance.")
        
    with col2:
        st.subheader("Predicted Category")
        st.metric(label="Category", value=cat_map.get(predicted_category, "Unknown"))

    st.divider()

    # Visualizations
    st.subheader("Visual Analysis")
    
    tab1, tab2 = st.tabs(["Feature Importance", "Input vs Average"])
    
    with tab1:
        if hasattr(clf_model, 'feature_importances_'):
            importances = clf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots()
            ax.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=90)
            ax.set_title("Feature Importance (XGBoost Classifier)")
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model.")

    with tab2:
        # Radar Chart or simple Bar comparison
        avg_vals = df[feature_cols].mean()
        input_vals = pd.Series(user_input)
        
        comp_df = pd.DataFrame({'Metric': feature_cols, 'University': input_vals.values, 'Global Average': avg_vals.values})
        comp_df = comp_df.set_index('Metric')
        
        st.bar_chart(comp_df)

else:
    st.warning("Model training failed or data not loaded.")
