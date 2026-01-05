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
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("data/cwurData.csv")
    except:
        st.error("Data file not found. Please ensure 'data/cwurData.csv' exists.")
        return None, None, None, None, None, None

    # Cleaning
    numeric_cols = [
            'quality_of_education', 'alumni_employment', 'quality_of_faculty', 
            'publications', 'influence', 'citations', 'broad_impact', 'patents', 'score'
        ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Feature Engineering
    def categorize_rank(rank):
        if pd.isna(rank): return 2
        if rank <= 100: return 0
        elif rank <= 500: return 1
        else: return 2
        
    df['ranking_category'] = df['world_rank'].apply(categorize_rank)

    # Preprocessing
    feature_cols = [
        'quality_of_education', 'alumni_employment', 'quality_of_faculty',
        'publications', 'influence', 'citations', 'broad_impact', 'patents'
    ]
    
    # Validation: Ensure all cols exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y_reg = df['score']
    y_clf = df['ranking_category']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Models (Simplified for App)
    reg_model = RandomForestRegressor(n_estimators=50, random_state=42)
    reg_model.fit(X_scaled, y_reg)
    
    clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_model.fit(X_scaled, y_clf)
    
    return reg_model, clf_model, scaler, feature_cols, df, X_scaled

reg_model, clf_model, scaler, feature_cols, df, X_scaled = load_and_prep_data()

if reg_model is not None:
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
    
    predicted_score = reg_model.predict(input_scaled)[0]
    predicted_category = clf_model.predict(input_scaled)[0]
    
    cat_map = {0: "Elite (Top 100)", 1: "High (101-500)", 2: "Average (>500)"}

    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Score")
        st.metric(label="Score (0-100)", value=f"{predicted_score:.2f}")
        
    with col2:
        st.subheader("Predicted Category")
        st.metric(label="Category", value=cat_map.get(predicted_category, "Unknown"))

    st.divider()

    # Visualizations
    st.subheader("Visual Analysis")
    
    tab1, tab2 = st.tabs(["Feature Importance", "Input vs Average"])
    
    with tab1:
        importances = reg_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots()
        ax.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=90)
        ax.set_title("Feature Importance (Regression)")
        st.pyplot(fig)

    with tab2:
        # Radar Chart or simple Bar comparison
        avg_vals = df[feature_cols].mean()
        input_vals = pd.Series(user_input)
        
        comp_df = pd.DataFrame({'Metric': feature_cols, 'University': input_vals.values, 'Global Average': avg_vals.values})
        comp_df = comp_df.set_index('Metric')
        
        st.bar_chart(comp_df)

else:
    st.warning("Model training failed or data not loaded.")
