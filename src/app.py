import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn

# Set page config
st.set_page_config(page_title="University Rank Predictor", layout="wide")

st.title("🎓 University Global Rank Predictor")
st.markdown("Predict the **Score** and **Ranking Category** of a university based on its performance metrics.")

# Load Data (Cache for performance)
# Load Data (Cache for performance)
@st.cache_data
def load_data_v2():
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
    
    # Note: Model expects 'country_encoded' as the last feature.
    # We don't add it to 'feature_cols' here because that controls the sliders loop.
    # We will add it manually in the user input section.

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

reg_model, clf_model, scaler, feature_cols, df, _ = load_data_v2()

if clf_model is not None:
    # Sidebar: User Input
    st.sidebar.header("Input University Metrics")
    
    user_input = {}
    for col in feature_cols:
        if col != 'country_encoded':
            # Get min/max for range
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            
            user_input[col] = st.sidebar.number_input(
                f"{col.replace('_', ' ').title()}", 
                min_value=min_val, 
                max_value=max_val, 
                value=mean_val,
                step=1.0,
                format="%.0f"
            )
    
    # Add Country Input
    countries = df['country'].unique() if 'country' in df.columns else []
    # If Country is not in df columns (loaded as None), we need to handle it.
    # In load_and_prep_data, we return df. Let's make sure 'country' is in it.
    
    selected_country = st.sidebar.selectbox("Country", sorted(countries)) if len(countries) > 0 else "USA"
    
    # We need to encode the country. Ideally we should have saved the LabelEncoder.
    # Quick fix: Re-fit label encoder on the loaded df (assuming df covers all training countries or close enough)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Fit on all countries in DB
    if 'country' in df.columns:
        le.fit(df['country'].astype(str))
        try:
            country_encoded = le.transform([selected_country])[0]
        except:
            country_encoded = 0 # Default if unknown
    else:
        country_encoded = 0
            
    user_input['country_encoded'] = country_encoded

    # Predict
    # Predict
    # Dynamic feature alignment using scaler's expected features
    if hasattr(scaler, 'feature_names_in_'):
        ordered_features = scaler.feature_names_in_
    else:
        # Fallback if attribute missing (older sklearn), though we confirmed it exists.
        ordered_features = [
            'quality_of_education', 'alumni_employment', 'quality_of_faculty',
            'publications', 'influence', 'citations', 'broad_impact', 'patents',
            'country_encoded'
        ]
    
    input_df = pd.DataFrame([user_input])
    
    # Fill missing columns with 0 (safety net)
    for col in ordered_features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns matches scaler
    input_df = input_df[ordered_features]
    
    # DEBUG: Visible proof of update
    st.write("Columns sent to model:", input_df.columns.tolist())
    
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        
        # Debugging block
        with st.expander("Show Debug Details", expanded=True):
            st.write("Sklearn Version:", sklearn.__version__)
            if hasattr(scaler, 'feature_names_in_'):
                st.write("Scaler expects:", scaler.feature_names_in_)
            else:
                st.write("Scaler has no feature_names_in_")
            
            st.write("Input Columns:", input_df.columns.tolist())
            
            # Check for mismatch
            if hasattr(scaler, 'feature_names_in_'):
                missing = set(scaler.feature_names_in_) - set(input_df.columns)
                extra = set(input_df.columns) - set(scaler.feature_names_in_)
                st.write("Missing:", missing)
                st.write("Extra:", extra)
                
        st.stop()
    
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
            # Use ordered_features for labels (includes country)
            feature_names = ordered_features if 'ordered_features' in locals() else feature_cols
            # If strictly using feature_cols (8 items) but importances has 9, we need to match.
            # Best to use scaler.feature_names_in_ if available
            if hasattr(clf_model, 'feature_importances_'):
                 if hasattr(scaler, 'feature_names_in_'):
                     feature_names = scaler.feature_names_in_
            
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            ax.set_title("Feature Importance (XGBoost Classifier)")
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model.")

    with tab2:
        # Radar Chart or simple Bar comparison
        avg_vals = df[feature_cols].mean()
        
        # Filter input to match valid feature columns (exclude country_encoded)
        input_vals = [user_input[col] for col in feature_cols]
        
        comp_df = pd.DataFrame({
            'Metric': feature_cols, 
            'University': input_vals, 
            'Global Average': avg_vals.values
        })
        comp_df = comp_df.reset_index()
        import altair as alt
        
        # Melt for Altair
        df_melt = comp_df.melt(id_vars='Metric', var_name='Type', value_name='Value')
        
        chart = alt.Chart(df_melt).mark_bar().encode(
            x=alt.X('Metric:N', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Value:Q', title='Rank / Value'),
            color='Type:N',
            xOffset='Type:N',
            tooltip=['Metric', 'Type', 'Value']
        ).properties(
            title="University vs Global Average Comparison"
        )
        
        st.altair_chart(chart, use_container_width=True)

else:
    st.warning("Model training failed or data not loaded.")
