import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_loader import load_data
from preprocessing import clean_data, feature_engineering, preprocess_for_model
import xgboost

def run_shap_analysis():
    print("Starting SHAP Analysis...")

    # 1. Load Data & Preprocess (Same as main.py)
    try:
        df = load_data()
        df = clean_data(df)
        df = feature_engineering(df)
        X_df, y, features, scaler = preprocess_for_model(df, target_col='ranking_category', classification=True)
    except Exception as e:
        print(f"Error loading/processing data: {e}")
        return

    # 2. Load Model
    model_path = 'results/xgboost_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run main.py first.")
        return
    
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # 3. Create SHAP Explainer
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # 4. Generate & Save Plots
    if not os.path.exists("eda_plots"):
        os.makedirs("eda_plots")

    # Summary Plot (Bar) - Feature overall importance
    print("Generating Summary Bar Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("eda_plots/shap_summary_bar.png")
    plt.close()

    # Summary Plot (Beeswarm) - Feature impact direction
    print("Generating Beeswarm Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()
    plt.savefig("eda_plots/shap_beeswarm.png")
    plt.close()

    print("SHAP plots saved to 'eda_plots/'.")

if __name__ == "__main__":
    run_shap_analysis()
