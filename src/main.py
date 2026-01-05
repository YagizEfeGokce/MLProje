import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import clean_data, feature_engineering, preprocess_for_model
from eda_report import generate_eda_report, save_confusion_matrix, save_feature_importance
from models import train_regressors, train_classifiers_extended, tune_hyperparameters
from evaluate import evaluate_regressors, evaluate_classifiers
import os

def main():
    print("Starting ML Pipeline...")
    
    if not os.path.exists("results"):
        os.makedirs("results")
    
    results_file = open("results/metrics.txt", "w")
    
    def log_print(msg):
        print(msg)
        results_file.write(msg + "\n")

    # 1. Load Data
    try:
        df = load_data()
    except FileNotFoundError as e:
        log_print(str(e))
        return

    # 2. EDA & Preprocessing
    df = clean_data(df)
    df = feature_engineering(df)
    generate_eda_report(df)
    
    # 3. Model Training (Classification Focus for Template)
    log_print("\n=== Classification Task (Predicting Category) ===")
    X_df, y, features, scaler = preprocess_for_model(df, target_col='ranking_category', classification=True)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train 6 Models
    clf_models = train_classifiers_extended(X_train, y_train)
    
    # Evaluate All
    log_print("\n--- All Models Results (Testing) ---")
    results_test = evaluate_classifiers(clf_models, X_test, y_test, set_name="Testing")
    
    log_print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'AUC':<10}")
    log_print("-" * 90)
    for name, metrics in results_test.items():
        log_print(f"{name:<25} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1']:<10.4f} | {metrics['AUC']:<10.4f}")
        
        # Save plots
        y_pred = clf_models[name].predict(X_test)
        save_confusion_matrix(y_test, y_pred, name)
        save_feature_importance(clf_models[name], features, name)

    # Save Best Model (XGBoost) & Scaler
    import joblib
    print("\nSaving XGBoost model and scaler to results/...")
    joblib.dump(clf_models['XGBoost'], 'results/xgboost_model.pkl')
    joblib.dump(scaler, 'results/scaler.pkl')
    print("Saved 'results/xgboost_model.pkl' and 'results/scaler.pkl'")

    # 4. Hyperparameter Tuning
    log_print("\n--- Hyperparameter Tuning (Random Forest) ---")
    best_rf, best_params = tune_hyperparameters(X_train, y_train)
    log_print(f"Best Params: {best_params}")
    
    # Evaluate Tuned Model
    tuned_models = {"Tuned Random Forest": best_rf}
    results_tuned = evaluate_classifiers(tuned_models, X_test, y_test, set_name="Tuned Testing")
    
    for name, metrics in results_tuned.items():
        log_print(f"{name:<25} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1']:<10.4f} | {metrics['AUC']:<10.4f}")

    print("\nPipeline Completed Successfully.")
    results_file.close()

if __name__ == "__main__":
    main()
