from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd

def calculate_mbe(y_true, y_pred):
    """Calculates Mean Bias Error"""
    return np.mean(y_true - y_pred)

def evaluate_regressors(models, X, y, set_name="Testing"):
    """
    Evaluates regression models and returns a dictionary of metrics.
    """
    results = {}
    print(f"\n--- Regression Results ({set_name}) ---")
    print(f"{'Model':<25} | {'MSE':<10} | {'R2':<10} | {'MAE':<10} | {'MBE':<10}")
    print("-" * 75)
    
    for name, model in models.items():
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mbe = calculate_mbe(y, y_pred)
        
        results[name] = {"MSE": mse, "R2": r2, "MAE": mae, "MBE": mbe}
        print(f"{name:<25} | {mse:<10.4f} | {r2:<10.4f} | {mae:<10.4f} | {mbe:<10.4f}")
        
    return results

def evaluate_classifiers(models, X, y, set_name="Testing"):
    """
    Evaluates classification models and returns a dictionary of metrics.
    """
    results = {}
    print(f"\n--- Classification Results ({set_name}) ---")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'AUC':<10}")
    print("-" * 90)
    
    for name, model in models.items():
        y_pred = model.predict(X)
        try:
            # For AUC, we need probabilities. Not all models support it easily or multi-class AUC is tricky.
            # We'll try to get probas if possible, otherwise NaN.
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
                # Handle multi-class AUC
                if y_prob.shape[1] > 2:
                    auc = roc_auc_score(y, y_prob, multi_class='ovr')
                else:
                    auc = roc_auc_score(y, y_prob[:, 1])
            else:
                auc = float('nan')
        except Exception as e:
            auc = float('nan')

        acc = accuracy_score(y, y_pred)
        # Weighted average for multi-class
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}
        print(f"{name:<25} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {auc:<10.4f}")
        
    return results
