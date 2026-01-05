import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_eda_report(df, output_dir="eda_plots"):
    """
    Generates EDA plots and saves them to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating EDA plots in {output_dir}...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 2. Distribution of Score
    if 'score' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['score'], kde=True, bins=30)
        plt.title("Distribution of University Scores")
        plt.xlabel("Score")
        plt.savefig(f"{output_dir}/score_distribution.png")
        plt.close()
    
    # 3. Scatter: Quality of Education vs Score
    if 'quality_of_education' in df.columns and 'score' in df.columns:
        plt.figure(figsize=(8, 6))
        hue = 'ranking_category' if 'ranking_category' in df.columns else None
        sns.scatterplot(x='quality_of_education', y='score', data=df, hue=hue, palette='viridis')
        plt.title("Quality of Education vs Score")
        plt.savefig(f"{output_dir}/education_vs_score.png")
        plt.close()

    print("EDA report generated.")

def save_confusion_matrix(y_test, y_pred, model_name, output_dir="eda_plots"):
    """
    Saves confusion matrix for a classifier.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    # Sanitize model name for filename
    filename = model_name.replace(" ", "_").lower()
    plt.savefig(f"{output_dir}/confusion_matrix_{filename}.png")
    plt.close()

def save_feature_importance(model, feature_names, model_name, output_dir="eda_plots"):
    """
    Saves feature importance plot for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances: {model_name}")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    filename = model_name.replace(" ", "_").lower()
    plt.savefig(f"{output_dir}/feature_importance_{filename}.png")
    plt.close()
