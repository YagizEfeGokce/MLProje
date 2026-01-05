import joblib
import pandas as pd
import numpy as np
import os
import sys

def load_artifacts():
    if not os.path.exists("results/xgboost_model.pkl") or not os.path.exists("results/scaler.pkl"):
        print("Error: Model files not found in 'results/'")
        print("Please run 'python src/main.py' first to generate the model and scaler.")
        sys.exit(1)
        
    print("Loading model and scaler...")
    model = joblib.load("results/xgboost_model.pkl")
    scaler = joblib.load("results/scaler.pkl")
    return model, scaler

def get_user_input():
    print("\n--- University Success Predictor (CLI) ---")
    print("Please enter the following details:")
    
    try:
        quality_of_education = float(input("Quality of Education (Rank, e.g., 1-1000): "))
        alumni_employment = float(input("Alumni Employment (Rank): "))
        quality_of_faculty = float(input("Quality of Faculty (Rank): "))
        publications = float(input("Publications (Rank): "))
        influence = float(input("Influence (Rank): "))
        citations = float(input("Citations (Rank): "))
        broad_impact = float(input("Broad Impact (Rank): "))
        patents = float(input("Patents (Rank): "))
        country_encoded = float(input("Country Code (0-100, enter 0 if unknown): "))
    except ValueError:
        print("\nInvalid input! Please enter numeric values.")
        return None

    # Feature order must match training
    features = [
        quality_of_education, alumni_employment, quality_of_faculty,
        publications, influence, citations, broad_impact, patents,
        country_encoded
    ]
    
    return np.array(features).reshape(1, -1)

def main():
    model, scaler = load_artifacts()
    
    while True:
        input_data = get_user_input()
        
        if input_data is not None:
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            # Map prediction to label
            labels = {0: "Elite (Top 100)", 1: "High (101-500)", 2: "Average (>500)"}
            result = labels.get(prediction, "Unknown")
            
            print(f"\n>>> PREDICTION: The predicted category is: {result}\n")
            
        cont = input("Do you want to predict another? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()
