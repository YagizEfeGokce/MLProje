import pandas as pd
import os

def load_data(filepath="data/cwurData.csv"):
    """
    Loads the CWUR dataset from the specified filepath.
    """
    if not os.path.exists(filepath):
        try:
            # Fallback for different CWD
            filepath = os.path.join(os.path.dirname(__file__), "..", "data", "cwurData.csv")
            if not os.path.exists(filepath):
                 raise FileNotFoundError(f"Dataset not found at {filepath}")
        except:
             raise FileNotFoundError(f"Dataset not found at {filepath}")

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df
