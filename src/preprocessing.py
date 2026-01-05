import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df):
    """
    Basic data cleaning: handling missing values (imputation).
    """
    print("Cleaning data (Robust Mode)...")
    
    # Explicitly list columns that SHOULD be numeric
    numeric_cols = [
        'world_rank', 'national_rank', 'quality_of_education', 'alumni_employment',
        'quality_of_faculty', 'publications', 'influence', 'citations', 
        'broad_impact', 'patents', 'score', 'year'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            # Force numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN with median
            try:
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(0)
            except Exception as e:
                print(f"Error filling {col}: {e}")
                df[col] = df[col].fillna(0)

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)
    return df

def feature_engineering(df):
    """
    Creates new features including 'ranking_category'.
    """
    print("Feature Engineering...")
    
    def categorize_rank(rank):
        if pd.isna(rank): return 2
        if rank <= 100:
            return 0 # Elite
        elif rank <= 500:
            return 1 # High
        else:
            return 2 # Average
            
    df['ranking_category'] = df['world_rank'].apply(categorize_rank)
    
    return df

def preprocess_for_model(df, target_col='score', classification=False):
    """
    Prepares X and y for modeling.
    """
    print("Preprocessing for model...")
    
    # Encode 'country'
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['country'].astype(str))
    
    # Define feature cols
    feature_cols = [
        'quality_of_education', 'alumni_employment', 'quality_of_faculty',
        'publications', 'influence', 'citations', 'broad_impact', 'patents',
        'country_encoded'
    ]
    
    # Ensure all feature cols exist (if some dropped)
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    
    if classification:
        # Check if ranking_category exists
        if 'ranking_category' not in df.columns:
             feature_engineering(df)
        y = df['ranking_category']
    else:
        y = df[target_col]
    
    # Fill any remaining NaNs in X
    X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_df, y, feature_cols, scaler
