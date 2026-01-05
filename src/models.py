from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb

def train_regressors(X_train, y_train):
    """
    Trains regression models.
    """
    print("\nTraining Regression Models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def train_classifiers_extended(X_train, y_train):
    """
    Trains 3 Non-Ensemble and 3 Ensemble classification models.
    """
    print("\nTraining Extended Classification Models...")
    
    # Non-Ensemble
    non_ensemble = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    # Ensemble
    ensemble = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    trained_models = {}
    
    print("--- Non-Ensemble ---")
    for name, model in non_ensemble.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    print("--- Ensemble ---")
    for name, model in ensemble.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning for Random Forest.
    """
    print("\nRunning Hyperparameter Tuning for Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    # Using RandomizedSearchCV for speed
    search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                n_iter=5, cv=3, verbose=1, random_state=42, n_jobs=-1)
    
    search.fit(X_train, y_train)
    
    print(f"Best Parameters: {search.best_params_}")
    return search.best_estimator_, search.best_params_
