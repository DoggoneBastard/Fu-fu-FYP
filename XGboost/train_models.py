import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def clean_target_column(series):
    """
    Cleans a target column by extracting numerical values from strings that may
    contain error margins (e.g., '0.85 ± 0.05').

    Args:
        series (pd.Series): The pandas Series (column) to clean.

    Returns:
        pd.Series: A new Series containing only the cleaned numerical values.
    """
    cleaned_values = []
    for item in series:
        try:
            # If the item is already a number, just append it
            cleaned_values.append(float(item))
            continue
        except (ValueError, TypeError):
            # If it's a string containing '±', split it and take the first part
            if isinstance(item, str) and '±' in item:
                try:
                    main_value = float(item.split('±')[0].strip())
                    cleaned_values.append(main_value)
                except (ValueError, IndexError):
                    # If parsing fails, append NaN
                    cleaned_values.append(np.nan)
            else:
                # For any other non-numeric type, append NaN
                cleaned_values.append(np.nan)
    return pd.Series(cleaned_values, index=series.index)

def train_and_save_models():
    """
    Trains separate XGBoost models for Viability and Recovery, evaluates them,
    and saves the trained models to the 'trained_models' directory.
    """
    print("--- Starting Model Training ---")
    
    # --- Data Loading and Preprocessing ---
    try:
        # Load the optimized dataset from the CV optimizer
        df = pd.read_csv('../Viability_Optimization_Project/best_swapped_data_cv.csv', encoding='utf-8-sig')
        print("Successfully loaded 'best_swapped_data_cv.csv'.")
    except FileNotFoundError:
        print("ERROR: 'best_swapped_data_cv.csv' not found. Please ensure the file is present.")
        return

    # Standardize column names to lowercase
    df.rename(columns=lambda c: c.lower().strip(), inplace=True)

    # Clean the target columns ('viability', 'recovery')
    targets = ['viability', 'recovery']
    for target in targets:
        if target in df.columns:
            df[target] = clean_target_column(df[target])

    # --- Feature Engineering ---
    # One-hot encode categorical features
    if 'cooling_rate' in df.columns:
        df = pd.get_dummies(df, columns=['cooling_rate'], dummy_na=False)
        print("Successfully one-hot encoded 'cooling_rate'.")

    # Define columns that are not features
    non_feature_cols = ['doubling time', 'all_ingredient'] + targets
    # Select potential numerical features
    potential_feature_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    # Final list of features by excluding non-feature columns
    feature_cols = [col for col in potential_feature_cols if col not in non_feature_cols]
    
    # Identify and remove constant columns, but explicitly keep 'dmso' if it exists
    constant_cols = [
        col for col in feature_cols 
        if df[col].nunique(dropna=False) <= 1 and col != 'dmso'
    ]
    if constant_cols:
        print(f"Identified and will ignore {len(constant_cols)} constant columns (excluding 'dmso').")
        feature_cols = [col for col in feature_cols if col not in constant_cols]
    
    print(f"Found {len(feature_cols)} potential features for training.")

    # --- Model Training and Evaluation Loop ---
    models_dir = 'trained_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: '{models_dir}'")

    for target in targets:
        print(f"\n--- Processing model for: {target.replace('_', ' ').title()} ---")
        
        # Prepare data for the current target
        df_target = df.dropna(subset=[target])
        
        if len(df_target) < 10:
            print(f"Insufficient data for '{target}' (only {len(df_target)} rows). Skipping.")
            continue
            
        X = df_target[feature_cols]
        y = df_target[target]
        
        # --- Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

        # Initialize the XGBoost Regressor model
        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=200, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        
        # --- Model Training and Evaluation ---
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        
        print(f"Single Train/Test Split R² score: {score:.4f}")

        # --- Final Model Training ---
        print("\nTraining final model on the entire dataset...")
        model.fit(X, y)
        
        # --- Feature Importance ---
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print(f"\n--- Top 5 Most Important Features for: {target.replace('_', ' ').title()} ---")
        print(feature_importance_df.head(5).to_string(index=False))
        
        # --- Save the Final Trained Model ---
        model_filename = f"{target.replace('_', ' ').title()}_model.joblib"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"\nFinal model saved to: {model_path}")

    print(f"\n{'='*50}")
    print("Success! Models have been evaluated with CV and final versions saved.")
    print(f"{'='*50}")

if __name__ == '__main__':
    train_and_save_models()