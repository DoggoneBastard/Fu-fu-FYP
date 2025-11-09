import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    Trains separate Random Forest models for Viability and Recovery, evaluates them,
    and saves the trained models to the 'trained_rf_models' directory.
    """
    print("--- Starting Model Training ---")
    
    # --- Data Loading and Preprocessing ---
    try:
        # Load data from the parent directory
        df = pd.read_csv('../final_data.csv', encoding='utf-8-sig')
        print("Successfully loaded 'final_data.csv'.")
    except FileNotFoundError:
        print("ERROR: 'final_data.csv' not found. Please ensure the file is present.")
        return

    # Standardize column names to lowercase
    df.rename(columns=lambda c: c.lower().strip(), inplace=True)

    # Clean the target columns ('viability', 'recovery')
    targets = ['viability', 'recovery']
    for target in targets:
        if target in df.columns:
            df[target] = clean_target_column(df[target])
    
    # --- Feature Engineering ---
    # Define columns that are not features
    non_feature_cols = ['name', 'cooling_rate', 'doubling_time_h', 'doubling time', 'all_ingredient'] + targets
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

    # --- Model Training Loop ---
    # Create a directory to save the models if it doesn't exist
    models_dir = 'trained_rf_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: '{models_dir}'")

    # Loop through each target to train a separate model
    for target in targets:
        print(f"\n--- Training model for: {target.replace('_', ' ').title()} ---")
        
        # Check if the target column exists
        if target not in df.columns:
            print(f"Target column '{target}' not found in data. Skipping.")
            continue

        # Prepare data for the current target: drop rows where the target is missing
        df_target = df.dropna(subset=[target])
        
        # Ensure there is enough data to train a model
        if len(df_target) < 10:
            print(f"Insufficient data for '{target}' (only {len(df_target)} rows). Skipping model training.")
            continue
            
        print(f"Using {len(df_target)} rows for training.")
        
        # Define features (X) and target (y)
        X = df_target[feature_cols].fillna(0) # Fill any remaining NaNs in features with 0
        y_target = df_target[target]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
        
        # Initialize and train the Random Forest Regressor model
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1 # Use all available CPU cores
        )
        model.fit(X_train, y_train)
        
        # Evaluate the model on the test set
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        print(f"Model for {target.replace('_', ' ').title()} trained. R² score on test set: {r2:.4f}")
        
        # --- Save the Trained Model ---
        model_filename = f"{target.replace('_', ' ').title()}_model_rf.joblib"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

    print(f"\n{'='*50}")
    print("Success! Viability and Recovery models have been trained and saved.")
    print(f"{'='*50}")

if __name__ == '__main__':
    train_and_save_models()