import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import joblib
import random
# import matplotlib.pyplot as plt
from collections import Counter
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- Configuration ---
DATA_FILE = '../final_data.csv'
TARGET_COLUMN = 'viability'
TARGET_R2_SCORE = 0.7
MAX_ITERATIONS = 200  # Safety break to prevent infinite loops
PATIENCE = 50         # Stop if no improvement after this many iterations
N_SPLITS_KFOLD = 5    # Number of splits for K-Fold Cross-Validation
OUTPUT_DATA_FILE = 'best_swapped_data_cv.csv'
OUTPUT_MODEL_FILE = 'viability_model_from_swapped_data_cv.joblib'
EXCLUDE_COLS = ['all_ingredient', 'viability', 'recovery', 'doubling time']

# --- New Configuration for Wrapper Method ---
# List of training data ratios to test. 0.5 means 50% for training, 50% for holding.
TRAIN_RATIOS_TO_TEST = [0.4, 0.5, 0.6, 0.7] 
OUTPUT_DATA_FILE_TEMPLATE = 'best_swapped_data_ratio_{}.csv'
OUTPUT_MODEL_FILE_TEMPLATE = 'viability_model_ratio_{}.joblib'


# --- 1. Load and Prepare Data (Done once) ---
print("Step 1: Loading and preparing data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, DATA_FILE)
df = pd.read_csv(data_file_path)

df.rename(columns=lambda c: c.lower().strip(), inplace=True)
df.dropna(subset=[TARGET_COLUMN], inplace=True)
print(f"Initial data shape after dropping NaN in target: {df.shape}")

X_full = df.drop(columns=EXCLUDE_COLS)
y_full = df[TARGET_COLUMN]
X_full = pd.get_dummies(X_full, columns=['cooling rate'], dummy_na=False)
y_full = y_full.loc[X_full.index]


# --- Helper Function for Cross-Validation ---
def evaluate_with_cv(X_pool, y_pool, n_splits=N_SPLITS_KFOLD):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    if len(X_pool) < n_splits:
        return -np.inf, np.inf
    
    fold_r2s, fold_rmses = [], []
    for train_idx, test_idx in kf.split(X_pool):
        X_train, X_test = X_pool.iloc[train_idx], X_pool.iloc[test_idx]
        y_train, y_test = y_pool.iloc[train_idx], y_pool.iloc[test_idx]
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fold_r2s.append(r2_score(y_test, preds))
        fold_rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        
    return np.mean(fold_r2s), np.mean(fold_rmses)

# --- Main Loop for Testing Ratios ---
overall_best_results = []

print(f"\nStarting optimization wrapper for ratios: {TRAIN_RATIOS_TO_TEST}")
print("="*70)

for ratio in TRAIN_RATIOS_TO_TEST:
    print(f"\n\n===== TESTING TRAINING RATIO: {ratio*100:.0f}% =====\n")
    
    # --- 2. Initial Data Split for the current ratio ---
    print(f"Step 2: Splitting data with {ratio*100:.0f}% for training...")
    initial_indices = X_full.index.tolist()
    
    if len(initial_indices) < 2:
        print("Not enough data to split. Skipping ratio.")
        continue

    training_indices, holding_indices = train_test_split(
        initial_indices, train_size=ratio, random_state=42
    )
    print(f"Training pool size: {len(training_indices)}, Holding pool size: {len(holding_indices)}")

    # --- 3. Iterative Swapping Optimization for the current ratio ---
    print("\nStep 3: Starting iterative swapping optimization...")
    best_composite_score = -np.inf
    best_metrics_for_ratio = {}
    best_training_indices_for_ratio = None
    patience_counter = 0

    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} (Ratio {ratio}) ---")
        
        X_train, y_train = X_full.loc[training_indices], y_full.loc[training_indices]
        current_r2, current_rmse = evaluate_with_cv(X_train, y_train)
        current_viability = y_train.mean()
        current_score = (0.5 * current_r2) + (0.5 * current_viability) - (2.0 * current_rmse)
        
        print(f"Current Composite Score: {current_score:.4f} (R2: {current_r2:.4f}, Viability: {current_viability:.4f}, RMSE: {current_rmse:.4f})")

        if current_score > best_composite_score:
            best_composite_score = current_score
            best_training_indices_for_ratio = list(training_indices)
            best_metrics_for_ratio = {
                'r2': current_r2, 'viability': current_viability, 'rmse': current_rmse,
                'composite_score': current_score, 'training_ratio': ratio,
                'final_training_size': len(training_indices)
            }
            patience_counter = 0
            print("*** New best composite score for this ratio! ***")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nNo improvement in {PATIENCE} iterations. Stopping optimization for this ratio.")
            break

        r2_gap = TARGET_R2_SCORE - current_r2
        swap_ratio = max(0, r2_gap / TARGET_R2_SCORE)
        k = int(np.clip(len(training_indices) * 0.5 * swap_ratio, 1, len(training_indices) * 0.25))
        k = min(k, len(training_indices), len(holding_indices))
        if k == 0: k = 1

        indices_to_swap_out = random.sample(training_indices, k)
        indices_to_swap_in = random.sample(holding_indices, k)

        temp_training_indices = list(set(training_indices) - set(indices_to_swap_out)) + indices_to_swap_in
        
        X_train_new, y_train_new = X_full.loc[temp_training_indices], y_full.loc[temp_training_indices]
        new_r2, new_rmse = evaluate_with_cv(X_train_new, y_train_new)
        new_viability = y_train_new.mean()
        new_score = (0.5 * new_r2) + (0.5 * new_viability) - (2.0 * new_rmse)

        if new_score > current_score:
            training_indices = temp_training_indices
            holding_indices = list((set(initial_indices) - set(training_indices)))
            print("Swap was beneficial. Keeping the new training pool.")
        else:
            print("Swap was not beneficial. Reverting.")

    if best_metrics_for_ratio:
        overall_best_results.append({
            "metrics": best_metrics_for_ratio,
            "indices": best_training_indices_for_ratio
        })
    print(f"\nFinished testing for ratio {ratio}. Best score found: {best_metrics_for_ratio.get('composite_score', -np.inf):.4f}")
    print("-"*70)

# --- 4. Final Summary and Saving the Overall Best Result ---
print("\n\n===== OVERALL OPTIMIZATION FINISHED =====\n")

if not overall_best_results:
    print("Error: No successful optimization runs were completed.")
else:
    # Sort results by the composite score to find the best
    overall_best_results.sort(key=lambda x: x['metrics']['composite_score'], reverse=True)
    
    # Print summary table
    summary_df = pd.DataFrame([res['metrics'] for res in overall_best_results])
    print("--- Summary of Results Across All Ratios ---")
    print(summary_df[['training_ratio', 'final_training_size', 'composite_score', 'r2', 'viability', 'rmse']].round(4).to_string())
    
    # Get the absolute best result
    absolute_best_run = overall_best_results[0]
    best_metrics = absolute_best_run['metrics']
    best_indices = absolute_best_run['indices']
    
    print("\n--- Absolute Best Configuration Found ---")
    print(f"Training Ratio: {best_metrics['training_ratio']*100:.0f}%")
    print(f"Best Composite Score: {best_metrics['composite_score']:.4f}")
    print(f"  - R2 Score: {best_metrics['r2']:.4f}")
    print(f"  - Avg Viability: {best_metrics['viability']:.4f}")
    print(f"  - RMSE: {best_metrics['rmse']:.4f}")
    print(f"Optimal dataset size: {len(best_indices)}")

    # Save the best dataset
    best_ratio_str = f"{best_metrics['training_ratio']:.1f}".replace('.', '_')
    output_data_file = os.path.join(script_dir, OUTPUT_DATA_FILE_TEMPLATE.format(best_ratio_str))
    best_df = df.loc[best_indices]
    best_df.to_csv(output_data_file, index=False)
    print(f"\nOptimal dataset saved to '{output_data_file}'")

    # Train and save the final model on the absolute best dataset
    output_model_file = os.path.join(script_dir, OUTPUT_MODEL_FILE_TEMPLATE.format(best_ratio_str))
    X_best_final, y_best_final = X_full.loc[best_indices], y_full.loc[best_indices]
    final_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    final_model.fit(X_best_final, y_best_final)
    joblib.dump(final_model, output_model_file)
    print(f"Final model saved to '{output_model_file}'")