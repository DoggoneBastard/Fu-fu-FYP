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

# --- 1. Load and Prepare Data ---
print("Step 1: Loading and preparing data...")
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, DATA_FILE)
df = pd.read_csv(data_file_path)

# Standardize column names to be consistent with the training script
df.rename(columns=lambda c: c.lower().strip(), inplace=True)


# a. Remove rows where the target variable is missing
df.dropna(subset=[TARGET_COLUMN], inplace=True)
print(f"Shape after dropping NaN in target: {df.shape}")

# b. Separate features and target
X = df.drop(columns=EXCLUDE_COLS)
y = df[TARGET_COLUMN]

# c. One-hot encode categorical features
X = pd.get_dummies(X, columns=['cooling rate'], dummy_na=False)
print(f"Features shape after one-hot encoding: {X.shape}")

# d. Align X and y indices
y = y.loc[X.index]

# --- New Helper Function for Cross-Validation ---
def evaluate_with_cv(X_pool, y_pool, n_splits=N_SPLITS_KFOLD):
    """
    Evaluates a model using K-Fold cross-validation on the given data pool.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_r2s, fold_rmses, fold_maes, fold_mapes = [], [], [], []

    # Ensure there's enough data for the specified number of splits
    if len(X_pool) < n_splits:
        print(f"Warning: Pool size ({len(X_pool)}) is smaller than n_splits ({n_splits}). Cannot perform CV.")
        return -np.inf, np.inf, np.inf, np.inf

    for train_idx, test_idx in kf.split(X_pool):
        X_train, X_test = X_pool.iloc[train_idx], X_pool.iloc[test_idx]
        y_train, y_test = y_pool.iloc[train_idx], y_pool.iloc[test_idx]

        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_r2s.append(r2_score(y_test, preds))
        fold_rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        fold_maes.append(mean_absolute_error(y_test, preds))
        fold_mapes.append(mean_absolute_percentage_error(y_test, preds))

    return np.mean(fold_r2s), np.mean(fold_rmses), np.mean(fold_maes), np.mean(fold_mapes)

# --- 2. Initial Data Split ---
print("\nStep 2: Splitting data into training and holding pools...")
# We use indices to manage the pools efficiently
initial_indices = X.index.tolist()

training_indices, holding_indices = train_test_split(
    initial_indices, test_size=0.5, random_state=42
)

print(f"Training pool size: {len(training_indices)}")
print(f"Holding pool size: {len(holding_indices)}")

# --- 3. Iterative Swapping Optimization ---
print("\nStep 3: Starting iterative swapping optimization...")

best_composite_score = -np.inf
best_metrics = {}
best_training_indices = None
patience_counter = 0
r2_history = []
viability_history = []
mae_history = []
mape_history = []
rmse_history = []
swapped_in_history = []
swapped_out_history = []

# Calculate and store initial viability
initial_viability = y.loc[training_indices].mean()
print(f"Initial average viability of training pool: {initial_viability:.4f}")

for i in range(MAX_ITERATIONS):
    print(f"\n--- Iteration {i+1}/{MAX_ITERATIONS} ---")
    
    # a. Train and evaluate on the current training pool using K-Fold Cross-Validation
    X_train, y_train = X.loc[training_indices], y.loc[training_indices]
    
    print(f"Evaluating current pool of size {len(X_train)} with {N_SPLITS_KFOLD}-Fold CV...")
    current_r2, current_rmse, current_mae, current_mape = evaluate_with_cv(X_train, y_train)
    
    # --- Multi-objective Evaluation ---
    current_viability = y.loc[training_indices].mean()
    
    # Calculate the composite score
    current_score = (0.5 * current_r2) + (0.5 * current_viability) - (2.0 * current_rmse)

    # Track history
    r2_history.append(current_r2)
    viability_history.append(current_viability)
    mae_history.append(current_mae)
    mape_history.append(current_mape)
    rmse_history.append(current_rmse)
    
    print(f"Current Composite Score: {current_score:.4f} (R2: {current_r2:.4f}, Viability: {current_viability:.4f}, RMSE: {current_rmse:.4f})")

    # b. Check for improvement and update best results based on the composite score
    if current_score > best_composite_score:
        best_composite_score = current_score
        best_training_indices = list(training_indices) # Save a copy
        best_metrics = {
            'r2': current_r2,
            'viability': current_viability,
            'mae': current_mae,
            'mape': current_mape,
            'rmse': current_rmse,
            'composite_score': current_score
        }
        patience_counter = 0
        print(f"*** New best composite score found! ***")
    else:
        patience_counter += 1

    # c. Check for termination conditions
    if patience_counter >= PATIENCE:
        print(f"\nNo improvement in {PATIENCE} iterations. Stopping optimization.")
        break

    # d. Determine dynamic swap size (k)
    # We can still use R2 gap as a heuristic for swap size
    r2_gap = TARGET_R2_SCORE - current_r2
    # The swap size is proportional to the gap, with a minimum of 1
    # and a max of 25% of the pool size.
    swap_ratio = max(0, r2_gap / TARGET_R2_SCORE) # Can be > 1 if R2 is negative
    k = int(np.clip(len(training_indices) * 0.5 * swap_ratio, 1, len(training_indices) * 0.25))
    print(f"R2 Gap: {r2_gap:.4f}. Swap size k: {k}")

    # e. Perform swap
    # If k is larger than the pool, use the max possible size
    k = min(k, len(training_indices), len(holding_indices))
    if k == 0: k = 1 # Ensure we always swap at least one

    # Select indices to swap
    indices_to_swap_out = random.sample(training_indices, k)
    indices_to_swap_in = random.sample(holding_indices, k)

    # Create the new potential pools by swapping
    temp_training_indices = list(training_indices)
    temp_holding_indices = list(holding_indices)

    for out_idx, in_idx in zip(indices_to_swap_out, indices_to_swap_in):
        temp_training_indices.remove(out_idx)
        temp_training_indices.append(in_idx)
        temp_holding_indices.remove(in_idx)
        temp_holding_indices.append(out_idx)

    # f. Evaluate the swap using K-Fold Cross-Validation
    X_train_new, y_train_new = X.loc[temp_training_indices], y.loc[temp_training_indices]
    print(f"Evaluating proposed swap pool of size {len(X_train_new)} with {N_SPLITS_KFOLD}-Fold CV...")
    new_r2, new_rmse, new_mae, new_mape = evaluate_with_cv(X_train_new, y_train_new)
    
    new_viability = y.loc[temp_training_indices].mean()
    
    # Calculate the composite score for the new set
    new_score = (0.5 * new_r2) + (0.5 * new_viability) - (2.0 * new_rmse)
    print(f"Score after proposed swap: {new_score:.4f} (R2: {new_r2:.4f}, Viability: {new_viability:.4f}, RMSE: {new_rmse:.4f})")

    # g. Decide whether to keep the swap based on the composite score
    if new_score > current_score:
        training_indices = temp_training_indices
        holding_indices = temp_holding_indices
        # Record the successful swap
        swapped_in_history.extend(indices_to_swap_in)
        swapped_out_history.extend(indices_to_swap_out)
        print("Swap was beneficial. Keeping the new training pool.")
    else:
        print("Swap was not beneficial. Reverting.")

# --- 4. Final Results and Saving ---
print("\n--- Optimization Finished ---")

if best_training_indices is None:
    print("Error: Optimization did not run. No results to show.")
else:
    print(f"Best Composite Score achieved: {best_metrics.get('composite_score', -1):.4f}")
    print(f"  - Corresponding R2 Score: {best_metrics.get('r2', -1):.4f}")
    print(f"  - Corresponding Avg Viability: {best_metrics.get('viability', -1):.4f}")
    print(f"  - Corresponding RMSE: {best_metrics.get('rmse', -1):.4f}")
    print(f"Number of data points in the final optimal set: {len(best_training_indices)}")

    # Save the best dataset found
    output_data_path = os.path.join(script_dir, OUTPUT_DATA_FILE)
    best_df = df.loc[best_training_indices]
    best_df.to_csv(output_data_path, index=False)
    print(f"Optimal dataset saved to '{output_data_path}'")

    # Train and save the final model on the full best dataset
    print("Training final model on the entire optimal dataset...")
    output_model_path = os.path.join(script_dir, OUTPUT_MODEL_FILE)
    X_best_final, y_best_final = X.loc[best_training_indices], y.loc[best_training_indices]
    final_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    final_model.fit(X_best_final, y_best_final)
    joblib.dump(final_model, output_model_path)
    print(f"Final model saved to '{output_model_path}'")

    # --- 5. Analysis of Swapped Components ---
    print("\n--- Analysis of Swapped Components ---")
    swapped_in_counts = Counter(swapped_in_history)
    swapped_out_counts = Counter(swapped_out_history)
    print("Top 5 most frequently swapped-in data points:")
    for idx, count in swapped_in_counts.most_common(5):
        print(f"  Index {idx}: swapped in {count} times")
    print("\nTop 5 most frequently swapped-out data points:")
    for idx, count in swapped_out_counts.most_common(5):
        print(f"  Index {idx}: swapped out {count} times")

    # --- 6. Final Metrics Summary ---
    print("\n--- Final Metrics Summary ---")
    final_viability = y.loc[best_training_indices].mean()
    print(f"Initial Average Viability: {initial_viability:.4f}")
    print(f"Final Average Viability: {final_viability:.4f}")
    print(f"Best R2 Score Achieved: {best_metrics.get('r2', -1):.4f}")
    if r2_history:
        print(f"Final R2 Score (at last iteration): {r2_history[-1]:.4f}")
    if mae_history:
        print(f"Final MAE (at last iteration): {mae_history[-1]:.4f}")
    if mape_history:
        print(f"Final MAPE (at last iteration): {mape_history[-1]:.4f}")
    if rmse_history:
        print(f"Final RMSE (at last iteration): {rmse_history[-1]:.4f}")