import pandas as pd
import numpy as np
import os
import joblib
from scipy.optimize import differential_evolution

def get_feature_bounds(feature_cols, data_df):
    """
    Defines the search space (bounds) for each feature for the optimization algorithm.

    The bounds are determined by the minimum and maximum values of each feature in the
    provided dataset. The upper bound is slightly increased (by 10%) to allow the
    optimization to explore values just outside the observed range.

    Args:
        feature_cols (list): A list of column names to be used as features.
        data_df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        list: A list of tuples, where each tuple represents the (min, max) bounds for a feature.
    """
    bounds = []
    for feature in feature_cols:
        min_val = data_df[feature].min()
        max_val = data_df[feature].max()
        # Set the upper bound to 110% of the max value to allow exploration
        bounds.append((min_val, max_val * 1.1)) 
    return bounds

def objective_function(formulation, models, feature_cols, weights, dmso_index):
    """
    Calculates a single score to be minimized by the optimization algorithm.

    The score is a weighted combination of predicted viability, predicted recovery, and
    the amount of DMSO. The goal is to maximize viability and recovery while minimizing
    DMSO, so the viability and recovery predictions are negated in the score calculation.

    Args:
        formulation (np.array): A numpy array of feature values representing one candidate solution.
        models (tuple): A tuple containing the trained viability and recovery models.
        feature_cols (list): The list of feature names, matching the order in `formulation`.
        weights (dict): A dictionary of weights for 'viability', 'recovery', and 'dmso'.
        dmso_index (int): The index of the DMSO feature in the `feature_cols` list.

    Returns:
        float: The calculated objective score. A lower score is better.
    """
    model_v, model_r = models
    w_v, w_r, w_dmso = weights['viability'], weights['recovery'], weights['dmso']

    # Create a DataFrame from the current formulation for prediction
    input_df = pd.DataFrame([formulation], columns=feature_cols)

    # Predict viability and recovery
    pred_v = model_v.predict(input_df)[0]
    pred_r = model_r.predict(input_df)[0]
    
    # Get the DMSO value from the formulation
    dmso_value = formulation[dmso_index]

    # Calculate the final score. We negate predictions because the optimizer minimizes.
    score = (-pred_v * w_v) + (-pred_r * w_r) + (dmso_value * w_dmso)
    
    return score

def run_optimization_instance(weights, models, feature_cols, bounds, dmso_index, scenario_name):
    """
    Runs a single instance of the differential evolution optimization for a given set of weights.

    Args:
        weights (dict): A dictionary of weights for the objective function.
        models (tuple): The trained machine learning models.
        feature_cols (list): The list of feature names.
        bounds (list): The search space bounds for the features.
        dmso_index (int): The index of the DMSO feature.
        scenario_name (str): The name of the current optimization scenario (e.g., 'Balanced').

    Returns:
        pd.DataFrame: A DataFrame containing the results of this optimization run.
    """
    print(f"\n--- Running Scenario: {scenario_name} ---")
    print(f"Weights: {weights}")
    
    # Run the differential evolution algorithm
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(models, feature_cols, weights, dmso_index),
        strategy='best1bin', maxiter=130, popsize=20, tol=0.01,
        mutation=(0.5, 1), recombination=0.7, seed=42, disp=True
    )

    # Extract the best formulation found by the optimizer
    optimal_formulation = result.x
    
    # Re-predict using the optimal formulation to get final viability and recovery values
    input_df = pd.DataFrame([optimal_formulation], columns=feature_cols)
    pred_v = models[0].predict(input_df)[0]
    pred_r = models[1].predict(input_df)[0]

    # Store the results in a dictionary
    result_data = {
        'scenario': [scenario_name],
        'weight_viability': [weights['viability']],
        'weight_recovery': [weights['recovery']],
        'weight_dmso': [weights['dmso']],
        'predicted_viability': [pred_v],
        'predicted_recovery': [pred_r],
    }
    # Add the values of the optimal formulation to the results dictionary
    for i, col in enumerate(feature_cols):
        result_data[col] = [optimal_formulation[i]]
        
    return pd.DataFrame(result_data)

def main():
    """
    Main function to load resources, prompt the user for the optimization mode,
    and run the optimization scenarios.
    """
    print("--- Optimization Script ---")
    
    # Define paths for models and data
    models_dir = 'trained_models'
    data_path = '../final_data.csv' # Relative path to the parent directory
    
    # Check if required files and directories exist
    if not (os.path.exists(models_dir) and os.path.exists(data_path)):
        print(f"ERROR: '{models_dir}' folder or '{data_path}' not found.")
        print("Please ensure you have run the 'train_models.py' script first.")
        return

    # Load the trained models
    try:
        model_v = joblib.load(os.path.join(models_dir, 'Viability_model.joblib'))
        model_r = joblib.load(os.path.join(models_dir, 'Recovery_model.joblib'))
        models = (model_v, model_r)
        print("Successfully loaded Viability and Recovery models.")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}. Please check file names in '{models_dir}'.")
        return

    # Load and preprocess the data
    df = pd.read_csv(data_path)
    df.rename(columns=lambda c: c.lower().strip(), inplace=True)

    # --- Feature Engineering ---
    # Define target and non-feature columns
    targets = ['viability', 'recovery']
    non_feature_cols = ['name', 'cooling_rate', 'doubling_time_h', 'doubling time', 'all_ingredient'] + targets
    # Identify potential numerical features
    potential_feature_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    # Filter out non-feature columns to get the final feature list
    feature_cols = [col for col in potential_feature_cols if col not in non_feature_cols]
    
    # Identify the DMSO column, which needs special handling
    dmso_col_name = None
    if 'dmso(%)' in feature_cols:
        dmso_col_name = 'dmso(%)'
    elif 'dmso' in feature_cols:
        dmso_col_name = 'dmso'

    # Identify and remove constant columns, but explicitly keep the DMSO column
    constant_cols = [
        col for col in feature_cols 
        if df[col].nunique(dropna=False) <= 1 and col != dmso_col_name
    ]
    if constant_cols:
        print(f"Identified and will ignore {len(constant_cols)} constant columns (excluding '{dmso_col_name}').")
        feature_cols = [col for col in feature_cols if col not in constant_cols]

    # Get the index of the DMSO column for the objective function
    try:
        if dmso_col_name:
            dmso_index = feature_cols.index(dmso_col_name)
        else:
            raise ValueError
    except ValueError:
        print("ERROR: 'dmso' or 'dmso(%)' column not found in features. Cannot apply DMSO penalty.")
        return

    # Get the feature bounds for the optimizer
    bounds = get_feature_bounds(feature_cols, df)
    print(f"Identified {len(feature_cols)} features for optimization.")

    # --- User Interaction for Scenario Selection ---
    while True:
        choice = input("\nChoose the mode: [1] Balanced or [2] Multi-scenario: ")
        if choice in ['1', '2']:
            break
        else:
            print("Invalid input. Please enter '1' or '2'.")

    all_results = []

    # Define scenarios based on user choice
    if choice == '1':
        # Single scenario with balanced weights
        scenarios = { 'Balanced': {'viability': 0.5, 'recovery': 0.5, 'dmso': 0.1} }
    else:
        # Multiple scenarios with different weightings
        print("\nThree weighting of paprameter will be shown...")
        scenarios = {
            'Balanced': {'viability': 0.5, 'recovery': 0.5, 'dmso': 0.1},
            'Viability First': {'viability': 0.8, 'recovery': 0.2, 'dmso': 0.1},
            'Low Toxicity': {'viability': 0.4, 'recovery': 0.4, 'dmso': 0.5},
        }

    # Run optimization for each defined scenario
    for name, weights in scenarios.items():
        result_df = run_optimization_instance(weights, models, feature_cols, bounds, dmso_index, name)
        all_results.append(result_df)

    # --- Finalize and Save Results ---
    # Concatenate results from all scenarios into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Standardize the DMSO column name in the final output
    if dmso_col_name and dmso_col_name != 'dmso':
        final_df.rename(columns={dmso_col_name: 'dmso'}, inplace=True)

    # Save the results to a CSV file
    output_path = 'best_results.csv'
    final_df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"\n{'='*50}")
    print("Success! Optimization complete.")
    print(f"All results have been saved to: {output_path}")
    print(f"{'='*50}\n")
    
    # Display a summary of the best results to the console
    display_cols = ['scenario', 'predicted_viability', 'predicted_recovery', 'dmso']
    print("The summary of best results:")
    print(final_df[display_cols].round(4))

if __name__ == '__main__':
    main()