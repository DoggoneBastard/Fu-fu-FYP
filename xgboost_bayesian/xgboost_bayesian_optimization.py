import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import warnings
import os

# Ignore warnings from scikit-optimize and FutureWarning for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='skopt')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Global Configuration ---
# Path to the input file, relative to the script's location
INPUT_FILE = '../final_data.csv' 
# Path to the output file that will be created by the preprocessing step
PROCESSED_FILE = 'ml_ready_data.csv'
# Define non-feature columns to accurately identify features in various stages
NON_FEATURE_COLS = ['viability', 'recovery', 'doubling time (h)', 'cooling rate', 'all_ingredient']


def preprocess_data_for_ml(input_path, output_path):
    """
    Stage 1: Preprocess data for machine learning.
    This involves loading the data, cleaning column names, handling missing values,
    and processing target columns that may contain non-numeric values.

    Args:
        input_path (str): The path to the raw data file.
        output_path (str): The path to save the processed data file.

    Returns:
        bool: True if preprocessing is successful, False otherwise.
    """
    print("--- Stage 1: Starting Data Preprocessing ---")
    
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}. Please ensure second_process.py has been run successfully.")
        return False

    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Rename 'dmso(%)' to a more consistent format if it exists
    if 'dmso(%)' in df.columns:
        df.rename(columns={'dmso(%)': 'dmso (%)'}, inplace=True)

    # Identify feature columns and fill missing values with 0
    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS and col not in df.columns[:1]]
    df[feature_cols] = df[feature_cols].fillna(0)
    print(f"1. Filled missing values with 0 in {len(feature_cols)} feature columns.")

    def clean_target_column(series):
        """Helper function to clean target columns like 'viability' and 'recovery'."""
        cleaned_values = []
        for item in series:
            try:
                cleaned_values.append(float(item))
            except (ValueError, TypeError):
                if isinstance(item, str) and '±' in item:
                    try:
                        cleaned_values.append(float(item.split('±')[0].strip()))
                    except (ValueError, IndexError):
                        cleaned_values.append(np.nan)
                else:
                    cleaned_values.append(np.nan)
        return pd.Series(cleaned_values, index=series.index)

    # Apply the cleaning function to the target columns
    for col in ['viability', 'recovery']:
        if col in df.columns:
            original_type = df[col].dtype
            df[col] = clean_target_column(df[col])
            if original_type == 'object' and df[col].dtype == 'float64':
                print(f"2. Successfully processed error values in '{col}' column.")

    # Save the preprocessed data
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n--- Preprocessing Complete ---")
    print(f"File '{output_path}' has been generated and is ready for machine learning modeling.")
    return True


def find_optimal_protocol(df, target_cols, weights):
    """
    Stage 2: Use machine learning and Bayesian optimization to find the optimal protocol.
    This function trains a surrogate XGBoost model and then uses Bayesian optimization
    to find the input parameters that maximize a composite score.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_cols (list): The target columns to optimize (e.g., ['viability', 'recovery']).
        weights (dict): The weights for each target and for the DMSO penalty.

    Returns:
        float: The predicted theoretical maximum composite score from the optimization.
    """
    print(f"\n--- Stage 2: Finding the Optimal Protocol ---")
    
    # Prepare the data for this specific task
    df_task = df.dropna(subset=target_cols).copy()
    
    # Handle categorical 'cooling rate' feature
    df_task['cooling rate'].fillna('unknown', inplace=True)
    cooling_rate_categories = df_task['cooling rate'].unique().tolist()
    
    # One-hot encode the 'cooling rate'
    base_feature_cols = [col for col in df_task.columns if col not in NON_FEATURE_COLS and col not in df_task.columns[:1]]
    df_encoded = pd.get_dummies(df_task, columns=['cooling rate'], prefix='cr')
    encoded_cr_cols = [col for col in df_encoded.columns if col.startswith('cr_')]
    feature_cols = base_feature_cols + encoded_cr_cols
    
    X = df_encoded[feature_cols].copy()
    
    # Include DMSO in the 'y' DataFrame for scaling, even if it's not a primary target
    all_targets_for_scaling = list(dict.fromkeys(target_cols + ['dmso (%)']))
    y = df_encoded[all_targets_for_scaling].copy()

    # Remove constant columns that provide no information
    constant_cols = [col for col in feature_cols if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"Note: Found and automatically removed {len(constant_cols)} constant feature columns.")
        feature_cols = [col for col in feature_cols if col not in constant_cols]
        X = X[feature_cols]

    # Scale features and targets to a [0, 1] range
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=feature_cols)
    y_scaled = pd.DataFrame(scaler_y.fit_transform(y), columns=y.columns)

    # --- Surrogate Model Training ---
    # Create a composite score to be used as the target for the surrogate model
    composite_score = 0
    for col in target_cols:
        composite_score += weights.get(col, 1.0) * y_scaled[col]
    
    # Apply a penalty for DMSO concentration
    dmso_penalty = weights.get('dmso', 0.0) * y_scaled['dmso (%)'] if 'dmso (%)' in y_scaled.columns else 0
    composite_score -= dmso_penalty
    
    # Train an XGBoost model to predict the composite score
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_scaled, composite_score)
    
    # --- Bayesian Optimization ---
    # Define the search space for the optimizer
    search_space_base = [col for col in base_feature_cols if col in feature_cols]
    search_space = [Real(low=df_task[col].min(), high=df_task[col].max(), name=col) for col in search_space_base]
    search_space.append(Categorical(cooling_rate_categories, name='cooling rate'))

    @use_named_args(search_space)
    def objective_function(**params):
        """The objective function for the Bayesian optimizer to minimize."""
        # Constraint: the sum of percentage-based ingredients should not exceed 100
        percent_sum = sum(v for k, v in params.items() if k.endswith(' (%)'))
        if percent_sum > 100:
            return 1e9 # Return a large value to penalize invalid solutions

        # Prepare the input for the surrogate model
        cooling_rate_val = params.pop('cooling rate')
        point_dict = params.copy()
        for cat in cooling_rate_categories:
            point_dict[f'cr_{cat}'] = 1 if cat == cooling_rate_val else 0
        
        point = pd.DataFrame([point_dict])[feature_cols]
        point_scaled = scaler_X.transform(point)
        
        # Predict the score and negate it because gp_minimize minimizes
        score = model.predict(point_scaled)
        return -score[0]

    # Run the optimization
    result = gp_minimize(func=objective_function, dimensions=search_space, n_calls=200, random_state=42, n_jobs=-1)
    
    print("\n--- Search Complete: Optimal Protocol Recommendation ---")
    optimal_protocol = dict(zip([dim.name for dim in search_space], result.x))
    predicted_score = -result.fun
    
    print(f"Optimal Cooling Rate: {optimal_protocol.pop('cooling rate')}")
    print("Optimal Ingredient Combination (concentration > 0.01):")
    for component, value in optimal_protocol.items():
        if value > 0.01:
            print(f"  - {component}: {value:.4f}")
            
    print(f"\nModel's Predicted Theoretical Maximum Composite Score: {predicted_score:.4f}")
    return predicted_score


def evaluate_potential(df, target_cols, weights, theoretical_best_score):
    """
    Stage 3: Evaluate the performance of existing data and calculate the potential for optimization.
    This function compares the best result in the existing dataset with the theoretical
    optimum found by the Bayesian optimization.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_cols (list): The target columns being evaluated.
        weights (dict): The weights used for scoring.
        theoretical_best_score (float): The score of the theoretical best protocol.
    """
    print("\n--- Stage 3: Evaluating Existing Data and Potential ---")
    
    df_task = df.dropna(subset=target_cols).copy()
    df_task.reset_index(drop=True, inplace=True)

    # Identify feature columns, excluding constant ones
    base_feature_cols = [col for col in df_task.columns if col not in NON_FEATURE_COLS and col not in df_task.columns[:1]]
    constant_cols = [col for col in base_feature_cols if df_task[col].nunique(dropna=False) <= 1]
    if constant_cols:
        base_feature_cols = [col for col in base_feature_cols if col not in constant_cols]

    # Scale features and targets to calculate the composite score for existing data
    all_targets_for_scaling = list(dict.fromkeys(target_cols + ['dmso (%)']))
    X = df_task[base_feature_cols].copy()
    y = df_task[all_targets_for_scaling].copy()
    
    scaler_y = MinMaxScaler()
    y_scaled = pd.DataFrame(scaler_y.fit_transform(y), columns=y.columns)

    # Calculate the composite score for every data point in the existing dataset
    composite_score = 0
    for col in target_cols:
        composite_score += weights.get(col, 1.0) * y_scaled[col]
    dmso_penalty = weights.get('dmso', 0.0) * y_scaled['dmso (%)'] if 'dmso (%)' in y_scaled.columns else 0
    composite_score -= dmso_penalty
    
    # Find the best score and protocol within the existing data
    max_score_in_data = composite_score.max()
    best_protocol_position = composite_score.idxmax()
    best_protocol_info = df_task.iloc[best_protocol_position]

    print(f"\nAmong {len(df_task)} valid experimental data points:")
    print(f"The composite score of the best existing protocol is: {max_score_in_data:.4f}")
    
    print("\nFor comparison:")
    print(f"The model's predicted score for the theoretical optimal protocol is: {theoretical_best_score:.4f}")
    
    # Calculate the percentage improvement potential
    improvement_potential = ((theoretical_best_score - max_score_in_data) / max_score_in_data) * 100 if max_score_in_data > 0 else float('inf')
    print(f"\nThis implies a theoretical performance improvement potential of up to: {improvement_potential:.2f}%")
    
    # Display details of the best protocol found in the data
    print("\nDetails of the best existing protocol:")
    print(f"  - Cooling Rate: {best_protocol_info.get('cooling rate', 'N/A')}")
    for col in target_cols:
        print(f"  - {col.capitalize()}: {best_protocol_info.get(col, 'N/A'):.2f}")
    print(f"  - DMSO (%): {best_protocol_info.get('dmso (%)', 'N/A'):.2f}")


def run_optimization_task(df, target_cols, weights, task_description):
    """
    Coordinates a full optimization and evaluation task, from finding the optimal
    protocol to evaluating the potential improvement.

    Args:
        df (pd.DataFrame): The main DataFrame.
        target_cols (list): The target columns for this task.
        weights (dict): The weights for this task.
        task_description (str): A description of the task being run.
    """
    print("\n" + "="*80)
    print(f"Starting Task: {task_description}")
    print("="*80)

    print("\n--- Preparing Task Data ---")
    
    # Filter data for the current task
    task_df = df.dropna(subset=target_cols).copy()
    
    if 'dmso (%)' in task_df.columns and task_df['dmso (%)'].isnull().any():
        task_df['dmso (%)'].fillna(0, inplace=True)

    print(f"Original data has {len(df)} rows. After filtering for '{', '.join(target_cols)}', {len(task_df)} rows remain.")
    if len(task_df) < 2: # Need at least 2 data points for training/evaluation
        print("Insufficient data to perform this optimization task.")
        return

    # Run the core optimization and evaluation stages
    theoretical_best_score = find_optimal_protocol(task_df.copy(), target_cols, weights)
    evaluate_potential(task_df.copy(), target_cols, weights, theoretical_best_score)


def main():
    """
    Main function: Coordinates running all optimization tasks and provides an interactive menu.
    """
    print("Welcome to the Cell Cryopreservation Protocol Optimization Suite")
    
    # --- Stage 1 ---
    if not preprocess_data_for_ml(INPUT_FILE, PROCESSED_FILE):
        return # Stop if preprocessing fails
        
    try:
        df = pd.read_csv(PROCESSED_FILE, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: Preprocessed file not found at {PROCESSED_FILE}.")
        return

    # --- Interactive Menu ---
    while True:
        print("\n" + "*"*80)
        print("Please select the optimization task you want to run:")
        print("1. Comprehensive Optimization (Maximize Viability and Recovery, Minimize DMSO)")
        print("2. Specialized Optimization (Maximize Viability, Minimize DMSO)")
        print("3. Specialized Optimization (Maximize Recovery, Minimize DMSO)")
        print("4. Run all of the above tasks")
        print("5. Exit program")
        print("*"*80)

        choice = input("Please enter your choice (1-5): ")

        if choice == '1':
            if 'viability' in df.columns and 'recovery' in df.columns:
                run_optimization_task(
                    df=df.copy(),
                    target_cols=['viability', 'recovery'],
                    weights={'viability': 1.0, 'recovery': 1.0, 'dmso': 1.0},
                    task_description="Comprehensive Optimization: Maximize Viability and Recovery, while minimizing DMSO"
                )
            else:
                print("Cannot perform this task because 'viability' or 'recovery' column is missing from the data.")
        
        elif choice == '2':
            if 'viability' in df.columns:
                run_optimization_task(
                    df=df.copy(),
                    target_cols=['viability'],
                    weights={'viability': 1.0, 'dmso': 1.0},
                    task_description="Specialized Optimization: Maximize Viability, while minimizing DMSO"
                )
            else:
                print("Cannot perform this task because 'viability' column is missing from the data.")

        elif choice == '3':
            if 'recovery' in df.columns:
                run_optimization_task(
                    df=df.copy(),
                    target_cols=['recovery'],
                    weights={'recovery': 1.0, 'dmso': 1.0},
                    task_description="Specialized Optimization: Maximize Recovery, while minimizing DMSO"
                )
            else:
                print("Cannot perform this task because 'recovery' column is missing from the data.")

        elif choice == '4':
            # Run all tasks sequentially
            if 'viability' in df.columns and 'recovery' in df.columns:
                run_optimization_task(
                    df=df.copy(),
                    target_cols=['viability', 'recovery'],
                    weights={'viability': 1.0, 'recovery': 1.0, 'dmso': 1.0},
                    task_description="Comprehensive Optimization: Maximize Viability and Recovery, while minimizing DMSO"
                )
            else:
                print("Skipping comprehensive optimization task because 'viability' or 'recovery' column is missing.")
            
            if 'viability' in df.columns:
                run_optimization_task(
                    df=df.copy(),
                    target_cols=['viability'],
                    weights={'viability': 1.0, 'dmso': 1.0},
                    task_description="Specialized Optimization: Maximize Viability, while minimizing DMSO"
                )
            else:
                print("Skipping Viability specialized optimization task because 'viability' column is missing.")
            
            if 'recovery' in df.columns:
                run_optimization_task(
                    df=df.copy(),
                    target_cols=['recovery'],
                    weights={'recovery': 1.0, 'dmso': 1.0},
                    task_description="Specialized Optimization: Maximize Recovery, while minimizing DMSO"
                )
            else:
                print("Skipping Recovery specialized optimization task because 'recovery' column is missing.")

        elif choice == '5':
            print("Thank you for using the program. Exiting.")
            break
        
        else:
            print("Invalid input. Please enter a number between 1 and 5.")


if __name__ == '__main__':
    main()