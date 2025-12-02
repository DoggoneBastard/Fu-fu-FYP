import pandas as pd
import os
import numpy as np

# --- Configuration Constants ---
INPUT_PATH = 'processed_data.csv'
OUTPUT_PATH = 'final_data.csv'
COLS_TO_CLEAN_KEYWORDS = ['viability', 'recovery', 'doubling time']

def load_data(file_path):
    """Loads data from the specified CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please run the first process script to generate it.")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded '{file_path}'.")
        return df
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return None

def preprocess_data(df, keywords):
    """
    Cleans specific columns by extracting numbers and replaces all zero values with NaN.
    """
    print("--- Data Pre-processing ---")
    
    cols_to_process = [col for col in df.columns if any(keyword in col.lower() for keyword in keywords)]
    
    if cols_to_process:
        print(f"Found columns to clean for non-numeric values: {cols_to_process}")
        for col in cols_to_process:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Extracted leading numbers and converted to numeric in specified columns.")
    else:
        print(f"No columns containing keywords {keywords} found to clean.")

    num_zeros = (df.select_dtypes(include=np.number) == 0).sum().sum()
    if num_zeros > 0:
        df.replace(0, np.nan, inplace=True)
        print(f"Replaced {num_zeros} zero value(s) with NaN across the dataset.")
    else:
        print("No zero values found to replace.")
        
    print("Pre-processing complete.\n")
    return df

def clean_sparse_columns(df):
    """
    Interactively identifies and removes feature columns with few entries.
    Also removes rows that have data in the columns being deleted.
    """
    feature_columns = [col for col in df.columns if '(' in col and ')' in col]
    print(f"--- Column-based cleaning ---\nTotal number of feature columns: {len(feature_columns)}\n")
    
    try:
        min_entries_col_str = input("Enter the minimum number of filled values a feature column must have to be kept (e.g., 5). Press Enter to skip: ")
        if min_entries_col_str:
            min_entries_col = int(min_entries_col_str)
            col_counts = df[feature_columns].notna().sum()
            cols_to_check = col_counts[col_counts < min_entries_col].index.tolist()
            
            if cols_to_check:
                print(f"\nFound {len(cols_to_check)} feature columns with fewer than {min_entries_col} entries:")
                for col in cols_to_check:
                    print(f"- {col} (has {col_counts[col]} entries)")

                # Find rows that have data in these sparse columns
                rows_to_delete_mask = df[cols_to_check].notna().any(axis=1)
                rows_to_delete_indices = df[rows_to_delete_mask].index
                num_rows_to_delete = len(rows_to_delete_indices)

                if num_rows_to_delete > 0:
                    print(f"\nWarning: These columns contain data in {num_rows_to_delete} row(s).")
                    print("If you choose to delete these columns, the corresponding rows containing their data will also be deleted.")

                confirm_delete = input("Do you want to delete these columns and their corresponding rows? (yes/no): ").lower().strip()
                if confirm_delete in ['yes', 'y']:
                    # Delete rows first
                    if num_rows_to_delete > 0:
                        df.drop(index=rows_to_delete_indices, inplace=True)
                        print(f"Deleted {num_rows_to_delete} corresponding rows.")
                    
                    # Then delete columns
                    df.drop(columns=cols_to_check, inplace=True)
                    print(f"Deleted {len(cols_to_check)} columns.")
                else:
                    print("Skipped deleting columns and rows.")
            else:
                print(f"No feature columns found with fewer than {min_entries_col} entries.")
        else:
            print("Skipped column cleaning.")
    except ValueError:
        print("Invalid input. Skipping column cleaning.")
    
    print("-" * 20)
    return df

def clean_invalid_rows(df):
    """
    Interactively identifies and removes rows where both 'viability' and 'recovery' are missing or zero.
    """
    print("\n--- Row-based cleaning: Checking for missing 'viability' and 'recovery' ---")

    if 'viability' not in df.columns or 'recovery' not in df.columns:
        print("Warning: 'viability' and/or 'recovery' columns not found. Skipping this cleaning step.")
        return df

    invalid_rows_mask = df['viability'].isna() & df['recovery'].isna()
    rows_to_check_indices = df[invalid_rows_mask].index

    if not rows_to_check_indices.empty:
        print(f"\nFound {len(rows_to_check_indices)} rows where both 'viability' and 'recovery' are missing or zero.")
        print(f"Row numbers in CSV: {[i + 2 for i in rows_to_check_indices]}")

        try:
            confirm_delete_row = input("Do you want to delete these rows? (yes/no): ").lower().strip()
            if confirm_delete_row in ['yes', 'y']:
                df.drop(index=rows_to_check_indices, inplace=True)
                print(f"Deleted {len(rows_to_check_indices)} rows.")
            else:
                print("Skipped deleting rows.")
        except Exception as e:
            print(f"An error occurred during row deletion: {e}")

    else:
        print("No rows found where both 'viability' and 'recovery' are missing or zero.")
        
    return df

def add_culture_medium(df):
    """
    Adds a 'culture medium(%)' column.
    Warns about rows where the calculated culture medium is >= 100%.
    Corrects negative culture medium values (from ingredient sums > 100%) to 0.
    Row numbers reported are relative to the final output file.
    """
    print("\n--- Feature Engineering: Adding 'culture medium(%)' column ---")

    feature_cols = [col for col in df.columns if '(%)' in col]
    
    if not feature_cols:
        print("Warning: No feature columns with '(%)' found. Skipping this step.")
        return df

    print(f"Calculating sum based on {len(feature_cols)} feature columns.")
    
    # Calculate the sum of features for each row
    feature_sum = df[feature_cols].sum(axis=1)

    # Calculate 'culture medium(%)' based on the sum
    df['culture medium(%)'] = 100 - feature_sum

    # --- Check 1: Culture medium is 100% or more (sum of ingredients is <= 0) ---
    high_medium_rows = df[df['culture medium(%)'] >= 100]
    if not high_medium_rows.empty:
        print(f"\nWarning: Found {len(high_medium_rows)} rows where 'culture medium(%)' is 100% or more.")
        print("This might indicate rows with no listed ingredients. Row numbers refer to the final output file.")
        for index in high_medium_rows.index:
            # Get the integer position of the row in the current dataframe
            positional_index = df.index.get_loc(index)
            # Final CSV line number = positional_index + 1 (for 1-based) + 1 (for header)
            final_line_num = positional_index + 2
            print(f"  - Check row (final output line ~{final_line_num}): Calculated 'culture medium(%)' is {df.loc[index, 'culture medium(%)']:.2f}%")

    # --- Check 2 & Interactive Deletion: Culture medium is negative (sum of ingredients > 100) ---
    negative_medium_rows = df[df['culture medium(%)'] < 0]
    if not negative_medium_rows.empty:
        print(f"\nCritical Warning: Found {len(negative_medium_rows)} rows where the sum of ingredients exceeds 100%.")
        print("Row numbers refer to the potential final output file if no deletions are made.")
        
        for index in negative_medium_rows.index:
            positional_index = df.index.get_loc(index)
            final_line_num = positional_index + 2
            original_sum = 100 - df.loc[index, 'culture medium(%)']
            print(f"  - Row (final output line ~{final_line_num}): Ingredient sum is {original_sum:.2f}%")

        while True:
            choice = input(f"\nDo you want to delete these {len(negative_medium_rows)} rows? (yes/no): ").lower().strip()
            if choice in ['yes', 'no']:
                break
            print("Invalid input. Please enter 'yes' or 'no'.")

        if choice == 'yes':
            rows_to_delete = negative_medium_rows.index
            df.drop(rows_to_delete, inplace=True)
            print(f"\n{len(rows_to_delete)} rows have been deleted.")
        else:
            # Correct negative values to 0
            df.loc[df['culture medium(%)'] < 0, 'culture medium(%)'] = 0
            print("\nNo rows were deleted. 'culture medium(%)' for these rows has been corrected to 0.")
    
    print("\n'culture medium(%)' column added and checked successfully.")
    return df

def fill_feature_nan_with_zero(df):
    """Fills NaN values in feature columns (containing '%') with 0."""
    print("\n--- Pre-processing: Filling NaN in feature columns with 0 ---")
    feature_cols = [col for col in df.columns if '(%)' in col]
    
    if not feature_cols:
        print("Warning: No feature columns with '(%)' found. Skipping NaN fill.")
        return df

    # Check for NaNs only in feature columns
    nan_counts = df[feature_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if not nan_cols.empty:
        print(f"Found NaN values in the following {len(nan_cols)} feature columns. They will be filled with 0.")
        for col, count in nan_cols.items():
            print(f"  - Column '{col}': {count} NaN values")
        
        # Fill NaNs only in the identified feature columns
        df[feature_cols] = df[feature_cols].fillna(0)
        print("\nNaN values in feature columns have been successfully filled with 0.")
    else:
        print("No NaN values found in any feature columns.")
        
    return df

def clean_rapid_freeze_rows(df):
    """
    Interactively finds and removes rows where 'cooling rate' is 'rapid freeze'
    (handles spaces, underscores, and case variations).
    """
    print("\n--- Row-based cleaning: Checking for 'rapid freeze' in 'cooling rate' ---")

    if 'cooling rate' not in df.columns:
        print("Warning: 'cooling rate' column not found. Skipping this cleaning step.")
        return df

    # Create a cleaned series for robust matching
    # Fill NA with empty string to avoid errors, convert to lower, replace underscore, and strip whitespace
    cleaned_series = df['cooling rate'].fillna('').astype(str).str.lower().str.replace('_', ' ', regex=False).str.strip()
    
    # Find rows where 'cooling rate' matches 'rapid freeze'
    rapid_freeze_mask = cleaned_series == 'rapid freeze'
    rows_to_delete_indices = df[rapid_freeze_mask].index

    if not rows_to_delete_indices.empty:
        print(f"\nFound {len(rows_to_delete_indices)} rows with 'cooling rate' as 'rapid freeze'.")
        print(f"Row numbers in CSV: {[i + 2 for i in rows_to_delete_indices]}")

        try:
            confirm_delete = input("Do you want to delete these rows? (yes/no): ").lower().strip()
            if confirm_delete in ['yes', 'y']:
                df.drop(index=rows_to_delete_indices, inplace=True)
                print(f"Deleted {len(rows_to_delete_indices)} rows.")
            else:
                print("Skipped deleting rows.")
        except Exception as e:
            print(f"An error occurred during row deletion: {e}")
    else:
        print("No rows found with 'cooling rate' as 'rapid freeze'.")
        
    return df

def verify_ingredient_counts(df):
    """
    Verifies that the number of active feature columns matches the number of
    ingredients listed in the 'all_ingredient' column for each row.
    """
    print("\n--- Verification: Checking ingredient counts against feature columns ---")

    if 'all_ingredient' not in df.columns:
        print("Warning: 'all_ingredient' column not found. Skipping this verification step.")
        return df

    feature_columns = [col for col in df.columns if '(' in col and ')' in col]
    
    if not feature_columns:
        print("Warning: No feature columns found. Skipping verification.")
        return df

    print(f"Found {len(feature_columns)} feature columns to check.")
    
    mismatches = []

    for index, row in df.iterrows():
        feature_count = row[feature_columns].notna().sum()
        all_ingredient_str = row['all_ingredient']
        
        ingredient_count = 0
        if pd.notna(all_ingredient_str):
            ingredients = [ing.strip() for ing in str(all_ingredient_str).split('+') if ing.strip()]
            ingredient_count = len(ingredients)

        if feature_count != ingredient_count:
            mismatches.append({
                'index': index,
                'feature_count': feature_count,
                'ingredient_count': ingredient_count,
                'all_ingredient_str': all_ingredient_str
            })

    if not mismatches:
        print("Verification successful. All rows have consistent ingredient counts.")
    else:
        print(f"\nVerification finished. Found {len(mismatches)} rows with inconsistent ingredient counts:")
        for mismatch in mismatches:
            print(f"  - Mismatch in row (CSV line {mismatch['index'] + 2}):")
            print(f"    > 'all_ingredient': '{mismatch['all_ingredient_str']}' (Parsed as {mismatch['ingredient_count']} ingredients)")
            print(f"    > Actual feature columns with values: {mismatch['feature_count']}")
        
    return df

def save_data(df, output_path):
    """Saves the final DataFrame to a CSV file."""
    try:
        # Reset index to make it clean and sequential for the final output
        df.reset_index(drop=True, inplace=True)
        df.to_csv(output_path, index=False)
        print(f"\nSuccessfully saved the processed data to '{output_path}'.")
        print(f"Final dataset contains {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"Error saving data to '{output_path}': {e}")


if __name__ == "__main__":
    # Main execution block
    dataframe = load_data(INPUT_PATH)
    
    if dataframe is not None:
        dataframe = preprocess_data(dataframe, COLS_TO_CLEAN_KEYWORDS)
        dataframe = clean_sparse_columns(dataframe)
        dataframe = clean_invalid_rows(dataframe)
        dataframe = clean_rapid_freeze_rows(dataframe)
        
        # --- Final Validation and Feature Engineering ---
        dataframe = fill_feature_nan_with_zero(dataframe)
        dataframe = add_culture_medium(dataframe)

        # --- Save Final Data ---
        save_data(dataframe, OUTPUT_PATH)