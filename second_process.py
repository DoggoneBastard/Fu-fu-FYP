import pandas as pd
import os
import numpy as np

# --- Configuration Constants ---
INPUT_PATH = '/Users/lifudu/Desktop/FYP/GitHub/processed_data.csv'
OUTPUT_PATH = '/Users/lifudu/Desktop/FYP/GitHub/final_data.csv'
COLS_TO_CLEAN_KEYWORDS = ['viability', 'recovery', 'doubling time']

def load_data(file_path):
    """Loads data from the specified CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please run the first process script to generate it.")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading '{file_path}': {e}")
        return None

def preprocess_data(df, keywords):
    """
    Cleans specific columns by extracting numbers and replaces all zero values with NaN.
    """
    print("--- Data Pre-processing ---")
    
    # 1. Convert specific columns to numeric
    cols_to_process = [col for col in df.columns if any(keyword in col.lower() for keyword in keywords)]
    
    if cols_to_process:
        print(f"Found columns to clean for non-numeric values: {cols_to_process}")
        for col in cols_to_process:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Extracted leading numbers and converted to numeric in specified columns.")
    else:
        print(f"No columns containing keywords {keywords} found to clean.")

    # 2. Replace all 0 values with NaN
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
    """
    feature_columns = [col for col in df.columns if '(' in col and ')' in col]
    print(f"--- Initial State ---\nTotal number of feature columns: {len(feature_columns)}\n")
    
    print("--- Column-based cleaning ---")
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
                
                confirm_delete_col = input("Do you want to delete these columns? (yes/no): ").lower()
                if confirm_delete_col == 'yes':
                    df.drop(columns=cols_to_check, inplace=True)
                    print(f"Deleted {len(cols_to_check)} columns.")
                else:
                    print("Skipped deleting columns.")
            else:
                print(f"No feature columns found with fewer than {min_entries_col} entries.")
        else:
            print("Skipped column cleaning.")
    except ValueError:
        print("Invalid input. Skipping column cleaning.")
    except Exception as e:
        print(f"An error occurred during column cleaning: {e}")
    
    print("-" * 20)
    return df

def clean_sparse_rows(df):
    """
    Interactively identifies and removes rows with few filled features.
    """
    feature_columns = [col for col in df.columns if '(' in col and ')' in col]
    
    print("\n--- Row-based cleaning ---")
    try:
        min_features_row_str = input("Enter the minimum number of filled features a row must have to be kept (e.g., 2). Press Enter to skip: ")
        if min_features_row_str:
            min_features_row = int(min_features_row_str)
            
            if not feature_columns:
                print("No feature columns available for row-based cleaning.")
                return df

            row_counts = df[feature_columns].notna().sum(axis=1)
            rows_to_check_indices = df[row_counts < min_features_row].index
            
            if not rows_to_check_indices.empty:
                print(f"\nFound {len(rows_to_check_indices)} rows with fewer than {min_features_row} filled features:")
                print(f"Row numbers in CSV: {[i + 2 for i in rows_to_check_indices]}")

                confirm_delete_row = input("Do you want to delete these rows? (yes/no): ").lower()
                if confirm_delete_row == 'yes':
                    df.drop(index=rows_to_check_indices, inplace=True)
                    print(f"Deleted {len(rows_to_check_indices)} rows.")
                else:
                    print("Skipped deleting rows.")
            else:
                print(f"No rows found with fewer than {min_features_row} filled features.")
        else:
            print("Skipped row cleaning.")
    except ValueError:
        print("Invalid input. Skipping row cleaning.")
    except Exception as e:
        print(f"An error occurred during row cleaning: {e}")
        
    return df

def save_data(df, output_path):
    """Saves the cleaned dataframe to a new CSV file."""
    print("\n--- Final State ---")
    final_feature_columns = [col for col in df.columns if '(' in col and ')' in col]
    print(f"Final number of feature columns: {len(final_feature_columns)}")

    try:
        df.to_csv(output_path, index=False)
        print(f"\nCleaning complete. The final data has been saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving updated data to '{output_path}': {e}")

def main():
    """
    Main function to orchestrate the data cleaning pipeline.
    """
    df = load_data(INPUT_PATH)
    if df is None:
        return

    df = preprocess_data(df, COLS_TO_CLEAN_KEYWORDS)
    df = clean_sparse_columns(df)
    df = clean_sparse_rows(df)
    
    save_data(df, OUTPUT_PATH)

if __name__ == "__main__":
    main()


import pandas as pd

def combine_and_clean_data(input_path='processed_data.csv', output_path='final_data.csv'):
    """
    Combines similar ingredient columns and performs final cleaning steps on the dataset.

    This function reads a processed CSV file, merges columns representing the same
    ingredient (e.g., 'FBS' and 'fetal bovine serum'), removes the original columns,
    and saves the result to a new, final CSV file.

    Args:
        input_path (str): Path to the processed CSV file. Defaults to 'processed_data.csv'.
        output_path (str): Path for the final output CSV file. Defaults to 'final_data.csv'.
    """
    try:
        # Load the processed data
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"Successfully loaded '{input_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return

    # --- Column Combination ---
    # Define a mapping for columns that should be combined.
    # The key is the new column name, and the value is a list of old column names.
    combination_map = {
        'FBS (%)': ['FBS', 'fetal bovine serum'],
        'DMSO (%)': ['DMSO', 'dimethyl sulfoxide'],
        'D-glucose (g/L)': ['D-glucose', 'glucose'],
    }

    print("Combining similar ingredient columns...")
    for new_col, old_cols in combination_map.items():
        # Find which of the old columns actually exist in the DataFrame
        existing_cols = [col for col in old_cols if col in df.columns]
        
        if existing_cols:
            # Create the new column by taking the maximum value across the existing old columns for each row.
            # This assumes that for any given row, only one of the related columns will have a non-zero value.
            df[new_col] = df[existing_cols].max(axis=1)
            
            # Drop the original columns that have been combined
            df.drop(columns=existing_cols, inplace=True)
            print(f"  - Combined {existing_cols} into '{new_col}'.")
        else:
            print(f"  - No columns found to combine for '{new_col}'.")

    # --- Final Data Saving ---
    try:
        # Save the finalized DataFrame to a new CSV file
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nData combination and cleaning complete. Final data saved to '{output_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == '__main__':
    # Run the function when the script is executed
    combine_and_clean_data()