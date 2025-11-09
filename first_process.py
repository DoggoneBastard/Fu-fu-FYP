import pandas as pd
import re
from collections import defaultdict

# --- Configuration Constants ---
INPUT_PATH = '/Users/lifudu/Desktop/FYP/GitHub/Data_raw.csv'
OUTPUT_PATH = '/Users/lifudu/Desktop/FYP/GitHub/processed_data.csv'

# Columns from the original dataframe to keep in the final output
FINAL_COLS_TO_KEEP = ['viability', 'recovery', 'doubling time', 'cooling rate']

# Define valid units for parsing and units to be ignored
VALID_UNITS = {'M', 'mM', '%', 'µM', 'µg/mL', 'ng/mL', 'mg/ml', 'nM', 'mmol/L', 'mol/L'}
UNITS_TO_IGNORE = ['wt', '(wt)', 'v/v']

# Define synonym groups for ingredients
SYNONYM_GROUPS = [
    ['1,2-propanediol', 'propylene glycol'],
    ['dmso', 'me2so'],
    ['ectoin', 'ectoine'],
    ['eg', 'ethylene glycol'],
    ['fbs', 'fcs', 'fetal bovine serum', 'fetal calf serum'],
    ['hes', 'hydroxyethyl starch'],
    ['hs', 'human serum'],
    ['hsa', 'human albumin', 'human serum albumin'],
    ['mc', 'methylcellulose']
]

def load_and_prepare_data(input_path):
    """
    Loads data from CSV, normalizes columns, and handles empty rows interactively.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return None

    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={df.columns[0]: 'all_ingredient'}, inplace=True)

    nan_mask = df['all_ingredient'].isna()
    str_mask = df['all_ingredient'].fillna('').astype(str).str.strip() == ''
    empty_mask = nan_mask | str_mask
    empty_rows = df[empty_mask]

    if not empty_rows.empty:
        empty_row_numbers = [index + 2 for index in empty_rows.index]
        print("\n--- INTERACTIVE ACTION REQUIRED ---")
        print(f"Warning: Found empty or null 'all_ingredient' values in rows: {empty_row_numbers}")
        user_input = input("Delete these rows? (yes/no): ").lower().strip()
        if user_input in ['yes', 'y']:
            df.drop(empty_rows.index, inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"Deleted {len(empty_row_numbers)} rows.")
        else:
            print("Skipping deletion.")
    return df

def parse_rows(df, valid_units, units_to_ignore):
    """
    Iterates through dataframe rows, parsing ingredient strings into features.
    """
    sorted_units = sorted(list(valid_units), key=len, reverse=True)
    units_pattern = '|'.join(re.escape(u) for u in sorted_units)
    
    # --- New Dual-Regex Strategy ---
    # Stricter regex: requires a space separator. More robust for complex names.
    strict_ingredient_regex = re.compile(r'^\s*([\d\.]+)\s*(' + units_pattern + r')\s+(.*)\s*$')
    # Lenient fallback regex: original version for cases like "10%Glycerol" (no space).
    lenient_ingredient_regex = re.compile(r'^\s*([\d\.]+)\s*(' + units_pattern + r')\s*(.*)\s*$')
    
    sorted_units_to_ignore = sorted(units_to_ignore, key=len, reverse=True)

    new_features_data = []
    parsing_errors_info = []
    mismatch_rows_info = []

    for index, row in df.iterrows():
        ingredient_str = row['all_ingredient']
        if not isinstance(ingredient_str, str) or not ingredient_str.strip():
            new_features_data.append({})
            continue

        ingredients = [s.strip() for s in ingredient_str.split('+')]
        is_row_valid, parsed_row_ingredients = True, []

        for ingredient in ingredients:
            processed_ingredient = ingredient
            for ignored in sorted_units_to_ignore:
                processed_ingredient = processed_ingredient.replace(ignored, '').strip()
            
            # --- Apply Dual-Regex Strategy ---
            # 1. Try the strict regex first, which is more robust for complex names.
            match = strict_ingredient_regex.match(processed_ingredient)
            if not match:
                # 2. If it fails, fall back to the lenient regex for edge cases.
                match = lenient_ingredient_regex.match(processed_ingredient)

            if not match:
                is_row_valid = False
                parsing_errors_info.append({
                    'index': row.name,
                    'message': f"Row {row.name + 2}: Invalid format for '{ingredient}' in '{ingredient_str}'"
                })
                break
            parsed_row_ingredients.append(match.groups())

        if not is_row_valid:
            new_features_data.append({})
            continue

        current_row_features, seen_substances = {}, {}
        for value_str, unit, substance in parsed_row_ingredients:
            substance = substance.strip().lower()
            try:
                value = float(value_str)
            except ValueError:
                continue
            
            feature_name = f"{substance}({unit})"
            if substance in seen_substances and seen_substances[substance] != unit:
                pass # Allow multiple entries, handled by mismatch check
            seen_substances[substance] = unit
            current_row_features[feature_name] = value

        if len(ingredients) != len(current_row_features):
            mismatch_rows_info.append({
                'index': row.name,
                'message': f"Row {row.name + 2}: Mismatch - {len(ingredients)} ingredients vs {len(current_row_features)} features. (Likely duplicates)"
            })
        
        new_features_data.append(current_row_features)

    return new_features_data, parsing_errors_info, mismatch_rows_info

def run_interactive_merges(features_df, synonym_groups):
    """
    Performs interactive merging for M/mM units and synonyms using a more robust,
    integrated approach that handles synonyms across different convertible units.
    """
    synonym_map = {synonym: group[0] for group in synonym_groups for synonym in group}
    merged_substances_log, synonym_merge_log = [], []

    # --- Helper function to parse column names ---
    def parse_col(col):
        match = re.match(r'^(.*)\((.*)\)$', col)
        if not match:
            return None, None
        substance, unit = match.group(1), match.group(2)
        canonical_sub = synonym_map.get(substance, substance)
        return canonical_sub, unit

    # --- Step 1: Group all columns by their canonical substance name ---
    canonical_groups = defaultdict(list)
    for col in features_df.columns:
        canonical_sub, _ = parse_col(col)
        if canonical_sub:
            canonical_groups[canonical_sub].append(col)

    # --- Step 2: Interactive M/mM merge (handles synonyms internally) ---
    m_mm_candidates = []
    for sub, cols in canonical_groups.items():
        has_m = any('(M)' in c for c in cols)
        has_mm = any('(mM)' in c for c in cols)
        if has_m and has_mm:
            m_mm_candidates.append(sub)

    if m_mm_candidates:
        print("\n--- INTERACTIVE ACTION: M/mM MERGE (Synonym-Aware) ---")
        print("The following substances (including their synonyms) have both 'M' and 'mM' units:")
        print(", ".join(f"'{s}'" for s in m_mm_candidates))
        user_input = input("Merge 'M' into 'mM' for ALL of them? (yes/no): ").lower().strip()
        if user_input in ['yes', 'y']:
            for sub in m_mm_candidates:
                cols_to_process = canonical_groups[sub]
                m_cols = [c for c in cols_to_process if '(M)' in c]
                mm_cols = [c for c in cols_to_process if '(mM)' in c]
                
                target_mm_col = f"{sub}(mM)"
                if target_mm_col not in features_df.columns:
                    features_df[target_mm_col] = 0.0

                # Consolidate all M and mM columns into the single canonical mM column
                for col in sorted(list(set(m_cols + mm_cols))): # Use sorted set to ensure consistent order
                    if col == target_mm_col:
                        continue
                    
                    # Check for conflicts before merging
                    if ((features_df[target_mm_col] > 0) & (features_df[col] > 0)).any():
                        conflict_rows = features_df[(features_df[target_mm_col] > 0) & (features_df[col] > 0)].index + 2
                        log_msg = f"ERROR: Merge of '{col}' into '{target_mm_col}' aborted! Conflict in rows: {list(conflict_rows)}."
                        print(log_msg)
                        merged_substances_log.append(log_msg)
                        continue

                    value_to_add = features_df[col]
                    log_type = "Synonym"
                    if '(M)' in col:
                        value_to_add = value_to_add * 1000
                        log_type = "Unit (M->mM)"

                    features_df[target_mm_col] += value_to_add
                    features_df.drop(columns=[col], inplace=True)
                    log_msg = f"Success ({log_type}): Merged '{col}' into '{target_mm_col}'."
                    print(log_msg)
                    merged_substances_log.append(log_msg)

    # --- Step 3: Interactive Synonym Merge (for all other units) ---
    # Re-build canonical groups as columns have been dropped/created
    current_canonical_groups = defaultdict(list)
    for col in features_df.columns:
        canonical_sub, _ = parse_col(col)
        if canonical_sub:
            current_canonical_groups[canonical_sub].append(col)

    synonym_candidates = {}
    for sub, cols in current_canonical_groups.items():
        # Group by unit to find synonyms within the same unit
        cols_by_unit = defaultdict(list)
        for col in cols:
            _, unit = parse_col(col)
            if unit:
                cols_by_unit[unit].append(col)
        
        for unit, unit_cols in cols_by_unit.items():
            if len(unit_cols) > 1:
                synonym_candidates[(sub, unit)] = sorted(unit_cols)

    if synonym_candidates:
        print("\n--- INTERACTIVE ACTION: SYNONYM MERGE (Remaining) ---")
        print("The following synonym groups with matching units were found:")
        for (sub, unit), cols in synonym_candidates.items():
            target_col = f"{sub}({unit})"
            syn_cols = [c for c in cols if c != target_col]
            if syn_cols:
                print(f"- Merge {syn_cols} into '{target_col}'")

        user_input = input("Merge ALL of them? (yes/no): ").lower().strip()
        if user_input in ['yes', 'y']:
            for (sub, unit), cols in synonym_candidates.items():
                target_col = f"{sub}({unit})"
                if target_col not in features_df.columns:
                    features_df[target_col] = 0.0
                
                for syn_col in [c for c in cols if c != target_col]:
                    if ((features_df[target_col] > 0) & (features_df[syn_col] > 0)).any():
                        conflict_rows = features_df[(features_df[target_col] > 0) & (features_df[syn_col] > 0)].index + 2
                        log_msg = f"ERROR: Merge of '{syn_col}' into '{target_col}' aborted! Conflict in rows: {list(conflict_rows)}."
                        print(log_msg)
                        synonym_merge_log.append(log_msg)
                        continue
                    
                    features_df[target_col] += features_df[syn_col]
                    features_df.drop(columns=[syn_col], inplace=True)
                    log_msg = f"Success: Merged '{syn_col}' into '{target_col}'."
                    print(log_msg)
                    synonym_merge_log.append(log_msg)

    return features_df, merged_substances_log, synonym_merge_log

def run_interactive_deletion(final_df, parsing_errors_info, mismatch_rows_info):
    """
    Prints reports of problematic rows and prompts the user for deletion.
    """
    error_indices = set()
    has_errors = False
    if parsing_errors_info:
        has_errors = True
        print("\n--- Rows with Parsing Errors ---")
        for info in parsing_errors_info:
            print(info['message'])
            error_indices.add(info['index'])
            
    if mismatch_rows_info:
        has_errors = True
        print("\n--- Rows with Ingredient Count Mismatch ---")
        for info in mismatch_rows_info:
            print(info['message'])
            error_indices.add(info['index'])

    if has_errors:
        print("\n--- INTERACTIVE ACTION: REMOVE FAILED ROWS ---")
        print(f"Found {len(error_indices)} rows with the issues listed above.")
        user_input = input("Do you want to REMOVE these rows from the final dataset? (yes/no): ").lower().strip()
        if user_input in ['yes', 'y']:
            final_df.drop(index=list(error_indices), inplace=True)
            print(f"Success: Removed {len(error_indices)} rows.")
        else:
            print("Skipping removal. Rows with errors will be kept.")
    return final_df

def generate_summary_reports(final_df, merged_substances_log, synonym_merge_log):
    """
    Generates and prints post-processing and inconsistency reports.
    """
    if merged_substances_log or synonym_merge_log:
        print("\n--- Post-Processing Summary ---")
        if merged_substances_log:
            print("\nM/mM Merges:")
            for log in merged_substances_log: print(f"- {log}")
        if synonym_merge_log:
            print("\nSynonym Merges:")
            for log in synonym_merge_log: print(f"- {log}")

    final_substance_map = defaultdict(set)
    for col in final_df.columns:
        if match := re.match(r'^(.*)\((.*)\)$', col):
            final_substance_map[match.group(1)].add(match.group(2))
    
    remaining_inconsistencies = [
        f"Substance '{sub}': Has multiple units: {', '.join(sorted(units))}" 
        for sub, units in final_substance_map.items() if len(units) > 1
    ]
    
    if remaining_inconsistencies:
        print("\nRemaining Inconsistencies:")
        for warning in remaining_inconsistencies: print(f"- {warning}")

def run_interactive_unit_conflict_removal(df):
    """
    Finds substances with multiple units and asks the user if they want to remove
    the ones with fewer data points.
    """
    print("\n" + "="*80)
    print("Step 5a: Interactive Removal of Columns with Unit Conflicts")
    print("This step will find substances that have multiple different units and suggest")
    print("removing the columns with fewer data entries.")
    print("="*80)

    # Re-use the parsing logic
    def parse_col_name(col_name):
        match = re.match(r'^(.*?)\s*\((.*?)\)$', col_name)
        if match:
            return match.groups()
        return None, None

    # Group columns by substance name
    feature_cols = [col for col in df.columns if '(' in col and ')' in col]
    substance_groups = {}
    for col in feature_cols:
        name, unit = parse_col_name(col)
        if name:
            if name not in substance_groups:
                substance_groups[name] = []
            substance_groups[name].append(col)

    cols_to_drop = []
    # Find groups with more than one unit (conflicts)
    for name, cols in substance_groups.items():
        if len(cols) > 1:
            # Calculate data points for each column
            col_counts = {col: (df[col] > 0).sum() for col in cols}
            
            # Sort by count, descending
            sorted_cols = sorted(col_counts.items(), key=lambda item: item[1], reverse=True)
            
            # The one with the most data is the one to keep
            primary_col, primary_count = sorted_cols[0]
            
            # The rest are candidates for removal
            secondary_cols = sorted_cols[1:]

            print(f"\nFound conflict for substance '{name}':")
            print(f"  - Keeping primary: '{primary_col}' ({primary_count} entries)")
            
            for col, count in secondary_cols:
                print(f"  - Suggest removing: '{col}' ({count} entries)")
                while True:
                    choice = input(f"    Do you want to remove '{col}'? [y/n]: ").lower()
                    if choice in ['y', 'n']:
                        break
                    print("    Invalid input. Please enter 'y' or 'n'.")
                
                if choice == 'y':
                    cols_to_drop.append(col)

    if not cols_to_drop:
        print("\nNo columns were selected for removal.")
        return df

    print(f"\nThe following columns will be removed: {', '.join(cols_to_drop)}")
    while True:
        final_choice = input("Confirm removal? [y/n]: ").lower()
        if final_choice in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    if final_choice == 'y':
        df.drop(columns=cols_to_drop, inplace=True)
        print("Columns successfully removed.")
    else:
        print("Removal cancelled.")
        
    return df


def main():
    """
    Main orchestration function.
    """
    df = load_and_prepare_data(INPUT_PATH)
    if df is None:
        return

    new_features_data, parsing_errors, mismatch_rows = parse_rows(df, VALID_UNITS, UNITS_TO_IGNORE)

    if not new_features_data:
        df[['all_ingredient'] + FINAL_COLS_TO_KEEP].to_csv(OUTPUT_PATH, index=False)
        print(f"Processed data saved to {OUTPUT_PATH}. No new features created.")
        return

    features_df = pd.DataFrame(new_features_data, index=df.index).fillna(0)
    
    # Step 4: Interactive merging of synonyms and M/mM units
    features_df, merged_log, synonym_log = run_interactive_merges(features_df, SYNONYM_GROUPS)

    # Step 5a: Interactive unit conflict removal (THE FIX)
    features_df = run_interactive_unit_conflict_removal(features_df)

    # Reorder columns after all merges and removals
    ordered_features = sorted(list(features_df.columns))
    features_df = features_df.reindex(columns=ordered_features, fill_value=0)

    # Assemble the final dataframe
    existing_cols_to_keep = [col for col in FINAL_COLS_TO_KEEP if col in df.columns]
    final_df = pd.concat([df[['all_ingredient']], features_df, df[existing_cols_to_keep]], axis=1)

    # Step 6: Interactive deletion of problematic rows
    final_df = run_interactive_deletion(final_df, parsing_errors, mismatch_rows)

    # Step 7: Save and report
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nProcessed data saved to {OUTPUT_PATH}")

    generate_summary_reports(final_df, merged_log, synonym_log)
    
    print("\nfinish the first process")


if __name__ == '__main__':
    main()


def process_ingredients(input_path='Data_raw.csv', output_path='processed_data.csv'):
    """
    Processes the 'all_ingredient' column in a CSV file to expand it into multiple
    columns, one for each unique ingredient.

    This function reads a CSV, extracts ingredient and concentration pairs from the
    'all_ingredient' column, creates a new column for each unique ingredient, and
    populates it with the corresponding concentration. The processed data is then
    saved to a new CSV file.

    Args:
        input_path (str): The path to the input CSV file. Defaults to 'Data_raw.csv'.
        output_path (str): The path for the output CSV file. Defaults to 'processed_data.csv'.
    """
    try:
        # Read the input CSV file with a specific encoding to handle potential BOM
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"Successfully loaded '{input_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        return

    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Check if the required 'all_ingredient' column exists
    if 'all_ingredient' not in df.columns:
        print("Error: 'all_ingredient' column not found in the data.")
        return

    # --- Ingredient Extraction and Processing ---
    print("Processing the 'all_ingredient' column...")
    
    # Fill any missing values in the 'all_ingredient' column with an empty string
    df['all_ingredient'] = df['all_ingredient'].fillna('')

    # Use a regular expression to find all ingredient-concentration pairs
    # This pattern looks for: (Ingredient Name) (Concentration Value) (Unit)
    df['parsed_ingredients'] = df['all_ingredient'].apply(
        lambda x: re.findall(r'([a-zA-Z\s\d\(\)-]+?)\s*(\d+\.?\d*)\s*(g/L|%|mM)', x)
    )

    # Create a set of all unique ingredients found
    all_ingredients = set()
    for ingredients_list in df['parsed_ingredients']:
        for ingredient, _, _ in ingredients_list:
            all_ingredients.add(ingredient.strip())

    # Create a new column for each unique ingredient and initialize with 0
    for ingredient in all_ingredients:
        df[ingredient] = 0

    # --- Populate Concentration Values ---
    print("Populating concentration values for each ingredient...")
    for index, row in df.iterrows():
        for ingredient, value, unit in row['parsed_ingredients']:
            ingredient_name = ingredient.strip()
            # Convert value to float and store it in the corresponding ingredient column
            df.at[index, ingredient_name] = float(value)

    # --- Finalizing the DataFrame ---
    # Drop the intermediate 'parsed_ingredients' column
    df.drop(columns=['parsed_ingredients'], inplace=True)
    
    # Save the processed DataFrame to the output CSV file
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nProcessing complete. The data has been saved to '{output_path}'.")
        print(f"A total of {len(all_ingredients)} unique ingredients were identified and expanded into columns.")
    except Exception as e:
        print(f"Error saving the file: {e}")

if __name__ == '__main__':
    # Run the processing function when the script is executed
    process_ingredients()