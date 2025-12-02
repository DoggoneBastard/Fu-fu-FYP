import pandas as pd

def clean_data(file_path='Data_raw.xlsx'):
    """
    Cleans the dataset by identifying and optionally removing rows with insufficient data.

    This function reads an Excel file, checks each row to see if it's empty or has fewer 
    than 5 non-null values. If such "problematic" rows are found, it reports them to the 
    user and asks for confirmation to delete them. If the user agrees, these rows are 
    removed from the dataset, and the original file is overwritten.

    Args:
        file_path (str): The path to the Excel file to be cleaned. 
                         Defaults to 'Data_raw.xlsx'.
    """
    try:
        # Attempt to read the specified Excel file
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        # If the file is not found, print an error message and exit the function
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        # If any other error occurs during file reading (e.g., 'openpyxl' not installed),
        # print an error message
        print(f"Error reading file: {e}")
        print("Please ensure you have the 'openpyxl' library installed (pip install openpyxl) to read Excel files.")
        return

    print(f"--- Cleaning and validating data for '{file_path}' ---")
    total_rows = len(df)
    print(f"Total number of rows in the file: {total_rows}")

    # Find rows that are empty or have fewer than 5 non-null values
    problematic_rows = []
    for index, row in df.iterrows():
        # Check if the row is entirely null or has a count of non-null values less than 5
        # +2 is used to match Excel's 1-based row numbering and account for the header row
        if row.isnull().all() or row.count() < 5:
            problematic_rows.append(index + 2) 

    if problematic_rows:
        # If problematic rows are found, display them to the user
        print("\nFound rows that are either empty or have fewer than 5 values:")
        print(", ".join(map(str, problematic_rows)))
        
        # Ask the user if they want to delete these rows
        user_input = input("\nDo you want to delete these rows? (yes/no): ").lower()
        if user_input in ['yes', 'y']:
            # If the user agrees, drop the problematic rows
            # -2 converts the Excel row number back to pandas' 0-based index
            df_cleaned = df.drop(index=[r - 2 for r in problematic_rows])
            try:
                # Save the cleaned DataFrame back to the original file
                df_cleaned.to_excel(file_path, index=False)
                print(f"\nThe problematic rows have been deleted. The original file '{file_path}' has been overwritten.")
                print(f"New total number of rows: {len(df_cleaned)}")
            except Exception as e:
                # If an error occurs while saving, print an error message
                print(f"Error saving file: {e}")
        else:
            # If the user does not agree, do nothing
            print("\nNo rows were deleted.")
    else:
        # If no problematic rows are found, inform the user
        print("\nNo empty or sparse rows found.")

    # Provide guidance for the user on the next steps
    print("\n--- Next Steps: Data Formatting ---")
    print("Please use an AI assistant to format the following columns based on the instructions in your README and paste them back:")
    print("Here suggest you to use Deepseek or chatgpt,or use Doubao")
    print("- cooling rate")
    print("- Recovery")
    print("- Viability")
    print("PS:if the data is too big, you can seperate it into smaller batches to fit the AI token")
    print("make sure that the rows produced by ai is the same as the data, you can put order before each ingredient to match the data")
    print("\nScript finished.")

if __name__ == '__main__':
    # When the script is run as the main program, call the clean_data function
    clean_data()