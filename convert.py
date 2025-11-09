import pandas as pd

def convert_excel_to_csv(excel_path='Data_raw.xlsx', csv_path='Data_raw.csv'):
    """
    Converts an Excel file to a CSV file.

    This function reads data from a specified Excel file and saves it into a CSV file
    with UTF-8 encoding. It handles potential errors like the file not being found
    or missing required libraries.

    Args:
        excel_path (str): The path to the input Excel file. Defaults to 'Data_raw.xlsx'.
        csv_path (str): The path for the output CSV file. Defaults to 'Data_raw.csv'.
    """
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(excel_path)
        
        # Save the DataFrame to a CSV file, without the index and with UTF-8 encoding
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Successfully converted '{excel_path}' to '{csv_path}'.")
        
    except FileNotFoundError:
        # Handle the case where the input Excel file does not exist
        print(f"Error: File not found at '{excel_path}'.")
    except Exception as e:
        # Handle other potential exceptions during the process
        print(f"An error occurred: {e}")
        print("Please ensure you have 'pandas' and 'openpyxl' libraries installed (e.g., pip install pandas openpyxl).")

if __name__ == '__main__':
    # When the script is run as the main program, call the conversion function
    convert_excel_to_csv()