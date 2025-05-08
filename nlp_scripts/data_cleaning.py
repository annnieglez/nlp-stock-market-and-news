'''This file groups functions for data cleaning in dataframes, such as 
    formatting columns to a consistent format. Also, it includes functions 
    to clean text data, remove timestamps, and create PDFs from JSON files.
    It also includes functions to scrape data from websites and Wikipedia 
    and create PDF file with the extracted information.'''

# Import necessary libraries

# Standard Libraries
import pandas as pd
import re
import os
from collections import defaultdict


# ==============================
# Data Cleaning Functions
# ==============================

def convert_to_str(data_frame, columns):
    """
    Convert column to string

    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with input columns converted to str.
    """

    # If a single column is provided, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    # Iterate through the columns and convert them to string
    for col in columns:
        data_frame[col] = data_frame[col].astype(object)

    return data_frame

def convert_to_int(data_frame, columns):
    """
    Convert column to int

    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with input columns converted to str.
    """

    # If a single column is provided, convert it to a list
    if isinstance(columns, str):
        columns = [columns]

    # Iterate through the columns and convert them to int
    for col in columns:
        data_frame[col] = data_frame[col].astype(int)

    return data_frame

def drop_rows_with_nan(data_frame, columns):
    """
    Drops rows from the DataFrame where the specified column(s) have NaN values.

    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - columns (str or list): Column name or list of column names to check for NaN values.

    Returns:
        - pd.DataFrame: A new DataFrame with the rows removed.
    """

    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]

    # Drop rows where any of the specified columns have NaN values
    data_frame = data_frame.dropna(subset=columns)

    return data_frame

def drop_col(data_frame, columns):
    """
    Drops specified columns from a DataFrame.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame from which columns will be dropped.
        - columns (list or str): A list of column names or a single column name to be dropped.
    
    Returns:
        - pd.DataFrame: The DataFrame with the specified columns removed.
    """

    # Check for columns that do not exist in the DataFrame
    missing_cols = [col for col in columns if col not in data_frame.columns]

    # If there are missing columns, print a message and exclude them from the drop list
    if missing_cols:
        print(f"Warning: The following columns were not found and will be skipped: {', '.join(missing_cols)}")
        columns = [col for col in columns if col in data_frame.columns]  # Keep only existing columns
    
    # Drop the existing columns
    data_frame = data_frame.drop(columns, axis=1)

    return data_frame

def snake(data_frame):
    """
    Converts column names to snake_case (lowercase with underscores).
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with column names in snake_case.
    """

    # Convert column names to snake_case
    data_frame.columns = [column.lower().replace(" ", "_").replace(")", "").replace("(", "") for column in data_frame.columns]

    return data_frame

def column_name(data_frame, columns, word_to_remove):
    """
    Formats columns name.

    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - columns (list): List of column names to modify.
        - word_to_remove (str): The word to remove from the column names.

    Returns:
        - pd.DataFrame: The DataFrame with the updated column name.
    """

    for column in columns:
        # If the column exists in the DataFrame, remove the word from the column name
        if column in data_frame.columns:
            new_column = column.replace(word_to_remove, '')
            data_frame = data_frame.rename(columns={column: new_column})

    return data_frame

def drop_columns_with_prefix(dataframe, prefix):
    """
    Drops columns from a DataFrame whose names start with the given prefix.

    Parameters:
        - df (pd.DataFrame): The DataFrame to modify.
        - prefix (str): The prefix to match column names.

    Returns:
        - pd.DataFrame: A new DataFrame with the specified columns removed.
    """

    # Copy the DataFrame to avoid modifying the original
    df = dataframe.copy()

    # Get the list of columns to drop
    columns_to_drop = [col for col in df.columns if col.startswith(prefix)]
    df = drop_col(df, columns_to_drop)

    return df

def columns_with_missing_data(df):
    """
    Identifies columns in a DataFrame with more than 50% missing data.

    Parameters:
        - df (pd.DataFrame): The DataFrame to check for missing data.

    Returns:
        - list: A list of column names with more than 50% missing data.
    """
    
    # List to store columns with more than 50% missing data
    columns_with_missing = []
    
    # Iterate over each column
    for col in df.columns:
        # Calculate the percentage of missing values for the column
        missing_percentage = df[col].isnull().mean() * 100
        
        # If the missing percentage is greater than 50, add the column to the list
        if missing_percentage > 50:
            columns_with_missing.append(col)
    
    return columns_with_missing

def clean_text(text):
    """
    Remove newlines and excessive spaces from the text.

    Parameters:
        - text (str): The input text to clean.

    Returns:
        - str: The cleaned text.
    """

    if text is not None:
        return " ".join(text.split())
    else:
        return "No text available."

def remove_timestamps(subtitle):
    """
    Remove timestamps from subtitles using regex.
    
    Parameters:
        - subtitle (list): List of subtitle strings with timestamps.

    Returns:
        - cleaned_subtitles (str): Cleaned subtitle string without timestamp.
    """

    # Step 1: Remove the timestamp lines (anything matching the timecode format)
    cleaned_subtitles = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', subtitle)
    cleaned_subtitles = re.sub(r'\d+\n\d{2}:\d{2}:\d{1},\d{3} --> \d{2}:\d{2}:\d{1},\d{3}', ' ', cleaned_subtitles)
    cleaned_subtitles = re.sub(r'\d+\n\d{2}:\d{2}:\d{1},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', ' ', cleaned_subtitles)
    cleaned_subtitles = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{1},\d{3}', ' ', cleaned_subtitles)
    cleaned_subtitles = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:[a-zA-Z]+:[a-zA-Z]+,[a-zA-Z]+\n', ' ', cleaned_subtitles)
    cleaned_subtitles = re.sub(r'\d+\n\d{2}:\d{2}:\d{1},\d{3} --> \d{2}:[a-zA-Z]+:[a-zA-Z]+,[a-zA-Z]+\n', ' ', cleaned_subtitles)
    cleaned_subtitles = re.sub(r'\[Music\]', '', cleaned_subtitles)
    cleaned_subtitles = re.sub(r'\[Applause\]', '', cleaned_subtitles)

    # Step 2: Remove empty lines and extra newlines
    cleaned_subtitles = re.sub(r'\n+', ' ', cleaned_subtitles)

    # Step 3: Remove any remaining blank spaces or unnecessary characters
    cleaned_subtitles = cleaned_subtitles.strip()

    # Remove excessive spaces from the subtitles
    cleaned_subtitles = re.sub(r'\s{2,}', ' ', cleaned_subtitles)

    return cleaned_subtitles

def clean_text_text(text):
    """
    Clean the input text by removing unwanted characters and formatting.

    Parameters:
        - text (str): The input text to clean.

    Returns:
        - str: The cleaned text.
    """
    # Remove unwanted symbols
    text = re.sub(r'[\x7f]+', '', text)

    # Remove emojis and other non-text characters
    text = re.sub(r'[^\w\s,.\'\"!?-]', '', text)

    # Remove all URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove timestamps (e.g., 0:00 - Iceland Intro)
    text = re.sub(r'\d{1,2}:\d{2} - [\w\s\(\)\&\.-]+', '', text)
    text = re.sub(r'\d{1,2}:\d{2}', '', text)

    # Optional: Remove unnecessary extra spaces and fix formatting
    text = re.sub(r'\n+', '\n', text)  
    text = re.sub(r'^\s*|\s*$', '', text)

    return text

def load_news_dataframes(folder_path):
    """
    Load news dataframes from a specified folder and return them as a dictionary.

    Parameters:
        - folder_path (str): The path to the folder containing the news CSV files.

    Returns:
        - dict: A dictionary where keys are the ticker and source names, and values are the corresponding DataFrames.
    """

    # Initialize an empty dictionary to store DataFrames
    dataframes = {}

    # Dictionary to map stock names to their tickers
    stock_names = {'Apple': 'AAPL', 'Amazon': 'AMZN', 'Tesla': 'TSLA', 'Bank of America': 'BAC',
                  'GameStop': 'GME', 'Goldman Sachs': 'GS', 'NVIDIA': 'NVDA'}

    # Iterate through all files in the specified folder
    for file in os.listdir(folder_path):
        if file.startswith("news") and file.endswith(".csv"):
            # Remove 'news_' prefix and '.csv' suffix
            name_body = file.replace("news_", "").replace(".csv", "")

            # Extract parts
            parts = name_body.split("_")
            ticker = parts[0]
            source = "_".join(parts[1:])  # Handles cases like 'google_search'

            if source == 'google_search':
                # Check if the ticker is in the stock_names dictionary
                if ticker in stock_names:
                    ticker = stock_names[ticker]


            # Construct the key name
            key = f"{ticker}_{source}"

            # Read the CSV into a DataFrame
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # Store in the dictionary
            dataframes[key] = df

    return dataframes

def load_stocks_dataframes(folder_path, start='stocks' ,end='.csv'):
    """
    Load stocks dataframes from a specified folder and return them as a dictionary.

    Parameters:
        - folder_path (str): The path to the folder containing the stocks CSV files.

    Returns:
        - dict: A dictionary where keys are the ticker and source names, and values are the corresponding DataFrames.
    """

    # Initialize an empty dictionary to store DataFrames
    dataframes = {}

    # Loop through all files in the specified folder
    for file in os.listdir(folder_path):
        if file.startswith(start) and file.endswith(end):
            # Remove 'news_' prefix and '.csv' suffix
            name_body = file.replace("news_", "").replace("stocks_", "").replace(".csv", "")

            # Extract parts
            parts = name_body.split("_")
            ticker = parts[0]
            source = "_".join(parts[1:])  # Handles cases like 'google_search'

            # Construct the key name
            key = f"{ticker}_{source}"

            # Read the CSV into a DataFrame
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # Store in the dictionary
            dataframes[key] = df

    return dataframes

def extract_alpha_vantage_per_stock(dataframes):
    """
    Extracts relevant columns from Alpha Vantage dataframes and renames them.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames with keys as stock tickers.

    Returns:
        - tuple: Two dictionaries containing full and minimal dataframes for each stock.
    """

    # Define the columns to keep
    selected_cols = ['title', 'time_published', 'summary', 
                     'overall_sentiment_score', 'overall_sentiment_label', 
                     'ticker_sentiment']
    minimal_cols = ['title', 'time_published']

    full_data_by_stock = {}
    minimal_data_by_stock = {}

    # Loop through the dataframes and extract relevant columns
    for key, df in dataframes.items():
        if "alpha_vantage" in key:
            # Extract stock ticker from the key
            ticker = key.split("_")[0]

            # Create copies with only the selected columns
            df_full = df[selected_cols].copy()
            df_min = df[minimal_cols].copy()

            # Rename 'time_published' to 'date'
            df_full.rename(columns={'time_published': 'date'}, inplace=True)
            df_min.rename(columns={'time_published': 'date'}, inplace=True)

            df_full['source'] = 'Alpha Vantage'
            df_min['source'] = 'Alpha Vantage'

            # Saving the DataFrames
            full_data_by_stock[ticker] = df_full
            minimal_data_by_stock[ticker] = df_min

    return full_data_by_stock, minimal_data_by_stock

def convert_date(dataframes):
    """
    Convert the 'date' column in each DataFrame to datetime format and print the first and last date.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames with keys as stock tickers.

    Returns:
        - dict: Updated DataFrames with 'date' column converted to datetime format.
    """

    # Loop through the dataframes and convert 'date' column to datetime
    for ticker, df in dataframes.items():
        # Check if 'date' column exists
        if 'date' in df.columns:
            # Convert 'date' column to datetime format (keeping only the date part)
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Print the first and last date
            print(f"Stock: {ticker}")
            print(f"First date: {df['date'].min()}")
            print(f"Last date: {df['date'].max()}")
            print("-" * 30)
        else:
            print(f"Warning: 'date' column not found in {ticker} data.")

    return dataframes

def extract_markets_insider(dataframes):
    """
    Extracts relevant columns from Markets Insider dataframes and renames them.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames with keys as stock tickers.

    Returns:
        - dict: Dictionary containing minimal dataframes for each stock.
    """

    minimal_data_by_stock = {}

    for key, df in dataframes.items():
        if "markets_insider" in key:
            # Extract stock ticker from the key
            ticker = key.split("_")[0]

            # Create copies with only the selected columns
            df_min = df.copy()

            # Rename 'time_published' to 'date'
            df_min.rename(columns={'publish_date': 'date'}, inplace=True)

            # Saving the DataFrames
  
            minimal_data_by_stock[ticker] = df_min

    return minimal_data_by_stock

def extract_google_search(dataframes):
    """
    Extracts relevant columns from Google Search dataframes and renames them.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames with keys as stock tickers.   

    Returns:
        - dict: Dictionary containing minimal dataframes for each stock.
    """

    minimal_data_by_stock = {}

    for key, df in dataframes.items():
        if "google_search" in key:
            # Extract stock ticker from the key
            ticker = key.split("_")[0]

            # Create copies with only the selected columns
            df_min = df.copy()

            # Rename 'time_published' to 'date'
            df_min.rename(columns={'publish_date': 'date'}, inplace=True)

            # Saving the DataFrames
  
            minimal_data_by_stock[ticker] = df_min

    return minimal_data_by_stock

def standardize_datetime_format(dataframes, column='date'):
    """
    Standardizes the datetime format in the specified column of each DataFrame.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames with keys as stock tickers.
        - column (str): The name of the column to standardize (default is 'date').

    Returns:
        - dict: Updated DataFrames with the specified column standardized to datetime format.
    """

    for ticker, df in dataframes.items():
        if column in df.columns:
            # Convert to datetime, ensuring 24-hour format
            df[column] = pd.to_datetime(df[column], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

            # Check if conversion had errors
            if df[column].isnull().any():
                print(f"Warning: There are invalid date values in {ticker}'s {column} column.")
            
        else:
            print(f"Warning: '{column}' column not found in {ticker}.")

    return dataframes

def clean_column_names(df):
    """
    Cleans the column names of a DataFrame by removing leading numbers and periods.

    Parameters:
        - df (pd.DataFrame): The input DataFrame whose columns need to be cleaned.

    Returns:
        - pd.DataFrame: DataFrame with cleaned column names.
    """

    # Rename columns to remove the leading number and period (if any)
    df.columns = df.columns.str.replace(r'^\d+\.', '', regex=True)
    return df

def rename_keys_to_ticker(data, pos = 0):
    """
    Rename the keys of a dictionary to the word before the first underscore.

    Parameters:
        - data (dict): The input dictionary with keys to be renamed.

    Returns:
        - dict: A new dictionary with renamed keys.
    """

    # Rename the keys to the word before the first underscore
    renamed_data = {key.split('_')[pos]: value for key, value in data.items()}
    return renamed_data

def save_cleaned_dataframes(dataframes, input_word, output_dir, prefix_word=''):
    """
    Save cleaned DataFrames to CSV files in the specified output directory.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames to save.
        - input_word (str): The word to include in the filename.
        - output_dir (str): The directory where the CSV files will be saved.
        - prefix_word (str): Optional prefix for the filenames.
    
    Returns:
        - None: The function saves the DataFrames to CSV files.
    """

    for key, df in dataframes.items():
        filename = f"{prefix_word}_cleaned_dataframe_{key}_{input_word}.csv"
        filepath = f"{output_dir}/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Saved {filename}")

def move_date_column_to_front(dataframes, column_name='date'):
    """
    Move the specified column to the front of each DataFrame in the dictionary.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames to modify.
        - column_name (str): The name of the column to move to the front.

    Returns:
        - dict: Updated DataFrames with the specified column moved to the front.
    """	

    updated_dataframes = {}
    for key, df in dataframes.items():
        if column_name in df.columns:
            # Reorder columns: 'date' + all other columns except 'date'
            cols = [column_name] + [col for col in df.columns if col != column_name]
            updated_dataframes[key] = df[cols]
        else:
            print(f"Warning: Column '{column_name}' not found in DataFrame '{key}'")
            updated_dataframes[key] = df  # Keep original if 'date' not found
    return updated_dataframes

def concat_and_sort_dataframes_by_key(dict_list, date_column='date'):
    """
    Concatenate DataFrames in a list of dictionaries by their keys and sort by date.

    Parameters:
        - dict_list (list): List of dictionaries containing DataFrames.
        - date_column (str): The name of the date column to sort by.

    Returns:
        - dict: A dictionary where keys are the original keys and values are the concatenated and sorted DataFrames.
    """

    combined_data = defaultdict(list)

    # Collect all dataframes under each key
    for d in dict_list:
        for key, df in d.items():
            combined_data[key].append(df)

    # Concatenate and sort
    final_data = {}
    for key, df_list in combined_data.items():
        combined_df = pd.concat(df_list, ignore_index=True)
        if date_column in combined_df.columns:
            combined_df[date_column] = pd.to_datetime(combined_df[date_column])
            combined_df = combined_df.sort_values(by=date_column)
        final_data[key] = combined_df

    return final_data

def remove_duplicates_subset(dataframes, subset_columns):
    """
    Remove duplicates from each DataFrame in the dictionary based on a subset of columns.

    Parameters:
        - dataframes (dict): Dictionary of DataFrames to clean.
        - subset_columns (list): List of columns to consider for identifying duplicates.

    Returns:
        - dict: Updated DataFrames with duplicates removed based on the specified subset of columns.
    """

    for key, df in dataframes.items():
        # Remove duplicates based on the specified subset of columns
        dataframes[key] = df.drop_duplicates(subset=subset_columns)

    return dataframes

def load_csv_with_column_names(filepath, column_names = ["bool_and_titles"]):
    """
    Load a CSV file without headers and assign new column names.

    Parameters:
        - filepath (str): Path to the CSV file.
        - column_names (list): List of new column names.
    
    Returns:
        - pd.DataFrame: DataFrame with the new column names assigned.
    """

    df = pd.read_csv(filepath, header=None)  # read without headers
    df.columns = column_names                # assign new column names
    return df

def split_bool_and_titles_column(df, column_name='bool_and_titles'):
    """
    Splits a column in the DataFrame into two separate columns.

    Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column_name (str): The name of the column to split.

    Returns:
        - pd.DataFrame: DataFrame with the specified column split into two new columns.
    """

    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    df[['fake_news', 'title']] = df[column_name].str.split('\t', n=1, expand=True)
    df.drop(columns=[column_name], inplace=True)
    df['fake_news'] = df['fake_news']

    return df

def save_dataframe_to_csv(df, output_dir, filename):
    """
    Save a DataFrame to a CSV file in the specified directory.

    Parameters:
        - df (pd.DataFrame): The DataFrame to save.
        - output_dir (str): The directory where the CSV file will be saved.
        - filename (str): The name of the CSV file.

    Returns:
        - None: The function saves the DataFrame to a CSV file.
    """

    os.makedirs(output_dir, exist_ok=True) 
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved DataFrame: {filename}")

def remove_duplicates(dataframe):
    """
    Remove duplicate rows from the DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        - pd.DataFrame: The DataFrame with duplicates removed.
    """
    
    dataframe = dataframe.drop_duplicates()

    return dataframe

def remove_first_row(dataframe):
    """
    Remove the first row from the DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        - pd.DataFrame: The DataFrame with the first row removed.
    """
    
    dataframe = dataframe.iloc[1:]
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe

def process_news_data(news_dict):
    """
    Processes a dictionary of news DataFrames. Each key is a stock name, and each value is a DataFrame
    with columns ['date', 'title', 'source']. The function removes the 'source' column and joins all 
    titles per date into a single string.

    Parameters:
        - news_dict (dict): Dictionary where key is stock and value is corresponding news DataFrame.

    Returns:
        - dict: Processed dictionary with same keys and new DataFrames with combined titles per date.
    """
    processed = {}

    for stock, df in news_dict.items():
        if 'source' in df.columns:
            df = df.drop(columns=['source'])
        df_grouped = df.groupby('date')['title'].apply(lambda x: ' '.join(x)).reset_index()
        processed[stock] = df_grouped

    return processed

def merge_stock_and_news(stock_dict, news_dict):
    """
    Merges stock price data with processed news data for each stock by date.

    Parameters:
        - stock_dict (dict): Dictionary where each key is a stock and each value is a DataFrame
                         with date as index or a 'date' column and stock price data.
        - news_dict (dict): Dictionary where each key is a stock and each value is a DataFrame
                        with ['date', 'title'] columns.

    Returns:
        - dict: Dictionary with same keys as input, each value is a DataFrame combining stock and news data.
    """
    merged_dict = {}

    for stock, stock_df in stock_dict.items():
        stock_df = stock_df.copy()
        
        # Make sure date is a datetime and set as index or column
        if 'date' in stock_df.columns:
            stock_df['date'] = pd.to_datetime(stock_df['date'])
        else:
            stock_df = stock_df.reset_index()
            stock_df['date'] = pd.to_datetime(stock_df['date'])

        if stock in news_dict:
            news_df = news_dict[stock].copy()
            news_df['date'] = pd.to_datetime(news_df['date'])

            # Merge with left join to keep all stock dates
            merged_df = pd.merge(stock_df, news_df, on='date', how='left')
        else:
            # No news available for this stock
            stock_df['title'] = pd.NA
            merged_df = stock_df

        merged_dict[stock] = merged_df

    return merged_dict