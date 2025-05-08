import os
import pandas as pd
from tqdm import tqdm

from nltk.sentiment import SentimentIntensityAnalyzer
vd = SentimentIntensityAnalyzer()

# Natural Language Processing libraries
import nltk
from nltk.corpus import wordnet, stopwords
import contractions
import torch
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.remove('not')

# Data  folder paths
# Data folders
base_folder = os.path.join(os.path.dirname(__file__), 'data')

embeddings_folder = os.path.join(base_folder, "..", 'data/embeddings')
sentiments_folder = os.path.join(base_folder, "..", 'data/sentiments')

# Ensure the folders exist
os.makedirs(embeddings_folder, exist_ok=True)
os.makedirs(sentiments_folder, exist_ok=True)


def load_dataframes_news_from_folder(folder_path, startswith = 'combined_cleaned_dataframe_', key_place = 3, endswith = '.csv'):
    """
    Load dataframes from a folder into a dictionary.
    The keys of the dictionary are the ticker symbols extracted from the filenames.
    The values are the dataframes loaded from the CSV files.

    Parameters:
        - folder_path (str): The path to the folder containing the CSV files.
        - startswith (str): The prefix of the filenames to look for.  Default is 'combined_cleaned_dataframe_'.
        - key_place (int): The index of the part of the filename to use as the key. Default is 3.
        - endswith (str): The suffix of the filenames to look for. Default is '.csv'.

    Returns:
        - dict: A dictionary where the keys are ticker symbols and the values are dataframes.
    """

    dataframes_dict = {}

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith(startswith) and filename.endswith(endswith):
            # Extract the ticker symbol (assuming it's the part after 'combined_cleaned_dataframe_')
            ticker = filename.split('_')[key_place]  # Adjust this based on your naming pattern

            # Read the CSV file into a dataframe
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)  # You may want to adjust this if your files are not CSV

            # Add the dataframe to the dictionary with the ticker as the key
            dataframes_dict[ticker] = df

    return dataframes_dict

def load_dataframes_stocks_from_folder(folder_path):
    """
    Load dataframes from a folder into a dictionary.
    The keys of the dictionary are the ticker symbols extracted from the filenames.
    The values are the dataframes loaded from the CSV files.

    Parameters:
        - folder_path (str): The path to the folder containing the CSV files.

    Returns:
        - dict: A dictionary where the keys are ticker symbols and the values are dataframes.
    """	

    dataframes_dict = {}

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith('stocks_') and filename.endswith('_alpha_vantage.csv'):
            # Extract the ticker symbol (assuming it's the part after 'stocks_' and before '_alpha_vantage')
            ticker = filename.split('_')[3]  # Adjust this based on your naming pattern

            # Read the CSV file into a dataframe
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)  # You may want to adjust this if your files are not CSV

            # Add the dataframe to the dictionary with the ticker as the key
            dataframes_dict[ticker] = df

    return dataframes_dict

def load_dataframes_fake_news_from_folder(folder_path, endswith='data_cleaned_fake_news.csv', key_place = 0):
    """
    Load dataframes from a folder into a dictionary.
    The keys of the dictionary are the first words of the filenames (before the first underscore).
    The values are the dataframes loaded from the CSV files.

    Parameters:
        - folder_path (str): The path to the folder containing the CSV files.
        - endswith (str): The suffix of the filenames to look for. Default is 'data_cleaned_fake_news.csv'.
        - key_place (int): The index of the part of the filename to use as the key. Default is 0.

    Returns:
        - dict: A dictionary where the keys are the first words of the filenames and the values are dataframes.
    """	

    dataframes_dict = {}

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(endswith):
            # Extract the first word of the filename (before the first underscore)
            key = filename.split('_')[key_place]  # Adjust this based on your naming pattern

            # Read the CSV file into a dataframe
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)  # You may want to adjust this if your files are not CSV

            # Add the dataframe to the dictionary with the first word as the key
            dataframes_dict[key] = df

    return dataframes_dict

def map_pos_tag(word):
    """
    Map POS tag to first character lemmatize() accepts.
    WordNet lemmatizer requires the first character of the POS tag.
    This function maps the POS tag to the first character of the tag.

    Parameters:
        - word (str): The word to be lemmatized.

    Returns:
        - str: The first character of the POS tag.
    """

    tag = nltk.pos_tag([word])[0][1][0].upper() # get the first character of the POS tag
        # [0] is the first word in the list
        # [1] is the POS tag of the word
        # [0] is the first character of the POS tag
    tag_dict = { # dictionary to map POS tags
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN) # return the value of the key or the default value

def text_processing(dataframes_dict, lemmatizer, text_column='title'):
    """
    Process text data in the DataFrames.
    This function performs the following operations:
    1. Remove contractions from the text.
    2. Tokenize the text into words.
    3. Remove punctuation and special characters.
    4. Lemmatize the words.
    5. Remove stop words.
    6. Join the words back into a single string.
    7. Create a new column with the processed text.

    Parameters:
        - dataframes_dict (dict): A dictionary where the keys are ticker symbols and the values are dataframes.
        - lemmatizer: The lemmatizer to use for lemmatization.
        - text_column (str): The name of the column containing the text to be processed. Default is 'title'.

    Returns:
        - dict: A dictionary where the keys are ticker symbols and the values are dataframes with the processed text.
    """

    # Iterate through the dictionary of DataFrames
    for key, df in dataframes_dict.items():
        # Ensure the text column exists in the DataFrame
        if text_column in df.columns:

            # Remove contractions
            df[text_column] = df[text_column].apply(lambda x: contractions.fix(x))

            # Crete a new column with the tokenized text
            df['text_column_tokens'] = df[text_column].apply(lambda x: nltk.word_tokenize(x.lower()))

            # Remove punctuation and special characters
            df['text_without_puntutation'] = df['text_column_tokens'].apply(lambda x: [word for word in x if word.replace(',', '').replace('.', '').replace('%', '').replace('-', '').isalnum() or '%' in word or '-' in word])
            df['lemmatizers'] = df['text_without_puntutation'].apply(lambda x: [lemmatizer.lemmatize(word, map_pos_tag(word)) for word in x])

            df['without_stop_words'] = df['lemmatizers'].apply(lambda x: [word for word in x if word not in stop_words])

            # Join the tokens back into a single string
            df['text_column'] = df['without_stop_words'].apply(lambda x: ' '.join(x))
        else:
            print(f"Warning: '{text_column}' not found in DataFrame with key '{key}'.")

    return dataframes_dict 


def apply_vader_polarity_score(dataframes_dict, text_column='title'):
    """
    Apply VADER sentiment analysis to the text data in the DataFrames.
    This function performs the following operations:
    1. Apply VADER sentiment analysis to the specified text column.
    2. Create new columns for the compound, positive, neutral, and negative scores.

    Parameters:
        - dataframes_dict (dict): A dictionary where the keys are ticker symbols and the values are dataframes.
        - text_column (str): The name of the column containing the text to be analyzed. Default is 'title'.

    Returns:
        - dict: A dictionary where the keys are ticker symbols and the values are dataframes with the sentiment scores.
    """

    # Iterate through the dictionary of DataFrames
    for key, df in dataframes_dict.items():
        # Ensure the text column exists in the DataFrame
        if text_column in df.columns:
            # Apply VADER sentiment analysis and get the polarity scores
            df['compound_score'] = df[text_column].apply(lambda x: vd.polarity_scores(str(x))['compound'])
            df['positive_score'] = df[text_column].apply(lambda x: vd.polarity_scores(str(x))['pos'])
            df['neutral_score'] = df[text_column].apply(lambda x: vd.polarity_scores(str(x))['neu'])
            df['negative_score'] = df[text_column].apply(lambda x: vd.polarity_scores(str(x))['neg'])
        else:
            print(f"Warning: '{text_column}' not found in DataFrame with key '{key}'.")

    return dataframes_dict

def convert_sentiment_labels(data_dict):
    """
    Convert sentiment labels in the DataFrames to numerical values.
    This function performs the following operations:
    1. Check if the 'sentiment_label' column exists in the DataFrame.
    2. If it exists, convert the labels to numerical values.
    3. Update the DataFrame in the dictionary.

    Parameters:
        - data_dict (dict): A dictionary where the keys are ticker symbols and the values are dataframes.

    Returns:
        - dict: A dictionary where the keys are ticker symbols and the values are dataframes with converted sentiment labels.
    """
    
    for key, df in data_dict.items():
        if 'sentiment_label' in df.columns:
            df['sentiment_label'] = df['sentiment_label'].apply(lambda x: int(x.split('_')[1]))
        data_dict[key] = df
    return data_dict