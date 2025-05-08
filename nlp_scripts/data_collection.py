'''This script collects financial news sentiment data and stock price data using the Alpha Vantage API.
It also scrapes news articles from Markets Insider and Google News for a list of stock symbols.'''

# Loading necessary libraries

# Environment Variables
import os 
from dotenv import load_dotenv
import time
import unicodedata
import re
import emoji

# Loading environment variables from .env file
api_key_alpha = os.getenv("API_KEY_ALPHA_VANTAGE")
BEARER_TOKEN = os.getenv("BEARER_TOKEN_X") 
load_dotenv()

# Data Manipulation 
import pandas as pd

# Web Scraping
import requests
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
from datetime import datetime, timedelta
import yfinance as yf

# Data folders
# Define the paths
base_folder = os.path.join(os.path.dirname(__file__), 'data')
sub_folder = os.path.join(base_folder, 'data_to_clean')

# Ensure the folders exist
os.makedirs(sub_folder, exist_ok=True)

# ==============================
# Data Collection Functions
# ==============================

def get_news_sentiment_alpha_vantage(symbol, start_date, end_date, limit = 1000):
    """
    Fetches news sentiment data for a given stock symbol from Alpha Vantage API.
    The data is filtered based on the provided start and end dates.

    Parameters:
        - symbol (str): The stock symbol to fetch news sentiment data for.
        - start_date (str): The start date for filtering news articles (format: YYYY-MM-DD).
        - end_date (str): The end date for filtering news articles (format: YYYY-MM-DD).
        - limit (int): The maximum number of news articles to fetch (default: 1000).

    Returns:
        - pd.DataFrame: A DataFrame containing the news sentiment data for the specified stock symbol.
    """

    # URL for Alpha Vantage API to fetch news sentiment data
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key_alpha}&limit={limit}&time_from={start_date}&time_to={end_date}"
    
    # Send GET request to the API
    response = requests.get(url)

    # Response
    data = response.json()
    
    # Extract relevant news articles
    if "feed" in data:

        # Get the news articles from the response
        news_articles = data["feed"]

        # Convert to DataFrame
        df = pd.DataFrame(news_articles)

        # Convert timestamps to datetime format
        df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")
        return df
    else:
        return pd.DataFrame()
    
def get_stock_data_alpha_vantage(symbol):
    """
    Fetches daily stock price data for a given stock symbol from Alpha Vantage API.

    Parameters:
        - symbol (str): The stock symbol to fetch data for.

    Returns:
        - pd.DataFrame: A DataFrame containing the daily stock price data for the specified stock symbol.
    """

    # URL for Alpha Vantage API to fetch daily stock price data
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key_alpha}&outputsize=full'

    # Send GET request to the API
    response = requests.get(url)

    # Response
    data = response.json()

    # Extract relevant stock price data
    if "Time Series (Daily)" in data:

        # Convert the data to a DataFrame
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={"index": "date"})
        #df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df
    else:
        return pd.DataFrame()

def collect_data_alpha_vantage(stock_symbols, start_date, end_date):   
    """
    Collects news sentiment data and stock price data for a list of stock symbols using Alpha Vantage API.

    Parameters:
        - stock_symbols (list): A list of stock symbols to collect data for.
        - start_date (str): The start date for filtering news articles (format: YYYY-MM-DD).
        - end_date (str): The end date for filtering news articles (format: YYYY-MM-DD).

    Returns:
        - dict: A dictionary containing news sentiment data and stock price data for each stock symbol.
    """

    # Initialize an empty dictionary to store data for each stock symbol
    stock_data_dict = {} 

    # Get the number of stocks to process
    number_of_stocks = len(stock_symbols)

    # Loop through all stocks and collect data
    for stock in tqdm(stock_symbols, total = number_of_stocks):
        print(f"Fetching news and stock data for {stock}...")
        
        # Get news sentiment data
        news_data = get_news_sentiment_alpha_vantage(stock, start_date, end_date)
        if news_data is not None and not news_data.empty:
            print(f"\u2705 Retrieved {len(news_data)} news articles for {stock}")
        else:
            print(f"\u274c No news articles found for {stock}")
        
        # Get stock price data
        stock_data = get_stock_data_alpha_vantage(stock)
        if stock_data is not None and not stock_data.empty:
            print(f"\u2705 Retrieved stock data for {stock} from {start_date} to {end_date}")
        else:
            print(f"\u274c No stock data found for {stock}")

        print("-" * 50)

        # Store the news sentiment data and stock price data in the dictionary
        stock_data_dict[stock] = {
            "news": news_data,
            "stocks": stock_data
        }

    return stock_data_dict

def get_news_from_markets_insider(stock_symbols):
    """
    Scrapes news articles from Markets Insider for a list of stock symbols.

    Parameters:
        - stock_symbols (list): A list of stock symbols to scrape news articles for.

    Returns:
        - dict: A dictionary containing news articles for each stock symbol.
    """

    # Initialize an empty dictionary to store news articles for each stock symbol
    news_data_dict = {}

    # Total number of stocks to process
    number_of_stocks = len(stock_symbols)

    # Loop through all stocks and scrape news articles
    for stock in tqdm(stock_symbols, total = number_of_stocks):

        #  Create a dictionary to store news articles for the current stock symbol
        df = pd.DataFrame(columns=["publish_date", "title", "source"])

        # Loop through each page of news articles (up to 230 pages)
        for page in range(1, 231):

            # Format the date into the stock and page number into the URL
            url = f"https://markets.businessinsider.com/news/{stock.lower()}-stock?p={page}"

            # Send request to Markets Insider
            response = requests.get(url, timeout=30)

            # Check if the response is successful (status code 200)
            if response.status_code != 200:
                print(f"Failed to retrieve data for {url}. Skipping.")
                # Move to the next page
                continue

            # Parse the page content using BeautifulSoup
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Find all article links in the Markets Insider page
            articles = soup.find_all('div', class_='latest-news__story')

            # Check if there are no articles on the page
            if not articles:
                print(f"No articles found for {stock} in page {page}. Skipping.")
            else:
                # Extract data for each article
                for article in articles:
                    # Extract the publish date, title, and source
                    publish_date = article.find('time', class_='latest-news__date').get('datetime')
                    title = article.find('a', class_='news-link').text
                    title = basic_cleanup(title)
                    source = "Markets Insider"

                    # Save the data to a DataFrame
                    new_data = pd.DataFrame([[publish_date, title, source]], columns=df.columns)

                    # Check if new_data is not empty
                    if not new_data.isnull().values.all():
                        df = pd.concat([df, new_data], ignore_index=True)

            # Sleep for 10 seconds to avoid overwhelming the server 
            time.sleep(10)   

        # Save the news articles to the dictionary
        news_data_dict[stock] = df

        # Check if the DataFrame is not empty
        if df is not None and not df.empty:
            print(f"\u2705 Retrieved {len(df)} news articles for {stock} from Markets Insider")
    
    return news_data_dict

def get_google_news_articles(stock_names, start_date, end_date, number_of_articles=10):
    """
    Scrapes news articles from Google News for a list of stock names within a specified date range.

    Parameters:
        - stock_names (list): A list of stock symbols to scrape news articles for.
        - start_date (datetime): The start date for filtering news articles.
        - end_date (datetime): The end date for filtering news articles.
        - number_of_articles (int): The maximum number of articles to scrape per stock symbol.

    Returns:
        - dict: A dictionary containing news articles for each stock symbol.
    """

    # Initialize an empty dictionary to store news articles for each stock symbol
    news_data_dict = {}

    # Get the number of stocks to process
    total_stocks = len(stock_names)

    # Loop through each stock in the list
    for stock in tqdm(stock_names, total=total_stocks):

        # Create a dictionary to store news articles for the current stock symbol
        df = pd.DataFrame(columns=["publish_date", "title", "source"])

        # Loop through each date in the range
        current_date = datetime.strptime(start_date, "%Y%m%dT%H%M").date()
        end_date_formated = datetime.strptime(end_date, "%Y%m%dT%H%M").date()
        while current_date <= end_date_formated:

            # url for Google News search
            url = f"https://news.google.com/search?q={stock}%20since%3A{current_date}%20until%3A{current_date}"

            # Send request to Google News
            response = requests.get(url)
            
            # Check if the response is successful (status code 200)
            if response.status_code != 200:
                print(f"Failed to retrieve data for {url}. Skipping.")
                current_date += timedelta(days=1)  # Move to the next day
                continue

            # Parse the page content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all article links in the Google News page
            articles = soup.find_all('a', {'class': 'JtKRv'})

            # Check if there are no articles on the page
            if not articles:
                print(f"No articles found for {stock} on date {current_date}. Skipping.")
                current_date += timedelta(days=1)
                continue
            else:
                # Extract data for each article
                for article in articles[:number_of_articles]:
                    title = article.get_text()
                    title = basic_cleanup(title)
                    publish_date = current_date
                    source = 'Google News'

                    # Sleep for 2 seconds to avoid overwhelming the server
                    time.sleep(2)  
                    
                    # Ensure valid data before appending to the DataFrame
                    new_data = pd.DataFrame([[publish_date, title, source]], columns=df.columns)

                    # Check if new_data is not empty
                    if not new_data.isnull().values.all():
                        df = pd.concat([df, new_data], ignore_index=True)

            # Move to the next day  
            current_date += timedelta(days=1)

        # Save the news articles to the dictionary
        news_data_dict[stock] = df

        # Check if the DataFrame is not empty
        if df is not None and not df.empty:
            print(f"\u2705 Retrieved {len(df)} news articles for {stock} from Google News")

    return news_data_dict

def basic_cleanup(text):
    """
    Basic text cleanup function to remove unwanted characters and normalize whitespace.

    Parameters:
        - text (str): The input text to be cleaned.

    Returns:
        - str: The cleaned text.
    """
    
    # Normalize unicode characters to ASCII and remove non-ASCII characters
    cleaned_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Remove unwanted characters and normalize whitespace
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"()\s]', '', cleaned_text)  
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def get_yahoo_stock_data(stock_symbols, start_date, end_date):
    """
    Fetches stock price data from Yahoo Finance for a list of stock symbols within a specified date range.

    Parameters:
        - stock_symbols (list): A list of stock symbols to fetch data for.
        - start_date (str): The start date for filtering stock prices (format: YYYY-MM-DD).
        - end_date (str): The end date for filtering stock prices (format: YYYY-MM-DD).

    Returns:
        - dict: A dictionary containing stock price data for each stock symbol.
    """

    # Initialize an empty dictionary to store stock price data for each stock symbol
    stock_data_dict = {}

    # Modify the date format to match Yahoo Finance requirements
    start_date = datetime.strptime(start_date, "%Y%m%dT%H%M").strftime("%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y%m%dT%H%M").strftime("%Y-%m-%d")

    # Get the number of stocks to process
    total_stocks = len(stock_symbols)

    # Loop through each stock in the list
    for stock in tqdm(stock_symbols, total=total_stocks):
        try:
            # Download daily historical data using yfinance
            stock_data = yf.download(stock, start=start_date, end=end_date, interval='1d')

            # Check if the DataFrame is not empty
            if not stock_data.empty:
                stock_data.reset_index(inplace=True)
                stock_data_dict[stock] = stock_data
                print(f"\u2705 Retrieved stock data for {stock} from Yahoo Finance")
            else:
                print(f"\u274c No stock data found for {stock}")
        except Exception as e:
            print(f"Error retrieving data for {stock}: {e}")

    return stock_data_dict
