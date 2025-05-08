import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import matplotlib as mpl
import os
import numpy as np

# ==============================
# Directory Setup
# ==============================

# Define the directory name for saving images
OUTPUT_DIR = "../images"

# Check if the directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================
# Plot Styling & Customization
# ==============================

# Set a Minimalist Style
sns.set_style("whitegrid")

# Customize Matplotlib settings for a modern look
mpl.rcParams.update({
    'axes.edgecolor': 'grey',       
    'axes.labelcolor': 'black',     
    'xtick.color': 'black',         
    'ytick.color': 'black',         
    'text.color': 'black'           
})

# ==============================
# Font Configuration
# ==============================

# Path to the custom font file
FONT_PATH = '../nlp_scripts/fonts/Montserrat-Regular.ttf'

# Add the font to matplotlib's font manager
font_manager.fontManager.addfont(FONT_PATH)

# Set the font family to Montserrat
plt.rcParams['font.family'] = 'Montserrat'


def fake_news_distribution(dataframe, word = 'training'):
    """
    Plots the distribution of fake news articles in the dataset.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the news articles with a 'fake_news' column.
    """
    # Count the number of fake and real articles
    fake_counts = dataframe['fake_news'].value_counts()
    
    # Create a bar plot for the distribution of fake vs real articles
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fake_counts.index, y=fake_counts.values, palette='rocket', hue = fake_counts.index)
    
    # Set labels and title
    plt.xlabel('Fake News (0 = Not Fake, 1 = Fake)', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.title('Distribution of Fake News Articles', fontsize=16)
    plt.legend().set_visible(False)

    # Show the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Custom grid on y-axis

    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"Distribution_of_Fake_News_Articles_{word}.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    plt.tight_layout()
    plt.show()

def plot_title_word_count_histogram(df, title_column='title', text_column='text_column', bins=50):
    """
    Plots a single histogram comparing the word count in the specified title column and text column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    title_column (str): The name of the column containing the titles. Default is 'title'.
    text_column (str): The name of the column containing the text. Default is 'text_column'.
    bins (int): The number of bins for the histogram. Default is 50.
    """
    # Count the number of words in each title and text
    word_counts_title = df[title_column].apply(lambda x: len(str(x).split()))
    word_counts_text = df[text_column].apply(lambda x: len(str(x).split()))

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts_title, bins=bins, color='blue', alpha=0.6, label=f'Before')
    plt.hist(word_counts_text, bins=20, color='green', alpha=0.6, label=f'After')
    plt.title('Histogram of Word Counts Before and After Text Preprocessing')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.7, linestyle='--', which='both')
    plt.savefig(os.path.join(OUTPUT_DIR, f"word_count_before_and_after.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    plt.tight_layout()
    plt.show()

# A function that plot how many news per stock
def plot_news_per_stock(dataframes_dict):
    """
    Plots the number of news articles per stock ticker.

    Parameters:
        - dataframes_dict (dict): A dictionary where keys are stock tickers and values are pandas DataFrames.
    
    Returns:
        - None: Displays a bar plot of the number of news articles per stock ticker.
    """
    # Create a DataFrame to hold the counts of news articles for each stock
    news_counts = {stock: len(df) for stock, df in dataframes_dict.items()}
    
    # Convert the dictionary to a DataFrame for easier plotting
    news_counts_df = pd.DataFrame(list(news_counts.items()), columns=['Stock', 'Number of News Articles'])
    
    # Create a bar plot for the number of news articles per stock
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Stock', y='Number of News Articles', data=news_counts_df, palette='rocket', hue='Stock')
    
    # Set labels and title
    plt.xlabel('Stock Ticker', fontsize=12)
    plt.ylabel('Number of News Articles', fontsize=12)
    plt.title('Number of News Articles per Stock', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Show the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Custom grid on y-axis

    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, "Number_of_News_Articles_per_Stock.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    plt.tight_layout()
    plt.show()

def plot_sentiment_scores_for_stock(dataframes_dict, score_columns = ['compound_score', 'positive_score', 'neutral_score', 'negative_score']):
    """
    Plots box plots for sentiment scores for each stock, with separate plots for each sentiment score (compound, positive, neutral, negative).
    
    Parameters:
        - dataframes_dict (dict): A dictionary where keys are stock tickers and values are pandas DataFrames.
        - score_columns (list): List of sentiment score columns to plot. Default is ['compound_score', 'positive_score', 'neutral_score', 'negative_score'].

    Returns:
        - None: Displays box plots of sentiment scores for each stock.
    """

    # Define a grid for the plots
    num_plots = len(score_columns)
    rows = int(np.ceil(num_plots / 2))
    cols = 2
    
    # Create a grid for plotting the sentiment scores 
    fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Loop through each sentiment score and create a plot
    for i, score in enumerate(score_columns):
        # Create a list of stock tickers (keys) and their corresponding sentiment values
        stock_names = list(dataframes_dict.keys())
        sentiment_values = [df[score] for df in dataframes_dict.values()]
        
        # Flatten the list of sentiment values for plotting
        sentiment_values_flat = [value for sublist in sentiment_values for value in sublist]
        stock_names_repeated = [stock for stock, values in zip(stock_names, sentiment_values) for _ in values]
        
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'Stock': stock_names_repeated,
            'Sentiment Value': sentiment_values_flat
        })
        
        # Plot the data
        sns.boxplot(x='Stock', y='Sentiment Value', hue='Stock', data=plot_df, palette='rocket', ax=axes[i])
        
        # Set the title for the plot (the name of the sentiment score)
        axes[i].set_title(score.replace('_', ' ').capitalize())
        axes[i].set_xlabel('Score', fontsize=12)  # You can customize this text
        axes[i].set_ylabel(f'{score.replace("_", " ").title()} Sentiment Value', fontsize=12)  # Use the key dynamically in ylabel
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)  # Custom grid on x-axis

        #axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    # Add a main title for the entire figure
    fig.suptitle('VADER Sentiment Score Distributions for Different Stocks Headlines', fontsize=16)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, "VADER_sentiment_score_per_stock_box_plot.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    plt.show()

def plot_sentiment_scores_for_fake(dataframes_dict, score_columns = ['compound_score', 'positive_score', 'neutral_score', 'negative_score']):
    """
    Plots box plots for sentiment scores for each stock, with separate plots for each sentiment score (compound, positive, neutral, negative).

    Parameters:
        - dataframes_dict (dict): A dictionary where keys are stock tickers and values are pandas DataFrames.
        - score_columns (list): List of sentiment score columns to plot. Default is ['compound_score', 'positive_score', 'neutral_score', 'negative_score'].

    Returns:
        - None: Displays box plots of sentiment scores for each stock.
    """

    # Extract the 'training' DataFrame from the dictionary
    df = dataframes_dict['training']

    # Define a grid for the plots
    num_plots = len(score_columns)
    rows = int(np.ceil(num_plots / 2))
    cols = 2
    
    # Create a grid for plotting the sentiment scores 
    fig, axes = plt.subplots(2, rows, figsize=(15, rows * 5))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Loop through each sentiment score and create a plot
    for i, score in enumerate(score_columns):
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'is_fake': df['fake_news'],  # Use the is_fake column (0 or 1)
            'Sentiment Value': df[score]  # The sentiment values for the current score
        })
        
        # Plot the data
        sns.boxplot(x='is_fake', y='Sentiment Value', hue='is_fake', data=plot_df, palette='rocket', ax=axes[i], legend=False)
        
        # Set the title for the plot (the name of the sentiment score)
        axes[i].set_title(score.replace('_', ' ').capitalize())
        axes[i].set_xlabel('Is Fake (0 = Not Fake, 1 = Fake)', fontsize=12)  # Custom x-axis label
        axes[i].set_ylabel(f'{score.replace("_", " ").title()} Sentiment Value', fontsize=12)  # y-axis label
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)  # Custom grid on x-axis

    # Add a main title for the entire figure
    fig.suptitle('VADER sentiment Score Distributions for Fake vs. Not Fake Articles', fontsize=16)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, "VADER_sentiment_score_per_fake_news_box_plot.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    

    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    plt.show()

def plot_sentiment_scores(news_dict, word = 'stocks'):
    """
    Function to plot sentiment scores, normalize frequencies, and save the images.

    Parameters:
        - news_dict (dict): Dictionary where the key is a label and the value is a dataframe.
        - word (str): Word to be used in the title of the plot.

    Returns:
        - None: Displays and saves the plot.
    """

    # Define a grid for the plots
    num_plots = len(news_dict)
    rows = int(np.ceil(num_plots / 2))
    cols = 2

    plt.figure(figsize=(15, rows * 5))
    
    for idx, (key, df) in enumerate(news_dict.items()):
        
        # Create a subplot for each dataframe in the dictionary
        plt.subplot(rows, cols, idx + 1)
        
        # Plot sentiment score distributions with KDE
        sns.histplot(df['sentiment_score'], kde=True, color=sns.color_palette("rocket")[0], label='Sentiment Score')
        sns.histplot(df['roberta_neg'], kde=True, color=sns.color_palette("rocket")[2], label='Negative Score')
        sns.histplot(df['roberta_neu'], kde=True, color=sns.color_palette("rocket")[4], label='Neutral Score')
        sns.histplot(df['roberta_pos'], kde=True, color=sns.color_palette("rocket")[5], label='Positive Score')
        
        # Add labels and title
        plt.title(f"{key}")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Normalized Frequency")
        
        # Add legend
        plt.legend()
        
        # Grid settings
        plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Add a main title for the entire figure
    plt.suptitle(f'roBERTa Sentiment Score for {word}', fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"roBERTa_sentiment_Score_Distribution_All_{word.replace(' ','_')}.png"), 
                bbox_inches='tight', facecolor='none', transparent=True)
    
    # Show the plot
    plt.show()

def plot_sentiment_scores_for_stock_roberta(dataframes_dict, score_columns = ['sentiment_score', 'roberta_pos', 'roberta_neu', 'roberta_neg'], word='sentiments', showfliers=True):
    """
    Plots box plots for sentiment scores for each stock, with separate plots for each sentiment score (compound, positive, neutral, negative).
    
    Parameters:
        - dataframes_dict (dict): A dictionary where keys are stock tickers and values are pandas DataFrames.
        - score_columns (list): List of sentiment score columns to plot. Default is ['sentiment_score', 'roberta_pos', 'roberta_neu', 'roberta_neg'].

    Returns:
        - None: Displays box plots of sentiment scores for each stock.
    """

    num_plots = len(score_columns)
    rows = int(np.ceil(num_plots / 2))
    cols = 2
    
    
    # Create a 2x2 grid for plotting the sentiment scores (4 subplots)
    fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Loop through each sentiment score and create a plot
    for i, score in enumerate(score_columns):
        # Create a list of stock tickers (keys) and their corresponding sentiment values
        stock_names = list(dataframes_dict.keys())
        sentiment_values = [df[score] for df in dataframes_dict.values()]
        
        # Flatten the list of sentiment values for plotting
        sentiment_values_flat = [value for sublist in sentiment_values for value in sublist]
        stock_names_repeated = [stock for stock, values in zip(stock_names, sentiment_values) for _ in values]
        
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'Stock': stock_names_repeated,
            'Sentiment Value': sentiment_values_flat
        })
        
        # Plot the data
        sns.boxplot(x='Stock', y='Sentiment Value', hue='Stock', data=plot_df, palette='rocket', ax=axes[i], showfliers=showfliers)
        
        # Set the title for the plot (the name of the sentiment score)
        axes[i].set_title(score.replace('_', ' ').capitalize())
        axes[i].set_xlabel('Score', fontsize=12)  # You can customize this text
        axes[i].set_ylabel(f'{score.replace("_", " ").title()} Sentiment Value', fontsize=12)  # Use the key dynamically in ylabel
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)  # Custom grid on x-axis

        #axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    # Add a main title for the entire figure
    fig.suptitle(f'roBERTa {word.title()} Score Distributions for Different Stocks Headlines', fontsize=16)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"roBERTa_{word}_score_per_stock_box_plot.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    plt.show()

def plot_sentiment_scores_for_fake_roberta(dataframes_dict, score_columns = ['sentiment_score', 'roberta_pos', 'roberta_neu', 'roberta_neg'], word='sentiments', showfliers=True):
    """
    Plots box plots for sentiment scores for each stock, with separate plots for each sentiment score (compound, positive, neutral, negative).

    Parameters:
        - dataframes_dict (dict): A dictionary where keys are stock tickers and values are pandas DataFrames.
        - score_columns (list): List of sentiment score columns to plot. Default is ['sentiment_score', 'roberta_pos', 'roberta_neu', 'roberta_neg'].

    Returns:
        - None: Displays box plots of sentiment scores for each stock.
    """

    # Extract the 'training' DataFrame from the dictionary
    df = dataframes_dict['training']
    
    num_plots = len(score_columns)
    rows = int(np.ceil(num_plots / 2))
    cols = 2

    # Create a 2x2 grid for plotting the sentiment scores (4 subplots)
    fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Loop through each sentiment score and create a plot
    for i, score in enumerate(score_columns):
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'is_fake': df['fake_news'],  # Use the is_fake column (0 or 1)
            'Sentiment Value': df[score]  # The sentiment values for the current score
        })
        
        # Plot the data
        sns.boxplot(x='is_fake', y='Sentiment Value', hue='is_fake', data=plot_df, palette='rocket', ax=axes[i], legend=False, showfliers=showfliers)
        
        # Set the title for the plot (the name of the sentiment score)
        axes[i].set_title(score.replace('_', ' ').capitalize())
        axes[i].set_xlabel('Is Fake (0 = Not Fake, 1 = Fake)', fontsize=12)  # Custom x-axis label
        axes[i].set_ylabel(f'{score.replace("_", " ").title()} Sentiment Value', fontsize=12)  # y-axis label
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)  # Custom grid on x-axis

    # Add a main title for the entire figure
    fig.suptitle(f'roBERTa {word.title()} Score Distributions for Fake vs. Not Fake Articles', fontsize=16)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"roBERTa_{word}_score_per_fake_news_box_plot.png"), 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    

    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    plt.show()

def plot_sentiment_corr(news_dict, word = 'stocks'):
    """
    Function to plot the correlation matrix for sentiment features across multiple stocks.

    Parameters:
        - news_dict (dict): Dictionary where the key is a label (e.g., stock name) and the value is a dataframe.
        - word (str): Word to be used in the title of the plot.

    Returns:
        - None: Displays and saves the plot.
    """

    # Define the sentiment columns for correlation
    sentiment_columns = ['anger', 'fear', 'sadness', 'disgust', 'surprise', 'neutral', 'sentiment_score', 'roberta_pos', 'roberta_neu', 'roberta_neg']
    
    # Define a grid for the plots
    num_plots = len(news_dict)
    rows = int(np.ceil(num_plots / 2))
    cols = 2

    plt.figure(figsize=(15, rows * 6))

    for idx, (key, df) in enumerate(news_dict.items()):
        
        # Create a subplot for each dataframe in the dictionary
        plt.subplot(rows, cols, idx + 1)
        
        # Compute the correlation matrix
        corr_matrix = df[sentiment_columns].corr()
        
        # Plot the heatmap
        sns.heatmap(corr_matrix, annot=True, cmap="magma_r", cbar_kws={'label': 'Correlation Coefficient'}, fmt='.2f')
        
        # Add title and labels
        plt.title(f"{key}")
        
        # Grid settings
        plt.grid(False)

    # Add a main title for the entire figure
    plt.suptitle(f'Sentiment Feature Correlation for {word}', fontsize=16, y=1.001)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"Sentiment_Feature_Correlation_All_for_{word.replace(' ', '_')}.png"), 
                bbox_inches='tight', facecolor='none', transparent=True)
    
    # Show the plot
    plt.show()

def plot_grouped_sentiment_distribution_for_stocks(news_dict, sentiment_column='sentiment_label'):
    """
    Function to plot grouped bar plots for sentiment score distribution across multiple stocks.
    Each stock will have bars for each sentiment score (0, 1, 2).

    Parameters:
        - news_dict (dict): Dictionary where the key is a label (e.g., stock name) and the value is a dataframe.
        - sentiment_column (str): The column name in the dataframe containing the sentiment labels (0, 1, 2).

    Returns:
        - None: Displays and saves the plot.
    """

    # Prepare data for grouped bar plot
    sentiment_scores = [0, 1, 2]  # Possible sentiment scores
    all_data = []

    # Loop through each dataframe and gather the data for each sentiment score
    for key, df in news_dict.items():
        sentiment_counts = df[sentiment_column].value_counts().reindex(sentiment_scores, fill_value=0)
        
        # Collect data for plotting
        for score in sentiment_scores:
            all_data.append({
                'Stock': key,
                'Sentiment Score': score,
                'Count': sentiment_counts[score]
            })
    
    # Convert to DataFrame for easy plotting
    plot_data = pd.DataFrame(all_data)
    
    # Plot grouped bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='Sentiment Score', y='Count', hue='Stock', palette='Set2')

    # Customize the plot
    plt.title("Sentiment Label Distribution Across Stocks")
    plt.xlabel("Sentiment Lable")
    plt.ylabel("Count")
    plt.legend(title="Stock", bbox_to_anchor=(1.02, 1), loc='upper left')  # Move the legend outside
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Custom grid on y-axis
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{OUTPUT_DIR}/Grouped_Sentiment_Score_Distribution_stocks.png", bbox_inches='tight', facecolor='none', transparent=True)
    
    # Show the plot
    plt.show()

def plot_stacked_sentiment_distribution(news_dict, selected_key = 'training', sentiment_column='sentiment_label', is_fake_column='fake_news'):
    """
    Function to plot a stacked bar plot for sentiment score distribution, separated by 'fake' and 'not fake'.
    
    Parameters:
        - news_dict (dict): Dictionary where the key is a label (e.g., stock name) and the value is a dataframe.
        - selected_key (str): The key to select the specific dataframe from the dictionary.
        - sentiment_column (str): The column name in the dataframe containing the sentiment labels (0, 1, 2).
        - is_fake_column (str): The column name in the dataframe indicating if the news is fake or not (0 = Not Fake, 1 = Fake).

    Returns:
        - None: Displays and saves the plot.
    """

    # Select the dataframe corresponding to the selected_key
    df = news_dict[selected_key]
    
    # Group by sentiment label and is_fake column, count the occurrences
    sentiment_counts = df.groupby([sentiment_column, is_fake_column]).size().unstack(fill_value=0)
    
    # Plot the stacked bar chart
    sentiment_counts.plot(kind='bar', stacked=True, figsize=(8, 6), color=['lightgreen', 'salmon'])
    
    # Customize the plot
    plt.title(f"Sentiment Score Distribution for {selected_key} (Fake vs Non-Fake)")
    plt.xlabel("Sentiment Label")
    plt.xticks(rotation=0)  # Rotate x-ticks for better readability
    plt.ylabel("Count")
    plt.legend(["Not Fake", "Fake"], loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Custom grid on y-axis
    
    # Adjust layout and show
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{OUTPUT_DIR}/Sentiment_Score_Distribution_Stacked_{selected_key}.png", bbox_inches='tight', facecolor='none', transparent=True)
    
    # Show the plot
    plt.show()

def plot_sentiment_difference_histograms(news_dict, word='stocks'):
    """
    Function to plot histograms of absolute differences between VADER and roBERTa sentiment scores.

    Parameters:
        - news_dict (dict): Dictionary where the key is a label (e.g., stock name) and the value is a dataframe.
        - word (str): Word to be used in the title of the plot.

    Returns:
        - None: Displays and saves the plot.
    """

    diff_neg = []
    diff_neu = []
    diff_pos = []

    # Gather differences
    for key, df in news_dict.items():
        if {'negative_score', 'roberta_neg', 'neutral_score', 'roberta_neu', 'positive_score', 'roberta_pos'}.issubset(df.columns):
            diff_neg.extend(np.abs(df['negative_score'] - df['roberta_neg']))
            diff_neu.extend(np.abs(df['neutral_score'] - df['roberta_neu']))
            diff_pos.extend(np.abs(df['positive_score'] - df['roberta_pos']))

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    sns.histplot(diff_neg, bins=30, kde=True, ax=axes[0], color='red')
    axes[0].set_title('Absolute Difference - Negative Sentiment')
    axes[0].set_xlabel('Abs(VADER - roBERTa)')
    axes[0].set_ylabel('Frequency')

    sns.histplot(diff_neu, bins=30, kde=True, ax=axes[1], color='gray')
    axes[1].set_title('Absolute Difference - Neutral Sentiment')
    axes[1].set_xlabel('Abs(VADER - roBERTa)')

    sns.histplot(diff_pos, bins=30, kde=True, ax=axes[2], color='green')
    axes[2].set_title('Absolute Difference - Positive Sentiment')
    axes[2].set_xlabel('Abs(VADER - roBERTa)')

    plt.savefig(f"{OUTPUT_DIR}/Difference_between_models_score{word.replace(' ','_')}.png", bbox_inches='tight', facecolor='none', transparent=True)
    plt.tight_layout()
    plt.show()

def plot_sentiment_over_time(news_dict, sentiment_columns, time_column='date'):
    """
    Plots average sentiment scores over time per sentiment column for multiple stocks.
    
    Parameters:
        - news_dict (dict): Dictionary where the key is a label (e.g., stock name) and the value is a dataframe.
        - sentiment_columns (list): List of sentiment columns to plot over time.
        - time_column (str): The column name in the dataframe containing the date/time information.

    Returns:
        - None: Displays and saves the plot.
    """

    # Filter dates: only keep data between April 1, 2022 and April 1, 2025
    start_date = pd.to_datetime("2022-04-01")
    end_date = pd.to_datetime("2025-04-01")

    num_sentiments = len(sentiment_columns)
    cols = 2
    rows = (num_sentiments + 1) // 2

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True)
    axs = axs.flatten()

    for i, sentiment in enumerate(sentiment_columns):
        ax = axs[i]
        for stock, df in news_dict.items():

            if sentiment in df.columns and time_column in df.columns:
                df_sorted = df.copy()
                # Convert to datetime and filter dates
                df_sorted[time_column] = pd.to_datetime(df_sorted[time_column], errors='coerce')

                df_sorted = df_sorted[(df_sorted[time_column] >= start_date) & (df_sorted[time_column] <= end_date)]

                # Sort by date and group by day
                df_sorted = df_sorted.sort_values(time_column)

                # Group by date and get mean sentiment
                daily_avg = df_sorted.groupby(df_sorted[time_column].dt.date)[sentiment].mean()
                daily_avg = daily_avg.reset_index()
                daily_avg = daily_avg.rename(columns={time_column: 'date'})

                # Convert 'date' to datetime
                daily_avg['date'] = pd.to_datetime(daily_avg['date'])

                # Set 'date' as index and resample monthly
                df_resample = daily_avg.set_index('date').resample('5ME').mean()

                df_resample[sentiment] = df_resample[sentiment].rolling(window=2, min_periods=1).mean()

                # Plot resampled data
                ax.plot(df_resample.index, df_resample[sentiment], label=stock)

        ax.set_title(f"{sentiment.replace('_',' ').title()}", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Score")
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend().set_visible(False)

    # Hide unused subplots if any
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    handles, labels = axs[0].get_legend_handles_labels()  # Get the legend from the first plot
    fig.legend(handles, labels, title="Stock", loc='upper right', bbox_to_anchor=(0.98, 1.02), ncol=2)

    # Add a main title for the entire figure
    fig.suptitle(f'Mean sentiments score over time', fontsize=16, y=1.001)

    plt.tight_layout()
    plt.show()

def plot_embedding_lengths_per_model(news_dict, length_columns, word='stocks'):
    """
    Plot separate histograms of embedding lengths per model using data from all stocks combined.

    Parameters:
        - news_dict (dict): Dictionary where the key is a label (e.g., stock name) and the value is a dataframe.
        - length_columns (list): List of columns containing the lengths of embeddings to plot.
        - word (str): Word to be used in the title of the plot.

    Returns:
        - None: Displays and saves the plot.
    """
    
    combined_lengths = {col: [] for col in length_columns}

    for df in news_dict.values():
        for col in length_columns:
            if col in df.columns:
                combined_lengths[col].extend(df[col].dropna())

    # Grid settings
    num_plots = len(length_columns)
    rows = 3
    cols = 2

    plt.figure(figsize=(12, 15))  # Adjust based on 2 cols × 3 rows

    for idx, col in enumerate(length_columns, 1):
        plt.subplot(rows, cols, idx)
        sns.histplot(combined_lengths[col], kde=True, color='skyblue')
        plt.title(f"{col.replace('_', ' ').title()}")
        plt.xlabel("Length")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.5)

    # In case there are fewer than 6 plots, hide the empty ones
    for idx in range(len(length_columns) + 1, rows * cols + 1):
        plt.subplot(rows, cols, idx)
        plt.axis('off')

    # Add a main title for the entire figure
    plt.suptitle(f'Embedding Length Distribution', fontsize=16, y=1.001)
    plt.savefig(f"{OUTPUT_DIR}/Embedings_{word.replace(' ','_')}.png", bbox_inches='tight', facecolor='none', transparent=True)
   
    plt.tight_layout()
    plt.show()

def plot_stock_prices_over_time(news_dict, time_column='date', price_column=' close'):
    """
    Plots stock prices over time for multiple stocks.
    
    Args:
        news_dict (dict): Dictionary of DataFrames. Keys = stock names.
        time_column (str): Name of the time column in the DataFrame (default='date').
        price_column (str): Name of the price column to plot (default='close').
    """

    # Filter dates: only keep data between April 1, 2022 and April 1, 2025
    start_date = pd.to_datetime("2022-04-01")
    end_date = pd.to_datetime("2025-04-01")

    # Set up the number of rows and columns for the subplots based on the number of stocks
    num_stocks = len(news_dict)
    cols = 2
    rows = (num_stocks + 1) // 2

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.flatten()

    for i, (stock, df) in enumerate(news_dict.items()):
        ax = axs[i]


        if price_column in df.columns and time_column in df.columns:

            df_sorted = df.copy()

            # Convert to datetime and filter dates
            df_sorted[time_column] = pd.to_datetime(df_sorted[time_column], errors='coerce')

            df_sorted = df_sorted[(df_sorted[time_column] >= start_date) & (df_sorted[time_column] <= end_date)]


            # Sort by date
            df_sorted = df_sorted.sort_values(time_column)

            # Set 'date' as index and resample monthly
            df_resample = df_sorted.set_index(time_column).resample('ME').mean()

            # Apply smoothing if needed (using rolling window)
            #df_resample[price_column] = df_resample[price_column].rolling(window=2, min_periods=1).mean()

            # Plot the price data
            ax.plot(df_resample.index, df_resample[price_column], label=stock)

        ax.set_title(f"{stock}", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{price_column.capitalize()} Price")
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend().set_visible(False)

    # Hide unused subplots if any
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    handles, labels = axs[0].get_legend_handles_labels()  # Get the legend from the first plot
    fig.legend(handles, labels, title="Stock", loc='upper right', bbox_to_anchor=(0.98, 1.02), ncol=2)

    # Add a main title for the entire figure
    fig.suptitle(f'Stock Prices Over Time', fontsize=16, y=1.001)

    plt.tight_layout()
    plt.show()